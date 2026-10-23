#!/usr/bin/env python3
"""Train the frozen Sequential RIVE-Ladder using source-only 1P+50N caches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rive_ladder_final.model import SequentialRiveLadder, SourceScale


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_verified(path: Path, required: tuple[str, ...]) -> tuple[dict, dict]:
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["npz_sha256"] != sha256(path):
        raise RuntimeError(f"cache SHA mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise KeyError(f"{path}: missing {sorted(missing)}")
        arrays = {key: archive[key] for key in archive.files}
    return arrays, manifest


def load_inputs(config: dict) -> tuple[dict, dict, np.ndarray, np.ndarray, dict]:
    required = (
        "triples", "labels", "valid_mask", "base_scores",
        "endpoint_features", "directions", "query_ids",
    )
    train, train_manifest = load_verified(resolve(config["train_cache"]), required)
    valid, valid_manifest = load_verified(resolve(config["valid_cache"]), required)
    shell_train, shell_train_manifest = load_verified(
        resolve(config["train_shell"]), ("shell_features", "triples", "query_ids")
    )
    shell_valid, shell_valid_manifest = load_verified(
        resolve(config["valid_shell"]), ("shell_features", "triples", "query_ids")
    )
    expected = int(config["candidate_count"])
    for name, arrays in (("train", train), ("valid", valid)):
        if arrays["labels"].shape[1] != expected:
            raise RuntimeError(f"{name}: candidate width is not {expected}")
        if not np.all((arrays["labels"] > 0.5).sum(axis=1) == 1):
            raise RuntimeError(f"{name}: each group must have one positive")
    for name, shell, arrays in (
        ("train", shell_train, train), ("valid", shell_valid, valid)
    ):
        np.testing.assert_array_equal(shell["triples"], arrays["triples"])
        np.testing.assert_array_equal(shell["query_ids"], arrays["query_ids"])
        values = np.asarray(shell["shell_features"], dtype=np.float32)
        if values.ndim == 4:
            values = values[:, :, 0]
        if values.shape[:2] != arrays["labels"].shape:
            raise RuntimeError(f"{name}: shell alignment failed")
        if name == "train":
            shell_train_values = values
        else:
            shell_valid_values = values
    manifests = {
        "train": train_manifest["npz_sha256"],
        "valid": valid_manifest["npz_sha256"],
        "train_shell": shell_train_manifest["npz_sha256"],
        "valid_shell": shell_valid_manifest["npz_sha256"],
    }
    return train, valid, shell_train_values, shell_valid_values, manifests


def estimate_scale(train: dict, config: dict) -> SourceScale:
    epsilon = float(config["normalization"]["epsilon"])
    stds = []
    for scores, mask in zip(train["base_scores"], train["valid_mask"], strict=True):
        values = np.asarray(scores, dtype=np.float64)[np.asarray(mask, dtype=bool)]
        stds.append(float(np.std(values)))
    positive = np.asarray(stds)[np.asarray(stds) > epsilon]
    if not len(positive):
        raise RuntimeError("source train base scores have degenerate query scales")
    low, high = config["normalization"]["clip_quantiles"]
    clip_low, clip_high = np.quantile(positive, (float(low), float(high)))
    return SourceScale(
        fallback=float(np.median(positive)),
        clip_low=max(float(clip_low), epsilon),
        clip_high=max(float(clip_high), max(float(clip_low), epsilon)),
        epsilon=epsilon,
    )


def normalize_numpy(arrays: dict, scale: SourceScale) -> np.ndarray:
    score = np.asarray(arrays["base_scores"], dtype=np.float64)
    mask = np.asarray(arrays["valid_mask"], dtype=bool)
    result = np.empty_like(score)
    for index, (row, valid) in enumerate(zip(score, mask, strict=True)):
        local = float(np.std(row[valid]))
        if local <= scale.epsilon:
            local = scale.fallback
        result[index] = row / np.clip(local, scale.clip_low, scale.clip_high)
    # Keep the monotone score transform in float64. Casting the normalized
    # anchor back to float32 can collapse distinct near-tied backbone scores
    # into an exact tie and violate conservative-rank identity.
    return result


def ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positive = np.asarray(labels).argmax(axis=1)
    positive_score = scores[np.arange(len(scores)), positive][:, None]
    return (scores >= positive_score).sum(axis=1)


def metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    rank = ranks(np.asarray(scores), np.asarray(labels))
    return {
        "mrr": float(np.mean(1.0 / rank)),
        "hits@1": float(np.mean(rank <= 1)),
        "hits@3": float(np.mean(rank <= 3)),
        "hits@5": float(np.mean(rank <= 5)),
        "hits@10": float(np.mean(rank <= 10)),
        "groups": int(len(rank)),
        "tie_policy": "conservative",
    }


def relation_macro(scores: np.ndarray, arrays: dict) -> float:
    positive = np.asarray(arrays["labels"]).argmax(axis=1)
    relation = np.asarray(arrays["triples"])[np.arange(len(positive)), positive, 2]
    return float(np.mean([
        metrics(scores[relation == value], arrays["labels"][relation == value])["mrr"]
        for value in np.unique(relation)
    ]))


def loader(length: int, batch_size: int, shuffle: bool, seed: int):
    return DataLoader(
        np.arange(length), batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), pin_memory=True,
    )


def trust_loss(correction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(correction.dtype)
    mean = (correction * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True).clamp_min(1)
    return (((correction - mean) * weight).square().sum() / weight.sum().clamp_min(1))


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    negative_infinity = torch.finfo(scores.dtype).min
    positive = (labels > 0.5) & mask
    return (
        torch.logsumexp(scores.masked_fill(~mask, negative_infinity), dim=1)
        - torch.logsumexp(scores.masked_fill(~positive, negative_infinity), dim=1)
    ).mean()


@torch.inference_mode()
def predict(model, arrays, shell, normalized, stage, batch_size, device):
    output_scores, endpoint_values, shell_values = [], [], []
    model.eval()
    relations = (arrays["endpoint_features"].shape[-1] - 4) // 7
    query_slice = slice(6 * relations + 4, 7 * relations + 4)
    for indices in loader(len(normalized), batch_size, False, 0):
        index = indices.numpy()
        anchor = torch.from_numpy(normalized[index]).to(device)
        endpoint = torch.from_numpy(arrays["endpoint_features"][index]).float().to(device)
        valid = torch.from_numpy(arrays["valid_mask"][index]).bool().to(device)
        endpoint_raw = model.endpoint_head(model.endpoint_encoder(endpoint)).squeeze(-1)
        endpoint_correction = model.endpoint_budget * torch.tanh(endpoint_raw)
        endpoint_correction = torch.where(valid, endpoint_correction, torch.zeros_like(endpoint_correction))
        scores = anchor + endpoint_correction
        shell_correction = torch.zeros_like(scores)
        if stage == 1:
            shell_tensor = torch.from_numpy(shell[index]).float().to(device)
            query = torch.from_numpy(arrays["endpoint_features"][index, :, query_slice]).float().to(device)
            shell_raw = model.shell_head(model.shell_encoder(shell_tensor, query)).squeeze(-1)
            shell_correction = model.shell_budget * torch.tanh(shell_raw)
            shell_correction = torch.where(valid, shell_correction, torch.zeros_like(shell_correction))
            scores = scores + shell_correction
        output_scores.append(scores.cpu().numpy())
        endpoint_values.append(endpoint_correction.cpu().numpy())
        shell_values.append(shell_correction.cpu().numpy())
    return tuple(map(np.concatenate, (output_scores, endpoint_values, shell_values)))


def fit_stage(model, train, valid, train_shell, valid_shell, train_norm, valid_norm, stage, config, device):
    training = config["training"]
    batch_size = int(training["batch_size"])
    epoch0, _, _ = predict(model, valid, valid_shell, valid_norm, stage, batch_size, device)
    previous_stage = 0 if stage == 1 else None
    anchor = (
        predict(model, valid, valid_shell, valid_norm, previous_stage, batch_size, device)[0]
        if previous_stage is not None else valid_norm
    )
    zero_error = float(np.max(np.abs(epoch0 - anchor)))
    if zero_error >= 1e-7:
        raise RuntimeError(f"stage{stage} epoch-zero identity failed: {zero_error}")
    parameters = (
        list(model.endpoint_encoder.parameters()) + list(model.endpoint_head.parameters())
        if stage == 0 else
        list(model.shell_encoder.parameters()) + list(model.shell_head.parameters()) + [model.shell_alpha]
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_scores = anchor.copy()
    best_metrics = metrics(best_scores, valid["labels"])
    best_key = (best_metrics["mrr"], best_metrics["hits@1"], best_metrics["hits@5"])
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, stale, history = 0, 0, []
    started = time.monotonic()
    relations = (train["endpoint_features"].shape[-1] - 4) // 7
    query_slice = slice(6 * relations + 4, 7 * relations + 4)
    for epoch in range(1, int(training["epochs_per_stage"]) + 1):
        model.train()
        losses = []
        for indices in loader(len(train_norm), batch_size, True, int(config["adapter_seed"]) + stage * 1000 + epoch):
            index = indices.numpy()
            base = torch.from_numpy(train_norm[index]).to(device)
            endpoint = torch.from_numpy(train["endpoint_features"][index]).float().to(device)
            mask = torch.from_numpy(train["valid_mask"][index]).bool().to(device)
            labels = torch.from_numpy(train["labels"][index]).float().to(device)
            endpoint_raw = model.endpoint_head(model.endpoint_encoder(endpoint)).squeeze(-1)
            endpoint_correction = model.endpoint_budget * torch.tanh(endpoint_raw)
            endpoint_correction = torch.where(mask, endpoint_correction, torch.zeros_like(endpoint_correction))
            score = base + endpoint_correction
            correction = endpoint_correction
            if stage == 1:
                shell_tensor = torch.from_numpy(train_shell[index]).float().to(device)
                query = torch.from_numpy(train["endpoint_features"][index, :, query_slice]).float().to(device)
                shell_raw = model.shell_head(model.shell_encoder(shell_tensor, query)).squeeze(-1)
                correction = model.shell_budget * torch.tanh(shell_raw)
                correction = torch.where(mask, correction, torch.zeros_like(correction))
                score = score + correction
            loss = listwise_loss(score, labels, mask)
            loss += float(training["residual_weight"]) * trust_loss(correction, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score, _, _ = predict(model, valid, valid_shell, valid_norm, stage, batch_size, device)
        current = metrics(score, valid["labels"])
        key = (current["mrr"], current["hits@1"], current["hits@5"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **current})
        if key > best_key:
            best_key, best_metrics = key, current
            best_scores = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, stale = epoch, 0
        else:
            stale += 1
            if stale >= int(training["patience"]):
                break
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "epoch0_max_abs_error": zero_error,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }


def bootstrap_mrr(labels, query_ids, before, after, samples, seed):
    before_rr, after_rr = 1.0 / ranks(before, labels), 1.0 / ranks(after, labels)
    unique = np.unique(query_ids)
    delta = np.asarray([
        (after_rr[query_ids == value] - before_rr[query_ids == value]).mean()
        for value in unique
    ])
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_delta_le_zero": float(np.mean(draws <= 0)),
        "clusters": int(len(unique)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config["status"] != "FROZEN_BEFORE_TRAINING":
        raise RuntimeError("training config is not frozen")
    if config["locked_test_access"] is not False:
        raise RuntimeError("locked test must remain closed")
    output = resolve(config["output"])
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"append-only output exists: {output} or {partial}")

    train, valid, train_shell, valid_shell, input_hashes = load_inputs(config)
    endpoint_dim = int(train["endpoint_features"].shape[-1])
    if (endpoint_dim - 4) % 7:
        raise RuntimeError("endpoint feature width is not 7R+4")
    relations = (endpoint_dim - 4) // 7
    if valid["endpoint_features"].shape[-1] != endpoint_dim:
        raise RuntimeError("endpoint feature width drift")
    expected_shell = 12 * relations + 10
    if train_shell.shape[-1] != expected_shell or valid_shell.shape[-1] != expected_shell:
        raise RuntimeError("shell feature width drift")

    seed_all(int(config["adapter_seed"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    scale = estimate_scale(train, config)
    train_norm, valid_norm = normalize_numpy(train, scale), normalize_numpy(valid, scale)
    if not np.array_equal(ranks(train["base_scores"], train["labels"]), ranks(train_norm, train["labels"])):
        raise RuntimeError("source train normalization changed ranks")
    if not np.array_equal(ranks(valid["base_scores"], valid["labels"]), ranks(valid_norm, valid["labels"])):
        raise RuntimeError("source validation normalization changed ranks")

    model = SequentialRiveLadder(endpoint_dim, relations).to(device)
    stage0 = fit_stage(model, train, valid, train_shell, valid_shell, train_norm, valid_norm, 0, config, device)
    model.freeze_stage0()
    stage1 = fit_stage(model, train, valid, train_shell, valid_shell, train_norm, valid_norm, 1, config, device)
    batch_size = int(config["training"]["batch_size"])
    stage0_scores, endpoint_correction, _ = predict(model, valid, valid_shell, valid_norm, 0, batch_size, device)
    final_scores, _, shell_correction = predict(model, valid, valid_shell, valid_norm, 1, batch_size, device)
    score_map = {
        "base": np.asarray(valid["base_scores"]),
        "stage0": stage0_scores,
        "sequential": final_scores,
    }
    metric_map = {}
    for name, score in score_map.items():
        metric_map[name] = metrics(score, valid["labels"])
        metric_map[name]["relation_macro_mrr"] = relation_macro(score, valid)

    bootstrap = {
        "stage0_minus_base": bootstrap_mrr(
            valid["labels"], valid["query_ids"], score_map["base"], score_map["stage0"],
            int(config["bootstrap"]["samples"]), int(config["bootstrap"]["seed"]),
        ),
        "sequential_minus_base": bootstrap_mrr(
            valid["labels"], valid["query_ids"], score_map["base"], score_map["sequential"],
            int(config["bootstrap"]["samples"]), int(config["bootstrap"]["seed"]) + 1,
        ),
        "sequential_minus_stage0": bootstrap_mrr(
            valid["labels"], valid["query_ids"], score_map["stage0"], score_map["sequential"],
            int(config["bootstrap"]["samples"]), int(config["bootstrap"]["seed"]) + 2,
        ),
    }

    partial.mkdir(parents=True)
    checkpoint = partial / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "scale": scale.__dict__,
            "endpoint_dim": endpoint_dim,
            "relations": relations,
            "adapter_seed": int(config["adapter_seed"]),
        },
        checkpoint,
    )
    score_path = partial / "source_validation_scores.npz"
    np.savez_compressed(
        score_path,
        labels=valid["labels"], query_ids=valid["query_ids"], directions=valid["directions"],
        scores_base=score_map["base"], scores_stage0=score_map["stage0"],
        scores_sequential=score_map["sequential"],
    )
    payload = {
        "protocol": config["protocol"],
        "claim_scope": "source validation only; no target or locked-test result",
        "dataset": config["dataset"],
        "backbone": config["backbone"],
        "adapter_seed": int(config["adapter_seed"]),
        "candidate_count": int(config["candidate_count"]),
        "locked_test_access": False,
        "identity": {
            "raw_to_normalized_rank_identity": True,
            "stage0_epoch0_max_abs_error": stage0["epoch0_max_abs_error"],
            "stage1_epoch0_max_abs_error": stage1["epoch0_max_abs_error"],
        },
        "scale": scale.__dict__,
        "metrics": metric_map,
        "bootstrap": bootstrap,
        "fit": {"stage0": stage0, "stage1": stage1},
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_after_stage0_freeze": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "shell_budget": float(model.shell_budget.detach().cpu()),
            "endpoint_saturation_rate": float(np.mean(np.abs(endpoint_correction) >= 0.475)),
            "shell_saturation_rate": float(
                np.mean(np.abs(shell_correction) >= 0.95 * float(model.shell_budget.detach().cpu()))
            ),
        },
        "compute": {
            "device": str(device),
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0,
            "stage0_seconds": stage0["elapsed_seconds"],
            "stage1_seconds": stage1["elapsed_seconds"],
        },
        "input_sha256": input_hashes,
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint),
        "scores_sha256": sha256(score_path),
    }
    result_path = partial / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    files = sorted(path for path in partial.iterdir() if path.is_file())
    (partial / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    os.replace(partial, output)
    print(json.dumps({"output": str(output), "metrics": metric_map, "bootstrap": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
