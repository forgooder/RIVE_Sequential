#!/usr/bin/env python3
"""Create deterministic synthetic caches for an end-to-end contract smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


RELATIONS = 4
CANDIDATES = 51
ENDPOINT_DIM = 7 * RELATIONS + 4
SHELL_DIM = 12 * RELATIONS + 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_split(rng: np.random.Generator, queries: int) -> tuple[dict, dict]:
    relations = rng.integers(0, RELATIONS, size=queries, dtype=np.int64)
    directions = np.arange(queries, dtype=np.int64) % 2
    query_ids = np.arange(queries, dtype=np.int64) // 2
    positive = rng.integers(0, CANDIDATES, size=queries, dtype=np.int64)

    triples = np.empty((queries, CANDIDATES, 3), dtype=np.int64)
    entities = np.arange(CANDIDATES, dtype=np.int64)
    for index, (relation, direction) in enumerate(zip(relations, directions, strict=True)):
        if direction == 0:
            triples[index, :, 0] = CANDIDATES + index
            triples[index, :, 1] = entities
        else:
            triples[index, :, 0] = entities
            triples[index, :, 1] = CANDIDATES + index
        triples[index, :, 2] = relation

    labels = np.zeros((queries, CANDIDATES), dtype=np.float32)
    labels[np.arange(queries), positive] = 1.0
    valid = np.ones_like(labels, dtype=bool)
    base = rng.normal(0.0, 1.0, size=labels.shape).astype(np.float64)
    base[np.arange(queries), positive] += 0.35

    endpoint = rng.normal(
        0.0, 0.45, size=(queries, CANDIDATES, ENDPOINT_DIM)
    ).astype(np.float32)
    endpoint[..., 6 * RELATIONS + 4 :] = 0.0
    endpoint[
        np.arange(queries), :, 6 * RELATIONS + 4 + relations
    ] = 1.0
    endpoint_signal = (query_ids % 2) == 0
    endpoint[
        np.arange(queries)[endpoint_signal], positive[endpoint_signal], 0
    ] += 1.5

    shell = rng.normal(
        0.0, 0.35, size=(queries, CANDIDATES, SHELL_DIM)
    ).astype(np.float32)
    row = np.arange(queries)
    for block in range(6):
        shell[row, positive, block * RELATIONS + relations] += 3.0
        shell[
            row,
            positive,
            6 * RELATIONS + 5 + block * RELATIONS + relations,
        ] += 3.0

    backbone = {
        "triples": triples,
        "labels": labels,
        "valid_mask": valid,
        "base_scores": base,
        "endpoint_features": endpoint,
        "directions": directions,
        "query_ids": query_ids,
    }
    shell_cache = {
        "triples": triples,
        "query_ids": query_ids,
        "shell_features": shell,
    }
    return backbone, shell_cache


def write_cache(path: Path, arrays: dict[str, np.ndarray]) -> str:
    np.savez_compressed(path, **arrays)
    digest = sha256(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "npz_sha256": digest,
                "scope": "synthetic software contract only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()

    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"append-only output exists: {output} or {partial}")
    partial.mkdir(parents=True)

    rng = np.random.default_rng(args.seed)
    train, train_shell = make_split(rng, 64)
    valid, valid_shell = make_split(rng, 32)
    hashes = {
        "train_scores.npz": write_cache(partial / "train_scores.npz", train),
        "valid_scores.npz": write_cache(partial / "valid_scores.npz", valid),
        "train_shell.npz": write_cache(partial / "train_shell.npz", train_shell),
        "valid_shell.npz": write_cache(partial / "valid_shell.npz", valid_shell),
    }
    (partial / "MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "SYNTHETIC_CONTRACT_COMPLETE",
                "scientific_claim_authorized": False,
                "seed": args.seed,
                "relations": RELATIONS,
                "candidates": CANDIDATES,
                "hashes": hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output)
    print(json.dumps({"output": str(output), "hashes": hashes}, sort_keys=True))


if __name__ == "__main__":
    main()
