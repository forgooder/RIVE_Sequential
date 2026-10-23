#!/usr/bin/env python3
"""Analyze one append-only WWW full-ranking score artifact by fact cluster."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from rive_ladder_final.fullrank import (
    paired_fact_bootstrap,
    per_fact_rows,
    ranking_metrics,
    ranks_from_matrices,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = ("base", "stage0", "stage1_only", "sequential")
REQUIRED = (
    "fact_ids",
    "directions",
    "positive_triples",
    "positive_indices",
    "candidate_entity_ids",
    "valid_mask",
)


def atomic_output_directory(path: Path) -> tuple[Path, Path]:
    final = path.resolve()
    partial = final.with_name(final.name + ".partial")
    if final.exists() or partial.exists():
        raise FileExistsError(f"append-only output exists: {final} or {partial}")
    partial.mkdir(parents=True)
    return final, partial


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    score_path: Path,
    output: Path,
    *,
    samples: int = 10_000,
    seed: int = 20_261_021,
) -> Path:
    manifest_path = score_path.parent / "score_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"score provenance sidecar is missing: {manifest_path}")
    score_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if score_manifest.get("status") != "SCORES_COMPLETE_ANALYSIS_PENDING":
        raise ValueError("score provenance status is not analyzable")
    if score_manifest.get("locked_test_accessed") is not False:
        raise PermissionError("score artifact does not certify locked-test exclusion")
    if score_manifest.get("hashes", {}).get("scores") != sha256_file(score_path):
        raise ValueError("score artifact SHA does not match its provenance sidecar")
    final, partial = atomic_output_directory(output)
    with np.load(score_path, allow_pickle=False) as archive:
        missing = set(REQUIRED) - set(archive.files)
        if missing:
            raise ValueError(f"score artifact is missing {sorted(missing)}")
        arrays = {key: archive[key] for key in archive.files}
    observed_models = {
        key.removeprefix("score_") for key in arrays if key.startswith("score_")
    }
    if observed_models != set(EXPECTED_MODELS):
        raise ValueError(
            "artifact requires exactly Base/Stage0/Stage1-only/Sequential scores"
        )
    if score_manifest.get("variants") != list(EXPECTED_MODELS):
        raise ValueError("score provenance does not bind the four required variants")
    score_names = list(EXPECTED_MODELS)

    positive = np.asarray(arrays["positive_indices"], dtype=np.int64)
    valid = np.asarray(arrays["valid_mask"], dtype=bool)
    candidate_entity_ids = np.asarray(
        arrays["candidate_entity_ids"], dtype=np.int64
    ).reshape(-1)
    if not np.array_equal(candidate_entity_ids, np.arange(valid.shape[1])):
        raise ValueError("candidate entity order must be contiguous entity-ID order")
    ranks = {
        name: ranks_from_matrices(arrays[f"score_{name}"], positive, valid)
        for name in score_names
    }
    metrics = {name: ranking_metrics(value) for name, value in ranks.items()}
    direction = np.asarray(arrays["directions"], dtype=np.int8)
    direction_metrics = {
        name: {
            "tail": ranking_metrics(value[direction == 0]),
            "head": ranking_metrics(value[direction == 1]),
        }
        for name, value in ranks.items()
    }
    relations = np.asarray(arrays["positive_triples"], dtype=np.int64)[:, 2]
    relation_rows = []
    relation_macro = {}
    for name, value in ranks.items():
        per_relation = []
        for relation in np.unique(relations):
            metric = ranking_metrics(value[relations == relation])
            relation_rows.append(
                {"model": name, "relation": int(relation), **metric}
            )
            per_relation.append(float(metric["mrr"]))
        relation_macro[name] = float(np.mean(per_relation))
    facts = per_fact_rows(arrays["fact_ids"], direction, ranks)
    positive_triples = np.asarray(arrays["positive_triples"], dtype=np.int64)
    for row in facts:
        selected = np.flatnonzero(
            np.asarray(arrays["fact_ids"], dtype=np.int64) == int(row["fact_id"])
        )
        triples = np.unique(positive_triples[selected], axis=0)
        if triples.shape != (1, 3):
            raise ValueError(f"fact {row['fact_id']} directions do not share one triple")
        row.update(
            {
                "head": int(triples[0, 0]),
                "tail": int(triples[0, 1]),
                "relation": int(triples[0, 2]),
            }
        )
    comparisons = []
    for index, name in enumerate(score_names):
        if name == "base":
            continue
        before = np.asarray([row["base_fact_mrr"] for row in facts])
        after = np.asarray([row[f"{name}_fact_mrr"] for row in facts])
        comparison = {
                "comparison": f"{name}-base",
                **paired_fact_bootstrap(
                    before, after, samples=samples, seed=seed + index
                ),
            }
        delta = after - before
        comparison.update(
            {
                "fact_wins": int(np.sum(delta > 0)),
                "fact_ties": int(np.sum(delta == 0)),
                "fact_losses": int(np.sum(delta < 0)),
                "top1_gains": int(np.sum((ranks["base"] > 1) & (ranks[name] == 1))),
                "top1_losses": int(np.sum((ranks["base"] == 1) & (ranks[name] > 1))),
            }
        )
        comparisons.append(comparison)
    for reference in ("stage0", "stage1_only"):
        before = np.asarray([row[f"{reference}_fact_mrr"] for row in facts])
        after = np.asarray([row["sequential_fact_mrr"] for row in facts])
        delta = after - before
        comparisons.append(
            {
                "comparison": f"sequential-{reference}",
                **paired_fact_bootstrap(
                    before,
                    after,
                    samples=samples,
                    seed=seed + len(score_names) + (1 if reference == "stage0" else 2),
                ),
                "fact_wins": int(np.sum(delta > 0)),
                "fact_ties": int(np.sum(delta == 0)),
                "fact_losses": int(np.sum(delta < 0)),
                "top1_gains": int(
                    np.sum((ranks[reference] > 1) & (ranks["sequential"] == 1))
                ),
                "top1_losses": int(
                    np.sum((ranks[reference] == 1) & (ranks["sequential"] > 1))
                ),
            }
        )

    query_rows = []
    for query_index, (fact_id, query_direction) in enumerate(
        zip(arrays["fact_ids"], direction, strict=True)
    ):
        row = {
            "query_index": query_index,
            "fact_id": int(fact_id),
            "direction": "tail" if int(query_direction) == 0 else "head",
            "head": int(positive_triples[query_index, 0]),
            "tail": int(positive_triples[query_index, 1]),
            "relation": int(positive_triples[query_index, 2]),
            "valid_candidates": int(valid[query_index].sum()),
            "candidate_universe": int(valid.shape[1]),
        }
        for name in score_names:
            row[f"{name}_rank"] = int(ranks[name][query_index])
            row[f"{name}_rr"] = float(1.0 / ranks[name][query_index])
            score_row = np.asarray(
                arrays[f"score_{name}"][query_index], dtype=np.float64
            )
            positive_score = float(score_row[positive[query_index]])
            row[f"{name}_positive_score"] = positive_score
            row[f"{name}_tie_count"] = int(
                np.sum(valid[query_index] & (score_row == positive_score))
            )
        query_rows.append(row)

    write_csv(partial / "query_metrics.csv", query_rows)
    write_csv(partial / "per_fact_metrics.csv", facts)
    write_csv(partial / "per_relation_metrics.csv", relation_rows)
    write_csv(partial / "paired_fact_bootstrap.csv", comparisons)
    result = {
        "status": "COMPLETE",
        "scope": "fixed-subset full-candidate validation sanity",
        "score_artifact": str(score_path.resolve()),
        "score_artifact_sha256": sha256_file(score_path),
        "models": score_names,
        "facts": len(facts),
        "directional_queries": len(direction),
        "metrics": metrics,
        "direction_metrics": direction_metrics,
        "relation_macro_mrr": relation_macro,
        "comparisons": comparisons,
        "tie_policy": "conservative_count_ge",
        "bootstrap_unit": "original_fact",
        "locked_test_accessed": False,
    }
    (partial / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_261_021)
    args = parser.parse_args()
    output = analyze(
        args.scores.resolve(),
        args.output,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
