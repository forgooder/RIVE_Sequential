from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rive_ladder_final.fullrank import (
    conservative_rank,
    filtered_truth_mask,
    full_entity_candidates,
    paired_fact_bootstrap,
    per_fact_rows,
)


def test_full_candidates_use_entity_id_order_for_both_directions() -> None:
    positive = (2, 4, 1)
    tail = full_entity_candidates(positive, 0, 6)
    head = full_entity_candidates(positive, 1, 6)
    np.testing.assert_array_equal(tail[:, 1], np.arange(6))
    np.testing.assert_array_equal(head[:, 0], np.arange(6))
    np.testing.assert_array_equal(tail[4], positive)
    np.testing.assert_array_equal(head[2], positive)


def test_filtered_truths_retain_positive_in_both_directions() -> None:
    positive = (1, 3, 0)
    truths = {(1, 2, 0), (2, 3, 0), positive}
    tail = full_entity_candidates(positive, 0, 5)
    head = full_entity_candidates(positive, 1, 5)
    tail_valid = filtered_truth_mask(tail, positive, truths)
    head_valid = filtered_truth_mask(head, positive, truths)
    assert not tail_valid[2] and tail_valid[3]
    assert not head_valid[2] and head_valid[1]


def test_conservative_rank_counts_equal_scores() -> None:
    scores = np.asarray([2.0, 9.0, 2.0, 1.0])
    valid = np.asarray([True, False, True, True])
    assert conservative_rank(scores, 0, valid) == 2


def test_fact_cluster_requires_head_and_tail() -> None:
    ranks = {"base": np.asarray([1, 2]), "sequential": np.asarray([1, 1])}
    rows = per_fact_rows(np.asarray([7, 7]), np.asarray([0, 1]), ranks)
    assert rows[0]["base_fact_mrr"] == pytest.approx(0.75)
    assert rows[0]["sequential_fact_mrr"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match=r"exactly head\+tail"):
        per_fact_rows(np.asarray([7]), np.asarray([0]), {"base": np.asarray([1])})


def test_paired_bootstrap_uses_fact_values() -> None:
    result = paired_fact_bootstrap(
        np.asarray([0.25, 0.5, 0.75]),
        np.asarray([0.5, 0.75, 1.0]),
        samples=500,
        seed=17,
    )
    assert result["mean_delta"] == pytest.approx(0.25)
    assert result["ci_low"] == pytest.approx(0.25)
    assert result["ci_high"] == pytest.approx(0.25)
    assert result["fact_clusters"] == 3


def test_synthetic_score_artifact_runs_end_to_end(tmp_path: Path) -> None:
    score_path = tmp_path / "fullrank_scores.npz"
    valid = np.ones((4, 3), dtype=bool)
    np.savez_compressed(
        score_path,
        fact_ids=np.asarray([10, 10, 11, 11]),
        directions=np.asarray([0, 1, 0, 1]),
        positive_triples=np.asarray(
            [[0, 1, 0], [0, 1, 0], [1, 2, 1], [1, 2, 1]], dtype=np.int64
        ),
        positive_indices=np.asarray([0, 1, 0, 1]),
        candidate_entity_ids=np.arange(3, dtype=np.int64),
        valid_mask=valid,
        score_base=np.asarray(
            [[2, 1, 0], [2, 1, 0], [1, 2, 0], [2, 1, 0]], dtype=float
        ),
        score_stage0=np.asarray(
            [[2, 1, 0], [1, 2, 0], [2, 1, 0], [1, 2, 0]], dtype=float
        ),
        score_stage1_only=np.asarray(
            [[2, 1, 0], [0, 2, 1], [1, 2, 0], [0, 2, 1]], dtype=float
        ),
        score_sequential=np.asarray(
            [[2, 1, 0], [0, 2, 1], [2, 1, 0], [0, 2, 1]], dtype=float
        ),
    )
    digest = hashlib.sha256(score_path.read_bytes()).hexdigest()
    (tmp_path / "score_manifest.json").write_text(
        json.dumps(
            {
                "status": "SCORES_COMPLETE_ANALYSIS_PENDING",
                "locked_test_accessed": False,
                "variants": ["base", "stage0", "stage1_only", "sequential"],
                "hashes": {"scores": digest},
            }
        ),
        encoding="utf-8",
    )
    script = ROOT / "scripts/analyze_fullrank_scores.py"
    spec = importlib.util.spec_from_file_location("fullrank_analyzer", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    output = module.analyze(score_path, tmp_path / "analysis", samples=200, seed=3)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["facts"] == 2
    assert result["bootstrap_unit"] == "original_fact"
    assert result["locked_test_accessed"] is False
    assert result["models"] == ["base", "stage0", "stage1_only", "sequential"]
