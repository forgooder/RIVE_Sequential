"""Shared contracts for the frozen WWW full-candidate sanity evaluation.

The module is deliberately NumPy-only.  Backbones may use different runtimes,
but candidate construction, filtering, conservative ties, and fact-cluster
statistics must be identical.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


Triple = tuple[int, int, int]  # (head, tail, relation)


@dataclass(frozen=True)
class FullRankEnvironment:
    """Immutable identity of one full-ranking graph environment."""

    cell_id: str
    dataset_dir: Path
    query_file_sha256: str
    expected_entities: int
    expected_valid_facts: int
    track: str


@dataclass(frozen=True)
class FrozenFact:
    source_index: int
    fact_sha256: str
    head: str
    relation: str
    tail: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_fact_sha256(head: str, relation: str, tail: str) -> str:
    return hashlib.sha256(f"{head}\t{relation}\t{tail}".encode("utf-8")).hexdigest()


def load_frozen_facts(
    path: str | Path, query_file_sha256: str
) -> list[FrozenFact]:
    """Read only the facts belonging to an explicitly bound query file."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = []
    for row in rows:
        if row["query_file_sha256"] != query_file_sha256:
            continue
        fact = FrozenFact(
            source_index=int(row["source_index"]),
            fact_sha256=row["fact_sha256"],
            head=row["head"],
            relation=row["relation"],
            tail=row["tail"],
        )
        if fact.fact_sha256 != normalized_fact_sha256(
            fact.head, fact.relation, fact.tail
        ):
            raise ValueError(f"frozen fact digest mismatch at source_index={fact.source_index}")
        selected.append(fact)
    if not selected:
        raise ValueError("no frozen facts match the environment query-file SHA")
    if len({fact.source_index for fact in selected}) != len(selected):
        raise ValueError("frozen fact indices are not unique")
    return selected


def validate_environment(
    environment: FullRankEnvironment,
    *,
    entity_count: int,
    valid_rows: Sequence[tuple[str, str, str]],
    frozen_facts: Sequence[FrozenFact],
) -> None:
    """Fail closed on the transductive/inductive environment mix-up."""

    valid_path = environment.dataset_dir / "valid.txt"
    if sha256_file(valid_path) != environment.query_file_sha256:
        raise ValueError(f"query file SHA mismatch for {environment.cell_id}")
    if entity_count != environment.expected_entities:
        raise ValueError(
            f"{environment.cell_id}: entities={entity_count}, "
            f"expected={environment.expected_entities}"
        )
    if len(valid_rows) != environment.expected_valid_facts:
        raise ValueError(
            f"{environment.cell_id}: valid facts={len(valid_rows)}, "
            f"expected={environment.expected_valid_facts}"
        )
    for fact in frozen_facts:
        if fact.source_index < 0 or fact.source_index >= len(valid_rows):
            raise ValueError(f"frozen source index out of range: {fact.source_index}")
        if tuple(valid_rows[fact.source_index]) != (fact.head, fact.relation, fact.tail):
            raise ValueError(f"frozen fact no longer matches valid.txt: {fact.source_index}")


def full_entity_candidates(
    positive: Triple, direction: int, num_entities: int
) -> np.ndarray:
    """Return candidates in target entity-ID order.

    ``direction=0`` corrupts the tail and ``direction=1`` corrupts the head.
    The positive is therefore located at its entity ID, not forced to column 0.
    """

    if num_entities < 2:
        raise ValueError("full ranking requires at least two entities")
    head, tail, relation = map(int, positive)
    entities = np.arange(num_entities, dtype=np.int64)
    if direction == 0:
        return np.column_stack(
            (np.full(num_entities, head), entities, np.full(num_entities, relation))
        )
    if direction == 1:
        return np.column_stack(
            (entities, np.full(num_entities, tail), np.full(num_entities, relation))
        )
    raise ValueError("direction must be 0 (tail) or 1 (head)")


def filtered_truth_mask(
    candidates: np.ndarray, positive: Triple, truths: Iterable[Triple]
) -> np.ndarray:
    """Keep the positive and candidates absent from train+authorized-valid truth."""

    values = np.asarray(candidates, dtype=np.int64).reshape(-1, 3)
    positive = tuple(map(int, positive))
    truth_set = frozenset(tuple(map(int, item)) for item in truths)
    valid = np.fromiter(
        (
            tuple(map(int, candidate)) == positive
            or tuple(map(int, candidate)) not in truth_set
            for candidate in values
        ),
        count=len(values),
        dtype=bool,
    )
    positive_rows = np.all(values == np.asarray(positive, dtype=np.int64), axis=1)
    if int(positive_rows.sum()) != 1 or not bool(valid[positive_rows][0]):
        raise ValueError("the positive must occur exactly once and remain unfiltered")
    return valid


def conservative_rank(
    scores: np.ndarray, positive_index: int, valid_mask: np.ndarray
) -> int:
    """Rank by ``count(valid score >= positive score)`` (higher is better)."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if scores.shape != valid.shape:
        raise ValueError("score and filter mask shapes differ")
    if positive_index < 0 or positive_index >= len(scores) or not valid[positive_index]:
        raise ValueError("positive index is invalid or filtered")
    if not np.isfinite(scores[valid]).all():
        raise ValueError("valid candidate scores must be finite")
    return int(np.sum(valid & (scores >= scores[positive_index])))


def ranks_from_matrices(
    scores: np.ndarray, positive_indices: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    positive = np.asarray(positive_indices, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or scores.shape != valid.shape or len(positive) != len(scores):
        raise ValueError("full-ranking matrices have incompatible shapes")
    return np.asarray(
        [conservative_rank(row, int(index), mask) for row, index, mask in zip(scores, positive, valid, strict=True)],
        dtype=np.int64,
    )


def ranking_metrics(ranks: np.ndarray) -> dict[str, float | int | str]:
    rank = np.asarray(ranks, dtype=np.float64).reshape(-1)
    if not len(rank) or not np.all(rank >= 1):
        raise ValueError("ranks must be non-empty and positive")
    return {
        "mrr": float(np.mean(1.0 / rank)),
        "hits@1": float(np.mean(rank <= 1)),
        "hits@3": float(np.mean(rank <= 3)),
        "hits@5": float(np.mean(rank <= 5)),
        "hits@10": float(np.mean(rank <= 10)),
        "directional_queries": int(len(rank)),
        "tie_policy": "conservative_count_ge",
    }


def per_fact_rows(
    fact_ids: np.ndarray,
    directions: np.ndarray,
    ranks_by_model: Mapping[str, np.ndarray],
) -> list[dict[str, float | int]]:
    """Aggregate the two directional queries before statistical testing."""

    facts = np.asarray(fact_ids, dtype=np.int64).reshape(-1)
    directions = np.asarray(directions, dtype=np.int8).reshape(-1)
    if len(facts) != len(directions):
        raise ValueError("fact IDs and directions differ in length")
    models = {
        name: np.asarray(ranks, dtype=np.float64).reshape(-1)
        for name, ranks in ranks_by_model.items()
    }
    if any(len(ranks) != len(facts) for ranks in models.values()):
        raise ValueError("rank vector differs from directional query count")
    rows = []
    for fact_id in np.unique(facts):
        selected = facts == fact_id
        observed_directions = set(directions[selected].tolist())
        if observed_directions != {0, 1} or int(selected.sum()) != 2:
            raise ValueError(f"fact {fact_id} does not contain exactly head+tail queries")
        row: dict[str, float | int] = {"fact_id": int(fact_id)}
        for name, rank in models.items():
            rr = 1.0 / rank[selected]
            row[f"{name}_fact_mrr"] = float(rr.mean())
            row[f"{name}_tail_rr"] = float(rr[directions[selected] == 0][0])
            row[f"{name}_head_rr"] = float(rr[directions[selected] == 1][0])
        rows.append(row)
    return rows


def paired_fact_bootstrap(
    before: np.ndarray,
    after: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Percentile paired bootstrap over already clustered fact-level values."""

    before = np.asarray(before, dtype=np.float64).reshape(-1)
    after = np.asarray(after, dtype=np.float64).reshape(-1)
    if before.shape != after.shape or not len(before):
        raise ValueError("paired fact vectors must be non-empty and aligned")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    delta = after - before
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    # Chunking prevents an accidental multi-GB [10000, many-facts] allocation.
    for start in range(0, samples, 1024):
        stop = min(start + 1024, samples)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        draws[start:stop] = delta[indices].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_delta_le_zero": float(np.mean(draws <= 0)),
        "fact_clusters": int(len(delta)),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
    }


__all__ = [
    "FrozenFact",
    "FullRankEnvironment",
    "Triple",
    "conservative_rank",
    "filtered_truth_mask",
    "full_entity_candidates",
    "load_frozen_facts",
    "normalized_fact_sha256",
    "paired_fact_bootstrap",
    "per_fact_rows",
    "ranking_metrics",
    "ranks_from_matrices",
    "sha256_file",
    "validate_environment",
]
