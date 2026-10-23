from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/make_synthetic_cache.py"
SPEC = importlib.util.spec_from_file_location("synthetic_cache", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_synthetic_cache_matches_public_contract() -> None:
    backbone, shell = MODULE.make_split(np.random.default_rng(7), 8)
    assert backbone["labels"].shape == (8, 51)
    assert backbone["endpoint_features"].shape == (8, 51, 7 * 4 + 4)
    assert shell["shell_features"].shape == (8, 51, 12 * 4 + 10)
    assert np.all(backbone["labels"].sum(axis=1) == 1)
    assert np.array_equal(backbone["triples"], shell["triples"])
    assert np.array_equal(backbone["query_ids"], shell["query_ids"])
    query_one_hot = backbone["endpoint_features"][..., 6 * 4 + 4 :]
    assert np.all(query_one_hot.sum(axis=-1) == 1)
