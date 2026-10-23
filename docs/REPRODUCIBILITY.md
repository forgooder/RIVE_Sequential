# Reproducibility and artifact scope

This document separates what can be reproduced from the public repository
alone from what additionally requires external datasets, backbone checkpoints,
and frozen cache artifacts.

## Evidence levels

### Level 1: software contract

Available from this repository alone:

- model construction and tensor-shape checks;
- zero-initialized residual identity;
- base-ranking preservation under positive score scaling;
- bounded Stage 0 and Stage 1 corrections;
- Stage 0 parameter freezing before Stage 1;
- conservative-tie ranking;
- full-entity candidate ordering and filtered-truth handling;
- original-fact clustered paired bootstrap;
- append-only output and SHA256 verification.

Run:

```bash
bash scripts/verify_release.sh
```

### Level 2: end-to-end training smoke test

The deterministic synthetic generator produces all four required NPZ caches
and their SHA256 sidecars. It validates the complete public training path but
does not represent a benchmark result.

```bash
python examples/make_synthetic_cache.py --output data/synthetic_smoke
python scripts/train_adapter.py --config configs/synthetic_smoke.json
```

Expected append-only outputs:

```text
outputs/synthetic_smoke/seed43/
  model.pt
  source_validation_scores.npz
  result.json
  SHA256SUMS.txt
```

### Level 3: aggregate validation snapshot

The repository publishes small aggregate CSV files, figures, and their hashes
under `results/www_validation_snapshot/`. These verify the values reported in
the README without exposing private paths or large raw score arrays.

The snapshot covers three seeds on a fixed FB15k-237-v1 validation subset:

- GraIL: entity-inductive;
- NBFNet: entity-inductive;
- ComplEx: transductive compatibility diagnostic.

The snapshot is not a locked-test result and must not be described as a full
benchmark.

### Level 4: numerical rerun of real experiments

A numerical rerun additionally requires:

1. the original dataset under its own license;
2. the exact third-party backbone implementation and checkpoint;
3. frozen candidate ordering and filtered-truth manifests;
4. candidate-clean endpoint and shell feature caches;
5. the cache SHA256 sidecars described in `CACHE_CONTRACT.md`.

These assets are intentionally not redistributed here. The public trainer
fails closed on hash mismatch, feature misalignment, candidate-width drift,
multiple positives, target/locked-test authorization, and output overwrite.

## Frozen protocols

### Sampled-ranking development

- one positive and 50 filtered negatives per query;
- head and tail prediction;
- candidate fact and explicit inverse deleted before scoring/features;
- no synthetic query edge;
- conservative ties;
- query-group listwise cross entropy;
- source-validation model selection only.

### Full-candidate validation sanity

- fixed subset selected before scoring;
- all target entities in entity-ID order;
- known alternatives filtered for ranking;
- head and tail aggregated by original fact;
- 10,000 paired fact-cluster bootstrap draws;
- `locked_test_accessed=false` required by the analyzer.

## Determinism

The trainer seeds Python, NumPy, and PyTorch. Candidate caches and feature
matrices are immutable inputs bound by SHA256. GPU kernels may still vary
slightly across CUDA, cuDNN, driver, or PyTorch versions. Report the complete
runtime environment and preserve per-seed outputs instead of claiming bitwise
cross-hardware determinism.

## Tested environment

```text
Python       3.10.20
PyTorch      2.1.2+cu121
NumPy        1.26.4
pytest       9.1.1
CUDA runtime 12.1
```

CPU execution is supported for the public model, tests, analyzer, and smoke
training. Backbone inference may require its native GPU environment.

## Claim language

Supported by the public repository:

- the implementation satisfies the documented initialization and ranking
  contracts;
- the included validation snapshot contains positive Sequential-minus-Base
  point estimates for the three listed backbones;
- the public result files match their SHA256 manifest.

Not supported by the public repository alone:

- state-of-the-art performance;
- universal cross-backbone improvement;
- locked-test generalization;
- full official-query benchmark performance;
- causal interpretation of endpoint or shell evidence.
