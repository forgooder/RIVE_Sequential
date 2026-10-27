# Entity-inductive sampled-test snapshot

This directory publishes aggregate results from the frozen `TEST_IND` protocol. It is separate from `results/www_validation_snapshot/`: these rows evaluate official target test facts, while the older snapshot is validation-only.

## Frozen protocol

- Datasets: WN18RR v1-v4 and FB15k-237 v1-v4.
- Directions: head and tail prediction for every official test fact.
- Candidate set: one positive plus 50 unique, strict-filtered negatives (51 candidates total).
- Tie handling: conservative ranking.
- Candidate sampling seed: `20261024`.
- Adapter: source-selected seed `43`; target test labels are not used for training, tuning, checkpoint selection, or early stopping.
- Inference graph: target training graph only; the query fact and its explicit inverse are absent/removed, and no synthetic query edge is inserted.
- Uncertainty: 10,000 paired fact-cluster bootstrap replicates with seed `20261024`.

## Coverage and evidence boundary

The matrix covers nine backbones across eight dataset variants (72 possible cells):

- 61 cells have executed test artifacts and retained per-candidate scores;
- 11 cells remain missing because the required checkpoint or verifiable checkpoint pairing is unavailable; they are not rerun, imputed, or replaced with validation values;
- 57 completed cells are strictly entity-disjoint: 44 have a positive 95% confidence interval and 13 are statistically uncertain;
- four completed WN18RR-v3 cells are exploratory because the available local split contains eight source-target entity overlaps; all four have a positive 95% confidence interval;
- no completed cell has a significantly negative 95% confidence interval.

These are **sampled filtered test** results, not full-candidate benchmark results. They must not be compared numerically with the full-candidate validation snapshot as if the protocols were identical. Dataset versions v1-v4 are distinct split constructions, not an ordered difficulty scale.

## Files

- `aggregate_metrics.csv`: one row per completed cell, including Base, Stage 0, Sequential, confidence intervals, evidence status, artifact hash, and protocol hash. Local filesystem paths were intentionally removed for the public release.
- `completion_matrix.csv`: all 72 expected cells, including explicit missing states.
- `COMPLETION_MATRIX_ZH.md`: human-readable matrices, paired results, missing-cell reasons, and the protocol boundary.
- `SHA256SUMS.txt`: checksums for all four public snapshot files, including this README.

The hashes identify the private result artifacts used for aggregation, but the private datasets, third-party checkpoints, raw predictions, and per-candidate score files are not redistributed here.
