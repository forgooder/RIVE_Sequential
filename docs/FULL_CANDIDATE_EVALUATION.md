# Full-candidate validation evaluation

This release includes a backbone-independent evaluator for fixed-subset,
full-target-entity filtered ranking. It is intended as a validation sanity
check and must not be described as a complete benchmark or locked-test result.

## Contract

- Candidate order is the target entity-ID order.
- Both tail and head prediction are evaluated.
- The current query fact and its explicit inverse must be removed before a
  backbone produces scores.
- Other known true candidates are filtered for ranking, not deleted from the
  inference graph.
- Ties are conservative: the positive rank is the number of valid candidates
  whose score is greater than or equal to the positive score.
- Head and tail directions are aggregated by original fact before paired
  bootstrap resampling.
- The score sidecar must state `locked_test_accessed=false` and bind the score
  file by SHA256.

The reusable implementation lives in `rive_ladder_final.fullrank`. The
append-only analyzer is `scripts/analyze_fullrank_scores.py`.

## Score artifact schema

The analyzer expects a compressed NumPy archive containing:

```text
fact_ids                 [Q]
directions               [Q]       # 0 tail, 1 head
positive_triples         [Q, 3]
positive_indices         [Q]
candidate_entity_ids     [E]       # exactly 0..E-1
valid_mask               [Q, E]
score_base               [Q, E]
score_stage0             [Q, E]
score_stage1_only        [Q, E]
score_sequential         [Q, E]
```

The neighboring `score_manifest.json` must use this minimal schema:

```json
{
  "status": "SCORES_COMPLETE_ANALYSIS_PENDING",
  "locked_test_accessed": false,
  "variants": ["base", "stage0", "stage1_only", "sequential"],
  "hashes": {"scores": "<sha256-of-npz>"}
}
```

Run:

```bash
python scripts/analyze_fullrank_scores.py \
  --scores /path/to/fullrank_scores.npz \
  --output /new/append-only/output/directory \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20261021
```

The repository deliberately excludes dataset files, third-party checkpoints,
candidate caches, and raw score matrices. Their hashes remain part of the
private experiment audit trail.

## Included result snapshot

`results/www_validation_snapshot/` contains aggregate CSV files and plots from
three frozen seeds on FB15k-237-v1. GraIL and NBFNet are entity-inductive;
ComplEx is a transductive compatibility diagnostic. These are validation-only
results and do not authorize claims about a locked test.
