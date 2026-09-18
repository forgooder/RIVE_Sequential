# Cache contract

The trainer consumes four NPZ files and one JSON manifest beside each NPZ. If
the cache is `train.npz`, its manifest is `train.json` and must contain:

```json
{"npz_sha256": "<sha256 of train.npz>"}
```

## Backbone score cache

The training and validation backbone caches contain:

| Key | Shape | Dtype |
|---|---|---|
| `triples` | `[Q,C,3]` | integer |
| `labels` | `[Q,C]` | float/bool |
| `valid_mask` | `[Q,C]` | bool |
| `base_scores` | `[Q,C]` | float |
| `endpoint_features` | `[Q,C,7R+4]` | float |
| `directions` | `[Q]` | integer/string |
| `query_ids` | `[Q]` | integer/string |

Every query group must have exactly one positive among `C=51` candidates.

## Relation-shell cache

The training and validation shell caches contain:

| Key | Shape | Dtype |
|---|---|---|
| `triples` | `[Q,C,3]` | integer |
| `query_ids` | `[Q]` | integer/string |
| `shell_features` | `[Q,C,12R+10]` | float |

The `triples` and `query_ids` arrays must be elementwise identical to their
backbone-cache counterparts. Candidate ordering must be frozen before feature
extraction.

## Candidate-clean requirement

Before extracting evidence or scoring a candidate, remove the candidate fact
and its explicit inverse from the inference graph. Do not add a synthetic query
edge. Known true alternatives are used for filtered evaluation, not silently
treated as negatives.

