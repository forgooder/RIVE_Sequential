# Method

Let a frozen knowledge graph reasoner produce one score for each candidate in a
query group. Sequential RIVE-Ladder corrects those scores without entity ID
embeddings and without modifying the backbone.

## Inputs

For batch size `B`, candidate width `C`, and relation count `R`:

| Tensor | Shape | Meaning |
|---|---|---|
| `base_scores` | `[B,C]` | Frozen backbone scores |
| `valid_mask` | `[B,C]` | Valid candidate positions |
| `endpoint_features` | `[B,C,7R+4]` | Compact endpoint-role evidence |
| `shell_features` | `[B,C,12R+10]` | One-hop relation-shell evidence |
| `query_one_hot` | `[B,C,R]` | Query relation |

The endpoint vector contains head/tail incoming and outgoing relation counts,
two cross-endpoint relation-overlap vectors, four degree scalars, and the query
relation one-hot vector. Counts are expected to use `log1p` preprocessing.

The shell vector contains two endpoint sides. Each side has six statistics per
relation plus five scalar summaries. The shell encoder treats each relation as
a token and aggregates tokens with a shared DeepSets encoder.

## Source-frozen score normalization

For each query, the base scores are divided by their population standard
deviation. Degenerate values fall back to the source-training median positive
query standard deviation. Divisors are clipped to source-training quantiles.
No target statistic is fitted. Since each query is divided by one positive
scalar, candidate order is preserved.

## Sequential correction

Stage 0 is

```text
stage0 = normalized_base + 0.5 * tanh(g0(endpoint_features)).
```

Stage 1 is

```text
final = stage0 + B1 * tanh(g1(shell_features, query_relation)),
B1 = 0.5 * sigmoid(alpha).
```

The two output heads are initialized to zero. Stage 0 is trained first. Its
parameters are frozen before stage 1 begins, so the second stage can supplement
but cannot rewrite the learned endpoint correction.

## Objective

Both stages use query-group listwise cross entropy:

```text
L_rank = -s_positive + logsumexp(s_candidates).
```

A small within-query correction variance penalty can be enabled through the
configuration. Checkpoint selection uses source-validation ranking only.

## Backbone interface

Any backbone is compatible if it exports deterministic, higher-is-better scores
in the frozen candidate order. A fair plug-in comparison trains a new adapter
for each `backbone x dataset x seed`; adapters are not transferred from another
backbone.

