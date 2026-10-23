# Sequential RIVE-Ladder

## Problem setting

Let a frozen knowledge graph reasoner assign a higher-is-better score
`s_b(q,c)` to candidate `c` in query group `q=(h,r,?)` or `q=(?,r,t)`.
Sequential RIVE-Ladder learns a small correction without changing the
backbone and without introducing entity-ID embeddings.

For batch size `B`, candidate width `C`, and relation count `R`, the public
interface is:

| Tensor | Shape | Meaning |
|---|---:|---|
| `base_scores` | `[B,C]` | Frozen backbone scores |
| `valid_mask` | `[B,C]` | Valid candidate positions |
| `endpoint_features` | `[B,C,7R+4]` | Compact endpoint-role evidence |
| `shell_features` | `[B,C,12R+10]` | One-hop relation-shell evidence |
| `query_one_hot` | `[B,C,R]` | Query relation |

Every candidate is processed under the same candidate-clean graph contract:
remove the candidate fact and its explicit inverse before scoring or feature
extraction, and do not add a synthetic query edge.

## Stage 0: compact endpoint-role evidence

The endpoint vector is

```text
x_endpoint = [
  head_out[R], head_in[R], tail_out[R], tail_in[R],
  overlap(head_out, tail_in)[R],
  overlap(head_in, tail_out)[R],
  endpoint_degree[4], query_relation_one_hot[R]
]
```

All count channels use `log1p` preprocessing. The four degree values retain
the directional endpoint totals used by the formal extractor. The total width
is `7R+4`.

The encoder is a two-hidden-layer MLP with LayerNorm and ReLU. Its scalar
output head is initialized to zero. With fixed budget `B_0=0.5`,

```text
delta_0(q,c) = 0.5 * tanh(g_0(x_endpoint(q,c))).
```

## Stage 1: neighbor-resolved relation-shell evidence

For each endpoint, the exact one-hop frontier is summarized without entity
identifiers. Each relation has six channels:

```text
sum_out, sum_in, frontier_out, frontier_in, max_out, max_in
```

Each endpoint also has five scalar summaries:

```text
frontier_size, incident_mass, outward_mass,
mean_log_degree, max_log_degree
```

The two endpoint sides therefore produce `2 * (6R+5) = 12R+10` features.

The shell encoder treats each relation as a token. A shared token MLP embeds
the 12 head/tail statistics for that relation, adds a learned relation token,
and mean-pools across relation tokens. The pooled vector is concatenated with
the query-relation embedding and encoded scalar summaries.

The second correction is

```text
delta_1(q,c) = B_1 * tanh(g_1(x_shell(q,c), r_q)),
B_1          = 0.5 * sigmoid(alpha).
```

Thus `0 < B_1 < 0.5`. The shell branch can supplement Stage 0 but remains a
bounded correction rather than an unconstrained replacement scorer.

## Source-frozen score normalization

For each query, the backbone scores are divided by their population standard
deviation. Degenerate values use the median positive standard deviation from
source-training groups. Divisors are clipped to source-training quantiles and
are never re-estimated from target validation or test data.

```text
sigma_q = clip(std(valid scores in q), source_q05, source_q95)
s_norm  = s_b / sigma_q
```

The transform uses one strictly positive divisor per query and therefore
preserves candidate order. It is performed in float64 to avoid creating new
ties through float32 rounding.

## Sequential optimization

The complete scorer is

```text
s_0     = s_norm + delta_0
s_final = s_0 + delta_1.
```

Training proceeds in two stages:

1. train the endpoint encoder and head;
2. restore the best source-validation Stage 0 checkpoint;
3. freeze every Stage 0 parameter;
4. train the shell encoder, head, and bounded budget parameter.

Both stages use query-group listwise cross entropy:

```text
L_rank(q) = -s(q,c_positive) + logsumexp_c s(q,c).
```

An optional low-weight within-query correction-variance penalty discourages
unnecessarily large candidate rewrites. Early stopping is based on source
validation MRR, followed by Hits@1 and Hits@5 as deterministic tie-breakers.

## Initialization and degeneration properties

Both scalar output heads are zero-initialized. At construction:

```text
delta_0 = 0
delta_1 = 0
s_final = s_norm
rank(s_final) = rank(s_b).
```

Removing both correction branches recovers the normalized frozen backbone,
which has exactly the same ranking as the raw backbone. Removing Stage 1 yields
the original compact endpoint correction. Removing Stage 0 yields the
Stage-1-only control used in the public ablation summary.

## Backbone adapter contract

A compatible backbone must provide:

1. deterministic higher-is-better candidate scores;
2. a frozen candidate order shared by scores and features;
3. source-training and source-validation query groups;
4. candidate-clean feature extraction under the same graph protocol.

A separate adapter is trained for every `backbone x dataset x seed`. An
adapter fitted to GraIL is not reused for NBFNet, ComplEx, or another dataset.
This keeps the plug-in comparison about the evidence interface rather than
cross-backbone checkpoint transfer.

## Parameter accounting

The parameter count depends on `R`, because the endpoint input width and
relation-token table scale with the relation vocabulary. Report trainable
adapter parameters separately from frozen backbone parameters. The included
FB15k-237-v1 snapshot records Stage 0, Stage 1-only, Sequential, and
parameter-matched flat-control counts in
`results/www_validation_snapshot/parameter_contract.csv`.
