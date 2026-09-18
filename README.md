# Sequential RIVE-Ladder

Sequential RIVE-Ladder is a small, backbone-agnostic score correction module
for knowledge graph ranking. It keeps the base reasoner frozen and introduces
two independently bounded evidence stages:

1. compact endpoint-role evidence (`7R+4` features), and
2. neighbor-resolved one-hop relation-shell evidence (`12R+10` features).

The repository contains the clean PyTorch implementation, the source-only
adapter trainer, an example configuration, and contract tests. It deliberately
does not contain datasets, checkpoints, cached scores, third-party model code,
or locked-test artifacts.

## Model

For candidate `c` in query group `q`, the frozen architecture is

```text
s0(q,c) = normalize_source_frozen(s_base(q,c))
s1(q,c) = s0(q,c) + 0.5 * tanh(g0(x_endpoint(q,c)))
s2(q,c) = s1(q,c) + 0.5 * sigmoid(alpha) * tanh(g1(x_shell(q,c), r_q))
```

`g0` and `g1` have zero-initialized output layers. Consequently, initialization
preserves the base ranking exactly. Stage 0 is trained first and then frozen;
stage 1 is trained afterward. The base backbone remains frozen throughout.

See [docs/METHOD.md](docs/METHOD.md) for tensor shapes and training details.

## Installation

```bash
python -m pip install -e '.[test]'
pytest -q
```

## Minimal forward pass

```bash
python examples/minimal_forward.py
```

## Training from cached backbone scores

Prepare source-training and source-validation caches according to
[docs/CACHE_CONTRACT.md](docs/CACHE_CONTRACT.md), update paths in
`configs/example_adapter.json`, and run:

```bash
python scripts/train_adapter.py --config configs/example_adapter.json
```

The trainer verifies cache hashes, candidate width, one-positive-per-group,
feature alignment, epoch-zero identity, and rank preservation before fitting.
It writes checkpoints and metrics append-only to the configured output path.

## Evaluation boundary

The intended development protocol is candidate-clean ranking with one positive
and 50 filtered negatives, conservative ties, both prediction directions, and
source-only model selection. Full-candidate ranking and locked-test evaluation
are separate stages and are not represented by the included example.

Embedding-model experiments are transductive compatibility diagnostics; they
must not be interpreted as unseen-entity results.

## Repository scope

Included:

- the Sequential RIVE-Ladder PyTorch module;
- source-frozen score normalization;
- two-stage listwise adapter training;
- paired query-cluster bootstrap;
- architecture and cache-contract tests.

Not included:

- third-party GraIL, NBFNet, RED-GNN, AdaProp, ULTRA, or KGE code;
- datasets, negative samples, feature caches, checkpoints, or predictions;
- private paths, credentials, identities, or locked-test material.

