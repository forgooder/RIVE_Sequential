#!/usr/bin/env python3
"""Run an epoch-zero forward pass without datasets or third-party backbones."""

import torch

from rive_ladder_final.model import (
    SequentialRiveLadder,
    SourceScale,
    normalize_query_scores,
)


def main() -> None:
    torch.manual_seed(43)
    batch, candidates, relations = 2, 51, 11
    base = torch.randn(batch, candidates)
    valid = torch.ones_like(base, dtype=torch.bool)
    normalized = normalize_query_scores(base, valid, SourceScale(1.0, 0.1, 10.0))
    endpoint = torch.randn(batch, candidates, 7 * relations + 4)
    shell = torch.randn(batch, candidates, 12 * relations + 10)
    query_ids = torch.randint(relations, (batch, candidates))
    query = torch.nn.functional.one_hot(query_ids, relations).float()

    model = SequentialRiveLadder(7 * relations + 4, relations).eval()
    output = model(normalized, endpoint, shell, query, valid)
    torch.testing.assert_close(output["scores"], normalized, rtol=0, atol=0)
    assert torch.equal(base.argsort(dim=-1), output["scores"].argsort(dim=-1))
    print("epoch-zero identity and rank preservation: PASS")


if __name__ == "__main__":
    main()

