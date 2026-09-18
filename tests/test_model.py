from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rive_ladder_final.model import SequentialRiveLadder, SourceScale, normalize_query_scores


def test_epoch0_is_normalized_base_and_preserves_rank():
    torch.manual_seed(42)
    relations = 3
    model = SequentialRiveLadder(7 * relations + 4, relations).eval()
    base = torch.tensor([[0.2, 0.8, -0.1], [2.0, 1.0, 3.0]])
    valid = torch.ones_like(base, dtype=torch.bool)
    normalized = normalize_query_scores(base, valid, SourceScale(1.0, 0.1, 10.0))
    endpoint = torch.randn(2, 3, 7 * relations + 4)
    shell = torch.randn(2, 3, 2 * (6 * relations + 5))
    query = torch.nn.functional.one_hot(torch.tensor([[0, 1, 2], [1, 1, 0]]), relations).float()
    output = model(normalized, endpoint, shell, query, valid)
    torch.testing.assert_close(output["scores"], normalized, rtol=0, atol=0)
    assert torch.equal(base.argsort(dim=-1), normalized.argsort(dim=-1))


def test_budgets_and_stage0_freezing():
    model = SequentialRiveLadder(25, 3)
    assert model.endpoint_budget == 0.5
    torch.testing.assert_close(model.shell_budget, torch.tensor(0.25))
    model.freeze_stage0()
    assert not any(parameter.requires_grad for parameter in model.endpoint_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.endpoint_head.parameters())
    assert any(parameter.requires_grad for parameter in model.shell_encoder.parameters())


def test_normalization_does_not_collapse_close_float32_scores_into_tie():
    base = torch.tensor(
        [[5.476570129394531, 5.476569652557373, -1.0]], dtype=torch.float32
    )
    valid = torch.ones_like(base, dtype=torch.bool)
    normalized = normalize_query_scores(
        base, valid, SourceScale(2.4920440775, 2.4920440775, 2.4920440775)
    )
    assert normalized.dtype == torch.float64
    assert normalized[0, 0] > normalized[0, 1]
    assert torch.equal(base.argsort(dim=-1), normalized.argsort(dim=-1))
