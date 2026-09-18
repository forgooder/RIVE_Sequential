"""Self-contained frozen implementation of Sequential RIVE-Ladder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SourceScale:
    """Source-estimated bounds for query-local score normalization."""

    fallback: float
    clip_low: float
    clip_high: float
    epsilon: float = 1e-8


def normalize_query_scores(
    scores: Tensor,
    valid_mask: Tensor,
    scale: SourceScale,
) -> Tensor:
    """Divide by query-local std clipped with source-only frozen bounds."""

    if scores.ndim != 2 or valid_mask.shape != scores.shape:
        raise ValueError("scores and valid_mask must have shape [batch, candidates]")
    # A positive per-query divisor is order preserving mathematically. Perform
    # the transform in float64 so float32 rounding cannot create new ties.
    scores = scores.to(torch.float64)
    valid = valid_mask.bool() & torch.isfinite(scores)
    count = valid.sum(dim=-1).clamp_min(1)
    clean = torch.where(valid, scores, torch.zeros_like(scores))
    mean = clean.sum(dim=-1) / count
    variance = torch.where(valid, (scores - mean.unsqueeze(-1)).square(), torch.zeros_like(scores)).sum(dim=-1) / count
    local = variance.sqrt()
    fallback = torch.as_tensor(scale.fallback, dtype=scores.dtype, device=scores.device)
    local = torch.where(local > scale.epsilon, local, fallback)
    divisor = local.clamp(min=scale.clip_low, max=scale.clip_high)
    return scores / divisor.unsqueeze(-1)


class FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim, elementwise_affine=False),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        features = torch.where(torch.isfinite(features), features, torch.zeros_like(features))
        return self.network(features)


def unpack_shell_tokens(shell: Tensor, relations: int) -> tuple[Tensor, Tensor]:
    side_width = 6 * relations + 5
    if shell.shape[-1] != 2 * side_width:
        raise ValueError(f"shell width {shell.shape[-1]} != {2 * side_width}")
    head, tail = shell[..., :side_width], shell[..., side_width:]
    head_tokens = head[..., : 6 * relations].reshape(*head.shape[:-1], 6, relations).transpose(-1, -2)
    tail_tokens = tail[..., : 6 * relations].reshape(*tail.shape[:-1], 6, relations).transpose(-1, -2)
    tokens = torch.cat((head_tokens, tail_tokens), dim=-1)
    scalars = torch.cat((head[..., 6 * relations :], tail[..., 6 * relations :]), dim=-1)
    return tokens, scalars


class ShellDeepSetsEncoder(nn.Module):
    def __init__(self, relations: int, token_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.relations = relations
        self.relation_embedding = nn.Parameter(torch.empty(relations, token_dim))
        nn.init.normal_(self.relation_embedding, std=token_dim**-0.5)
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(12, elementwise_affine=False),
            nn.Linear(12, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.scalar_encoder = nn.Sequential(
            nn.LayerNorm(10, elementwise_affine=False), nn.Linear(10, token_dim)
        )
        self.output = nn.Sequential(
            nn.Linear(3 * token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, shell: Tensor, query_one_hot: Tensor) -> Tensor:
        tokens, scalars = unpack_shell_tokens(shell, self.relations)
        encoded = self.token_encoder(tokens) + self.relation_embedding
        query = query_one_hot @ self.relation_embedding
        return self.output(torch.cat((encoded.mean(dim=-2), query, self.scalar_encoder(scalars)), dim=-1))


class SequentialRiveLadder(nn.Module):
    """Fixed-budget endpoint stage followed by an independently bounded shell stage."""

    def __init__(self, endpoint_dim: int, relations: int) -> None:
        super().__init__()
        if endpoint_dim != 7 * relations + 4:
            raise ValueError("endpoint_dim must equal 7R+4")
        self.endpoint_encoder = FeatureMLP(endpoint_dim, 64)
        self.endpoint_head = nn.Linear(64, 1)
        self.shell_encoder = ShellDeepSetsEncoder(relations, 16, 32)
        self.shell_head = nn.Linear(32, 1)
        self.shell_alpha = nn.Parameter(torch.zeros(()))
        self.endpoint_budget = 0.5
        self.shell_maximum_budget = 0.5
        nn.init.zeros_(self.endpoint_head.weight)
        nn.init.zeros_(self.endpoint_head.bias)
        nn.init.zeros_(self.shell_head.weight)
        nn.init.zeros_(self.shell_head.bias)

    @property
    def shell_budget(self) -> Tensor:
        return self.shell_maximum_budget * torch.sigmoid(self.shell_alpha)

    def freeze_stage0(self) -> None:
        for module in (self.endpoint_encoder, self.endpoint_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        normalized_base_scores: Tensor,
        endpoint_features: Tensor,
        shell_features: Tensor,
        query_one_hot: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        valid = valid_mask.bool()
        endpoint_raw = self.endpoint_head(self.endpoint_encoder(endpoint_features)).squeeze(-1)
        endpoint_correction = self.endpoint_budget * torch.tanh(endpoint_raw)
        endpoint_correction = torch.where(valid, endpoint_correction, torch.zeros_like(endpoint_correction))
        stage0_scores = normalized_base_scores + endpoint_correction
        shell_raw = self.shell_head(self.shell_encoder(shell_features, query_one_hot)).squeeze(-1)
        shell_correction = self.shell_budget * torch.tanh(shell_raw)
        shell_correction = torch.where(valid, shell_correction, torch.zeros_like(shell_correction))
        return {
            "scores": stage0_scores + shell_correction,
            "stage0_scores": stage0_scores,
            "endpoint_correction": endpoint_correction,
            "shell_correction": shell_correction,
            "shell_budget": self.shell_budget,
        }


__all__ = ["SequentialRiveLadder", "SourceScale", "normalize_query_scores", "unpack_shell_tokens"]
