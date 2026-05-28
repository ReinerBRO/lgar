from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LGARParams, routed_layer_indices


class SharedQueryRouter(nn.Module):
    def __init__(self, hidden_size: int, hidden_dim: int) -> None:
        super().__init__()
        width = min(int(hidden_dim), int(hidden_size))
        self.norm = nn.RMSNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden_f = hidden.float()
        return torch.sigmoid(self.net(self.norm(hidden_f)).squeeze(-1))


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().float().flatten()
    labels = labels.detach().bool().flatten()
    pos = labels.sum().item()
    neg = labels.numel() - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)
    rank_sum_pos = ranks[labels].sum()
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / max(1.0, pos * neg)
    return float(auc.item())


def _precision_at_budget(scores: torch.Tensor, labels: torch.Tensor, budget: float) -> float:
    if scores.numel() == 0:
        return float("nan")
    k = max(1, int(math.ceil(scores.numel() * float(budget))))
    k = min(k, scores.numel())
    top = torch.topk(scores.float(), k=k).indices
    return float(labels.bool()[top].float().mean().item())


def router_aux_loss(
    router: SharedQueryRouter,
    hidden_states: tuple[torch.Tensor, ...],
    label_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    model_config: Any,
    params: LGARParams,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    num_layers = int(getattr(model_config, "num_hidden_layers"))
    indices = routed_layer_indices(num_layers, params.routed_layer_fraction)
    scores_by_layer = []
    bce_losses = []
    budget_losses = []
    entropies = []
    valid = valid_mask.bool()
    targets = label_targets.float()
    if not valid.any():
        z = hidden_states[0].new_zeros(())
        return z, {"router/valid_tokens": 0.0, "router/positive_fraction": 0.0}, targets.new_zeros(targets.shape)
    for layer_idx in indices:
        if layer_idx >= len(hidden_states):
            raise IndexError(f"hidden_states does not include input for layer {layer_idx}")
        scores = router(hidden_states[layer_idx])
        scores_by_layer.append(scores)
        scores_valid = scores[valid].float().clamp(1e-6, 1.0 - 1e-6)
        targets_valid = targets[valid]
        bce_losses.append(F.binary_cross_entropy(scores_valid, targets_valid))
        budget_losses.append((scores_valid.mean() - float(params.router_target_budget)).pow(2))
        entropies.append(-(scores_valid * scores_valid.log() + (1.0 - scores_valid) * (1.0 - scores_valid).log()).mean())
    avg_scores = torch.stack(scores_by_layer, dim=0).mean(dim=0)
    bce = torch.stack(bce_losses).mean()
    budget = torch.stack(budget_losses).mean()
    entropy = torch.stack(entropies).mean()
    loss = (
        float(params.lambda_router_final) * bce
        + float(params.lambda_budget) * budget
        - float(params.entropy_weight) * entropy
    )
    valid_scores = avg_scores[valid].detach()
    valid_labels = label_targets[valid].detach().bool()
    metrics = {
        "router/loss_bce": float(bce.detach().float().item()),
        "router/loss_budget": float(budget.detach().float().item()),
        "router/loss_aux_weighted": float(loss.detach().float().item()),
        "router/entropy": float(entropy.detach().float().item()),
        "router/score_mean": float(valid_scores.float().mean().item()),
        "router/positive_fraction": float(valid_labels.float().mean().item()),
        "router/auc": _binary_auc(valid_scores, valid_labels),
        "router/precision_at_5pct": _precision_at_budget(valid_scores, valid_labels, 0.05),
        "router/precision_at_10pct": _precision_at_budget(valid_scores, valid_labels, 0.10),
        "router/random_precision": float(valid_labels.float().mean().item()),
        "router/routed_layer_count": float(len(indices)),
        "router/valid_tokens": float(valid.float().sum().item()),
    }
    return loss, metrics, avg_scores
