from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import LGARParams, routed_layer_indices
from .mining import lgar_routed_attention_mask, select_topk_global_queries
from .router import SharedQueryRouter


@dataclass
class LGARForwardOutput:
    logits: torch.Tensor
    router_loss: torch.Tensor
    router_scores: torch.Tensor | None
    metrics: dict[str, float]


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
    return float(((rank_sum_pos - pos * (pos + 1) / 2.0) / max(1.0, pos * neg)).item())


def _precision_at_budget(scores: torch.Tensor, labels: torch.Tensor, budget: float) -> float:
    if scores.numel() == 0:
        return float("nan")
    k = max(1, min(scores.numel(), int(torch.ceil(torch.tensor(scores.numel() * float(budget))).item())))
    top = torch.topk(scores.float(), k=k).indices
    return float(labels.bool()[top].float().mean().item())


def _position_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)


def qwen_lgar_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    doc_ids: torch.Tensor,
    router: SharedQueryRouter | None,
    params: LGARParams,
    label_targets: torch.Tensor | None = None,
    label_valid_mask: torch.Tensor | None = None,
    routing_valid_mask: torch.Tensor | None = None,
    force_global_query_mask: torch.Tensor | None = None,
    mode: str = "full",
    target_budget: float | None = None,
    router_budget_target: float | None = None,
) -> LGARForwardOutput:
    """Forward Qwen with optional upper-layer local/global query routing.

    `mode`:
    - `full`: no router, full document-aware attention in every layer.
    - `router_aux`: full attention everywhere, but train/log the upper-layer router.
    - `routed`: lower layers full, selected upper layers use per-layer top-k routed masks.
    """

    if mode not in {"full", "router_aux", "routed"}:
        raise ValueError(f"unknown L-GAR mode: {mode}")
    if mode in {"router_aux", "routed"} and router is None:
        raise ValueError(f"{mode} requires a router")

    backbone = model.model
    hidden_states = backbone.embed_tokens(input_ids)
    position_ids = _position_ids(input_ids)
    position_embeddings = backbone.rotary_emb(hidden_states, position_ids)
    selected_layers = set(routed_layer_indices(int(model.config.num_hidden_layers), params.routed_layer_fraction))
    budget = float(params.final_global_budget if target_budget is None else target_budget)
    budget_target = float(params.router_target_budget if router_budget_target is None else router_budget_target)

    router_losses: list[torch.Tensor] = []
    budget_losses: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    layer_scores: list[torch.Tensor] = []
    actual_budgets: list[float] = []

    if routing_valid_mask is None:
        routing_valid_mask = doc_ids >= 0
    routing_valid_mask = routing_valid_mask.bool()
    forced_global = force_global_query_mask.bool() if force_global_query_mask is not None else None
    if forced_global is not None and forced_global.shape != routing_valid_mask.shape:
        raise ValueError("force_global_query_mask must have the same shape as routing_valid_mask")
    label_valid = label_valid_mask.bool() if label_valid_mask is not None else None
    label_float = label_targets.float() if label_targets is not None else None

    for layer_idx, decoder_layer in enumerate(backbone.layers[: int(model.config.num_hidden_layers)]):
        attention_mask = full_attention_mask
        if router is not None and layer_idx in selected_layers:
            # The router is supervised from Qwen features, but its auxiliary
            # loss should not backpropagate through every upper backbone layer.
            # Routed attention remains driven by the live router scores.
            scores = router(hidden_states.detach())
            layer_scores.append(scores)
            if label_valid is not None and label_float is not None and label_valid.any():
                scores_valid = scores[label_valid].float().clamp(1e-6, 1.0 - 1e-6)
                labels_valid = label_float[label_valid]
                router_losses.append(F.binary_cross_entropy(scores_valid, labels_valid))
                budget_losses.append((scores_valid.mean() - budget_target).pow(2))
                entropies.append(
                    -(scores_valid * scores_valid.log() + (1.0 - scores_valid) * (1.0 - scores_valid).log()).mean()
                )
            if mode == "routed":
                global_queries = select_topk_global_queries(scores.detach(), routing_valid_mask, budget)
                if forced_global is not None:
                    global_queries = global_queries | (forced_global & routing_valid_mask)
                mask_format = "bool" if full_attention_mask.dtype == torch.bool else "additive"
                attention_mask = lgar_routed_attention_mask(
                    doc_ids,
                    global_queries,
                    params.local_window,
                    mask_format=mask_format,
                )
                if routing_valid_mask.any():
                    actual_budgets.append(float(global_queries.float()[routing_valid_mask].mean().item()))

        def layer_forward(layer_hidden_states: torch.Tensor) -> torch.Tensor:
            return decoder_layer(
                layer_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )

        if torch.is_grad_enabled() and bool(getattr(model, "is_gradient_checkpointing", False)) and hidden_states.requires_grad:
            hidden_states = checkpoint(layer_forward, hidden_states, use_reentrant=False, preserve_rng_state=False)
        else:
            hidden_states = layer_forward(hidden_states)

    hidden_states = backbone.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    z = logits.new_zeros(())
    avg_scores = torch.stack(layer_scores, dim=0).mean(dim=0) if layer_scores else None
    router_zero = avg_scores.float().sum() * 0.0 if avg_scores is not None else z
    if router_losses:
        bce = torch.stack(router_losses).mean()
        budget_loss = torch.stack(budget_losses).mean()
        entropy = torch.stack(entropies).mean()
        router_loss = (
            float(params.lambda_router_final) * bce
            + float(params.lambda_budget) * budget_loss
            - float(params.entropy_weight) * entropy
        )
    else:
        bce = z
        budget_loss = z
        entropy = z
        router_loss = router_zero

    metrics = {
        "router/loss_bce": float(bce.detach().float().item()),
        "router/loss_budget": float(budget_loss.detach().float().item()),
        "router/loss_aux_weighted": float(router_loss.detach().float().item()),
        "router/entropy": float(entropy.detach().float().item()),
        "router/routed_layer_count": float(len(selected_layers)),
        "router/actual_budget": float(sum(actual_budgets) / len(actual_budgets)) if actual_budgets else 1.0,
        "router/score_mean": 0.0,
        "router/positive_fraction": 0.0,
        "router/auc": 0.5,
        "router/precision_at_5pct": 0.0,
        "router/precision_at_10pct": 0.0,
        "router/random_precision": 0.0,
    }
    if avg_scores is not None and label_valid is not None and label_float is not None and label_valid.any():
        scores_valid = avg_scores[label_valid].detach()
        labels_valid_bool = label_targets[label_valid].detach().bool()
        metrics.update(
            {
                "router/score_mean": float(scores_valid.float().mean().item()),
                "router/positive_fraction": float(labels_valid_bool.float().mean().item()),
                "router/auc": _binary_auc(scores_valid, labels_valid_bool),
                "router/precision_at_5pct": _precision_at_budget(scores_valid, labels_valid_bool, 0.05),
                "router/precision_at_10pct": _precision_at_budget(scores_valid, labels_valid_bool, 0.10),
                "router/random_precision": float(labels_valid_bool.float().mean().item()),
            }
        )
    return LGARForwardOutput(logits=logits, router_loss=router_loss, router_scores=avg_scores, metrics=metrics)
