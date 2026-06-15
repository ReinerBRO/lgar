from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss averaged over positions where mask is True.

    Gathers the masked positions BEFORE the vocab softmax so memory scales with
    the number of selected tokens, not B*T. With a sparse mask (e.g. high-utility
    positions) this avoids materializing a full [B, T, V] float tensor.

    Args:
        logits: [B, T, V]
        labels: [B, T]
        mask: [B, T] bool
    """
    if mask.any():
        sel_logits = logits[mask]  # [N, V]
        sel_labels = labels[mask]  # [N]
        return F.cross_entropy(sel_logits.float(), sel_labels)
    # Grad-connected zero: keep the autograd graph intact when no positions are
    # selected (e.g. a batch with no high-utility tokens), so downstream
    # torch.autograd.grad / .backward does not see a grad-less constant.
    return logits.sum() * 0.0


def select_cure_main_loss_mask(
    loss_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    high_utility_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Select the CURE main CE positions.

    Modes:
      - all: the original CE baseline objective over all real tokens.
      - valid_remote: mined long-short candidate tokens (remote + filters).
      - high_utility: high-utility tokens within the mined candidate set.
      - none: no main CE; returns an empty mask so CE is a grad-connected zero.
    """
    loss_mask = loss_mask.bool()
    valid_mask = valid_mask.bool()
    high_utility_mask = high_utility_mask.bool()
    mode = str(mode)
    if mode == "all":
        return loss_mask
    if mode == "valid_remote":
        return valid_mask & loss_mask
    if mode == "high_utility":
        return high_utility_mask & valid_mask & loss_mask
    if mode == "none":
        return torch.zeros_like(loss_mask, dtype=torch.bool)
    raise ValueError(
        f"unsupported cure_main_loss_mask={mode!r}; "
        "expected one of: all, valid_remote, high_utility, none"
    )


def masked_kl_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """KL(teacher || student) averaged over positions where mask is True.

    Teacher logits are detached (no gradient to teacher). Student logits retain
    gradient. Masked positions are gathered BEFORE the softmax so memory scales
    with the number of selected tokens, not B*T*V.
    """
    if mask.any():
        t_sel = teacher_logits[mask]  # [N, V]
        s_sel = student_logits[mask]  # [N, V]
        with torch.no_grad():
            teacher_logp = F.log_softmax(t_sel.float(), dim=-1)
            teacher_p = teacher_logp.exp()
        student_logp = F.log_softmax(s_sel.float(), dim=-1)
        kl = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)  # [N]
        return kl.mean()
    # Grad-connected zero (depends on student_logits) so the adapter grad path
    # stays intact when no high-utility positions are present this step.
    return student_logits.sum() * 0.0


def coverage_loss(
    retrieval_scores: torch.Tensor,
    chunk_attention_mass: torch.Tensor,
    positive_chunks: torch.Tensor,
) -> torch.Tensor:
    """Coverage loss: different retrieval heads should cover different positive chunks.

    Args:
        retrieval_scores: [B, T] mean router scores (not used in v1, placeholder)
        chunk_attention_mass: [n_rh, n_chunks] attention mass per head per chunk
        positive_chunks: [n_chunks] bool

    Returns:
        scalar loss
    """
    if not positive_chunks.any():
        return chunk_attention_mass.new_tensor(0.0)

    pos_mass = chunk_attention_mass[:, positive_chunks]  # [n_rh, n_pos]
    max_per_chunk = pos_mass.max(dim=0).values  # [n_pos]
    # Penalize if no head covers a chunk well
    return -torch.log(max_per_chunk.clamp(min=1e-6)).mean()


def cure_ce_loss(
    logits_full: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_CE: standard CE on all valid tokens. Backprop through all params."""
    l_ce = masked_cross_entropy(logits_full, labels, valid_mask)
    return l_ce, {"loss/ce": float(l_ce.detach().item())}


def cure_rh_loss(
    logits_full: torch.Tensor,
    logits_rh_only: torch.Tensor,
    labels: torch.Tensor,
    high_utility_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_rh_ce: float,
    lambda_rh_kd: float,
    lambda_cov: float,
    step_tokens: int,
    cov_warmup_tokens: int,
    chunk_attention_mass: torch.Tensor | None = None,
    positive_chunks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_RH-CE + L_RH-KD + L_cov: adapter-only losses.

    Gradient from this loss should only flow to retrieval-head adapters.
    Use torch.autograd.grad with inputs=adapter_params.
    """
    hu_valid = high_utility_mask & valid_mask
    # L_RH-CE: CE on logits_rh_only (adapter-affected) at high-utility positions
    # Gradient flows only to adapters since logits_rh_only depends on adapters
    l_rh_ce = masked_cross_entropy(logits_rh_only, labels, hu_valid)
    # L_RH-KD: KL(full || RH-only), teacher detached, gradient to adapters only
    l_rh_kd = masked_kl_divergence(logits_full, logits_rh_only, hu_valid)

    l_cov_val = logits_full.new_tensor(0.0)
    cov_active = step_tokens >= cov_warmup_tokens and lambda_cov > 0
    if cov_active and chunk_attention_mass is not None and positive_chunks is not None:
        l_cov_val = coverage_loss(None, chunk_attention_mass, positive_chunks)

    total = lambda_rh_ce * l_rh_ce + lambda_rh_kd * l_rh_kd + lambda_cov * l_cov_val

    hu_count = float(hu_valid.sum().item())
    metrics = {
        "loss/rh_ce": float(l_rh_ce.detach().item()),
        "loss/rh_kd": float(l_rh_kd.detach().item()),
        "loss/cov": float(l_cov_val.detach().item()),
        "loss/rh_total": float(total.detach().item()),
        "utility/high_utility_count": hu_count,
        "utility/high_utility_fraction": hu_count / max(1.0, float(valid_mask.sum().item())),
    }
    return total, metrics


def cure_full_hu_loss(
    logits_full: torch.Tensor,
    labels: torch.Tensor,
    high_utility_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_full_hu_ce: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full-path high-utility CE used as an adapter-only auxiliary loss.

    This loss is computed on the normal full-attention logits, not on the
    RH-bottleneck path. The caller must apply it with
    `torch.autograd.grad(..., inputs=adapter_params)` so it updates adapters
    without changing base-model gradients.
    """
    hu_valid = high_utility_mask & valid_mask
    l_full_hu_ce = masked_cross_entropy(logits_full, labels, hu_valid)
    total = float(lambda_full_hu_ce) * l_full_hu_ce
    hu_count = float(hu_valid.sum().item())
    valid_count = float(valid_mask.sum().item())
    return total, {
        "loss/full_hu_ce": float(l_full_hu_ce.detach().item()),
        "loss/full_hu_total": float(total.detach().item()),
        "utility/full_hu_count": hu_count,
        "utility/full_hu_fraction": hu_count / max(1.0, valid_count),
    }


def cure_nonhu_logp_consistency_loss(
    active_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    high_utility_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_nonhu_logp: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep adapters from drifting on non-high-utility remote candidates.

    The teacher is the adapter-off CE checkpoint path. Matching only the gold
    token log-prob keeps this loss cheap enough for the online mined setting
    while directly preserving the probability of ordinary target tokens.
    """
    nonhu_valid = (~high_utility_mask.bool()) & valid_mask.bool()
    if nonhu_valid.any():
        chunk_size = max(1, int(os.environ.get("CURE_LOGP_CHUNK_SIZE", "512")))
        coords = nonhu_valid.nonzero(as_tuple=False)
        active_chunks: list[torch.Tensor] = []
        teacher_chunks: list[torch.Tensor] = []
        for start in range(0, coords.shape[0], chunk_size):
            end = min(start + chunk_size, coords.shape[0])
            rows = coords[start:end, 0]
            cols = coords[start:end, 1]
            chunk_labels = labels[rows, cols]
            active_chunks.append(
                F.log_softmax(active_logits[rows, cols].float(), dim=-1)
                .gather(-1, chunk_labels.unsqueeze(-1))
                .squeeze(-1)
            )
            with torch.no_grad():
                teacher_chunks.append(
                    F.log_softmax(teacher_logits[rows, cols].float(), dim=-1)
                    .gather(-1, chunk_labels.unsqueeze(-1))
                    .squeeze(-1)
                )
        active_logp = torch.cat(active_chunks, dim=0)
        with torch.no_grad():
            teacher_logp = torch.cat(teacher_chunks, dim=0)
        mse = (active_logp - teacher_logp).float().pow(2).mean()
    else:
        mse = active_logits.sum() * 0.0
    total = float(lambda_nonhu_logp) * mse
    count = float(nonhu_valid.sum().item())
    valid_count = float(valid_mask.sum().item())
    return total, {
        "loss/nonhu_logp_mse": float(mse.detach().item()),
        "loss/nonhu_logp_total": float(total.detach().item()),
        "utility/nonhu_count": count,
        "utility/nonhu_fraction": count / max(1.0, valid_count),
    }


def apply_isolated_adapter_grads(
    l_ce: torch.Tensor,
    adapter_losses: dict[str, torch.Tensor],
    adapter_params: list[nn.Parameter],
    base_frozen: bool = False,
) -> dict[str, float]:
    """Backprop L_CE into all params, then add auxiliary grads to adapters only.

    Gradient isolation contract:
      - L_CE updates every trainable param (base model + adapters).
      - Auxiliary losses update ONLY retrieval-head adapters. Their gradients
        must never reach base-model params.

    When the base model is frozen, all trainable params are adapters, so the
    memory-safe path is a single backward on `L_CE + auxiliary_losses`. When
    the base model is trainable, this uses `l_ce.backward(retain_graph=...)`
    followed by explicit `torch.autograd.grad(loss, adapter_params)` calls.
    Since the second stage restricts `inputs=adapter_params`, autograd never
    populates `.grad` on base params from auxiliary losses.

    Caller must zero grads before this and run optimizer.step() after.

    Returns adapter CE/auxiliary gradient norms so CURE can detect whether the
    auxiliary objective is numerically dominated by the base CE objective.
    """
    active_losses = {
        name: loss for name, loss in adapter_losses.items()
        if loss is not None
    }
    if base_frozen:
        total = l_ce + sum(active_losses.values(), start=l_ce.new_tensor(0.0))
        total.backward()
        total_sq = l_ce.new_tensor(0.0)
        for param in adapter_params:
            if param.grad is not None:
                total_sq = total_sq + param.grad.detach().float().pow(2).sum()
        total_norm = total_sq.sqrt()
        metrics = {
            "grad/adapter_ce_norm": float(total_norm.detach().item()),
            "grad/adapter_aux_norm": 0.0,
            "grad/adapter_aux_to_ce_ratio": 0.0,
            "grad/adapter_total_norm": float(total_norm.detach().item()),
            "grad/adapter_frozen_single_backward": 1.0,
        }
        for name in active_losses:
            metrics[f"grad/adapter_{name}_norm"] = 0.0
            metrics[f"grad/adapter_{name}_to_ce_ratio"] = 0.0
        return metrics

    if os.environ.get("CURE_REQUIRE_FREEZE_BASE", "").strip().lower() not in {"", "0", "false", "no", "off"}:
        raise RuntimeError(
            "CURE_REQUIRE_FREEZE_BASE=1 but apply_isolated_adapter_grads received "
            "base_frozen=False. The launcher did not pass --freeze-base-model."
        )

    l_ce.backward(retain_graph=bool(active_losses))
    ce_sq = l_ce.new_tensor(0.0)
    for param in adapter_params:
        if param.grad is not None:
            ce_sq = ce_sq + param.grad.detach().float().pow(2).sum()
    ce_norm = ce_sq.sqrt()

    metrics = {
        "grad/adapter_ce_norm": float(ce_norm.detach().item()),
        "grad/adapter_aux_norm": 0.0,
        "grad/adapter_aux_to_ce_ratio": 0.0,
    }
    if not active_losses or not adapter_params:
        return metrics

    aux_sq_total = l_ce.new_tensor(0.0)
    items = list(active_losses.items())
    for idx, (name, loss) in enumerate(items):
        loss_sq = l_ce.new_tensor(0.0)
        if loss.requires_grad:
            grads = torch.autograd.grad(
                loss,
                adapter_params,
                retain_graph=idx < len(items) - 1,
                allow_unused=True,
            )
            for param, grad in zip(adapter_params, grads):
                if grad is None:
                    continue
                loss_sq = loss_sq + grad.detach().float().pow(2).sum()
                param.grad = (param.grad + grad) if param.grad is not None else grad
        loss_norm = loss_sq.sqrt()
        aux_sq_total = aux_sq_total + loss_sq
        ratio = loss_norm / ce_norm.clamp(min=1.0e-12)
        metrics[f"grad/adapter_{name}_norm"] = float(loss_norm.detach().item())
        metrics[f"grad/adapter_{name}_to_ce_ratio"] = float(ratio.detach().item())

    aux_norm = aux_sq_total.sqrt()
    aux_ratio = aux_norm / ce_norm.clamp(min=1.0e-12)
    metrics["grad/adapter_aux_norm"] = float(aux_norm.detach().item())
    metrics["grad/adapter_aux_to_ce_ratio"] = float(aux_ratio.detach().item())
    return metrics


def apply_isolated_grads(
    l_ce: torch.Tensor,
    l_rh: torch.Tensor,
    adapter_params: list[nn.Parameter],
) -> dict[str, float]:
    """Backward-compatible wrapper for the original RH-only isolation helper."""
    return apply_isolated_adapter_grads(l_ce, {"rh": l_rh}, adapter_params)
