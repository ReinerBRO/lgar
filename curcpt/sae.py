from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import masked_kl_divergence


class SparseAutoencoder(nn.Module):
    """Minimal ReLU sparse autoencoder for residual-stream features."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        input_mean: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(int(input_dim), int(feature_dim))
        self.decoder = nn.Linear(int(feature_dim), int(input_dim))
        mean = torch.zeros(int(input_dim)) if input_mean is None else input_mean.float().clone()
        scale = torch.ones(int(input_dim)) if input_scale is None else input_scale.float().clone()
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_scale", scale.clamp(min=1.0e-6))

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        x = (residual.float() - self.input_mean) / self.input_scale
        return F.relu(self.encoder(x))

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        x = self.decoder(acts.float())
        return x * self.input_scale + self.input_mean

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(residual)
        return self.decode(acts), acts


@dataclass(frozen=True)
class EvidenceFeatureSet:
    layer: int
    hook_point: str
    feature_ids: tuple[int, ...]
    sae_checkpoint: str | None
    validation_passed: bool
    scores: dict[str, Any]
    config: dict[str, Any]


def _state_dict_from_checkpoint(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "sae", "sae_state_dict"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise ValueError("cannot find SAE state_dict in checkpoint")


def load_sparse_autoencoder(path: str | Path, device: torch.device) -> SparseAutoencoder:
    ckpt = torch.load(str(path), map_location="cpu")
    state = _state_dict_from_checkpoint(ckpt)
    enc_w = state.get("encoder.weight")
    dec_w = state.get("decoder.weight")
    if enc_w is None or dec_w is None:
        raise ValueError("SAE checkpoint must contain encoder.weight and decoder.weight")
    input_dim = int(enc_w.shape[1])
    feature_dim = int(enc_w.shape[0])
    mean = state.get("input_mean")
    scale = state.get("input_scale")
    sae = SparseAutoencoder(input_dim, feature_dim, mean, scale)
    sae.load_state_dict(state, strict=False)
    sae.to(device)
    sae.eval()
    for param in sae.parameters():
        param.requires_grad_(False)
    return sae


def load_evidence_feature_set(
    path: str | Path,
    require_validated: bool = True,
) -> EvidenceFeatureSet:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(
                "training currently expects one validated evidence-feature layer; "
                f"got {len(payload)} entries"
            )
        payload = payload[0]
    if "layers" in payload:
        layers = payload["layers"]
        if not isinstance(layers, list) or len(layers) != 1:
            raise ValueError(
                "training currently expects one validated evidence-feature layer; "
                f"got layers={layers!r}"
            )
        payload = layers[0]

    feature_ids = tuple(int(x) for x in payload.get("feature_ids", []))
    if not feature_ids:
        raise ValueError("evidence feature set has no feature_ids")
    hook_point = str(payload.get("hook_point", "resid_mid"))
    if hook_point != "resid_mid":
        raise ValueError(f"unsupported SAE hook_point={hook_point!r}; expected resid_mid")
    validation = payload.get("validation", {})
    passed = bool(payload.get("validation_passed", validation.get("passed", False)))
    if require_validated and not passed:
        raise ValueError(
            "SAE steering requires causally validated evidence features. "
            "Set --allow-unvalidated-sae-features only for debugging, not for reported runs."
        )
    if require_validated and passed:
        required_logp_checks = (
            "selected_ablation_delta_logp",
            "masked_head_answer_delta_logp",
        )
        missing = [key for key in required_logp_checks if key not in validation]
        if missing:
            raise ValueError(
                "SAE steering requires answer-level causal validation fields: "
                + ", ".join(missing)
            )
        if float(validation["selected_ablation_delta_logp"]) <= 0.0:
            raise ValueError("SAE feature ablation did not lower answer logprob")
        if float(validation["masked_head_answer_delta_logp"]) <= 0.0:
            raise ValueError("masked retrieval heads did not lower answer logprob")
    return EvidenceFeatureSet(
        layer=int(payload["layer"]),
        hook_point=hook_point,
        feature_ids=feature_ids,
        sae_checkpoint=payload.get("sae_checkpoint"),
        validation_passed=passed,
        scores=dict(payload.get("scores", {})),
        config=dict(payload.get("mining_config", payload.get("config", {}))),
    )


def _selected_feature_acts(
    sae: SparseAutoencoder,
    residual: torch.Tensor,
    feature_ids: tuple[int, ...],
) -> torch.Tensor:
    acts = sae.encode(residual)
    index = torch.as_tensor(feature_ids, device=acts.device, dtype=torch.long)
    return acts.index_select(dim=-1, index=index)


def sae_decoder_feature_contribution(
    sae: SparseAutoencoder,
    feature_values: torch.Tensor,
    feature_ids: tuple[int, ...],
) -> torch.Tensor:
    """Decode selected feature coefficients into residual-space contribution."""
    index = torch.as_tensor(feature_ids, device=feature_values.device, dtype=torch.long)
    vectors = sae.decoder.weight[:, index].T.to(device=feature_values.device, dtype=feature_values.dtype)
    vectors = vectors * sae.input_scale.to(device=feature_values.device, dtype=feature_values.dtype)[None, :]
    return torch.einsum("...k,kh->...h", feature_values, vectors)


def sae_feature_losses(
    residual_by_layer: dict[int, torch.Tensor],
    sae: SparseAutoencoder,
    features: EvidenceFeatureSet,
    target_full: torch.Tensor,
    target_neg: torch.Tensor,
    target_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    high_utility_mask: torch.Tensor,
    beta_margin: float,
    beta_match: float,
    gamma: float,
    match_clip: float | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Feature margin/matching losses for validated evidence features.

    The loss is defined only on tokens that are valid, high-utility, and marked
    by the precomputed SAE target mask. It never invents targets online.
    """

    if features.layer not in residual_by_layer:
        raise KeyError(f"missing captured resid_mid for SAE layer {features.layer}")
    residual = residual_by_layer[features.layer]
    theta = _selected_feature_acts(sae, residual, features.feature_ids)

    if target_full.shape != theta.shape or target_neg.shape != theta.shape:
        raise ValueError(
            "SAE target arrays must have shape [batch, seq, num_features] matching "
            f"feature_ids; got full={tuple(target_full.shape)} neg={tuple(target_neg.shape)} "
            f"theta={tuple(theta.shape)}"
        )
    token_target_mask = target_mask.bool()
    if token_target_mask.ndim == 3:
        token_target_mask = token_target_mask.any(dim=-1)
    token_mask = token_target_mask & valid_mask.bool() & high_utility_mask.bool()
    if token_mask.any():
        theta_sel = theta[token_mask]
        neg_sel = target_neg.detach()[token_mask].to(theta_sel.dtype)
        full_sel = target_full.detach()[token_mask].to(theta_sel.dtype)
        raw_margin = theta_sel - neg_sel
        margin = F.relu(float(gamma) - raw_margin).mean()
        if match_clip is not None:
            full_sel = full_sel.clamp(max=float(match_clip))
        match = (theta_sel - full_sel).float().pow(2).mean()
        margin_value = float(raw_margin.detach().mean().item())
        theta_mean = float(theta_sel.detach().mean().item())
        full_mean = float(full_sel.detach().mean().item())
        neg_mean = float(neg_sel.detach().mean().item())
    else:
        margin = theta.sum() * 0.0
        match = theta.sum() * 0.0
        margin_value = 0.0
        theta_mean = 0.0
        full_mean = 0.0
        neg_mean = 0.0

    losses: dict[str, torch.Tensor] = {}
    if beta_margin > 0.0:
        losses["sae_margin"] = float(beta_margin) * margin
    if beta_match > 0.0:
        losses["sae_match"] = float(beta_match) * match

    selected_count = float(token_mask.sum().item())
    metrics = {
        "sae/enabled": 1.0,
        "sae/layer": float(features.layer),
        "sae/num_features": float(len(features.feature_ids)),
        "sae/target_token_count": selected_count,
        "sae/target_token_fraction": selected_count / max(1.0, float(valid_mask.sum().item())),
        "sae/feature_margin": margin_value,
        "sae/theta_feature_mean": theta_mean,
        "sae/full0_feature_mean": full_mean,
        "sae/neg0_feature_mean": neg_mean,
        "loss/sae_margin_raw": float(margin.detach().item()),
        "loss/sae_match_raw": float(match.detach().item()),
        "loss/sae_margin_total": float(losses.get("sae_margin", margin.new_tensor(0.0)).detach().item()),
        "loss/sae_match_total": float(losses.get("sae_match", match.new_tensor(0.0)).detach().item()),
    }
    return losses, metrics


def short_kl_loss(
    active_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    high_utility_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_short_kl: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    short_mask = (~high_utility_mask.bool()) & valid_mask.bool()
    kl = masked_kl_divergence(teacher_logits.detach(), active_logits, short_mask)
    total = float(lambda_short_kl) * kl
    count = float(short_mask.sum().item())
    valid_count = float(valid_mask.sum().item())
    return total, {
        "loss/short_kl": float(kl.detach().item()),
        "loss/short_kl_total": float(total.detach().item()),
        "utility/short_kl_count": count,
        "utility/short_kl_fraction": count / max(1.0, valid_count),
    }


def _context_mask_from_selected(
    doc_ids: torch.Tensor,
    selected: torch.Tensor,
    window: int,
) -> torch.Tensor:
    bsz, seq_len = selected.shape
    context = torch.zeros_like(selected, dtype=torch.bool)
    coords = selected.nonzero(as_tuple=False)
    for row in coords:
        b = int(row[0].item())
        t = int(row[1].item())
        start = max(0, t - int(window) + 1)
        same_doc = doc_ids[b, start : t + 1] == doc_ids[b, t]
        context[b, start : t + 1] |= same_doc & (doc_ids[b, start : t + 1] >= 0)
    return context


def lsor_loss(
    adapter_writes_by_layer: dict[int, torch.Tensor],
    residual_by_layer: dict[int, torch.Tensor],
    doc_ids: torch.Tensor,
    selected_mask: torch.Tensor,
    top_k: int,
    window: int,
    max_context_tokens: int,
    lambda_lsor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize adapter residual writes inside the local dominant subspace."""

    total: torch.Tensor | None = None
    ratios: list[torch.Tensor] = []
    active_layers = 0
    selected = selected_mask.bool() & (doc_ids >= 0)
    for layer, dz in adapter_writes_by_layer.items():
        if layer not in residual_by_layer or not selected.any():
            continue
        dz_sel = dz[selected].float()
        if dz_sel.numel() == 0:
            continue
        context_mask = _context_mask_from_selected(doc_ids, selected, window)
        h_local = residual_by_layer[layer][context_mask].detach().float()
        if h_local.shape[0] < 2:
            continue
        if h_local.shape[0] > int(max_context_tokens):
            h_local = h_local[: int(max_context_tokens)]
        h_local = h_local - h_local.mean(dim=0, keepdim=True)
        rank = min(int(top_k), int(h_local.shape[0]) - 1, int(h_local.shape[1]))
        if rank <= 0:
            continue
        _, _, vh = torch.linalg.svd(h_local, full_matrices=False)
        basis = vh[:rank].detach().to(device=dz_sel.device, dtype=dz_sel.dtype)
        proj = dz_sel @ basis.T @ basis
        ratio = proj.float().pow(2).sum(dim=-1) / dz_sel.float().pow(2).sum(dim=-1).clamp(min=1.0e-12)
        layer_loss = ratio.mean()
        total = layer_loss if total is None else total + layer_loss
        ratios.append(layer_loss.detach())
        active_layers += 1

    if total is None:
        base = next(iter(adapter_writes_by_layer.values()), None)
        if base is None:
            base = next(iter(residual_by_layer.values()), None)
        if base is None:
            raise ValueError("LSOR requested but no adapter writes or residual captures were provided")
        raw = base.sum() * 0.0
        ratio_mean = 0.0
    else:
        raw = total / max(1, active_layers)
        ratio_mean = float(torch.stack(ratios).mean().item()) if ratios else 0.0
    weighted = float(lambda_lsor) * raw
    return weighted, {
        "loss/lsor": float(raw.detach().item()),
        "loss/lsor_total": float(weighted.detach().item()),
        "lsor/projection_ratio": ratio_mean,
        "lsor/active_layers": float(active_layers),
    }
