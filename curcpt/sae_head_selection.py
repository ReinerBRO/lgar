from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lgar_cpt.modeling import qwen_layers

from .sae import EvidenceFeatureSet, SparseAutoencoder


@dataclass
class HeadWriteStats:
    high_sum: float = 0.0
    high_abs_sum: float = 0.0
    high_count: int = 0
    nonhigh_sum: float = 0.0
    nonhigh_abs_sum: float = 0.0
    nonhigh_count: int = 0

    def update(self, token_scores: torch.Tensor, high_mask: torch.Tensor, nonhigh_mask: torch.Tensor) -> None:
        scores = token_scores.detach().float()
        high = high_mask.bool()
        nonhigh = nonhigh_mask.bool()
        if high.any():
            selected = scores[high]
            self.high_sum += float(selected.sum().item())
            self.high_abs_sum += float(selected.abs().sum().item())
            self.high_count += int(selected.numel())
        if nonhigh.any():
            selected = scores[nonhigh]
            self.nonhigh_sum += float(selected.sum().item())
            self.nonhigh_abs_sum += float(selected.abs().sum().item())
            self.nonhigh_count += int(selected.numel())

    def as_metrics(self) -> dict[str, float]:
        high_mean = self.high_sum / max(1, self.high_count)
        nonhigh_mean = self.nonhigh_sum / max(1, self.nonhigh_count)
        high_abs_mean = self.high_abs_sum / max(1, self.high_count)
        nonhigh_abs_mean = self.nonhigh_abs_sum / max(1, self.nonhigh_count)
        specificity = high_mean - nonhigh_mean
        return {
            "score": specificity,
            "specificity": specificity,
            "high_mean": high_mean,
            "nonhigh_mean": nonhigh_mean,
            "high_abs_mean": high_abs_mean,
            "nonhigh_abs_mean": nonhigh_abs_mean,
            "high_count": float(self.high_count),
            "nonhigh_count": float(self.nonhigh_count),
        }


def parse_layer_spec(value: str, max_layers: int) -> list[int]:
    """Parse layer specs like '16-21,23'."""
    text = str(value).strip()
    if text in {"", "all"}:
        return list(range(int(max_layers)))
    layers: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"invalid layer range {part!r}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    unique = sorted(set(layers))
    bad = [layer for layer in unique if layer < 0 or layer >= int(max_layers)]
    if bad:
        raise ValueError(f"layer ids out of range 0..{int(max_layers) - 1}: {bad}")
    return unique


def load_candidate_heads(
    *,
    candidate_layers: list[int],
    num_heads: int,
    heads_json: str | Path | None = None,
) -> list[tuple[int, int]]:
    if heads_json is None:
        return [(int(layer), int(head)) for layer in candidate_layers for head in range(int(num_heads))]
    payload = json.loads(Path(heads_json).read_text(encoding="utf-8"))
    heads = [(int(layer), int(head)) for layer, head in payload.get("retrieval_heads", [])]
    layer_set = set(int(layer) for layer in candidate_layers)
    selected = [(layer, head) for layer, head in heads if layer in layer_set]
    bad = [(layer, head) for layer, head in selected if head < 0 or head >= int(num_heads)]
    if bad:
        raise ValueError(f"candidate head ids out of range 0..{int(num_heads) - 1}: {bad}")
    return selected


def evidence_decoder_vectors(
    sae: SparseAutoencoder,
    features: EvidenceFeatureSet,
    *,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    feature_ids = torch.as_tensor(features.feature_ids, dtype=torch.long, device=device)
    vectors = sae.decoder.weight[:, feature_ids].T.to(device=device, dtype=torch.float32)
    vectors = vectors * sae.input_scale.to(device=device, dtype=torch.float32)[None, :]
    if normalize:
        vectors = F.normalize(vectors, dim=-1, eps=1.0e-6)
    return vectors


def head_feature_projection_weight(
    o_proj_weight: torch.Tensor,
    decoder_vectors: torch.Tensor,
    *,
    head_id: int,
    head_dim: int,
) -> torch.Tensor:
    start = int(head_id) * int(head_dim)
    end = start + int(head_dim)
    if start < 0 or end > int(o_proj_weight.shape[1]):
        raise ValueError(
            f"head {head_id} slice [{start}:{end}] is outside o_proj input dim {o_proj_weight.shape[1]}"
        )
    weight_slice = o_proj_weight[:, start:end].detach().float()
    return torch.matmul(decoder_vectors.detach().float(), weight_slice)


def select_top_heads(
    head_scores: dict[str, dict[str, float]],
    *,
    top_k: int,
    max_heads_per_layer: int | None = None,
) -> list[list[int]]:
    ranked = sorted(
        head_scores.items(),
        key=lambda item: (
            float(item[1].get("score", 0.0)),
            float(item[1].get("high_mean", 0.0)),
            int(item[0].split("_")[0]),
            int(item[0].split("_")[1]),
        ),
        reverse=True,
    )
    selected: list[list[int]] = []
    per_layer: dict[int, int] = {}
    for key, _metrics in ranked:
        layer_s, head_s = key.split("_", 1)
        layer = int(layer_s)
        head = int(head_s)
        if max_heads_per_layer is not None and per_layer.get(layer, 0) >= int(max_heads_per_layer):
            continue
        selected.append([layer, head])
        per_layer[layer] = per_layer.get(layer, 0) + 1
        if len(selected) >= int(top_k):
            break
    return selected


def build_selection_payload(
    *,
    retrieval_heads: list[list[int]],
    head_scores: dict[str, dict[str, float]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "retrieval_heads": [[int(layer), int(head)] for layer, head in retrieval_heads],
        "num_selected": int(len(retrieval_heads)),
        "source": "sae_guided_evidence_write_selection",
        "selection_rule": "rank heads by O-output projection onto validated SAE evidence decoder directions on LSD-positive tokens",
        "head_scores": head_scores,
        "metadata": metadata,
    }


class SAEHeadWriteScorer:
    """Accumulate per-head evidence-write scores from o_proj inputs.

    A Qwen attention block concatenates per-head attention outputs before
    ``o_proj``. For a head slice x_h and base W_O slice W_h, the residual write is
    x_h @ W_h.T. Projecting that write onto selected SAE decoder directions
    tells us whether this head naturally writes the evidence features.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        candidate_heads: list[tuple[int, int]],
        decoder_vectors: torch.Tensor,
        *,
        head_dim: int,
    ) -> None:
        self.model = model
        self.head_dim = int(head_dim)
        self.stats = {f"{int(layer)}_{int(head)}": HeadWriteStats() for layer, head in candidate_heads}
        self.heads_by_layer: dict[int, list[int]] = {}
        for layer, head in candidate_heads:
            self.heads_by_layer.setdefault(int(layer), []).append(int(head))
        self.projection_weights: dict[int, dict[int, torch.Tensor]] = {}
        layers = qwen_layers(model)
        for layer, heads in self.heads_by_layer.items():
            o_proj_weight = layers[int(layer)].self_attn.o_proj.weight
            self.projection_weights[int(layer)] = {
                int(head): head_feature_projection_weight(
                    o_proj_weight,
                    decoder_vectors,
                    head_id=int(head),
                    head_dim=self.head_dim,
                ).to(device=decoder_vectors.device, dtype=torch.float32)
                for head in sorted(set(heads))
            }

    def register_hooks(
        self,
        *,
        high_mask: torch.Tensor,
        nonhigh_mask: torch.Tensor,
    ) -> list[torch.utils.hooks.RemovableHandle]:
        handles: list[torch.utils.hooks.RemovableHandle] = []
        layers = qwen_layers(self.model)

        for layer, heads in self.heads_by_layer.items():
            o_proj = layers[int(layer)].self_attn.o_proj

            def hook_fn(
                module,
                args,
                kwargs,
                output,
                *,
                _layer=int(layer),
                _heads=tuple(sorted(set(heads))),
            ):
                del module, kwargs, output
                x = args[0]
                shape = x.shape
                if shape[-1] % self.head_dim != 0:
                    raise RuntimeError(f"cannot split o_proj input shape {tuple(shape)} by head_dim={self.head_dim}")
                x_heads = x.reshape(shape[0], shape[1], shape[-1] // self.head_dim, self.head_dim)
                weights = self.projection_weights[_layer]
                for head in _heads:
                    proj_weight = weights[int(head)].to(device=x.device, dtype=torch.float32)
                    projected = torch.einsum("btd,kd->btk", x_heads[:, :, int(head)].float(), proj_weight)
                    token_scores = projected.mean(dim=-1)
                    self.stats[f"{_layer}_{int(head)}"].update(token_scores, high_mask, nonhigh_mask)

            handles.append(o_proj.register_forward_hook(hook_fn, with_kwargs=True))
        return handles

    def metrics(self) -> dict[str, dict[str, float]]:
        return {key: value.as_metrics() for key, value in sorted(self.stats.items())}
