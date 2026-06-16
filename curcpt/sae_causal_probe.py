from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from lgar_cpt.data import PackedSequenceSignalDataset
from lgar_cpt.mining import attention_mask_format_for_model, document_causal_attention_mask, gather_logprob
from lgar_cpt.modeling import unwrap_logits

from .adapters import HeadAdapterSet
from .forward import (
    _model_backbone,
    _register_adapter_hooks,
    _register_resid_mid_capture_hooks,
    _unwrap_parallel_module,
)
from .model_eval_utils import load_model_tokenizer_for_eval
from .sae import SparseAutoencoder, load_evidence_feature_set, load_sparse_autoencoder, sae_decoder_feature_contribution


@dataclass
class MeanStat:
    total: float = 0.0
    count: float = 0.0

    def add(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        if values.ndim == mask.ndim + 1:
            values = values.float().mean(dim=-1)
        selected = values[mask]
        if selected.numel() == 0:
            return
        self.total += float(selected.detach().float().sum().item())
        self.count += float(selected.numel())

    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")


@dataclass
class BucketStats:
    count: float = 0.0
    metrics: dict[str, MeanStat] = field(default_factory=dict)

    def add_mask(self, mask: torch.Tensor) -> None:
        self.count += float(mask.detach().sum().item())

    def add_metric(self, name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        self.metrics.setdefault(name, MeanStat()).add(values, mask)

    def as_dict(self) -> dict[str, float]:
        out = {"count": self.count}
        out.update({name: stat.mean() for name, stat in sorted(self.metrics.items())})
        return out


def _to_torch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value.dtype == bool:
            out[key] = torch.as_tensor(value, dtype=torch.bool, device=device)
        elif value.dtype.kind == "f":
            out[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
        else:
            out[key] = torch.as_tensor(value, dtype=torch.long, device=device)
    return out


def _feature_mean(values: torch.Tensor) -> torch.Tensor:
    return values.float().mean(dim=-1) if values.ndim >= 3 else values.float()


def _safe_ratio(numerator: float, denominator: float, eps: float = 1.0e-8) -> float:
    if abs(denominator) < eps:
        return float("nan")
    return numerator / denominator


def _choose_random_feature_ids(
    sae: SparseAutoencoder,
    exclude: tuple[int, ...],
    count: int,
    seed: int,
) -> tuple[int, ...]:
    feature_dim = int(sae.encoder.out_features)
    excluded = {int(x) for x in exclude}
    candidates = [idx for idx in range(feature_dim) if idx not in excluded]
    if len(candidates) < count:
        raise ValueError(
            f"not enough random SAE features outside selected set: need {count}, have {len(candidates)}"
        )
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(np.asarray(candidates, dtype=np.int64), size=int(count), replace=False)
    return tuple(int(x) for x in selected.tolist())


def _selected_feature_acts(
    sae: SparseAutoencoder,
    residual: torch.Tensor,
    feature_ids: tuple[int, ...],
) -> torch.Tensor:
    acts = sae.encode(residual)
    index = torch.as_tensor(feature_ids, device=acts.device, dtype=torch.long)
    return acts.index_select(dim=-1, index=index)


def _gather_logprob_chunked(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    if int(chunk_size) <= 0 or logits.shape[1] <= int(chunk_size):
        return gather_logprob(logits, labels)
    pieces = []
    for start in range(0, logits.shape[1], int(chunk_size)):
        end = min(start + int(chunk_size), logits.shape[1])
        pieces.append(gather_logprob(logits[:, start:end], labels[:, start:end]))
    return torch.cat(pieces, dim=1)


def _register_resid_mid_feature_ablation_hook(
    model: torch.nn.Module,
    layer: int,
    sae: SparseAutoencoder,
    feature_ids: tuple[int, ...],
    scale: float = 1.0,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Subtract selected SAE decoder directions from resid_mid and its skip path.

    Qwen-style decoder layers keep the post-attention residual in a local
    `residual` variable before calling post_attention_layernorm. A pre-hook on
    that layernorm only changes the MLP branch, not the final residual skip.
    The paired layer output hook subtracts the same delta from the decoder
    output, yielding: (resid_mid - delta) + mlp(norm(resid_mid - delta)).
    """
    backbone = _model_backbone(model)
    decoder_layer = _unwrap_parallel_module(backbone.layers[int(layer)])
    hook_point = getattr(decoder_layer, "post_attention_layernorm", None)
    if hook_point is None:
        raise AttributeError(f"Layer {layer} has no post_attention_layernorm hook point")

    state: dict[str, torch.Tensor] = {}

    def hook_fn(module, args, kwargs):
        residual = args[0]
        values = _selected_feature_acts(sae, residual, feature_ids)
        delta = sae_decoder_feature_contribution(sae, values, feature_ids)
        scaled_delta = float(scale) * delta.to(device=residual.device, dtype=residual.dtype)
        state["delta"] = scaled_delta
        new_residual = residual - scaled_delta
        return (new_residual, *args[1:]), kwargs

    def layer_output_hook(module, args, kwargs, output):
        scaled_delta = state.pop("delta", None)
        if scaled_delta is None:
            return output
        if torch.is_tensor(output):
            return output - scaled_delta.to(device=output.device, dtype=output.dtype)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            first = output[0] - scaled_delta.to(device=output[0].device, dtype=output[0].dtype)
            return (first, *output[1:])
        raise TypeError(f"unsupported decoder layer output type for SAE ablation: {type(output)!r}")

    return [
        hook_point.register_forward_pre_hook(hook_fn, with_kwargs=True),
        decoder_layer.register_forward_hook(layer_output_hook, with_kwargs=True),
    ]


@contextlib.contextmanager
def _temporary_handles(handles: list[torch.utils.hooks.RemovableHandle]) -> Iterator[None]:
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _attention_mask(model: torch.nn.Module, doc_ids: torch.Tensor) -> torch.Tensor:
    return document_causal_attention_mask(
        doc_ids,
        mask_format=attention_mask_format_for_model(model),
    )


def _forward_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
) -> torch.Tensor:
    out = model(
        input_ids,
        attention_mask=_attention_mask(model, doc_ids),
        use_cache=False,
    )
    return unwrap_logits(out)


def _active_forward(
    model: torch.nn.Module,
    adapters: HeadAdapterSet | None,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    *,
    capture_resid_layer: int | None = None,
    capture_write_layer: int | None = None,
    ablate_feature_ids: tuple[int, ...] | None = None,
    sae: SparseAutoencoder | None = None,
    ablation_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    resid_by_layer: dict[int, torch.Tensor] = {}
    writes_by_layer: dict[int, torch.Tensor] = {}
    phase_ref = {"name": "full"}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    if adapters is not None:
        write_layers = {int(capture_write_layer)} if capture_write_layer is not None else None
        handles.extend(
            _register_adapter_hooks(
                model,
                adapters,
                adapter_write_layers=write_layers,
                adapter_writes_by_layer=writes_by_layer,
                phase_ref=phase_ref,
            )
        )
    if capture_resid_layer is not None:
        handles.extend(
            _register_resid_mid_capture_hooks(model, {int(capture_resid_layer)}, resid_by_layer, phase_ref=phase_ref)
        )
    if ablate_feature_ids is not None:
        if sae is None:
            raise ValueError("sae is required when ablate_feature_ids is provided")
        handles.extend(
            _register_resid_mid_feature_ablation_hook(
                model,
                int(capture_resid_layer if capture_resid_layer is not None else 0),
                sae,
                ablate_feature_ids,
                scale=ablation_scale,
            )
        )
    with _temporary_handles(handles):
        logits = _forward_logits(model, input_ids, doc_ids)
    return logits, resid_by_layer, writes_by_layer


def _adapter_off_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    *,
    capture_resid_layer: int | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    resid_by_layer: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    if capture_resid_layer is not None:
        handles.extend(_register_resid_mid_capture_hooks(model, {int(capture_resid_layer)}, resid_by_layer))
    with _temporary_handles(handles):
        logits = _forward_logits(model, input_ids, doc_ids)
    return logits, resid_by_layer


def _bucket_masks(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    loss_mask = batch["loss_mask"].bool()
    high = batch["offline_lsd_labels"].bool()
    valid = batch["offline_lsd_valid"].bool()
    buckets = {
        "high_utility_remote": loss_mask & valid & high,
        "valid_nonhigh": loss_mask & valid & (~high),
        "local_or_invalid": loss_mask & (~valid),
        "all_loss": loss_mask,
    }
    if "offline_sae_feature_mask" in batch:
        sae_mask = batch["offline_sae_feature_mask"].bool()
        if sae_mask.ndim == 3:
            sae_mask = sae_mask.any(dim=-1)
        buckets["sae_target_tokens"] = loss_mask & valid & high & sae_mask
    return buckets


def _append_bucket_metrics(
    bucket_stats: dict[str, BucketStats],
    buckets: dict[str, torch.Tensor],
    metrics: dict[str, torch.Tensor],
) -> None:
    for bucket_name, mask in buckets.items():
        stats = bucket_stats.setdefault(bucket_name, BucketStats())
        stats.add_mask(mask)
        for metric_name, values in metrics.items():
            stats.add_metric(metric_name, values, mask)


@torch.no_grad()
def run_sae_causal_probes(
    *,
    model_path: str,
    checkpoint_path: str,
    sae_checkpoint: str,
    features_path: str,
    cache_dir: str,
    signal_dir: str,
    output_path: str,
    batch_size: int = 1,
    batches: int = 8,
    seed: int = 1337,
    dtype: str = "bf16",
    attn_implementation: str = "sdpa",
    ablation_scale: float = 1.0,
    logprob_chunk_size: int = 512,
    allow_unvalidated_features: bool = False,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = load_evidence_feature_set(features_path, require_validated=not allow_unvalidated_features)
    sae = load_sparse_autoencoder(sae_checkpoint, device=device)
    random_feature_ids = _choose_random_feature_ids(
        sae,
        exclude=features.feature_ids,
        count=len(features.feature_ids),
        seed=seed,
    )

    model, tokenizer, adapters, initial_handles, checkpoint = load_model_tokenizer_for_eval(
        model_path,
        checkpoint_path,
        device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    for handle in initial_handles:
        handle.remove()
    if adapters is None:
        raise ValueError("checkpoint has no adapters; SAE-CURE probes require an adapter checkpoint")

    dataset = PackedSequenceSignalDataset(
        cache_dir=cache_dir,
        signal_dir=signal_dir,
        pad_token_id=int(tokenizer.pad_token_id),
        seed=seed,
    )
    summary = checkpoint.get("summary", {}) if isinstance(checkpoint, dict) else {}
    retrieval_heads = summary.get("retrieval_heads", [])

    bucket_stats: dict[str, BucketStats] = {}
    global_stats: dict[str, MeanStat] = {
        "adapter_gain_logp": MeanStat(),
        "selected_ablation_delta_nll": MeanStat(),
        "random_ablation_delta_nll": MeanStat(),
        "selected_feature_active_mean": MeanStat(),
        "adapter_resid_selected_feature_delta": MeanStat(),
        "adapter_resid_random_feature_delta": MeanStat(),
        "adapter_resid_delta_norm": MeanStat(),
        "write_selected_feature_delta": MeanStat(),
        "write_random_feature_delta": MeanStat(),
        "write_norm": MeanStat(),
    }

    processed_batches = 0
    for _ in range(int(batches)):
        batch = _to_torch(dataset.sample(int(batch_size)), device=device)
        missing = [key for key in ("offline_lsd_labels", "offline_lsd_valid") if key not in batch]
        if missing:
            raise ValueError("SAE probes require offline signal masks in signal_dir: " + ", ".join(missing))

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        doc_ids = batch["doc_ids_full"][:, :-1]
        loss_mask = batch["loss_mask"].bool()
        buckets = _bucket_masks(batch)
        probe_mask = buckets.get("sae_target_tokens", buckets["high_utility_remote"])

        active_logits, resid_by_layer, writes_by_layer = _active_forward(
            model,
            adapters,
            input_ids,
            doc_ids,
            capture_resid_layer=features.layer,
            capture_write_layer=features.layer,
        )
        active_logp = _gather_logprob_chunked(active_logits, labels, chunk_size=logprob_chunk_size)
        del active_logits

        off_logits, off_resid_by_layer = _adapter_off_forward(
            model,
            input_ids,
            doc_ids,
            capture_resid_layer=features.layer,
        )
        off_logp = _gather_logprob_chunked(off_logits, labels, chunk_size=logprob_chunk_size)
        del off_logits

        selected_ablate_logits, _, _ = _active_forward(
            model,
            adapters,
            input_ids,
            doc_ids,
            capture_resid_layer=features.layer,
            ablate_feature_ids=features.feature_ids,
            sae=sae,
            ablation_scale=ablation_scale,
        )
        selected_ablate_logp = _gather_logprob_chunked(
            selected_ablate_logits,
            labels,
            chunk_size=logprob_chunk_size,
        )
        del selected_ablate_logits

        random_ablate_logits, _, _ = _active_forward(
            model,
            adapters,
            input_ids,
            doc_ids,
            capture_resid_layer=features.layer,
            ablate_feature_ids=random_feature_ids,
            sae=sae,
            ablation_scale=ablation_scale,
        )
        random_ablate_logp = _gather_logprob_chunked(
            random_ablate_logits,
            labels,
            chunk_size=logprob_chunk_size,
        )
        del random_ablate_logits

        if features.layer not in resid_by_layer:
            raise KeyError(f"missing captured resid_mid at SAE layer {features.layer}")
        resid_mid = resid_by_layer[features.layer]
        if features.layer not in off_resid_by_layer:
            raise KeyError(f"missing adapter-off resid_mid at SAE layer {features.layer}")
        off_resid_mid = off_resid_by_layer[features.layer]
        selected_acts = _selected_feature_acts(sae, resid_mid, features.feature_ids)
        random_acts = _selected_feature_acts(sae, resid_mid, random_feature_ids)
        off_selected_acts = _selected_feature_acts(sae, off_resid_mid, features.feature_ids)
        off_random_acts = _selected_feature_acts(sae, off_resid_mid, random_feature_ids)

        adapter_gain_logp = active_logp - off_logp
        selected_drop = active_logp - selected_ablate_logp
        random_drop = active_logp - random_ablate_logp

        adapter_resid_delta = resid_mid - off_resid_mid.to(device=resid_mid.device, dtype=resid_mid.dtype)
        adapter_resid_delta_norm = adapter_resid_delta.float().norm(dim=-1)
        adapter_resid_selected_delta = selected_acts - off_selected_acts
        adapter_resid_random_delta = random_acts - off_random_acts

        write = writes_by_layer.get(features.layer)
        if write is None:
            write = torch.zeros_like(resid_mid)
        write_base = resid_mid - write.to(device=resid_mid.device, dtype=resid_mid.dtype)
        selected_write_delta = (
            _selected_feature_acts(sae, write_base + write, features.feature_ids)
            - _selected_feature_acts(sae, write_base, features.feature_ids)
        )
        random_write_delta = (
            _selected_feature_acts(sae, write_base + write, random_feature_ids)
            - _selected_feature_acts(sae, write_base, random_feature_ids)
        )
        write_norm = write.float().norm(dim=-1)

        per_token_metrics = {
            "active_nll": -active_logp,
            "adapter_gain_logp": adapter_gain_logp,
            "selected_ablation_delta_nll": selected_drop,
            "random_ablation_delta_nll": random_drop,
            "selected_feature_active_mean": _feature_mean(selected_acts),
            "random_feature_active_mean": _feature_mean(random_acts),
            "adapter_resid_selected_feature_delta": _feature_mean(adapter_resid_selected_delta),
            "adapter_resid_random_feature_delta": _feature_mean(adapter_resid_random_delta),
            "adapter_resid_delta_norm": adapter_resid_delta_norm,
            "write_selected_feature_delta": _feature_mean(selected_write_delta),
            "write_random_feature_delta": _feature_mean(random_write_delta),
            "write_norm": write_norm,
        }
        if "offline_sae_feature_full" in batch:
            per_token_metrics["offline_sae_target_full_mean"] = _feature_mean(batch["offline_sae_feature_full"])
        if "offline_sae_feature_neg" in batch:
            per_token_metrics["offline_sae_target_neg_mean"] = _feature_mean(batch["offline_sae_feature_neg"])

        _append_bucket_metrics(bucket_stats, buckets, per_token_metrics)
        for metric_name, stat in global_stats.items():
            stat.add(per_token_metrics[metric_name], probe_mask & loss_mask)

        processed_batches += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    high_stats = bucket_stats.get("sae_target_tokens")
    if high_stats is None or high_stats.count == 0:
        high_stats = bucket_stats["high_utility_remote"]
    high_adapter_gain = high_stats.metrics.get("adapter_gain_logp", MeanStat()).mean()
    high_selected_drop = high_stats.metrics.get("selected_ablation_delta_nll", MeanStat()).mean()
    high_random_drop = high_stats.metrics.get("random_ablation_delta_nll", MeanStat()).mean()

    output = {
        "config": {
            "model_path": model_path,
            "checkpoint_path": checkpoint_path,
            "sae_checkpoint": sae_checkpoint,
            "features_path": features_path,
            "cache_dir": cache_dir,
            "signal_dir": signal_dir,
            "batch_size": int(batch_size),
            "batches": int(batches),
            "seed": int(seed),
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "ablation_scale": float(ablation_scale),
            "logprob_chunk_size": int(logprob_chunk_size),
        },
        "features": {
            "layer": int(features.layer),
            "hook_point": features.hook_point,
            "feature_ids": list(features.feature_ids),
            "random_control_feature_ids": list(random_feature_ids),
            "validation_passed": bool(features.validation_passed),
        },
        "retrieval_heads": retrieval_heads,
        "processed_batches": processed_batches,
        "probe_summary": {
            "feature_ablation": {
                "selected_delta_nll_on_sae_targets": high_selected_drop,
                "random_delta_nll_on_sae_targets": high_random_drop,
                "selected_minus_random_delta_nll": high_selected_drop - high_random_drop,
            },
            "adapter_mediation": {
                "adapter_gain_logp_on_sae_targets": high_adapter_gain,
                "selected_ablation_logp_drop_on_sae_targets": high_selected_drop,
                "mediation_ratio_drop_over_gain": _safe_ratio(high_selected_drop, high_adapter_gain),
            },
            "specificity": {
                name: stats.as_dict() for name, stats in sorted(bucket_stats.items())
            },
            "o_side_write_alignment": {
                "direct_layer_write_selected_feature_delta_on_sae_targets": high_stats.metrics.get(
                    "write_selected_feature_delta",
                    MeanStat(),
                ).mean(),
                "direct_layer_write_random_feature_delta_on_sae_targets": high_stats.metrics.get(
                    "write_random_feature_delta",
                    MeanStat(),
                ).mean(),
                "direct_layer_write_norm_on_sae_targets": high_stats.metrics.get("write_norm", MeanStat()).mean(),
                "total_adapter_resid_selected_feature_delta_on_sae_targets": high_stats.metrics.get(
                    "adapter_resid_selected_feature_delta",
                    MeanStat(),
                ).mean(),
                "total_adapter_resid_random_feature_delta_on_sae_targets": high_stats.metrics.get(
                    "adapter_resid_random_feature_delta",
                    MeanStat(),
                ).mean(),
                "total_adapter_resid_delta_norm_on_sae_targets": high_stats.metrics.get(
                    "adapter_resid_delta_norm",
                    MeanStat(),
                ).mean(),
            },
        },
        "global_probe_target_means": {name: stat.mean() for name, stat in sorted(global_stats.items())},
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run four SAE-CURE causal probes: selected feature ablation, adapter mediation, "
            "token specificity, and O-side residual-write alignment."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--sae-checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--signal-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--ablation-scale", type=float, default=1.0)
    parser.add_argument("--logprob-chunk-size", type=int, default=512)
    parser.add_argument(
        "--allow-unvalidated-sae-features",
        action="store_true",
        help="debug only: bypass answer-level feature validation checks",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run_sae_causal_probes(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        sae_checkpoint=args.sae_checkpoint,
        features_path=args.features,
        cache_dir=args.cache_dir,
        signal_dir=args.signal_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        batches=args.batches,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        ablation_scale=args.ablation_scale,
        logprob_chunk_size=args.logprob_chunk_size,
        allow_unvalidated_features=args.allow_unvalidated_sae_features,
    )
    print(json.dumps(output["probe_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
