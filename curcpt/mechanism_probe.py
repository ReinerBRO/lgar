from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lgar_cpt.config import LGARParams
from lgar_cpt.data import PackedFineWebDataset
from lgar_cpt.mining import (
    attention_mask_format_for_model,
    build_lsd_label_batch_from_scores,
    document_causal_attention_mask,
    gather_logprob,
    local_document_attention_mask,
)
from lgar_cpt.modeling import load_tokenizer, unwrap_logits

from .forward import _register_adapter_hooks
from .model_eval_utils import load_model_tokenizer_for_eval


class Accumulator:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0.0

    def add(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        selected = values[mask]
        if selected.numel() == 0:
            return
        self.sum += float(selected.detach().float().sum().item())
        self.count += float(selected.numel())

    def mean(self) -> float:
        return self.sum / self.count if self.count else float("nan")


def _parse_checkpoint_args(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--checkpoint must be NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    if not out:
        raise SystemExit("at least one --checkpoint NAME=PATH is required")
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


def _forward_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    *,
    local_window: int | None = None,
) -> torch.Tensor:
    mask_format = attention_mask_format_for_model(model)
    if local_window is None:
        mask = document_causal_attention_mask(doc_ids, mask_format=mask_format)
    else:
        mask = local_document_attention_mask(doc_ids, int(local_window), mask_format=mask_format)
    out = model(input_ids, attention_mask=mask, use_cache=False)
    return unwrap_logits(out)


def _full_short_logp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    doc_ids: torch.Tensor,
    short_window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full_logits = _forward_logits(model, input_ids, doc_ids, local_window=None)
    full_logp = gather_logprob(full_logits, labels)
    short_logits = _forward_logits(model, input_ids, doc_ids, local_window=short_window)
    short_logp = gather_logprob(short_logits, labels)
    return full_logits, full_logp.detach(), short_logp.detach()


def _special_token_ids(tokenizer: Any) -> set[int]:
    return {
        int(x)
        for x in [tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id]
        if x is not None
    }


@torch.no_grad()
def _build_reference_masks(
    model_path: str,
    checkpoint_path: str,
    batches: list[dict[str, np.ndarray]],
    device: torch.device,
    dtype: str,
    attn_implementation: str,
    params: LGARParams,
) -> list[dict[str, torch.Tensor]]:
    model, tokenizer, _adapters, handles, _checkpoint = load_model_tokenizer_for_eval(
        model_path,
        checkpoint_path,
        device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    special_ids = _special_token_ids(tokenizer)
    refs: list[dict[str, torch.Tensor]] = []
    try:
        for batch_np in batches:
            batch = _to_torch(batch_np, device)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            doc_ids = batch["doc_ids_full"][:, :-1]
            _logits, full_logp, short_logp = _full_short_logp(
                model, input_ids, labels, doc_ids, params.short_window
            )
            label_batch = build_lsd_label_batch_from_scores(
                long_logp=full_logp,
                short_logp=short_logp,
                labels=labels,
                loss_mask=batch["loss_mask"],
                doc_offsets_full=batch["doc_offsets_full"],
                source_doc_rows_full=batch.get("source_doc_rows_full"),
                special_token_ids=special_ids,
                params=params,
                positive_fraction=params.lsd_top_fraction,
            )
            refs.append(
                {
                    "valid": label_batch.valid.detach().cpu(),
                    "high": label_batch.labels.detach().cpu(),
                    "utility": label_batch.lsd.detach().cpu(),
                }
            )
    finally:
        for handle in handles:
            handle.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return refs


def _quantiles(values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan")}
    x = torch.cat([v.detach().float().cpu().flatten() for v in values if v.numel() > 0])
    if x.numel() == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan")}
    return {
        "mean": float(x.mean().item()),
        "p50": float(torch.quantile(x, 0.50).item()),
        "p90": float(torch.quantile(x, 0.90).item()),
        "p95": float(torch.quantile(x, 0.95).item()),
        "p99": float(torch.quantile(x, 0.99).item()),
    }


def _corrupt_far_prefix(input_ids: torch.Tensor, local_window: int) -> tuple[torch.Tensor, torch.Tensor]:
    corrupted = input_ids.clone()
    seq_len = int(input_ids.size(1))
    far_end = max(0, seq_len - int(local_window))
    position = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    suffix_query_mask = position >= far_end
    if far_end > 1:
        # Deterministic within-row roll keeps token distribution fixed while
        # breaking exact far-prefix evidence for suffix predictions.
        corrupted[:, :far_end] = torch.roll(corrupted[:, :far_end], shifts=max(1, far_end // 2), dims=1)
    return corrupted, suffix_query_mask


@torch.no_grad()
def _evaluate_one_model(
    name: str,
    checkpoint_path: str,
    model_path: str,
    batches: list[dict[str, np.ndarray]],
    refs: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: str,
    attn_implementation: str,
    params: LGARParams,
) -> dict[str, Any]:
    model, tokenizer, adapters, handles, _checkpoint = load_model_tokenizer_for_eval(
        model_path,
        checkpoint_path,
        device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    special_ids = _special_token_ids(tokenizer)
    acc = {
        "normal_ce": Accumulator(),
        "ref_valid_ce": Accumulator(),
        "ref_high_ce": Accumulator(),
        "ref_nonhigh_ce": Accumulator(),
        "self_high_ce": Accumulator(),
        "full_minus_local_utility": Accumulator(),
        "ref_high_utility": Accumulator(),
        "corrupt_suffix_delta_nll": Accumulator(),
        "corrupt_ref_high_delta_nll": Accumulator(),
        "adapter_active_minus_off_delta_nll": Accumulator(),
        "adapter_ref_high_active_minus_off_delta_nll": Accumulator(),
        "adapter_logit_l2": Accumulator(),
        "adapter_ref_high_logit_l2": Accumulator(),
        "adapter_top1_flip_rate": Accumulator(),
        "adapter_ref_high_top1_flip_rate": Accumulator(),
    }
    utility_values: list[torch.Tensor] = []
    ref_high_count = 0.0
    self_high_count = 0.0
    overlap_count = 0.0
    valid_count = 0.0
    offset_bins = {
        "0_1023": {"ce": Accumulator(), "utility": Accumulator()},
        "1024_2047": {"ce": Accumulator(), "utility": Accumulator()},
        "2048_4095": {"ce": Accumulator(), "utility": Accumulator()},
        "4096_plus": {"ce": Accumulator(), "utility": Accumulator()},
    }

    try:
        for batch_np, ref_cpu in zip(batches, refs):
            batch = _to_torch(batch_np, device)
            ref_valid = ref_cpu["valid"].to(device=device, dtype=torch.bool)
            ref_high = ref_cpu["high"].to(device=device, dtype=torch.bool)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            loss_mask = batch["loss_mask"].bool()
            doc_ids = batch["doc_ids_full"][:, :-1]

            full_logits, full_logp, short_logp = _full_short_logp(
                model, input_ids, labels, doc_ids, params.short_window
            )
            nll = -full_logp
            utility = full_logp - short_logp
            ref_nonhigh = ref_valid & ~ref_high

            label_batch = build_lsd_label_batch_from_scores(
                long_logp=full_logp,
                short_logp=short_logp,
                labels=labels,
                loss_mask=loss_mask,
                doc_offsets_full=batch["doc_offsets_full"],
                source_doc_rows_full=batch.get("source_doc_rows_full"),
                special_token_ids=special_ids,
                params=params,
                positive_fraction=params.lsd_top_fraction,
            )
            self_high = label_batch.labels.bool()

            acc["normal_ce"].add(nll, loss_mask)
            acc["ref_valid_ce"].add(nll, ref_valid)
            acc["ref_high_ce"].add(nll, ref_high)
            acc["ref_nonhigh_ce"].add(nll, ref_nonhigh)
            acc["self_high_ce"].add(nll, self_high)
            acc["full_minus_local_utility"].add(utility, ref_valid)
            acc["ref_high_utility"].add(utility, ref_high)
            utility_values.append(utility[ref_valid].detach().cpu())

            ref_high_count += float(ref_high.sum().item())
            self_high_count += float(self_high.sum().item())
            overlap_count += float((ref_high & self_high).sum().item())
            valid_count += float(ref_valid.sum().item())

            target_offsets = batch["doc_offsets_full"][:, 1 : input_ids.size(1) + 1]
            for key, mask in {
                "0_1023": target_offsets < 1024,
                "1024_2047": (target_offsets >= 1024) & (target_offsets < 2048),
                "2048_4095": (target_offsets >= 2048) & (target_offsets < 4096),
                "4096_plus": target_offsets >= 4096,
            }.items():
                m = loss_mask & mask
                offset_bins[key]["ce"].add(nll, m)
                offset_bins[key]["utility"].add(utility, m)

            corrupted, suffix_mask = _corrupt_far_prefix(input_ids, params.short_window)
            corrupt_logits = _forward_logits(model, corrupted, doc_ids, local_window=None)
            corrupt_logp = gather_logprob(corrupt_logits, labels)
            corrupt_delta = (-corrupt_logp) - nll
            suffix_valid = loss_mask & suffix_mask
            acc["corrupt_suffix_delta_nll"].add(corrupt_delta, suffix_valid)
            acc["corrupt_ref_high_delta_nll"].add(corrupt_delta, suffix_valid & ref_high)

            if adapters is not None:
                for handle in handles:
                    handle.remove()
                off_logits = _forward_logits(model, input_ids, doc_ids, local_window=None)
                off_logp = gather_logprob(off_logits, labels)
                active_minus_off = nll - (-off_logp)
                acc["adapter_active_minus_off_delta_nll"].add(active_minus_off, loss_mask)
                acc["adapter_ref_high_active_minus_off_delta_nll"].add(active_minus_off, ref_high)
                logit_delta_l2 = (full_logits.float() - off_logits.float()).pow(2).mean(dim=-1).sqrt()
                top1_flip = (full_logits.argmax(dim=-1) != off_logits.argmax(dim=-1)).float()
                acc["adapter_logit_l2"].add(logit_delta_l2, loss_mask)
                acc["adapter_ref_high_logit_l2"].add(logit_delta_l2, ref_high)
                acc["adapter_top1_flip_rate"].add(top1_flip, loss_mask)
                acc["adapter_ref_high_top1_flip_rate"].add(top1_flip, ref_high)
                handles = _register_adapter_hooks(model, adapters)
    finally:
        for handle in handles:
            handle.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "checkpoint": checkpoint_path,
        "adapters_active": adapters is not None,
        "num_retrieval_heads": adapters.num_heads() if adapters is not None else 0,
        "num_sequences": len(batches),
        "seq_len": int(params.seq_len),
        "short_window": int(params.short_window),
        "reference_valid_tokens": valid_count,
        "reference_high_tokens": ref_high_count,
        "self_high_tokens": self_high_count,
        "self_ref_high_overlap_over_ref": overlap_count / ref_high_count if ref_high_count else float("nan"),
        "self_ref_high_overlap_over_self": overlap_count / self_high_count if self_high_count else float("nan"),
        "metrics": {key: value.mean() for key, value in acc.items()},
        "utility_quantiles_on_ref_valid": _quantiles(utility_values),
        "offset_bins": {
            key: {
                "ce": bins["ce"].mean(),
                "utility": bins["utility"].mean(),
                "tokens": bins["ce"].count,
            }
            for key, bins in offset_bins.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CURE-CPT intrinsic mechanism probes on fixed FineWeb batches.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH. First name is used as reference unless --reference-name is set.")
    parser.add_argument("--reference-name", default=None)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--short-window", type=int, default=1024)
    parser.add_argument("--min-remote-margin", type=int, default=256)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--long-nll-max-quantile", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-sequences", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    checkpoints = _parse_checkpoint_args(args.checkpoint)
    reference_name = args.reference_name or next(iter(checkpoints))
    if reference_name not in checkpoints:
        raise SystemExit(f"reference checkpoint {reference_name!r} not found in --checkpoint list")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else int(device.index))

    tokenizer = load_tokenizer(args.model_path)
    params = LGARParams(
        seq_len=int(args.seq_len),
        short_window=int(args.short_window),
        min_remote_margin=int(args.min_remote_margin),
        local_window=int(args.short_window),
        lsd_top_fraction=float(args.top_fraction),
        long_nll_max_quantile=float(args.long_nll_max_quantile),
    )
    dataset = PackedFineWebDataset(
        args.cache_dir,
        seq_len=int(args.seq_len),
        pad_token_id=int(tokenizer.pad_token_id),
        split="val",
        seed=int(args.seed),
    )
    batches: list[dict[str, np.ndarray]] = []
    remaining = int(args.num_sequences)
    while remaining > 0:
        n = min(int(args.batch_size), remaining)
        batches.append(dataset.sample(n))
        remaining -= n

    refs = _build_reference_masks(
        args.model_path,
        checkpoints[reference_name],
        batches,
        device,
        args.dtype,
        args.attn_implementation,
        params,
    )

    results: dict[str, Any] = {
        "protocol": {
            "data": "Fixed FineWeb validation packed sequences shared by all checkpoints.",
            "reference_high_utility": f"{reference_name} top {args.top_fraction:.3f} LSD positions after LGAR validity filtering.",
            "prefix_corruption": "Deterministic roll of the far prefix; suffix-window gold NLL delta measures sensitivity to far-prefix content.",
            "adapter_delta": "active NLL minus adapter-off NLL; negative means adapters improve gold-token NLL.",
        },
        "reference_name": reference_name,
        "num_sequences": int(args.num_sequences),
        "seq_len": int(args.seq_len),
        "short_window": int(args.short_window),
        "models": {},
    }
    for name, checkpoint_path in checkpoints.items():
        print(json.dumps({"event": "probe_model_start", "model": name}, sort_keys=True), flush=True)
        results["models"][name] = _evaluate_one_model(
            name,
            checkpoint_path,
            args.model_path,
            batches,
            refs,
            device,
            args.dtype,
            args.attn_implementation,
            params,
        )
        print(json.dumps({"event": "probe_model_done", "model": name}, sort_keys=True), flush=True)

    ce = results["models"].get(reference_name)
    if ce is not None:
        deltas: dict[str, dict[str, float]] = {}
        for name, result in results["models"].items():
            if name == reference_name:
                continue
            deltas[name] = {
                key: float(value) - float(ce["metrics"][key])
                for key, value in result["metrics"].items()
                if isinstance(value, (int, float)) and key in ce["metrics"]
            }
        results["deltas_vs_reference"] = deltas

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
