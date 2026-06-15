from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lgar_cpt.config import LGARParams
from lgar_cpt.data import PackedFineWebDataset
from lgar_cpt.mining import build_lsd_label_batch_from_scores, gather_logprob
from lgar_cpt.modeling import load_tokenizer

from .forward import _register_adapter_hooks
from .mechanism_probe import _parse_checkpoint_args, _special_token_ids, _to_torch
from .model_eval_utils import load_model_tokenizer_for_eval
from .rh_bottleneck import (
    full_doc_logits,
    local_all_heads_logits,
    resolve_retrieval_heads,
    rh_bottleneck_logits,
)


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


def _init_bucket() -> dict[str, Accumulator]:
    keys = [
        "full_nll",
        "local_all_heads_nll",
        "rh_bottleneck_nll",
        "local_minus_full_nll",
        "rh_minus_full_nll",
        "full_vs_local_logit_l2",
        "full_vs_rh_logit_l2",
        "full_vs_local_top1_flip",
        "full_vs_rh_top1_flip",
        "rh_adapter_active_minus_off_nll",
        "rh_adapter_on_off_logit_l2",
        "rh_adapter_on_off_top1_flip",
    ]
    return {key: Accumulator() for key in keys}


def _add_bucket_metrics(
    bucket: dict[str, Accumulator],
    mask: torch.Tensor,
    *,
    full_nll: torch.Tensor,
    local_nll: torch.Tensor,
    rh_nll: torch.Tensor,
    full_logits: torch.Tensor,
    local_logits: torch.Tensor,
    rh_logits: torch.Tensor,
    rh_off_nll: torch.Tensor | None,
    rh_off_logits: torch.Tensor | None,
) -> None:
    bucket["full_nll"].add(full_nll, mask)
    bucket["local_all_heads_nll"].add(local_nll, mask)
    bucket["rh_bottleneck_nll"].add(rh_nll, mask)
    bucket["local_minus_full_nll"].add(local_nll - full_nll, mask)
    bucket["rh_minus_full_nll"].add(rh_nll - full_nll, mask)

    local_l2 = (local_logits.float() - full_logits.float()).pow(2).mean(dim=-1).sqrt()
    rh_l2 = (rh_logits.float() - full_logits.float()).pow(2).mean(dim=-1).sqrt()
    local_flip = (local_logits.argmax(dim=-1) != full_logits.argmax(dim=-1)).float()
    rh_flip = (rh_logits.argmax(dim=-1) != full_logits.argmax(dim=-1)).float()
    bucket["full_vs_local_logit_l2"].add(local_l2, mask)
    bucket["full_vs_rh_logit_l2"].add(rh_l2, mask)
    bucket["full_vs_local_top1_flip"].add(local_flip, mask)
    bucket["full_vs_rh_top1_flip"].add(rh_flip, mask)

    if rh_off_nll is not None and rh_off_logits is not None:
        on_off_l2 = (rh_logits.float() - rh_off_logits.float()).pow(2).mean(dim=-1).sqrt()
        on_off_flip = (rh_logits.argmax(dim=-1) != rh_off_logits.argmax(dim=-1)).float()
        bucket["rh_adapter_active_minus_off_nll"].add(rh_nll - rh_off_nll, mask)
        bucket["rh_adapter_on_off_logit_l2"].add(on_off_l2, mask)
        bucket["rh_adapter_on_off_top1_flip"].add(on_off_flip, mask)


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
            full_logits = full_doc_logits(model, input_ids, doc_ids)
            local_logits = local_all_heads_logits(model, input_ids, doc_ids, params.short_window)
            full_logp = gather_logprob(full_logits, labels)
            local_logp = gather_logprob(local_logits, labels)
            label_batch = build_lsd_label_batch_from_scores(
                long_logp=full_logp.detach(),
                short_logp=local_logp.detach(),
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
                }
            )
    finally:
        for handle in handles:
            handle.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return refs


@torch.no_grad()
def _evaluate_model(
    name: str,
    checkpoint_path: str,
    model_path: str,
    batches: list[dict[str, np.ndarray]],
    refs: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: str,
    attn_implementation: str,
    params: LGARParams,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model, tokenizer, adapters, handles, checkpoint = load_model_tokenizer_for_eval(
        model_path,
        checkpoint_path,
        device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    del tokenizer
    retrieval_heads = resolve_retrieval_heads(
        checkpoint,
        explicit_heads=args.retrieval_heads,
        heads_json=args.retrieval_heads_json,
        reference_checkpoint_path=args.reference_checkpoint_path,
    )
    if not retrieval_heads:
        raise SystemExit(f"{name}: no retrieval heads available for RH-bottleneck probe")

    buckets = {
        "all_valid": _init_bucket(),
        "high_utility": _init_bucket(),
        "non_high_utility": _init_bucket(),
        "offset_ge_1024": _init_bucket(),
        "offset_ge_2048": _init_bucket(),
    }
    counts = {key: 0 for key in buckets}
    try:
        for batch_np, ref_cpu in zip(batches, refs):
            batch = _to_torch(batch_np, device)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            doc_ids = batch["doc_ids_full"][:, :-1]
            ref_valid = ref_cpu["valid"].to(device=device, dtype=torch.bool)
            ref_high = ref_cpu["high"].to(device=device, dtype=torch.bool)
            target_offsets = batch["doc_offsets_full"][:, 1 : input_ids.size(1) + 1]

            full_logits = full_doc_logits(model, input_ids, doc_ids)
            local_logits = local_all_heads_logits(model, input_ids, doc_ids, params.short_window)
            rh_logits = rh_bottleneck_logits(
                model,
                input_ids,
                doc_ids,
                retrieval_heads,
                params.short_window,
            )
            full_nll = -gather_logprob(full_logits, labels)
            local_nll = -gather_logprob(local_logits, labels)
            rh_nll = -gather_logprob(rh_logits, labels)

            rh_off_logits = None
            rh_off_nll = None
            if adapters is not None:
                for handle in handles:
                    handle.remove()
                rh_off_logits = rh_bottleneck_logits(
                    model,
                    input_ids,
                    doc_ids,
                    retrieval_heads,
                    params.short_window,
                )
                rh_off_nll = -gather_logprob(rh_off_logits, labels)
                handles = _register_adapter_hooks(model, adapters)

            masks = {
                "all_valid": ref_valid,
                "high_utility": ref_high,
                "non_high_utility": ref_valid & ~ref_high,
                "offset_ge_1024": ref_valid & (target_offsets >= 1024),
                "offset_ge_2048": ref_valid & (target_offsets >= 2048),
            }
            for bucket_name, mask in masks.items():
                counts[bucket_name] += int(mask.sum().item())
                _add_bucket_metrics(
                    buckets[bucket_name],
                    mask,
                    full_nll=full_nll,
                    local_nll=local_nll,
                    rh_nll=rh_nll,
                    full_logits=full_logits,
                    local_logits=local_logits,
                    rh_logits=rh_logits,
                    rh_off_nll=rh_off_nll,
                    rh_off_logits=rh_off_logits,
                )
    finally:
        for handle in handles:
            handle.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "checkpoint": checkpoint_path,
        "adapters_active": adapters is not None,
        "checkpoint_num_adapter_heads": adapters.num_heads() if adapters is not None else 0,
        "rh_bottleneck_retrieval_heads": sorted([list(head) for head in retrieval_heads]),
        "buckets": {
            bucket_name: {
                "valid_count": counts[bucket_name],
                **{metric: acc.mean() for metric, acc in bucket.items()},
            }
            for bucket_name, bucket in buckets.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended CURE mechanism probe over full/local/RH-bottleneck paths.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--reference-name", default=None)
    parser.add_argument("--reference-checkpoint-path", default=None)
    parser.add_argument("--retrieval-heads", default=None)
    parser.add_argument("--retrieval-heads-json", default=None)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--short-window", type=int, default=1024)
    parser.add_argument("--min-remote-margin", type=int, default=256)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--long-nll-max-quantile", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    checkpoints = _parse_checkpoint_args(args.checkpoint)
    reference_name = args.reference_name or next(iter(checkpoints))
    if reference_name not in checkpoints:
        raise SystemExit(f"reference checkpoint {reference_name!r} not found")
    if args.reference_checkpoint_path is None:
        args.reference_checkpoint_path = checkpoints[reference_name]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else int(device.index))

    tokenizer = load_tokenizer(args.model_path)
    params = LGARParams(
        seq_len=int(args.seq_len),
        short_window=int(args.short_window),
        local_window=int(args.short_window),
        min_remote_margin=int(args.min_remote_margin),
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
    batches = []
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
            "paths": ["full", "local_all_heads", "rh_bottleneck"],
            "rh_bottleneck": "retrieval heads full document-causal; all non-retrieval heads local-window.",
            "buckets": ["all_valid", "high_utility", "non_high_utility", "offset_ge_1024", "offset_ge_2048"],
            "gap_sign": "rh_minus_full_nll > 0 means RH-bottleneck is worse than full on gold-token NLL.",
        },
        "num_sequences": int(args.num_sequences),
        "seq_len": int(args.seq_len),
        "short_window": int(args.short_window),
        "reference_name": reference_name,
        "models": {},
    }
    for name, checkpoint_path in checkpoints.items():
        print(json.dumps({"event": "extended_probe_start", "model": name}, sort_keys=True), flush=True)
        results["models"][name] = _evaluate_model(
            name,
            checkpoint_path,
            args.model_path,
            batches,
            refs,
            device,
            args.dtype,
            args.attn_implementation,
            params,
            args,
        )
        print(json.dumps({"event": "extended_probe_done", "model": name}, sort_keys=True), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
