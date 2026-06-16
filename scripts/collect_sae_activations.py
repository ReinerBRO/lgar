#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curcpt.forward import _register_resid_mid_capture_hooks
from curcpt.head_ablation import load_ablation_results
from scripts.precompute_sae_feature_targets import (
    _device_from_arg,
    _materialize_indices,
    _register_masked_head_hooks,
)


def _parse_layers(value: str, retrieval_heads: set[tuple[int, int]], max_layers: int) -> list[int]:
    value = str(value).strip()
    if value != "auto":
        return [int(x) for x in value.split(",") if x.strip()]
    if not retrieval_heads:
        raise ValueError("--layers auto requires --ablation-results")
    counts: dict[int, int] = {}
    for layer, _head in retrieval_heads:
        counts[int(layer)] = counts.get(int(layer), 0) + 1
    return [
        layer
        for layer, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: int(max_layers)]
    ]


@torch.no_grad()
def _capture_resid_mid(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: set[int],
    extra_handles: list[torch.utils.hooks.RemovableHandle] | None = None,
) -> dict[int, torch.Tensor]:
    resid_mid_by_layer: dict[int, torch.Tensor] = {}
    handles = _register_resid_mid_capture_hooks(model, layers, resid_mid_by_layer, phase_ref={"name": "full"})
    try:
        model(input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        for handle in extra_handles or []:
            handle.remove()
    missing = layers - set(resid_mid_by_layer)
    if missing:
        raise KeyError(f"missing captured resid_mid layers: {sorted(missing)}")
    return resid_mid_by_layer


def _append_view(
    store: dict[int, dict[str, list[np.ndarray]]],
    view: str,
    captured: dict[int, torch.Tensor],
    token_mask: torch.Tensor,
    out_dtype: np.dtype,
) -> None:
    for layer, residual in captured.items():
        view_values = residual[token_mask].detach().float().cpu().numpy().astype(out_dtype)
        store[layer].setdefault(view, []).append(view_values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect paired resid_mid activations for SAE training and evidence-feature mining."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--offline-signal-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ablation-results", default=None)
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--views", default="full,short,masked_head")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequences", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    views = {view.strip() for view in str(args.views).split(",") if view.strip()}
    unknown = views - {"full", "short", "masked_head"}
    if unknown:
        raise ValueError(
            "Unsupported view(s): "
            + ", ".join(sorted(unknown))
            + ". corrupt_context needs evidence-span metadata and is intentionally not approximated here."
        )
    if "full" not in views:
        raise ValueError("--views must include full so mining has a positive view")
    if "masked_head" in views and not args.ablation_results:
        raise ValueError("--views includes masked_head, so --ablation-results is required")

    retrieval_heads: set[tuple[int, int]] = set()
    if args.ablation_results:
        retrieval_heads, _ = load_ablation_results(args.ablation_results)
    layers = _parse_layers(args.layers, retrieval_heads, int(args.max_layers))
    layer_set = {int(layer) for layer in layers}

    from lgar_cpt.data import PackedSequenceLayoutDataset
    from lgar_cpt.mining import (
        attention_mask_format_for_model,
        document_causal_attention_mask,
        local_document_attention_mask,
    )
    from lgar_cpt.modeling import load_qwen_causal_lm, load_tokenizer

    device = _device_from_arg(args.device)
    tokenizer = load_tokenizer(args.model_path)
    model = load_qwen_causal_lm(
        args.model_path,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=False,
    ).to(device)
    model.eval()

    signal_dir = Path(args.offline_signal_dir)
    offline_labels = np.load(signal_dir / "offline_lsd_labels.npy", mmap_mode="r")
    offline_valid = np.load(signal_dir / "offline_lsd_valid.npy", mmap_mode="r")
    offline_lsd_path = signal_dir / "offline_lsd.npy"
    offline_lsd = np.load(offline_lsd_path, mmap_mode="r") if offline_lsd_path.exists() else None
    dataset = PackedSequenceLayoutDataset(
        cache_dir=args.cache_dir,
        signal_dir=signal_dir,
        pad_token_id=int(tokenizer.pad_token_id),
        seed=int(args.seed),
    )
    num_sequences = min(dataset.num_sequences, int(args.max_sequences))
    mask_format = attention_mask_format_for_model(model)
    out_dtype = np.float16 if args.output_dtype == "float16" else np.float32

    store: dict[int, dict[str, list[np.ndarray]]] = {layer: {} for layer in layers}
    sequence_chunks: list[np.ndarray] = []
    token_chunks: list[np.ndarray] = []
    target_token_chunks: list[np.ndarray] = []
    lsd_chunks: list[np.ndarray] = []
    token_total = 0

    for start in range(0, num_sequences, int(args.batch_size)):
        if token_total >= int(args.max_tokens):
            break
        end = min(start + int(args.batch_size), num_sequences)
        indices = np.arange(start, end, dtype=np.int64)
        batch = _materialize_indices(dataset, indices)
        input_ids = torch.as_tensor(batch["input_ids"], device=device)
        doc_ids = torch.as_tensor(batch["doc_ids_full"][:, :-1], device=device)
        loss_mask = torch.as_tensor(batch["loss_mask"], device=device).bool()
        token_mask = (
            torch.as_tensor(offline_labels[indices], device=device).bool()
            & torch.as_tensor(offline_valid[indices], device=device).bool()
            & loss_mask
        )
        if not token_mask.any():
            continue
        remaining = int(args.max_tokens) - token_total
        coords = token_mask.nonzero(as_tuple=False)
        if coords.shape[0] > remaining:
            keep = coords[:remaining]
            clipped = torch.zeros_like(token_mask)
            clipped[keep[:, 0], keep[:, 1]] = True
            token_mask = clipped
            coords = keep
        token_total += int(coords.shape[0])

        seq_np = indices[coords[:, 0].detach().cpu().numpy()]
        tok_np = coords[:, 1].detach().cpu().numpy().astype(np.int32)
        sequence_chunks.append(seq_np.astype(np.int64))
        token_chunks.append(tok_np)
        target_token_chunks.append(
            np.asarray(batch["labels"][coords[:, 0].detach().cpu().numpy(), tok_np], dtype=np.int64)
        )
        if offline_lsd is not None:
            lsd_chunks.append(np.asarray(offline_lsd[seq_np, tok_np], dtype=np.float32))

        full_mask = document_causal_attention_mask(doc_ids, mask_format=mask_format).to(device)
        if "full" in views:
            _append_view(
                store,
                "full",
                _capture_resid_mid(model, input_ids, full_mask, layer_set),
                token_mask,
                out_dtype,
            )
        if "short" in views:
            short_mask = local_document_attention_mask(
                doc_ids,
                int(args.local_window),
                mask_format=mask_format,
            ).to(device)
            _append_view(
                store,
                "short",
                _capture_resid_mid(model, input_ids, short_mask, layer_set),
                token_mask,
                out_dtype,
            )
        if "masked_head" in views:
            masked_handles = _register_masked_head_hooks(
                model,
                retrieval_heads,
                doc_ids,
                int(args.local_window),
                mask_format,
            )
            _append_view(
                store,
                "masked_head",
                _capture_resid_mid(model, input_ids, full_mask, layer_set, extra_handles=masked_handles),
                token_mask,
                out_dtype,
            )
        print(
            json.dumps(
                {
                    "event": "collect_sae_activations_progress",
                    "start": int(start),
                    "end": int(end),
                    "tokens": int(token_total),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_index = np.concatenate(sequence_chunks, axis=0) if sequence_chunks else np.empty((0,), dtype=np.int64)
    token_position = np.concatenate(token_chunks, axis=0) if token_chunks else np.empty((0,), dtype=np.int32)
    target_token_id = (
        np.concatenate(target_token_chunks, axis=0) if target_token_chunks else np.empty((0,), dtype=np.int64)
    )
    lsd = np.concatenate(lsd_chunks, axis=0) if lsd_chunks else np.empty((0,), dtype=np.float32)

    for layer in layers:
        payload = {
            view: np.concatenate(chunks, axis=0)
            for view, chunks in store[layer].items()
        }
        payload["sequence_index"] = sequence_index
        payload["token_position"] = token_position
        payload["target_token_id"] = target_token_id
        if lsd.size:
            payload["long_short_gap"] = lsd
        np.savez_compressed(output_dir / f"layer{int(layer)}_activations.npz", **payload)
    meta = {
        "layers": [int(layer) for layer in layers],
        "views": sorted(views),
        "num_tokens": int(token_total),
        "local_window": int(args.local_window),
        "ablation_results": str(args.ablation_results) if args.ablation_results else None,
        "hook_point": "resid_mid",
    }
    (output_dir / "activation_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "collect_sae_activations_done", **meta}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
