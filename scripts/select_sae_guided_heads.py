#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curcpt.sae import load_evidence_feature_set, load_sparse_autoencoder
from curcpt.sae_head_selection import (
    SAEHeadWriteScorer,
    build_selection_payload,
    evidence_decoder_vectors,
    load_candidate_heads,
    parse_layer_spec,
    select_top_heads,
)
from lgar_cpt.data import PackedSequenceSignalDataset
from lgar_cpt.mining import attention_mask_format_for_model, document_causal_attention_mask
from lgar_cpt.modeling import load_qwen_causal_lm, load_tokenizer, qwen_head_geometry, qwen_layers


def _device_from_arg(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _crop_signal_window(batch: dict[str, np.ndarray], max_tokens: int) -> dict[str, np.ndarray]:
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        return batch
    seq_len = int(batch["input_ids"].shape[1])
    if max_tokens >= seq_len:
        return batch

    high = np.asarray(batch["offline_lsd_labels"], dtype=bool)
    valid = np.asarray(batch["offline_lsd_valid"], dtype=bool)
    loss_mask = np.asarray(batch["loss_mask"], dtype=bool)
    signal = high & valid & loss_mask
    starts: list[int] = []
    for row in range(signal.shape[0]):
        positions = np.flatnonzero(signal[row])
        if positions.size:
            center = int(positions[positions.size // 2])
        else:
            center = seq_len // 2
        start = max(0, min(seq_len - max_tokens, center - max_tokens // 2))
        starts.append(int(start))

    cropped: dict[str, np.ndarray] = {}
    for key, value in batch.items():
        if not isinstance(value, np.ndarray) or value.ndim < 2 or value.shape[0] != len(starts):
            cropped[key] = value
            continue
        if value.shape[1] == seq_len:
            rows = [value[row, start : start + max_tokens] for row, start in enumerate(starts)]
            cropped[key] = np.stack(rows, axis=0)
        elif value.shape[1] == seq_len + 1:
            rows = [value[row, start : start + max_tokens + 1] for row, start in enumerate(starts)]
            cropped[key] = np.stack(rows, axis=0)
        else:
            cropped[key] = value
    cropped["crop_start"] = np.asarray(starts, dtype=np.int64)
    return cropped


def _tensor_batch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("input_ids", "doc_ids_full", "loss_mask", "offline_lsd_labels", "offline_lsd_valid")
    return {key: torch.as_tensor(batch[key], device=device) for key in keys}


def _attention_mask(
    model: torch.nn.Module,
    doc_ids: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor | None:
    if mode == "causal":
        return None
    mask_format = attention_mask_format_for_model(model)
    return document_causal_attention_mask(doc_ids, mask_format=mask_format).to(device=doc_ids.device)


def _jsonable_head_scores(head_scores: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, metrics in head_scores.items():
        out[key] = {metric: float(value) for metric, value in metrics.items()}
    return out


def _validate_candidate_heads(
    *,
    features_layer: int,
    candidate_heads: list[tuple[int, int]],
) -> None:
    if not candidate_heads:
        raise ValueError("no candidate heads to score")
    after_sae = [(layer, head) for layer, head in candidate_heads if int(layer) > int(features_layer)]
    if after_sae:
        raise ValueError(
            "SAE-guided selection scores head writes against the SAE residual layer, so candidate heads must be "
            f"at or before sae_layer={features_layer}; found after-SAE candidates like {after_sae[:5]}"
        )


def _validate_selected_heads(
    *,
    selected_heads: list[list[int]],
    top_k: int,
) -> None:
    if len(selected_heads) != int(top_k):
        raise ValueError(f"selected {len(selected_heads)} heads, expected top_k={top_k}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Select CURE retrieval heads by validated SAE evidence-write score.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--offline-signal-dir", required=True)
    parser.add_argument("--sae-checkpoint", required=True)
    parser.add_argument("--features", required=True, help="validated evidence_features.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-layers", default=None, help="Layer spec, e.g. 16-21. Defaults to sae_layer-5..sae_layer.")
    parser.add_argument("--candidate-heads-json", default=None, help="Optional JSON restricting candidate heads.")
    parser.add_argument("--top-k-heads", type=int, default=24)
    parser.add_argument("--max-heads-per-layer", type=int, default=0)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=0, help="0 means full sequence; otherwise crop around LSD-positive tokens.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--attention-mask-mode", choices=["causal", "document"], default="causal")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-unvalidated-sae-features", action="store_true")
    parser.add_argument("--no-normalize-feature-vectors", action="store_true")
    args = parser.parse_args()

    device = _device_from_arg(args.device)
    features = load_evidence_feature_set(args.features, require_validated=not args.allow_unvalidated_sae_features)
    sae = load_sparse_autoencoder(args.sae_checkpoint, device=device)

    tokenizer = load_tokenizer(args.model_path)
    dataset = PackedSequenceSignalDataset(
        cache_dir=args.cache_dir,
        signal_dir=args.offline_signal_dir,
        pad_token_id=int(tokenizer.pad_token_id),
        seed=int(args.seed),
    )
    model = load_qwen_causal_lm(
        args.model_path,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=False,
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    n_heads, _n_kv_heads, head_dim = qwen_head_geometry(model)
    num_layers = len(qwen_layers(model))
    if args.candidate_layers is None:
        start = max(0, int(features.layer) - 5)
        candidate_layers = list(range(start, int(features.layer) + 1))
    else:
        candidate_layers = parse_layer_spec(args.candidate_layers, max_layers=num_layers)
    candidate_heads = load_candidate_heads(
        candidate_layers=candidate_layers,
        num_heads=n_heads,
        heads_json=args.candidate_heads_json,
    )
    _validate_candidate_heads(
        features_layer=int(features.layer),
        candidate_heads=candidate_heads,
    )
    if len(candidate_heads) < int(args.top_k_heads):
        raise ValueError(f"only {len(candidate_heads)} candidate heads, cannot select top_k={args.top_k_heads}")

    decoder_vectors = evidence_decoder_vectors(
        sae,
        features,
        device=device,
        normalize=not args.no_normalize_feature_vectors,
    )
    scorer = SAEHeadWriteScorer(
        model,
        candidate_heads,
        decoder_vectors,
        head_dim=head_dim,
    )

    total_high = 0
    total_valid = 0
    for step in range(int(args.batches)):
        batch_np = _crop_signal_window(dataset.sample(int(args.batch_size)), int(args.max_tokens))
        batch = _tensor_batch(batch_np, device)
        input_ids = batch["input_ids"].long()
        doc_ids = batch["doc_ids_full"][:, :-1].long()
        loss_mask = batch["loss_mask"].bool()
        high_mask = batch["offline_lsd_labels"].bool() & batch["offline_lsd_valid"].bool() & loss_mask
        nonhigh_mask = (~batch["offline_lsd_labels"].bool()) & batch["offline_lsd_valid"].bool() & loss_mask
        total_high += int(high_mask.sum().item())
        total_valid += int((batch["offline_lsd_valid"].bool() & loss_mask).sum().item())
        attention_mask = _attention_mask(model, doc_ids, mode=args.attention_mask_mode)

        handles = scorer.register_hooks(high_mask=high_mask, nonhigh_mask=nonhigh_mask)
        try:
            model(input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "sae_head_selection_batch",
                    "step": step + 1,
                    "batches": int(args.batches),
                    "high_tokens": int(high_mask.sum().item()),
                    "valid_tokens": int((batch["offline_lsd_valid"].bool() & loss_mask).sum().item()),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if total_high <= 0:
        raise RuntimeError("sampled batches contained zero LSD-positive tokens; increase --batches or inspect signal cache")
    head_scores = _jsonable_head_scores(scorer.metrics())
    selected_heads = select_top_heads(
        head_scores,
        top_k=int(args.top_k_heads),
        max_heads_per_layer=int(args.max_heads_per_layer) if int(args.max_heads_per_layer) > 0 else None,
    )
    _validate_selected_heads(
        selected_heads=selected_heads,
        top_k=int(args.top_k_heads),
    )

    metadata: dict[str, Any] = {
        "model_path": str(args.model_path),
        "cache_dir": str(args.cache_dir),
        "offline_signal_dir": str(args.offline_signal_dir),
        "sae_checkpoint": str(args.sae_checkpoint),
        "features": str(args.features),
        "sae_layer": int(features.layer),
        "feature_ids": [int(x) for x in features.feature_ids],
        "num_features": int(len(features.feature_ids)),
        "candidate_layers": [int(x) for x in candidate_layers],
        "candidate_heads": [[int(layer), int(head)] for layer, head in candidate_heads],
        "num_candidate_heads": int(len(candidate_heads)),
        "top_k_heads": int(args.top_k_heads),
        "max_heads_per_layer": int(args.max_heads_per_layer),
        "batches": int(args.batches),
        "batch_size": int(args.batch_size),
        "max_tokens": int(args.max_tokens),
        "total_high_tokens": int(total_high),
        "total_valid_tokens": int(total_valid),
        "attention_mask_mode": str(args.attention_mask_mode),
        "dtype": str(args.dtype),
        "attn_implementation": str(args.attn_implementation),
        "normalized_feature_vectors": not args.no_normalize_feature_vectors,
    }
    payload = build_selection_payload(
        retrieval_heads=selected_heads,
        head_scores=head_scores,
        metadata=metadata,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "sae_guided_heads_selected",
                "output": str(output),
                "retrieval_heads": selected_heads,
                "num_selected": len(selected_heads),
                "total_high_tokens": int(total_high),
                "total_valid_tokens": int(total_valid),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
