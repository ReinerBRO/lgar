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
from curcpt.sae import load_evidence_feature_set, load_sparse_autoencoder


def _device_from_arg(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _materialize_indices(dataset, indices: np.ndarray) -> dict[str, np.ndarray]:
    layout = {
        "segment_doc_rows": np.asarray(dataset.segment_doc_rows[indices], dtype=np.int32),
        "segment_start_offsets": np.asarray(dataset.segment_start_offsets[indices], dtype=np.int32),
        "segment_lengths": np.asarray(dataset.segment_lengths[indices], dtype=np.int32),
        "sequence_indices": np.asarray(indices, dtype=np.int64),
    }
    return dataset.base.materialize_layout_batch(layout)


def _make_selected_heads_local_hook(
    q_heads: set[int],
    local_allowed: torch.Tensor,
    n_heads: int,
):
    q_head_ids = sorted(int(h) for h in q_heads)

    def hook_fn(module, args, kwargs):
        orig_mask = kwargs.get("attention_mask")
        if orig_mask is None and len(args) > 2:
            orig_mask = args[2]
        if orig_mask is None:
            return args, kwargs

        bsz = orig_mask.shape[0]
        seq_len = orig_mask.shape[-1]
        per_head_mask = orig_mask.expand(bsz, n_heads, seq_len, seq_len).clone()
        if orig_mask.dtype == torch.bool:
            per_head_mask[:, q_head_ids] = local_allowed[:, None, :, :]
        else:
            local_additive = torch.zeros(
                bsz,
                seq_len,
                seq_len,
                device=orig_mask.device,
                dtype=orig_mask.dtype,
            )
            local_additive.masked_fill_(~local_allowed, -1.0e4)
            per_head_mask[:, q_head_ids] = local_additive[:, None, :, :]
        kwargs["attention_mask"] = per_head_mask
        return args, kwargs

    return hook_fn


def _register_masked_head_hooks(
    model: torch.nn.Module,
    retrieval_heads: set[tuple[int, int]],
    doc_ids: torch.Tensor,
    local_window: int,
    mask_format: str,
) -> list[torch.utils.hooks.RemovableHandle]:
    from lgar_cpt.mining import local_document_attention_mask
    from lgar_cpt.modeling import qwen_layers

    if not retrieval_heads:
        raise ValueError("masked_head view requires non-empty --ablation-results")
    n_heads = int(model.config.num_attention_heads)
    local_allowed = local_document_attention_mask(doc_ids, int(local_window), mask_format="bool")
    local_allowed = local_allowed[:, 0]
    heads_by_layer: dict[int, set[int]] = {}
    for layer, head in retrieval_heads:
        heads_by_layer.setdefault(int(layer), set()).add(int(head))

    handles: list[torch.utils.hooks.RemovableHandle] = []
    layers = qwen_layers(model)
    for layer_idx, q_heads in heads_by_layer.items():
        attn = layers[layer_idx].self_attn
        hook = _make_selected_heads_local_hook(q_heads, local_allowed, n_heads)
        handles.append(attn.register_forward_pre_hook(hook, with_kwargs=True))
    return handles


@torch.no_grad()
def _capture_selected_features(
    model: torch.nn.Module,
    sae: torch.nn.Module,
    layer: int,
    feature_ids: tuple[int, ...],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    extra_handles: list[torch.utils.hooks.RemovableHandle] | None = None,
) -> torch.Tensor:
    resid_mid_by_layer: dict[int, torch.Tensor] = {}
    phase_ref = {"name": "full"}
    handles = _register_resid_mid_capture_hooks(
        model,
        {int(layer)},
        resid_mid_by_layer,
        phase_ref=phase_ref,
    )
    try:
        model(input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        for handle in extra_handles or []:
            handle.remove()
    if int(layer) not in resid_mid_by_layer:
        raise KeyError(f"missing captured resid_mid for layer {layer}")
    acts = sae.encode(resid_mid_by_layer[int(layer)])
    index = torch.as_tensor(feature_ids, device=acts.device, dtype=torch.long)
    return acts.index_select(dim=-1, index=index).detach().float()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute a_full0/aneg0 SAE targets for SAE-Steered CURE training."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--offline-signal-dir", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--sae-checkpoint", default=None)
    parser.add_argument("--ablation-results", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--views", default="short,masked_head")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--delta-feature", type=float, default=0.0)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--no-write-meta", action="store_true")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    views = {view.strip() for view in str(args.views).split(",") if view.strip()}
    unknown = views - {"short", "masked_head"}
    if unknown:
        raise ValueError(
            "Unsupported negative view(s): "
            + ", ".join(sorted(unknown))
            + ". corrupt_context needs evidence-span metadata and is intentionally not approximated here."
        )
    if "masked_head" in views and not args.ablation_results:
        raise ValueError("--views includes masked_head, so --ablation-results is required")

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

    features = load_evidence_feature_set(args.features, require_validated=True)
    sae_path = args.sae_checkpoint or features.sae_checkpoint
    if not sae_path:
        raise ValueError("--sae-checkpoint is required unless features JSON has sae_checkpoint")
    sae = load_sparse_autoencoder(sae_path, device=device)
    if int(sae.encoder.in_features) != int(model.config.hidden_size):
        raise ValueError(
            f"SAE input_dim={sae.encoder.in_features} does not match model hidden_size={model.config.hidden_size}"
        )

    retrieval_heads: set[tuple[int, int]] = set()
    ablation_payload = None
    if args.ablation_results:
        retrieval_heads, ablation_payload = load_ablation_results(args.ablation_results)

    signal_dir = Path(args.offline_signal_dir)
    output_dir = Path(args.output_dir) if args.output_dir else signal_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    offline_labels = np.load(signal_dir / "offline_lsd_labels.npy", mmap_mode="r")
    offline_valid = np.load(signal_dir / "offline_lsd_valid.npy", mmap_mode="r")
    dataset = PackedSequenceLayoutDataset(
        cache_dir=args.cache_dir,
        signal_dir=signal_dir,
        pad_token_id=int(tokenizer.pad_token_id),
        seed=1337,
    )
    num_sequences = dataset.num_sequences
    if args.max_sequences > 0:
        num_sequences = min(num_sequences, int(args.max_sequences))
    seq_len = int(dataset.seq_len)
    num_features = len(features.feature_ids)
    start_index = max(0, int(args.start_index))
    end_index = int(args.end_index) if int(args.end_index) > 0 else int(num_sequences)
    end_index = min(int(num_sequences), max(start_index, end_index))
    mode = "r+" if (output_dir / "offline_sae_feature_full.npy").exists() else "w+"

    full_mm = np.lib.format.open_memmap(
        output_dir / "offline_sae_feature_full.npy",
        mode=mode,
        dtype=np.float32,
        shape=(num_sequences, seq_len, num_features),
    )
    neg_mm = np.lib.format.open_memmap(
        output_dir / "offline_sae_feature_neg.npy",
        mode=mode,
        dtype=np.float32,
        shape=(num_sequences, seq_len, num_features),
    )
    mask_mm = np.lib.format.open_memmap(
        output_dir / "offline_sae_feature_mask.npy",
        mode=mode,
        dtype=bool,
        shape=(num_sequences, seq_len),
    )

    mask_format = attention_mask_format_for_model(model)
    for start in range(start_index, end_index, int(args.batch_size)):
        end = min(start + int(args.batch_size), end_index)
        indices = np.arange(start, end, dtype=np.int64)
        batch = _materialize_indices(dataset, indices)
        input_ids = torch.as_tensor(batch["input_ids"], device=device)
        doc_ids = torch.as_tensor(batch["doc_ids_full"][:, :-1], device=device)
        loss_mask = torch.as_tensor(batch["loss_mask"], device=device).bool()
        full_mask = document_causal_attention_mask(doc_ids, mask_format=mask_format).to(device)

        full_acts = _capture_selected_features(
            model,
            sae,
            int(features.layer),
            features.feature_ids,
            input_ids,
            full_mask,
        )
        neg_acts: list[torch.Tensor] = []
        if "short" in views:
            short_mask = local_document_attention_mask(
                doc_ids,
                int(args.local_window),
                mask_format=mask_format,
            ).to(device)
            neg_acts.append(
                _capture_selected_features(
                    model,
                    sae,
                    int(features.layer),
                    features.feature_ids,
                    input_ids,
                    short_mask,
                )
            )
        if "masked_head" in views:
            masked_handles = _register_masked_head_hooks(
                model,
                retrieval_heads,
                doc_ids,
                int(args.local_window),
                mask_format,
            )
            neg_acts.append(
                _capture_selected_features(
                    model,
                    sae,
                    int(features.layer),
                    features.feature_ids,
                    input_ids,
                    full_mask,
                    extra_handles=masked_handles,
                )
            )
        if not neg_acts:
            raise ValueError("at least one negative view is required")
        neg = torch.stack(neg_acts, dim=0).amax(dim=0)

        diff = full_acts - neg
        eligible = (
            torch.as_tensor(offline_labels[indices], device=device).bool()
            & torch.as_tensor(offline_valid[indices], device=device).bool()
            & loss_mask
            & (diff > float(args.delta_feature)).any(dim=-1)
        )
        full_mm[start:end] = full_acts.cpu().numpy()
        neg_mm[start:end] = neg.cpu().numpy()
        mask_mm[start:end] = eligible.cpu().numpy()
        print(
            json.dumps(
                {
                    "event": "precompute_sae_targets_progress",
                    "start": int(start),
                    "end": int(end),
                    "eligible_tokens": int(eligible.sum().item()),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    del full_mm, neg_mm, mask_mm
    if args.no_write_meta:
        return
    meta = {
        "features": str(args.features),
        "sae_checkpoint": str(sae_path),
        "layer": int(features.layer),
        "hook_point": features.hook_point,
        "feature_ids": list(features.feature_ids),
        "negative_views": sorted(views),
        "delta_feature": float(args.delta_feature),
        "local_window": int(args.local_window),
        "num_sequences": int(num_sequences),
        "start_index": int(start_index),
        "end_index": int(end_index),
        "seq_len": int(seq_len),
        "ablation_results": str(args.ablation_results) if args.ablation_results else None,
        "ablation_num_selected": (
            int(ablation_payload.get("num_selected", len(retrieval_heads)))
            if isinstance(ablation_payload, dict)
            else None
        ),
    }
    (output_dir / "offline_sae_feature_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"event": "precompute_sae_targets_done", **meta}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
