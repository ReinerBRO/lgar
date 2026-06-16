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

from curcpt.head_ablation import load_ablation_results
from curcpt.sae import load_evidence_feature_set, load_sparse_autoencoder
from scripts.precompute_sae_feature_targets import (
    _capture_selected_features,
    _device_from_arg,
    _materialize_indices,
    _register_masked_head_hooks,
)


def _scheduled_sequences(
    num_sequences: int,
    world_size: int,
    steps_per_rank: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    chunks = []
    for rank in range(int(world_size)):
        rng = np.random.default_rng(int(seed) + rank * 1009)
        draws = rng.integers(
            0,
            int(num_sequences),
            size=int(steps_per_rank) * int(batch_size),
            dtype=np.int64,
        )
        chunks.append(draws)
    return np.unique(np.concatenate(chunks, axis=0))


def _part_path(output_dir: Path, shard_id: int) -> Path:
    return output_dir / f"sparse_targets_part{int(shard_id):02d}.npz"


def _run_worker(args: argparse.Namespace) -> None:
    from lgar_cpt.data import PackedSequenceLayoutDataset
    from lgar_cpt.mining import (
        attention_mask_format_for_model,
        document_causal_attention_mask,
        local_document_attention_mask,
    )
    from lgar_cpt.modeling import load_qwen_causal_lm, load_tokenizer

    signal_dir = Path(args.offline_signal_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_path)
    dataset = PackedSequenceLayoutDataset(
        cache_dir=args.cache_dir,
        signal_dir=signal_dir,
        pad_token_id=int(tokenizer.pad_token_id),
        seed=int(args.seed),
    )
    scheduled = _scheduled_sequences(
        dataset.num_sequences,
        int(args.world_size),
        int(args.steps_per_rank),
        int(args.batch_size),
        int(args.seed),
    )
    shard_sequences = scheduled[int(args.shard_id) :: int(args.num_shards)]

    features = load_evidence_feature_set(args.features, require_validated=True)
    sae_path = args.sae_checkpoint or features.sae_checkpoint
    if not sae_path:
        raise ValueError("--sae-checkpoint is required unless features JSON has sae_checkpoint")

    device = _device_from_arg(args.device)
    model = load_qwen_causal_lm(
        args.model_path,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=False,
    ).to(device)
    model.eval()
    sae = load_sparse_autoencoder(sae_path, device=device)
    if int(sae.encoder.in_features) != int(model.config.hidden_size):
        raise ValueError(
            f"SAE input_dim={sae.encoder.in_features} does not match model hidden_size={model.config.hidden_size}"
        )

    retrieval_heads, _ = load_ablation_results(args.ablation_results)
    offline_labels = np.load(signal_dir / "offline_lsd_labels.npy", mmap_mode="r")
    offline_valid = np.load(signal_dir / "offline_lsd_valid.npy", mmap_mode="r")
    mask_format = attention_mask_format_for_model(model)

    seq_chunks: list[np.ndarray] = []
    pos_chunks: list[np.ndarray] = []
    full_chunks: list[np.ndarray] = []
    neg_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for cursor, seq_idx in enumerate(shard_sequences.tolist()):
            indices = np.asarray([int(seq_idx)], dtype=np.int64)
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
            short_mask = local_document_attention_mask(
                doc_ids,
                int(args.local_window),
                mask_format=mask_format,
            ).to(device)
            short_acts = _capture_selected_features(
                model,
                sae,
                int(features.layer),
                features.feature_ids,
                input_ids,
                short_mask,
            )
            masked_handles = _register_masked_head_hooks(
                model,
                retrieval_heads,
                doc_ids,
                int(args.local_window),
                mask_format,
            )
            masked_acts = _capture_selected_features(
                model,
                sae,
                int(features.layer),
                features.feature_ids,
                input_ids,
                full_mask,
                extra_handles=masked_handles,
            )
            neg = torch.stack([short_acts, masked_acts], dim=0).amax(dim=0)
            diff = full_acts - neg
            eligible = (
                torch.as_tensor(offline_labels[indices], device=device).bool()
                & torch.as_tensor(offline_valid[indices], device=device).bool()
                & loss_mask
                & (diff > float(args.delta_feature)).any(dim=-1)
            )
            positions = eligible[0].nonzero(as_tuple=False).reshape(-1)
            if positions.numel() > 0:
                seq_chunks.append(
                    np.full((int(positions.numel()),), int(seq_idx), dtype=np.int32)
                )
                pos_chunks.append(positions.detach().cpu().numpy().astype(np.int32, copy=False))
                full_chunks.append(
                    full_acts[0, positions].detach().float().cpu().numpy().astype(np.float16)
                )
                neg_chunks.append(
                    neg[0, positions].detach().float().cpu().numpy().astype(np.float16)
                )
            if cursor % int(args.log_interval) == 0:
                print(
                    json.dumps(
                        {
                            "event": "precompute_sae_sparse_progress",
                            "shard": int(args.shard_id),
                            "cursor": int(cursor),
                            "total": int(shard_sequences.shape[0]),
                            "sequence": int(seq_idx),
                            "stored_tokens": int(sum(chunk.shape[0] for chunk in pos_chunks)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if seq_chunks:
        sequence_ids = np.concatenate(seq_chunks, axis=0)
        token_positions = np.concatenate(pos_chunks, axis=0)
        full_values = np.concatenate(full_chunks, axis=0)
        neg_values = np.concatenate(neg_chunks, axis=0)
    else:
        num_features = len(features.feature_ids)
        sequence_ids = np.zeros((0,), dtype=np.int32)
        token_positions = np.zeros((0,), dtype=np.int32)
        full_values = np.zeros((0, num_features), dtype=np.float16)
        neg_values = np.zeros((0, num_features), dtype=np.float16)

    out = _part_path(output_dir, int(args.shard_id))
    np.savez(
        out,
        sequence_ids=sequence_ids,
        token_positions=token_positions,
        full_values=full_values,
        neg_values=neg_values,
        shard_sequences=shard_sequences.astype(np.int64, copy=False),
    )
    done = output_dir / f"sparse_targets_part{int(args.shard_id):02d}.done"
    done.write_text(
        json.dumps(
            {
                "part": str(out),
                "shard": int(args.shard_id),
                "scheduled_sequences": int(shard_sequences.shape[0]),
                "stored_tokens": int(sequence_ids.shape[0]),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "precompute_sae_sparse_worker_done",
                "part": str(out),
                "shard": int(args.shard_id),
                "scheduled_sequences": int(shard_sequences.shape[0]),
                "stored_tokens": int(sequence_ids.shape[0]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _run_merge(args: argparse.Namespace) -> None:
    signal_dir = Path(args.offline_signal_dir)
    output_dir = Path(args.output_dir)
    features_payload = json.loads(Path(args.features).read_text(encoding="utf-8"))
    if "layers" in features_payload:
        features_payload = features_payload["layers"][0]
    layout = json.loads((signal_dir / "layout_meta.json").read_text(encoding="utf-8"))
    num_sequences = int(layout["num_sequences"])
    seq_len = int(layout["seq_len"])
    feature_ids = [int(x) for x in features_payload["feature_ids"]]
    num_features = len(feature_ids)

    parts = [_part_path(output_dir, shard_id) for shard_id in range(int(args.num_shards))]
    missing = [str(path) for path in parts if not path.exists()]
    if missing:
        raise FileNotFoundError("missing sparse target part(s): " + ", ".join(missing))

    counts = np.zeros(num_sequences, dtype=np.int64)
    total = 0
    for part in parts:
        payload = np.load(part)
        sequence_ids = np.asarray(payload["sequence_ids"], dtype=np.int64)
        if sequence_ids.size:
            counts += np.bincount(sequence_ids, minlength=num_sequences)
            total += int(sequence_ids.size)

    offsets = np.zeros(num_sequences + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    write_pos = offsets[:-1].copy()

    offsets_path = signal_dir / "offline_sae_sparse_offsets.npy"
    token_positions_path = signal_dir / "offline_sae_sparse_token_positions.npy"
    full_path = signal_dir / "offline_sae_sparse_full.npy"
    neg_path = signal_dir / "offline_sae_sparse_neg.npy"
    for path in (offsets_path, token_positions_path, full_path, neg_path):
        if path.exists():
            path.unlink()

    offsets_mm = np.lib.format.open_memmap(
        offsets_path,
        mode="w+",
        dtype=np.int64,
        shape=(num_sequences + 1,),
    )
    offsets_mm[:] = offsets
    del offsets_mm
    token_positions_mm = np.lib.format.open_memmap(
        token_positions_path,
        mode="w+",
        dtype=np.int32,
        shape=(total,),
    )
    full_mm = np.lib.format.open_memmap(
        full_path,
        mode="w+",
        dtype=np.float16,
        shape=(total, num_features),
    )
    neg_mm = np.lib.format.open_memmap(
        neg_path,
        mode="w+",
        dtype=np.float16,
        shape=(total, num_features),
    )

    for part in parts:
        payload = np.load(part)
        sequence_ids = np.asarray(payload["sequence_ids"], dtype=np.int64)
        token_positions = np.asarray(payload["token_positions"], dtype=np.int32)
        full_values = np.asarray(payload["full_values"], dtype=np.float16)
        neg_values = np.asarray(payload["neg_values"], dtype=np.float16)
        for local_idx, seq_idx in enumerate(sequence_ids.tolist()):
            dst = int(write_pos[int(seq_idx)])
            token_positions_mm[dst] = token_positions[local_idx]
            full_mm[dst] = full_values[local_idx]
            neg_mm[dst] = neg_values[local_idx]
            write_pos[int(seq_idx)] += 1

    del token_positions_mm, full_mm, neg_mm
    meta = {
        "format": "sequence_sparse_v1",
        "features": str(args.features),
        "sae_checkpoint": str(args.sae_checkpoint),
        "layer": int(features_payload["layer"]),
        "hook_point": features_payload.get("hook_point", "resid_mid"),
        "feature_ids": feature_ids,
        "num_features": int(num_features),
        "num_sequences": int(num_sequences),
        "seq_len": int(seq_len),
        "world_size": int(args.world_size),
        "steps_per_rank": int(args.steps_per_rank),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "num_shards": int(args.num_shards),
        "stored_tokens": int(total),
        "negative_views": ["masked_head", "short"],
        "dtype": "float16",
    }
    (signal_dir / "offline_sae_sparse_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"event": "precompute_sae_sparse_merge_done", **meta}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute schedule-sparse SAE feature targets.")
    parser.add_argument("--mode", choices=["worker", "merge"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--offline-signal-dir", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--sae-checkpoint", required=True)
    parser.add_argument("--ablation-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--steps-per-rank", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--local-window", type=int, default=2048)
    parser.add_argument("--delta-feature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "worker":
        _run_worker(args)
    else:
        _run_merge(args)


if __name__ == "__main__":
    main()
