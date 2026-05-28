from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .config import LGARParams, Paths
from .data import (
    PackedSequenceLayoutDataset,
    packed_sequence_count_for_tokens,
    prepare_packed_sequence_layout_cache,
)
from .evaluate import to_torch_batch
from .mining import mine_lsd_labels
from .modeling import load_qwen_causal_lm, load_tokenizer
from .sharding import iter_shard_ranges
from .utils import ensure_dir, set_seed, write_json


def _distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(hours=12))
    return rank, world_size, local_rank, device


def _is_rank0(rank: int) -> bool:
    return int(rank) == 0


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _all_reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _open_signal_memmaps(
    output_dir: Path,
    num_sequences: int,
    seq_len: int,
    save_lsd: bool,
    create: bool,
) -> tuple[np.memmap, np.memmap, np.memmap | None]:
    mode = "w+" if create else "r+"
    labels_mm = np.lib.format.open_memmap(
        output_dir / "offline_lsd_labels.npy",
        mode=mode,
        dtype=np.bool_,
        shape=(int(num_sequences), int(seq_len)),
    )
    valid_mm = np.lib.format.open_memmap(
        output_dir / "offline_lsd_valid.npy",
        mode=mode,
        dtype=np.bool_,
        shape=(int(num_sequences), int(seq_len)),
    )
    lsd_mm = None
    if save_lsd:
        lsd_mm = np.lib.format.open_memmap(
            output_dir / "offline_lsd.npy",
            mode=mode,
            dtype=np.float16,
            shape=(int(num_sequences), int(seq_len)),
        )
    return labels_mm, valid_mm, lsd_mm


def _load_teacher(
    model_path: str,
    checkpoint_path: str | None,
    dtype_name: str,
    attn_implementation: str,
    device: torch.device,
) -> torch.nn.Module:
    model = load_qwen_causal_lm(
        model_path,
        dtype_name=dtype_name,
        attn_implementation=attn_implementation,
    ).to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline packed-sequence LSD signal cache.")
    defaults = Paths()
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--cache-dir", default=defaults.cache_dir)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--num-sequences", type=int, default=None)
    parser.add_argument("--layout-batch-size", type=int, default=1024)
    parser.add_argument("--mine-batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--short-window", type=int, default=1024)
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--min-remote-margin", type=int, default=256)
    parser.add_argument("--long-nll-max-quantile", type=float, default=0.60)
    parser.add_argument("--final-global-budget", type=float, default=0.25)
    parser.add_argument("--min-doc-tokens", type=int, default=256)
    parser.add_argument("--val-docs", type=int, default=1024)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--save-lsd", action="store_true")
    args = parser.parse_args()
    rank, world_size, _local_rank, device = _distributed_context()

    if args.num_sequences is None and args.target_tokens is None:
        raise SystemExit("provide --num-sequences or --target-tokens")

    num_sequences = (
        int(args.num_sequences)
        if args.num_sequences is not None
        else packed_sequence_count_for_tokens(int(args.target_tokens), int(args.seq_len))
    )
    set_seed(int(args.seed) + int(rank))
    output_dir = ensure_dir(args.output_dir)
    tokenizer = load_tokenizer(args.model_path)

    layout_meta = output_dir / "layout_meta.json"
    if _is_rank0(rank) and not layout_meta.exists():
        prepare_packed_sequence_layout_cache(
            cache_dir=args.cache_dir,
            output_dir=output_dir,
            seq_len=int(args.seq_len),
            pad_token_id=int(tokenizer.pad_token_id),
            split=args.split,
            num_sequences=num_sequences,
            seed=int(args.seed),
            min_doc_tokens=int(args.min_doc_tokens),
            val_docs=int(args.val_docs),
            write_batch_size=int(args.layout_batch_size),
        )
    _barrier()

    dataset = PackedSequenceLayoutDataset(
        cache_dir=args.cache_dir,
        signal_dir=output_dir,
        pad_token_id=int(tokenizer.pad_token_id),
        seed=int(args.seed),
    )
    if int(dataset.seq_len) != int(args.seq_len):
        raise SystemExit(f"layout cache seq_len={dataset.seq_len} does not match requested --seq-len={args.seq_len}")
    if dataset.num_sequences != num_sequences:
        num_sequences = int(dataset.num_sequences)

    if _is_rank0(rank):
        labels_mm, valid_mm, lsd_mm = _open_signal_memmaps(
            output_dir=output_dir,
            num_sequences=num_sequences,
            seq_len=int(dataset.seq_len),
            save_lsd=bool(args.save_lsd),
            create=True,
        )
        labels_mm.flush()
        valid_mm.flush()
        if lsd_mm is not None:
            lsd_mm.flush()
        del labels_mm, valid_mm, lsd_mm
    _barrier()
    labels_mm, valid_mm, lsd_mm = _open_signal_memmaps(
        output_dir=output_dir,
        num_sequences=num_sequences,
        seq_len=int(dataset.seq_len),
        save_lsd=bool(args.save_lsd),
        create=False,
    )

    model = _load_teacher(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        device=device,
    )
    params = LGARParams(
        seq_len=int(dataset.seq_len),
        short_window=int(args.short_window),
        min_remote_margin=int(args.min_remote_margin),
        local_window=int(args.local_window),
        lsd_top_fraction=float(args.final_global_budget),
        long_nll_max_quantile=float(args.long_nll_max_quantile),
        router_target_budget=float(args.final_global_budget),
        final_global_budget=float(args.final_global_budget),
    )
    supervision_start = min(int(args.local_window), int(args.short_window) + int(args.min_remote_margin))

    stats_accum: dict[str, float] = {}
    stat_count = 0
    local_done = 0
    global_stride = max(1, int(args.mine_batch_size)) * max(1, int(world_size))
    num_global_steps = max(1, (int(num_sequences) + global_stride - 1) // global_stride)
    with torch.no_grad():
        for global_step_idx in range(num_global_steps):
            start = int(global_step_idx) * global_stride + int(rank) * int(args.mine_batch_size)
            end = min(int(num_sequences), start + int(args.mine_batch_size))
            if start < int(num_sequences):
                batch = dataset.base.materialize_layout_batch(
                    {
                        "segment_doc_rows": np.asarray(dataset.segment_doc_rows[start:end], dtype=np.int32),
                        "segment_start_offsets": np.asarray(dataset.segment_start_offsets[start:end], dtype=np.int32),
                        "segment_lengths": np.asarray(dataset.segment_lengths[start:end], dtype=np.int32),
                        "sequence_indices": np.arange(start, end, dtype=np.int64),
                    }
                )
                torch_batch = to_torch_batch(batch, device)
                labels = mine_lsd_labels(
                    model=model,
                    batch=torch_batch,
                    tokenizer=tokenizer,
                    params=params,
                    valid_offset_threshold=supervision_start,
                    positive_fraction=float(args.final_global_budget),
                )
                labels_mm[start:end] = labels.labels.detach().cpu().numpy().astype(np.bool_)
                valid_mm[start:end] = labels.valid.detach().cpu().numpy().astype(np.bool_)
                if lsd_mm is not None:
                    lsd_mm[start:end] = labels.lsd.detach().cpu().numpy().astype(np.float16)
                for key, value in labels.stats.items():
                    stats_accum[key] = stats_accum.get(key, 0.0) + float(value)
                stat_count += 1
                local_done += int(end - start)
            if (global_step_idx + 1) % 32 == 0 or global_step_idx + 1 == num_global_steps:
                labels_mm.flush()
                valid_mm.flush()
                if lsd_mm is not None:
                    lsd_mm.flush()
                done_sequences = int(min(int(num_sequences), round(_all_reduce_sum(float(local_done), device))))
                if _is_rank0(rank):
                    print(
                        json.dumps(
                            {
                                "event": "offline_signal_progress",
                                "done_sequences": done_sequences,
                                "total_sequences": int(num_sequences),
                                "world_size": int(world_size),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    labels_mm.flush()
    valid_mm.flush()
    if lsd_mm is not None:
        lsd_mm.flush()
    del labels_mm, valid_mm, lsd_mm
    total_stat_count = _all_reduce_sum(float(stat_count), device)
    stat_keys = sorted(stats_accum)
    if dist.is_available() and dist.is_initialized():
        gathered_keys: list[list[str] | None] = [None for _ in range(int(world_size))]
        dist.all_gather_object(gathered_keys, stat_keys)
        stat_keys = sorted({key for keys in gathered_keys if keys is not None for key in keys})
    mean_stats = {
        key: _all_reduce_sum(float(stats_accum.get(key, 0.0)), device) / max(1.0, total_stat_count) for key in stat_keys
    }
    _barrier()
    signal_meta = {
        "output_dir": str(output_dir),
        "cache_dir": str(args.cache_dir),
        "model_path": str(args.model_path),
        "checkpoint_path": args.checkpoint_path,
        "split": str(args.split),
        "num_sequences": int(num_sequences),
        "seq_len": int(dataset.seq_len),
        "short_window": int(args.short_window),
        "local_window": int(args.local_window),
        "min_remote_margin": int(args.min_remote_margin),
        "long_nll_max_quantile": float(args.long_nll_max_quantile),
        "final_global_budget": float(args.final_global_budget),
        "save_lsd": bool(args.save_lsd),
        "world_size": int(world_size),
        "mean_stats": mean_stats,
    }
    if _is_rank0(rank):
        write_json(output_dir / "signal_meta.json", signal_meta)
        print(
            json.dumps(
                {
                    "signal_meta": str(output_dir / "signal_meta.json"),
                    "num_sequences": int(num_sequences),
                    "world_size": int(world_size),
                },
                sort_keys=True,
            )
        )
    _barrier()
    _destroy_process_group()


if __name__ == "__main__":
    main()
