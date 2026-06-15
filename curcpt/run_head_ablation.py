from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from lgar_cpt.config import Paths
from lgar_cpt.data import PackedFineWebDataset, PackedSequenceSignalDataset
from lgar_cpt.modeling import load_qwen_causal_lm, load_tokenizer
from lgar_cpt.utils import set_seed

from .config import CUREParams, ablation_candidate_layers
from .head_ablation import (
    run_ablation_calibration,
    select_heads_from_scores,
    save_ablation_results,
)


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
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
    return rank, world_size, local_rank, device


def main() -> None:
    parser = argparse.ArgumentParser(description="CURE-CPT head ablation calibration")
    parser.add_argument("--model-path", default="/gemini/space/private/zjc/models/Qwen2.5-0.5B")
    parser.add_argument("--checkpoint-path", default=None, help="CE-CPT checkpoint to load")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--offline-signal-dir", default=None, help="Offline LSD signal/layout dir with offline_lsd_labels.npy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ablation-sequences", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k-fraction", type=float, default=0.05)
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--min-remote-margin", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--no-split-half", action="store_true")
    args = parser.parse_args()

    rank, world_size, local_rank, device = _distributed_context()
    set_seed(args.seed)

    tokenizer = load_tokenizer(args.model_path)
    model = load_qwen_causal_lm(
        args.model_path,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=False,
    ).to(device)

    if args.checkpoint_path is not None:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    model.eval()

    params = CUREParams(
        seq_len=args.seq_len,
        local_window=args.local_window,
        min_remote_margin=args.min_remote_margin,
        ablation_batch_size=args.batch_size,
        ablation_calibration_sequences=args.ablation_sequences,
        ablation_top_k_fraction=args.top_k_fraction,
        ablation_split_half=not args.no_split_half,
    )

    def make_dataset(seed: int):
        if args.offline_signal_dir:
            return PackedSequenceSignalDataset(
                cache_dir=args.cache_dir,
                signal_dir=args.offline_signal_dir,
                pad_token_id=int(tokenizer.pad_token_id),
                seed=seed,
            )
        return PackedFineWebDataset(
            args.cache_dir, args.seq_len, int(tokenizer.pad_token_id), "val", seed=seed,
        )

    # Split-half: two different seeds for calibration sampling.
    # With offline_signal_dir, the sampled sequences carry offline_lsd_labels,
    # so head ablation does not recompute online long/short log-probs at 32k.
    half1 = make_dataset(args.seed)
    half2 = make_dataset(args.seed + 1)

    # Shard the candidate heads across ranks. Each rank ablates a disjoint
    # subset on the SAME calibration data (same seeds -> identical batches),
    # so per-head scores are directly comparable and need no numeric reduce.
    num_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    candidate_layers = ablation_candidate_layers(num_layers, params.ablation_layer_fraction)
    all_candidates = [(l, h) for l in candidate_layers for h in range(n_heads)]
    my_heads = all_candidates[rank::world_size]
    if rank == 0:
        print(f"[ablate] {len(all_candidates)} candidate heads across {world_size} rank(s); "
              f"rank0 handles {len(my_heads)}")
        if args.offline_signal_dir:
            print(f"[ablate] using offline_signal_dir={args.offline_signal_dir}", flush=True)
        else:
            print("[ablate] no offline_signal_dir; online HU mining is enabled", flush=True)

    sc1 = run_ablation_calibration(
        model,
        half1,
        params,
        device=device,
        batch_size=args.batch_size,
        candidate_heads=my_heads,
    )
    sc2 = run_ablation_calibration(
        model,
        half2,
        params,
        device=device,
        batch_size=args.batch_size,
        candidate_heads=my_heads,
    )
    local_h1 = {f"{l}_{h}": v["mean_delta"] for (l, h), v in sc1.items()}
    local_h2 = {f"{l}_{h}": v["mean_delta"] for (l, h), v in sc2.items()}

    # Gather disjoint score shards onto every rank, then merge.
    if world_size > 1:
        g1: list = [None] * world_size
        g2: list = [None] * world_size
        dist.all_gather_object(g1, local_h1)
        dist.all_gather_object(g2, local_h2)
        raw_h1, raw_h2 = {}, {}
        for d in g1:
            raw_h1.update(d)
        for d in g2:
            raw_h2.update(d)
    else:
        raw_h1, raw_h2 = local_h1, local_h2

    results = select_heads_from_scores(raw_h1, raw_h2, params)
    results["calibration_sequences_per_half"] = params.ablation_calibration_sequences
    results["split_half_stable"] = params.ablation_split_half

    if rank == 0:
        output_path = Path(args.output_dir) / "ablation_results.json"
        save_ablation_results(results, output_path)
        print(f"Ablation complete: {results['num_selected']} retrieval heads selected "
              f"(of {results['num_candidates']} candidates)")
        print(f"Results saved to {output_path}")
        for h in results["retrieval_heads"][:10]:
            key = f"{h[0]}_{h[1]}"
            score = results["head_scores"].get(key, 0)
            print(f"  layer {h[0]}, head {h[1]}: score={score:.4f}")

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
