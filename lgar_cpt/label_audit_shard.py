from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import LGARParams, Paths
from .data import PackedFineWebDataset
from .modeling import load_qwen_causal_lm, load_tokenizer
from .train_stage0 import collect_label_audit
from .utils import ensure_dir, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one L-GAR Stage0 label-audit shard.")
    defaults = Paths()
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--cache-dir", default=defaults.cache_dir)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--sequences", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--short-window", type=int, default=1024)
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    seed = int(args.seed) + int(args.shard_id) * 1009
    set_seed(seed)
    paths = Paths(model_path=args.model_path, cache_dir=args.cache_dir)
    params = LGARParams(seq_len=args.seq_len, short_window=args.short_window, local_window=args.local_window)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(paths.model_path)
    dataset = PackedFineWebDataset(paths.cache_dir, params.seq_len, int(tokenizer.pad_token_id), "train", seed=seed)
    model = load_qwen_causal_lm(paths.model_path, dtype_name=args.dtype, attn_implementation=args.attn_implementation).to(device)
    model.eval()
    result = collect_label_audit(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        params=params,
        device=device,
        sequences=args.sequences,
        batch_size=args.batch_size,
        example_limit=20,
    )
    result["shard_id"] = int(args.shard_id)
    result["num_shards"] = int(args.num_shards)
    result["seed"] = seed
    write_json(args.output, result)
    print(json.dumps({"output": args.output, "shard_id": args.shard_id, "examples": len(result["examples"])}, sort_keys=True))


if __name__ == "__main__":
    main()
