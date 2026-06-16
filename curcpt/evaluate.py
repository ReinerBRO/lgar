from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from lgar_cpt.config import LGARParams, Paths, TrainParams
from lgar_cpt.evaluate import (
    _destroy_process_group,
    _distributed_context,
    _is_rank0,
    evaluate_model,
)
from lgar_cpt.modeling import load_qwen_causal_lm, load_tokenizer
from lgar_cpt.utils import ensure_dir, write_json

from .adapters import HeadAdapterSet
from .forward import _register_adapter_hooks
from .model_eval_utils import infer_lora_alpha


def _heads_from_summary(summary: dict[str, Any]) -> set[tuple[int, int]]:
    return {(int(layer), int(head)) for layer, head in summary.get("retrieval_heads", [])}


def _heads_from_adapter_state(adapter_state: dict[str, torch.Tensor]) -> set[tuple[int, int]]:
    heads: set[tuple[int, int]] = set()
    for key in adapter_state:
        parts = key.split(".")
        if len(parts) < 2 or parts[0] not in {"q_loras", "o_loras"}:
            continue
        layer_head = parts[1].split("_")
        if len(layer_head) != 2:
            continue
        heads.add((int(layer_head[0]), int(layer_head[1])))
    return heads


def _infer_lora_rank(adapter_state: dict[str, torch.Tensor], default: int) -> int:
    for key, value in adapter_state.items():
        if key.endswith(".lora_A") and value.ndim == 2:
            return int(value.shape[0])
    return int(default)


def _load_adapters(
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    lora_rank: int,
    lora_alpha: float,
) -> HeadAdapterSet | None:
    adapter_state = checkpoint.get("adapters")
    if not adapter_state:
        return None

    retrieval_heads = _heads_from_summary(checkpoint.get("summary", {}))
    if not retrieval_heads:
        retrieval_heads = _heads_from_adapter_state(adapter_state)
    if not retrieval_heads:
        raise SystemExit("checkpoint contains adapters but no retrieval head metadata")

    head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
    adapters = HeadAdapterSet(
        retrieval_heads=retrieval_heads,
        head_dim=head_dim,
        rank=_infer_lora_rank(adapter_state, lora_rank),
        alpha=infer_lora_alpha(checkpoint, lora_alpha),
        hidden_size=int(model.config.hidden_size),
    )
    adapters.load_state_dict(adapter_state, strict=True)
    adapters.to(device)
    adapters.eval()
    return adapters


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CURE-CPT pilot checkpoints.")
    defaults = Paths()
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--cache-dir", default=defaults.cache_dir)
    parser.add_argument("--short-mc-dir", default=defaults.short_mc_dir)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--short-limit", type=int, default=16)
    parser.add_argument("--long-limit", type=int, default=8)
    parser.add_argument("--long-lengths", default="2048,4096")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--short-window", type=int, default=1024)
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    args = parser.parse_args()

    rank, _world_size, _local_rank, device = _distributed_context()
    paths = Paths(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        short_mc_dir=args.short_mc_dir,
    )
    lgar_params = LGARParams(
        seq_len=args.seq_len,
        short_window=args.short_window,
        local_window=args.local_window,
    )
    train_params = TrainParams(
        batch_size=args.batch_size,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    tokenizer = load_tokenizer(args.model_path)
    model = load_qwen_causal_lm(
        args.model_path,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=False,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(device)
    model.eval()

    adapters = _load_adapters(
        checkpoint,
        model,
        device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    handles = _register_adapter_hooks(model, adapters) if adapters is not None else []
    try:
        metrics = evaluate_model(
            model,
            tokenizer,
            paths,
            lgar_params,
            train_params,
            batches=args.eval_batches,
            short_limit=args.short_limit,
            long_limit=args.long_limit,
            long_lengths=[int(x.strip()) for x in str(args.long_lengths).split(",") if x.strip()],
            device=device,
            router=None,
            mode="full",
            target_budget=None,
        )
    finally:
        for handle in handles:
            handle.remove()
        _destroy_process_group()

    if _is_rank0(rank):
        payload = {
            "checkpoint_path": str(args.checkpoint_path),
            "adapters_active": adapters is not None,
            "num_retrieval_heads": adapters.num_heads() if adapters is not None else 0,
            "metrics": metrics,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        ensure_dir(Path(args.output).parent)
        write_json(args.output, payload)


if __name__ == "__main__":
    main()
