from __future__ import annotations

from typing import Any

import torch

from lgar_cpt.modeling import load_qwen_causal_lm, load_tokenizer

from .adapters import HeadAdapterSet
from .forward import _register_adapter_hooks


def heads_from_summary(summary: dict[str, Any]) -> set[tuple[int, int]]:
    return {(int(layer), int(head)) for layer, head in summary.get("retrieval_heads", [])}


def heads_from_adapter_state(adapter_state: dict[str, torch.Tensor]) -> set[tuple[int, int]]:
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


def infer_lora_rank(adapter_state: dict[str, torch.Tensor], default: int = 8) -> int:
    for key, value in adapter_state.items():
        if key.endswith(".lora_A") and value.ndim == 2:
            return int(value.shape[0])
    return int(default)


def infer_lora_alpha(checkpoint: dict[str, Any], default: float = 16.0) -> float:
    summary = checkpoint.get("summary", {})
    if isinstance(summary, dict) and summary.get("lora_alpha") is not None:
        return float(summary["lora_alpha"])
    return float(default)


def load_adapters_from_checkpoint(
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
) -> HeadAdapterSet | None:
    adapter_state = checkpoint.get("adapters")
    if not adapter_state:
        return None

    retrieval_heads = heads_from_summary(checkpoint.get("summary", {}))
    if not retrieval_heads:
        retrieval_heads = heads_from_adapter_state(adapter_state)
    if not retrieval_heads:
        raise SystemExit("checkpoint contains adapters but no retrieval head metadata")

    head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
    adapters = HeadAdapterSet(
        retrieval_heads=retrieval_heads,
        head_dim=head_dim,
        rank=infer_lora_rank(adapter_state, lora_rank),
        alpha=infer_lora_alpha(checkpoint, lora_alpha),
        hidden_size=int(model.config.hidden_size),
    )
    adapters.load_state_dict(adapter_state, strict=True)
    adapters.to(device)
    adapters.eval()
    return adapters


def load_model_tokenizer_for_eval(
    model_path: str,
    checkpoint_path: str,
    device: torch.device,
    dtype: str = "bf16",
    attn_implementation: str = "sdpa",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
) -> tuple[torch.nn.Module, Any, HeadAdapterSet | None, list[torch.utils.hooks.RemovableHandle], dict[str, Any]]:
    tokenizer = load_tokenizer(model_path)
    model = load_qwen_causal_lm(
        model_path,
        dtype_name=dtype,
        attn_implementation=attn_implementation,
        gradient_checkpointing=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(device)
    model.eval()

    adapters = load_adapters_from_checkpoint(
        checkpoint,
        model,
        device,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    handles = _register_adapter_hooks(model, adapters) if adapters is not None else []
    return model, tokenizer, adapters, handles, checkpoint
