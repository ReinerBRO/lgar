from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import dtype_from_name


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qwen_causal_lm(
    model_path: str,
    dtype_name: str = "bf16",
    attn_implementation: str | None = "eager",
    gradient_checkpointing: bool = False,
) -> torch.nn.Module:
    dtype = dtype_from_name(dtype_name)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def qwen_backbone(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "base_model"):
        return model.base_model
    raise AttributeError("cannot locate Qwen backbone")


def qwen_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    backbone = qwen_backbone(model)
    if hasattr(backbone, "layers"):
        return backbone.layers
    raise AttributeError("cannot locate Qwen decoder layers")


def qwen_head_geometry(model: torch.nn.Module) -> tuple[int, int, int]:
    cfg = model.config
    n_heads = int(cfg.num_attention_heads)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(cfg.hidden_size) // n_heads
    return n_heads, n_kv_heads, head_dim


def unwrap_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs.logits


def unwrap_hidden_states(outputs: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(outputs, dict):
        return outputs["hidden_states"]
    return outputs.hidden_states

