from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .adapters import HeadAdapterSet


@dataclass
class CUREForwardOutput:
    logits_full: torch.Tensor
    logits_rh_only: torch.Tensor | None
    metrics: dict[str, float]


def _unwrap_parallel_module(module: nn.Module) -> nn.Module:
    current = module
    while current.__class__.__name__ in {"FullyShardedDataParallel", "DistributedDataParallel"} and hasattr(current, "module"):
        current = current.module
    return current


def _model_config(model: nn.Module):
    base = _unwrap_parallel_module(model)
    config = getattr(base, "config", None)
    if config is None:
        raise AttributeError("cannot locate model config")
    return config


def _model_backbone(model: nn.Module) -> nn.Module:
    base = _unwrap_parallel_module(model)
    if hasattr(base, "model"):
        return base.model
    if hasattr(base, "base_model"):
        return base.base_model
    raise AttributeError("cannot locate model backbone")


def _register_adapter_hooks(
    model: nn.Module,
    adapters: HeadAdapterSet,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register hooks on q_proj and o_proj for each retrieval head.

    For each retrieval head (layer_id, q_head_id):
    - q_proj post-hook: adds LoRA delta to the target Q head's slice
    - o_proj pre-hook: adds LoRA delta to the target head's slice of o_proj input

    Returns list of handles for cleanup.
    """
    backbone = _model_backbone(model)
    config = _model_config(model)
    n_heads = int(config.num_attention_heads)
    head_dim = int(config.hidden_size) // n_heads
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_id, q_head_id in sorted(adapters.retrieval_heads):
        q_lora = adapters.get_q_lora(layer_id, q_head_id)
        o_lora = adapters.get_o_lora(layer_id, q_head_id)
        if q_lora is None or o_lora is None:
            continue

        decoder_layer = _unwrap_parallel_module(backbone.layers[layer_id])
        attn = decoder_layer.self_attn
        q_proj = attn.q_proj
        o_proj = attn.o_proj

        # Q hook: post-hook on q_proj to add LoRA to target head's slice
        # q_proj output: [B, T, hidden_size]
        # register_forward_hook: (module, input, output)
        def q_hook_fn(module, inp, output, *, _q=q_lora, _n=n_heads, _d=head_dim, _t=q_head_id):
            shape = output.shape  # [B, T, hidden_size]
            x = output.reshape(shape[0], shape[1], _n, _d)
            lora_delta = _q(x[:, :, _t])  # [B, T, head_dim]
            head_mask = torch.zeros(_n, device=output.device, dtype=output.dtype)
            head_mask[_t] = 1.0
            full_delta = lora_delta[:, :, None, :] * head_mask[None, None, :, None]
            return (x + full_delta).reshape(shape)

        handles.append(q_proj.register_forward_hook(q_hook_fn))

        # O hook: pre-hook on o_proj to add LoRA to target head's slice
        # o_proj input: [B, T, hidden_size]
        # register_forward_pre_hook with_kwargs: (module, args, kwargs)
        def o_hook_fn(module, args, kwargs, *, _o=o_lora, _n=n_heads, _d=head_dim, _t=q_head_id):
            x = args[0]  # [B, T, hidden_size]
            shape = x.shape
            x_r = x.reshape(shape[0], shape[1], _n, _d)
            lora_delta = _o(x_r[:, :, _t])  # [B, T, head_dim]
            head_mask = torch.zeros(_n, device=x.device, dtype=x.dtype)
            head_mask[_t] = 1.0
            full_delta = lora_delta[:, :, None, :] * head_mask[None, None, :, None]
            return ((x_r + full_delta).reshape(shape),) + args[1:], kwargs

        handles.append(o_proj.register_forward_pre_hook(o_hook_fn, with_kwargs=True))

    return handles


def _build_rh_bottleneck_mask(
    doc_ids: torch.Tensor,
    layer_retrieval_q_heads: set[int],
    n_heads: int,
    local_window: int,
    mask_format: str = "additive",
) -> torch.Tensor:
    """Build per-head attention mask for RH-bottleneck forward at one layer.

    Retrieval heads (by q_head_id): full document-causal mask
    Non-retrieval heads: local-only document-causal mask

    Args:
        doc_ids: [B, T]
        layer_retrieval_q_heads: set of q_head_ids that are retrieval heads at this layer
        n_heads: total number of Q heads
        local_window: local attention window
        mask_format: "bool" or "additive"

    Returns:
        mask: [B, n_heads, T, T]
    """
    bsz, seq_len = doc_ids.shape
    device = doc_ids.device

    q_idx = torch.arange(seq_len, device=device)[:, None]
    k_idx = torch.arange(seq_len, device=device)[None, :]

    causal = k_idx <= q_idx
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    real_query = doc_ids[:, :, None] >= 0
    self_query = k_idx == q_idx

    # Full document-causal
    full_allowed = (causal[None, :, :] & same_doc & real_query) | (
        self_query[None, :, :] & ~real_query
    )

    # Local-only
    local = (k_idx <= q_idx) & (k_idx >= q_idx - local_window + 1)
    local_allowed = (local[None, :, :] & same_doc & real_query) | (
        self_query[None, :, :] & ~real_query
    )

    if mask_format == "bool":
        mask = local_allowed.unsqueeze(1).expand(bsz, n_heads, seq_len, seq_len).clone()
        for q_head_id in layer_retrieval_q_heads:
            if q_head_id < n_heads:
                mask[:, q_head_id] = full_allowed.squeeze(1)
        return mask
    else:
        min_val = -1.0e4
        full_additive = torch.zeros(bsz, 1, seq_len, seq_len, device=device, dtype=torch.float32)
        full_additive.masked_fill_(~full_allowed.unsqueeze(1), min_val)
        local_additive = torch.zeros(bsz, 1, seq_len, seq_len, device=device, dtype=torch.float32)
        local_additive.masked_fill_(~local_allowed.unsqueeze(1), min_val)

        mask = local_additive.expand(bsz, n_heads, seq_len, seq_len).clone()
        for q_head_id in layer_retrieval_q_heads:
            if q_head_id < n_heads:
                mask[:, q_head_id] = full_additive.squeeze(1)
        return mask


def _make_bottleneck_mask_hook(
    retrieval_heads: set[tuple[int, int]],
    layer_idx: int,
    local_window: int,
    doc_ids: torch.Tensor,
    n_heads: int,
):
    """Return a pre-hook that provides per-head mask for one layer."""
    layer_q_heads = {h for l, h in retrieval_heads if l == layer_idx}
    if not layer_q_heads:
        return None

    def hook_fn(module, args, kwargs):
        mask = _build_rh_bottleneck_mask(
            doc_ids, layer_q_heads, n_heads, local_window,
            mask_format="additive",
        )
        ref_param = next(module.parameters())
        kwargs["attention_mask"] = mask.to(device=ref_param.device, dtype=ref_param.dtype)
        return args, kwargs

    return hook_fn


def _register_rh_bottleneck_mask_hooks(
    model: nn.Module,
    doc_ids: torch.Tensor,
    retrieval_heads: set[tuple[int, int]],
    local_window: int,
    scope: str = "all_layers",
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register RH-bottleneck masks on every decoder layer.

    This is the exact path used by evaluation: retrieval heads get full
    document-causal attention; all other heads, including all heads in layers
    with no retrieval head, get document-local-window attention.
    """
    backbone = _model_backbone(model)
    n_heads = int(_model_config(model).num_attention_heads)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    if scope not in {"all_layers", "routed_layers"}:
        raise ValueError(f"unsupported rh_bottleneck scope: {scope}")
    for layer_idx, decoder_layer in enumerate(backbone.layers):
        decoder_layer = _unwrap_parallel_module(decoder_layer)
        layer_q_heads = {h for l, h in retrieval_heads if l == layer_idx}
        if scope == "routed_layers" and not layer_q_heads:
            continue
        hook = _make_bottleneck_mask_hook(
            retrieval_heads={(layer_idx, h) for h in layer_q_heads},
            layer_idx=layer_idx,
            local_window=local_window,
            doc_ids=doc_ids,
            n_heads=n_heads,
        )
        if hook is None:
            # Empty set intentionally means all heads local in this layer.
            def hook(module, args, kwargs, *, _doc=doc_ids, _lh=local_window, _nh=n_heads):
                mask = _build_rh_bottleneck_mask(
                    _doc, set(), _nh, _lh, mask_format="additive",
                )
                ref_param = next(module.parameters())
                kwargs["attention_mask"] = mask.to(device=ref_param.device, dtype=ref_param.dtype)
                return args, kwargs

        handles.append(decoder_layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True))
    return handles


def _rh_bottleneck_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    retrieval_heads: set[tuple[int, int]],
    local_window: int,
    scope: str = "all_layers",
) -> torch.Tensor:
    """Forward pass with per-head bottleneck masks, bypassing _update_causal_mask.

    Directly iterates decoder layers, injecting per-head masks for routed layers
    and the standard mask for non-routed layers.
    """
    backbone = _model_backbone(model)
    config = _model_config(model)
    n_heads = int(config.num_attention_heads)
    lm_head = _unwrap_parallel_module(model).lm_head

    if scope not in {"all_layers", "routed_layers"}:
        raise ValueError(f"unsupported rh_bottleneck scope: {scope}")

    # Prepare per-head masks. Strict all_layers makes non-routed layers local-only;
    # routed_layers keeps layers without retrieval heads full-attention.
    layer_masks: dict[int, torch.Tensor] = {}
    for layer_idx, _decoder_layer in enumerate(backbone.layers):
        layer_q_heads = {h for l, h in retrieval_heads if l == layer_idx}
        if scope == "routed_layers" and not layer_q_heads:
            continue
        layer_masks[layer_idx] = _build_rh_bottleneck_mask(
            doc_ids, layer_q_heads, n_heads, local_window, mask_format="additive",
        )

    # Run backbone manually
    hidden_states = backbone.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    position_embeddings = backbone.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(backbone.layers):
        decoder_layer = _unwrap_parallel_module(decoder_layer)
        mask = layer_masks.get(layer_idx, full_attention_mask).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )[0]

    hidden_states = backbone.norm(hidden_states)
    logits = lm_head(hidden_states)
    return logits


def cure_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    retrieval_heads: set[tuple[int, int]],
    adapters: HeadAdapterSet | None,
    local_window: int,
    compute_rh_bottleneck: bool = True,
    rh_bottleneck_scope: str = "all_layers",
) -> CUREForwardOutput:
    """Run CURE dual-path forward.

    Both forward passes use adapter hooks so gradients flow to adapter params.
    The RH-bottleneck pass adds per-head local/full attention routing on top.

    1. Full forward: adapters active, standard document-causal mask
    2. RH-bottleneck: adapters active, retrieval heads full / non-retrieval heads local
    """
    adapter_handles: list[torch.utils.hooks.RemovableHandle] = []

    # Register adapter hooks for both passes
    if adapters is not None and retrieval_heads:
        adapter_handles = _register_adapter_hooks(model, adapters)

    try:
        # Forward 1: full (with adapters active)
        out_full = model(
            input_ids,
            attention_mask=full_attention_mask,
            use_cache=False,
        )
        logits_full = out_full.logits if hasattr(out_full, "logits") else out_full["logits"]

        logits_rh_only = None
        if compute_rh_bottleneck and retrieval_heads:
            mask_handles = _register_rh_bottleneck_mask_hooks(
                model, doc_ids, retrieval_heads, local_window, scope=rh_bottleneck_scope
            )

            try:
                out_rh = model(input_ids, attention_mask=full_attention_mask, use_cache=False)
                logits_rh_only = out_rh.logits if hasattr(out_rh, "logits") else out_rh["logits"]
            finally:
                for h in mask_handles:
                    h.remove()

    finally:
        for h in adapter_handles:
            h.remove()

    metrics = {
        "forward/retrieval_heads": float(len(retrieval_heads)),
        "forward/rh_bottleneck_computed": logits_rh_only is not None,
    }
    return CUREForwardOutput(
        logits_full=logits_full,
        logits_rh_only=logits_rh_only,
        metrics=metrics,
    )
