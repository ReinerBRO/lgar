from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from lgar_cpt.mining import (
    attention_mask_format_for_model,
    document_causal_attention_mask,
    local_document_attention_mask,
)
from lgar_cpt.modeling import unwrap_logits

from .forward import _build_rh_bottleneck_mask
from .model_eval_utils import heads_from_adapter_state, heads_from_summary


def parse_retrieval_heads(text: str | None) -> set[tuple[int, int]]:
    if not text:
        return set()
    heads: set[tuple[int, int]] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            layer, head = item.split(":", 1)
        elif "_" in item:
            layer, head = item.split("_", 1)
        else:
            raise ValueError(f"retrieval head must be L:H or L_H, got {item!r}")
        heads.add((int(layer), int(head)))
    return heads


def retrieval_heads_from_json(path: str | Path | None) -> set[tuple[int, int]]:
    if not path:
        return set()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(int(layer), int(head)) for layer, head in data.get("retrieval_heads", [])}


def retrieval_heads_from_checkpoint(checkpoint: dict[str, Any]) -> set[tuple[int, int]]:
    heads = heads_from_summary(checkpoint.get("summary", {}))
    if heads:
        return heads
    adapter_state = checkpoint.get("adapters")
    if adapter_state:
        return heads_from_adapter_state(adapter_state)
    return set()


def resolve_retrieval_heads(
    checkpoint: dict[str, Any],
    *,
    explicit_heads: str | None = None,
    heads_json: str | Path | None = None,
    reference_checkpoint_path: str | Path | None = None,
) -> set[tuple[int, int]]:
    heads = parse_retrieval_heads(explicit_heads)
    if heads:
        return heads
    heads = retrieval_heads_from_json(heads_json)
    if heads:
        return heads
    heads = retrieval_heads_from_checkpoint(checkpoint)
    if heads:
        return heads
    if reference_checkpoint_path:
        reference = torch.load(reference_checkpoint_path, map_location="cpu")
        heads = retrieval_heads_from_checkpoint(reference)
        if heads:
            return heads
    return set()


def full_doc_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
) -> torch.Tensor:
    mask = document_causal_attention_mask(
        doc_ids,
        mask_format=attention_mask_format_for_model(model),
    ).to(input_ids.device)
    return unwrap_logits(model(input_ids, attention_mask=mask, use_cache=False))


def local_all_heads_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    local_window: int,
) -> torch.Tensor:
    mask = local_document_attention_mask(
        doc_ids,
        int(local_window),
        mask_format=attention_mask_format_for_model(model),
    ).to(input_ids.device)
    return unwrap_logits(model(input_ids, attention_mask=mask, use_cache=False))


def rh_bottleneck_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    retrieval_heads: set[tuple[int, int]],
    local_window: int,
    bottleneck_scope: str = "all_layers",
) -> torch.Tensor:
    if not retrieval_heads:
        raise ValueError("rh_bottleneck mode requires at least one retrieval head")
    if bottleneck_scope not in {"all_layers", "routed_layers"}:
        raise ValueError(f"unsupported rh bottleneck scope: {bottleneck_scope}")

    backbone = model.model if hasattr(model, "model") else model.base_model
    n_heads = int(model.config.num_attention_heads)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_q_heads: set[int]):
        def hook_fn(module, args, kwargs):
            mask = _build_rh_bottleneck_mask(
                doc_ids,
                layer_q_heads,
                n_heads,
                int(local_window),
                mask_format="additive",
            )
            ref_param = next(module.parameters())
            kwargs["attention_mask"] = mask.to(device=ref_param.device, dtype=ref_param.dtype)
            return args, kwargs

        return hook_fn

    try:
        for layer_idx, layer in enumerate(backbone.layers):
            layer_q_heads = {head for layer_id, head in retrieval_heads if layer_id == layer_idx}
            if bottleneck_scope == "routed_layers" and not layer_q_heads:
                continue
            handles.append(layer.self_attn.register_forward_pre_hook(make_hook(layer_q_heads), with_kwargs=True))
        return full_doc_logits(model, input_ids, doc_ids)
    finally:
        for handle in handles:
            handle.remove()


def pack_sequences(
    sequences: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths = [len(ids) for ids in sequences]
    max_len = max(lengths)
    input_ids = torch.full((len(sequences), max_len), int(pad_token_id), dtype=torch.long, device=device)
    doc_ids = torch.full((len(sequences), max_len), -1, dtype=torch.long, device=device)
    for row_idx, ids in enumerate(sequences):
        row = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[row_idx, : len(ids)] = row
        doc_ids[row_idx, : len(ids)] = 0
    return input_ids, doc_ids, lengths


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    seq_len: int,
    max_new_tokens: int,
    device: torch.device,
    *,
    eval_mode: str = "full",
    retrieval_heads: set[tuple[int, int]] | None = None,
    local_window: int = 1024,
    stop_on_newline: bool = False,
) -> list[str]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must provide pad_token_id or eos_token_id")
    eos_token_id = tokenizer.eos_token_id
    sequences = [
        tokenizer.encode(prompt, add_special_tokens=False)[-int(seq_len) :]
        for prompt in prompts
    ]
    generated: list[list[int]] = [[] for _ in sequences]
    finished = [False for _ in sequences]

    for _step in range(int(max_new_tokens)):
        input_ids, doc_ids, lengths = pack_sequences(sequences, int(pad_token_id), device)
        if eval_mode == "full":
            logits = full_doc_logits(model, input_ids, doc_ids)
        elif eval_mode in {"rh_bottleneck", "rh_layer_bottleneck"}:
            logits = rh_bottleneck_logits(
                model,
                input_ids,
                doc_ids,
                retrieval_heads or set(),
                int(local_window),
                bottleneck_scope="routed_layers" if eval_mode == "rh_layer_bottleneck" else "all_layers",
            )
        else:
            raise ValueError(f"unsupported eval_mode: {eval_mode}")

        all_done = True
        for row_idx, length in enumerate(lengths):
            if finished[row_idx]:
                continue
            next_token = int(torch.argmax(logits[row_idx, int(length) - 1]).item())
            generated[row_idx].append(next_token)
            if eos_token_id is not None and next_token == int(eos_token_id):
                finished[row_idx] = True
                continue
            if stop_on_newline:
                decoded = tokenizer.decode(generated[row_idx], skip_special_tokens=True)
                if "\n" in decoded and decoded.strip():
                    finished[row_idx] = True
                    continue
            sequences[row_idx].append(next_token)
            if len(sequences[row_idx]) > int(seq_len):
                sequences[row_idx] = sequences[row_idx][-int(seq_len) :]
            all_done = False
        if all_done:
            break

    texts = [tokenizer.decode(tokens, skip_special_tokens=True).strip() for tokens in generated]
    if stop_on_newline:
        texts = [text.split("\n", 1)[0].strip() for text in texts]
    return texts
