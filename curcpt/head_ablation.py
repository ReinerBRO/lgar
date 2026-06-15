from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .config import CUREParams, ablation_candidate_layers


def _head_local_allowed(
    doc_ids: torch.Tensor,
    local_window: int,
) -> torch.Tensor:
    """Boolean mask [B, T, T] for document-causal local attention."""
    bsz, seq_len = doc_ids.shape
    del bsz
    device = doc_ids.device
    q = torch.arange(seq_len, device=device)[:, None]
    k = torch.arange(seq_len, device=device)[None, :]
    local = (k <= q) & (k >= q - int(local_window) + 1)
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    real_query = doc_ids[:, :, None] >= 0
    self_query = k == q
    return (local[None, :, :] & same_doc & real_query) | (self_query[None, :, :] & ~real_query)


def _remote_margin_valid_mask(
    doc_ids: torch.Tensor,
    doc_offsets_full: torch.Tensor | None,
    local_window: int,
    min_remote_margin: int,
) -> torch.Tensor:
    """Return label-position validity using intra-document target offsets."""
    valid = doc_ids >= 0
    threshold = int(local_window) + int(min_remote_margin)
    if doc_offsets_full is not None:
        target_offsets = doc_offsets_full[:, 1 : doc_ids.shape[1] + 1]
        return valid & (target_offsets >= threshold)

    offsets = torch.zeros_like(doc_ids)
    for row_idx in range(doc_ids.shape[0]):
        current_doc = None
        current_offset = -1
        for pos in range(doc_ids.shape[1]):
            doc = int(doc_ids[row_idx, pos].item())
            if doc < 0:
                current_doc = None
                current_offset = -1
                continue
            if current_doc != doc:
                current_doc = doc
                current_offset = 0
            else:
                current_offset += 1
            offsets[row_idx, pos] = current_offset
    return valid & (offsets >= threshold)


def _compute_position_nll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    doc_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run forward and return per-position NLL [B, T]."""
    out = model(
        input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = out.logits if hasattr(out, "logits") else out["logits"]
    nll_chunk_size = max(1, int(os.environ.get("CURE_NLL_CHUNK_SIZE", "512")))
    nll = torch.empty(labels.shape, device=labels.device, dtype=torch.float32)
    vocab_size = logits.size(-1)
    for start in range(0, labels.shape[1], nll_chunk_size):
        end = min(start + nll_chunk_size, labels.shape[1])
        nll[:, start:end] = F.cross_entropy(
            logits[:, start:end].float().reshape(-1, vocab_size),
            labels[:, start:end].reshape(-1),
            reduction="none",
        ).view(labels.shape[0], end - start)
    del logits, out
    return nll


def _make_head_ablation_hook(
    target_q_head: int,
    local_allowed: torch.Tensor,
    n_heads: int,
):
    """Return a forward hook that replaces the attention mask for one Q head.

    The hook modifies the attention_scores before softmax:
    - For the target head: apply local-only mask
    - For other heads: keep original mask (document-causal)

    This works by modifying the attention weights directly after Q@K^T,
    before the softmax. We hook into the attention module's forward.
    """

    def hook_fn(module, args, kwargs):
        # In Qwen2.5, the attention module receives attention_mask as a keyword arg.
        # We need to construct a per-head mask.
        # The standard attention_mask is [B, 1, T, T] (broadcast across heads).
        # We create [B, n_heads, T, T] with local-only for the target head.

        # NOTE: do not use `a or b` on tensors — bool(tensor) is ambiguous.
        orig_mask = kwargs.get("attention_mask")
        if orig_mask is None and len(args) > 2:
            orig_mask = args[2]
        if orig_mask is None:
            return args, kwargs

        # Determine mask format
        if orig_mask.dtype == torch.bool:
            # Boolean mask: True = attend
            B = orig_mask.shape[0]
            T = orig_mask.shape[-1]
            per_head_mask = orig_mask.expand(B, n_heads, T, T).clone()
            per_head_mask[:, target_q_head] = local_allowed
            kwargs["attention_mask"] = per_head_mask
        else:
            # Additive mask: 0 = attend, -inf = masked
            B = orig_mask.shape[0]
            T = orig_mask.shape[-1]
            per_head_mask = orig_mask.expand(B, n_heads, T, T).clone()
            # Build local-only additive mask for target head
            local_additive = torch.zeros(B, T, T, device=orig_mask.device, dtype=orig_mask.dtype)
            local_additive.masked_fill_(~local_allowed, -1.0e4)
            per_head_mask[:, target_q_head] = local_additive
            kwargs["attention_mask"] = per_head_mask

        return args, kwargs

    return hook_fn


@torch.no_grad()
def ablate_single_head(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    doc_ids: torch.Tensor,
    baseline_nll: torch.Tensor,
    layer_idx: int,
    q_head_id: int,
    local_window: int,
    high_utility_mask: torch.Tensor | None = None,
    full_attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mask remote attention for one Q head, return NLL delta.

    If high_utility_mask is provided, metrics are computed only at those positions.
    Returns:
        delta: [B, T] per-position NLL increase (ablated - baseline)
        metrics: dict with mean_delta, max_delta, etc.
    """
    n_heads = int(model.config.num_attention_heads)
    local_allowed = _head_local_allowed(doc_ids, local_window)

    backbone = model.model if hasattr(model, "model") else model.base_model
    target_attn = backbone.layers[layer_idx].self_attn

    hook = _make_head_ablation_hook(q_head_id, local_allowed, n_heads)
    handle = target_attn.register_forward_pre_hook(hook, with_kwargs=True)

    try:
        ablated_nll = _compute_position_nll(
            model, input_ids, labels, doc_ids, attention_mask=full_attention_mask
        )
    finally:
        handle.remove()

    delta = ablated_nll - baseline_nll

    # Filter to high-utility positions if provided, else use all valid positions
    if high_utility_mask is not None:
        eval_mask = high_utility_mask
    else:
        eval_mask = doc_ids >= 0
    delta_eval = delta[eval_mask]

    metrics = {
        "mean_delta": float(delta_eval.mean().item()) if delta_eval.numel() > 0 else 0.0,
        "max_delta": float(delta_eval.max().item()) if delta_eval.numel() > 0 else 0.0,
        "fraction_positive": float((delta_eval > 0).float().mean().item()) if delta_eval.numel() > 0 else 0.0,
    }
    return delta, metrics


@torch.no_grad()
def run_ablation_calibration(
    model: torch.nn.Module,
    dataset: Any,
    params: CUREParams,
    utility_cache: dict[int, torch.Tensor] | None = None,
    device: torch.device | None = None,
    batch_size: int | None = None,
    num_batches: int | None = None,
    mine_high_utility: bool = True,
    candidate_heads: list[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], dict[str, float]]:
    """Run head ablation on a calibration set.

    Args:
        model: loaded Qwen model
        dataset: PackedFineWebDataset with a .sample(batch_size) method
        params: CURE parameters
        utility_cache: optional precomputed utility masks per sequence index
        device: target device
        batch_size: sequences per sampled batch (defaults to params.ablation_batch_size)
        num_batches: number of batches to draw (defaults to ablation_calibration_sequences)
        mine_high_utility: mine high-utility positions online when no offline cache
        candidate_heads: restrict ablation to these (layer, q_head) pairs (for
            sharding the candidate set across ranks); defaults to all candidates

    Returns:
        dict mapping (layer_id, q_head_id) -> aggregated metrics
    """
    import numpy as np
    from lgar_cpt.mining import document_causal_attention_mask, attention_mask_format_for_model

    if device is None:
        device = next(model.parameters()).device
    bs = int(batch_size or params.ablation_batch_size)
    n_batches = int(num_batches or params.ablation_calibration_sequences)

    num_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    if candidate_heads is None:
        candidate_layers = ablation_candidate_layers(num_layers, params.ablation_layer_fraction)
        heads_to_ablate = [(l, h) for l in candidate_layers for h in range(n_heads)]
    else:
        heads_to_ablate = list(candidate_heads)
    mask_format = attention_mask_format_for_model(model)

    head_scores: dict[tuple[int, int], list[float]] = {hd: [] for hd in heads_to_ablate}

    def _to_t(v):
        return torch.as_tensor(v, device=device) if isinstance(v, np.ndarray) else v

    for _ in range(n_batches):
        batch = dataset.sample(bs)
        input_ids = _to_t(batch["input_ids"])
        labels = _to_t(batch["labels"])
        doc_ids = _to_t(batch["doc_ids_full"])[:, :-1]

        # High-utility positions: offline cache if present, else mine online.
        # CURE's premise is that retrieval work concentrates at high-utility
        # positions, so ablation delta MUST be measured there (averaging over
        # all positions dilutes the signal ~10x and is not a valid test).
        hu_mask = None
        if "offline_lsd_labels" in batch:
            hu_mask = _to_t(batch["offline_lsd_labels"]).bool()
            if "offline_lsd_valid" in batch:
                hu_mask = hu_mask & _to_t(batch["offline_lsd_valid"]).bool()
        elif mine_high_utility:
            from .utility_mining import compute_query_utility
            from lgar_cpt.mining import select_topk_global_queries
            utility, full_logp = compute_query_utility(
                model, input_ids, labels, doc_ids, params.local_window
            )
            doc_offsets = _to_t(batch["doc_offsets_full"]) if "doc_offsets_full" in batch else None
            valid = _remote_margin_valid_mask(
                doc_ids,
                doc_offsets,
                params.local_window,
                params.min_remote_margin,
            )
            hu_mask = select_topk_global_queries(
                utility, valid, params.utility_top_fraction_ablation
            )

        full_mask = document_causal_attention_mask(doc_ids, mask_format=mask_format).to(device)
        baseline_nll = _compute_position_nll(
            model, input_ids, labels, doc_ids, attention_mask=full_mask
        )

        for (layer_idx, q_head) in heads_to_ablate:
            _, metrics = ablate_single_head(
                model, input_ids, labels, doc_ids, baseline_nll,
                layer_idx, q_head, params.local_window,
                high_utility_mask=hu_mask,
                full_attention_mask=full_mask,
            )
            head_scores[(layer_idx, q_head)].append(metrics["mean_delta"])

    aggregated: dict[tuple[int, int], dict[str, float]] = {}
    for key, scores in head_scores.items():
        if scores:
            aggregated[key] = {
                "mean_delta": sum(scores) / len(scores),
                "num_batches": len(scores),
            }
    return aggregated


@torch.no_grad()
def run_split_half_ablation(
    model: torch.nn.Module,
    dataloader_half1: Any,
    dataloader_half2: Any,
    params: CUREParams,
    device: torch.device | None = None,
    candidate_heads: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Run ablation on two halves and select heads stable across both.

    If candidate_heads is given, only those heads are ablated (used for
    sharding the candidate set across ranks). Returns raw per-half scores in
    addition to the selection so shards can be merged before selecting.

    Returns:
        dict with retrieval_heads, head_scores, split_half details
    """
    scores_h1 = run_ablation_calibration(
        model, dataloader_half1, params, device=device, candidate_heads=candidate_heads
    )
    scores_h2 = run_ablation_calibration(
        model, dataloader_half2, params, device=device, candidate_heads=candidate_heads
    )
    raw_h1 = {f"{l}_{h}": v["mean_delta"] for (l, h), v in scores_h1.items()}
    raw_h2 = {f"{l}_{h}": v["mean_delta"] for (l, h), v in scores_h2.items()}
    results = select_heads_from_scores(raw_h1, raw_h2, params)
    results["calibration_sequences_per_half"] = params.ablation_calibration_sequences
    results["split_half_stable"] = params.ablation_split_half
    return results


def select_heads_from_scores(
    raw_h1: dict[str, float],
    raw_h2: dict[str, float],
    params: CUREParams,
) -> dict[str, Any]:
    """Select retrieval heads from per-half mean-delta score dicts.

    Pure function over score dicts ("L_H" -> mean_delta) so it can run after
    merging sharded results. A head qualifies only if its delta exceeds
    ablation_min_delta in BOTH halves (split-half stability); the survivors are
    ranked by averaged delta and the top ablation_top_k_fraction are selected.
    """
    all_keys = set(raw_h1) & set(raw_h2)
    combined: dict[str, float] = {}
    for k in all_keys:
        if raw_h1[k] > params.ablation_min_delta and raw_h2[k] > params.ablation_min_delta:
            combined[k] = (raw_h1[k] + raw_h2[k]) / 2.0

    sorted_heads = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    num_candidates = len(all_keys)
    k = max(1, int(math.ceil(num_candidates * params.ablation_top_k_fraction)))
    selected = sorted_heads[:k]

    retrieval_heads = [[int(x) for x in key.split("_")] for key, _ in selected]
    all_sorted = sorted(
        ((kk, (raw_h1[kk] + raw_h2[kk]) / 2.0) for kk in all_keys),
        key=lambda x: x[1], reverse=True,
    )
    return {
        "retrieval_heads": retrieval_heads,
        "head_scores": {kk: v for kk, v in all_sorted},
        "head_scores_half1": {kk: raw_h1[kk] for kk in all_keys},
        "head_scores_half2": {kk: raw_h2[kk] for kk in all_keys},
        "num_selected": len(retrieval_heads),
        "num_candidates": num_candidates,
    }


def save_ablation_results(results: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


def load_ablation_results(path: str | Path) -> tuple[set[tuple[int, int]], dict[str, Any]]:
    """Load ablation results and return (retrieval_heads_set, full_results)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    heads = {(h[0], h[1]) for h in data["retrieval_heads"]}
    return heads, data
