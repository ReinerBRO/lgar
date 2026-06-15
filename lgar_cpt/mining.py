from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .config import LGARParams
from .modeling import unwrap_logits


@dataclass
class LSDLabelBatch:
    labels: torch.Tensor
    valid: torch.Tensor
    lsd: torch.Tensor
    long_logp: torch.Tensor
    short_logp: torch.Tensor
    stats: dict[str, float]


def gather_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def ce_from_logits(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    chunk_size = max(1, int(os.environ.get("LGAR_CE_CHUNK_SIZE", "512")))
    vocab_size = logits.size(-1)
    labels = labels.contiguous()
    mask = mask.bool().contiguous()

    use_mask = bool(mask.any().item())
    total = logits.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, labels.shape[1], chunk_size):
        end = min(start + chunk_size, labels.shape[1])
        losses = F.cross_entropy(
            logits[:, start:end].reshape(-1, vocab_size).float(),
            labels[:, start:end].reshape(-1),
            reduction="none",
        ).view(labels.shape[0], end - start)
        if use_mask:
            mask_chunk = mask[:, start:end]
            total = total + losses[mask_chunk].sum()
            count += int(mask_chunk.sum().item())
        else:
            total = total + losses.sum()
            count += losses.numel()
    return total / max(1, count)


def _additive_mask(allowed: torch.Tensor) -> torch.Tensor:
    # Use a finite sentinel. PyTorch SDPA can produce NaNs with very large
    # negative additive masks after dtype/kernel conversions.
    min_value = -1.0e4
    out = torch.zeros(allowed.shape, dtype=torch.float32, device=allowed.device)
    out.masked_fill_(~allowed, min_value)
    return out


def _format_mask(allowed: torch.Tensor, mask_format: str = "additive") -> torch.Tensor:
    expanded = allowed[:, None, :, :]
    if mask_format == "additive":
        return _additive_mask(expanded)
    if mask_format == "bool":
        return expanded
    raise ValueError(f"unknown attention mask format: {mask_format}")


def attention_mask_format_for_model(model: torch.nn.Module) -> str:
    config = getattr(model, "config", None)
    impl = getattr(config, "_attn_implementation", None) or getattr(config, "attn_implementation", None)
    return "bool" if impl == "sdpa" else "additive"


def document_causal_attention_mask(doc_ids: torch.Tensor, mask_format: str = "additive") -> torch.Tensor:
    bsz, seq_len = doc_ids.shape
    del bsz
    device = doc_ids.device
    q = torch.arange(seq_len, device=device)[:, None]
    k = torch.arange(seq_len, device=device)[None, :]
    causal = k <= q
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    real_query = doc_ids[:, :, None] >= 0
    self_query = k == q
    # SDPA returns NaNs for rows with every key masked. Padding queries are
    # ignored by the loss, but still need a finite attention row.
    allowed = (causal[None, :, :] & same_doc & real_query) | (self_query[None, :, :] & ~real_query)
    return _format_mask(allowed, mask_format)


def local_document_attention_mask(doc_ids: torch.Tensor, window: int, mask_format: str = "additive") -> torch.Tensor:
    bsz, seq_len = doc_ids.shape
    del bsz
    device = doc_ids.device
    q = torch.arange(seq_len, device=device)[:, None]
    k = torch.arange(seq_len, device=device)[None, :]
    local = (k <= q) & (k >= q - int(window) + 1)
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    real_query = doc_ids[:, :, None] >= 0
    self_query = k == q
    allowed = (local[None, :, :] & same_doc & real_query) | (self_query[None, :, :] & ~real_query)
    return _format_mask(allowed, mask_format)


def lgar_routed_attention_mask(
    doc_ids: torch.Tensor,
    global_queries: torch.Tensor,
    local_window: int,
    mask_format: str = "additive",
) -> torch.Tensor:
    """Build a document-aware local/global query mask.

    This mask is intentionally separate from the Qwen forward path. Passing it
    to all layers would not satisfy L-GAR's "upper routed layers only" mechanism.
    """

    bsz, seq_len = doc_ids.shape
    if global_queries.shape != (bsz, seq_len):
        raise ValueError("global_queries must have shape [batch, seq_len]")
    device = doc_ids.device
    q = torch.arange(seq_len, device=device)[:, None]
    k = torch.arange(seq_len, device=device)[None, :]
    causal = k <= q
    local = k >= q - int(local_window) + 1
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    real_query = doc_ids[:, :, None] >= 0
    self_query = k == q
    query_global = global_queries[:, :, None]
    allowed = (
        causal[None, :, :]
        & same_doc
        & real_query
        & (query_global | local[None, :, :])
    ) | (self_query[None, :, :] & ~real_query)
    return _format_mask(allowed, mask_format)


@torch.no_grad()
def compute_long_short_logp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    doc_ids: torch.Tensor,
    short_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_format = attention_mask_format_for_model(model)
    full_mask = document_causal_attention_mask(doc_ids, mask_format=mask_format)
    full_out = model(input_ids, attention_mask=full_mask, use_cache=False)
    full_logits = unwrap_logits(full_out)
    full_logp = gather_logprob(full_logits, labels)
    del full_out, full_logits
    short_mask = local_document_attention_mask(doc_ids, short_window, mask_format=mask_format)
    short_out = model(input_ids, attention_mask=short_mask, use_cache=False)
    short_logits = unwrap_logits(short_out)
    short_logp = gather_logprob(short_logits, labels)
    del short_out, short_logits
    return full_logp.detach(), short_logp.detach()


def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(float("nan"))
    return torch.quantile(values.float(), float(q))


def _stats_from_labels(
    lsd: torch.Tensor,
    long_logp: torch.Tensor,
    short_logp: torch.Tensor,
    positive: torch.Tensor,
    valid: torch.Tensor,
    source_doc_rows: torch.Tensor | None,
) -> dict[str, float]:
    valid_lsd = lsd[valid]
    pos = positive & valid
    stats = {
        "lsd_mean": float(valid_lsd.mean().item()) if valid_lsd.numel() else 0.0,
        "lsd_p50": float(_quantile(valid_lsd, 0.50).item()) if valid_lsd.numel() else 0.0,
        "lsd_p90": float(_quantile(valid_lsd, 0.90).item()) if valid_lsd.numel() else 0.0,
        "lsd_p95": float(_quantile(valid_lsd, 0.95).item()) if valid_lsd.numel() else 0.0,
        "lsd_p99": float(_quantile(valid_lsd, 0.99).item()) if valid_lsd.numel() else 0.0,
        "positive_label_fraction": float(pos.float().sum().item() / max(1.0, valid.float().sum().item())),
        "labels_per_sequence": float(pos.float().sum().item() / max(1, lsd.shape[0])),
        "labels_per_batch": float(pos.float().sum().item()),
        "valid_tokens": float(valid.float().sum().item()),
    }
    if pos.any():
        stats["long_nll_positive_mean"] = float((-long_logp[pos]).mean().item())
        stats["short_nll_positive_mean"] = float((-short_logp[pos]).mean().item())
        stats["mean_lsd_positive"] = float(lsd[pos].mean().item())
    else:
        stats["long_nll_positive_mean"] = 0.0
        stats["short_nll_positive_mean"] = 0.0
        stats["mean_lsd_positive"] = 0.0
    if source_doc_rows is not None and pos.any():
        docs = source_doc_rows[pos].detach().cpu()
        stats["documents_with_labels"] = float(torch.unique(docs).numel())
    else:
        stats["documents_with_labels"] = 0.0
    return stats


def build_lsd_label_batch_from_scores(
    long_logp: torch.Tensor,
    short_logp: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    doc_offsets_full: torch.Tensor,
    source_doc_rows_full: torch.Tensor | None,
    special_token_ids: set[int],
    params: LGARParams,
    valid_offset_threshold: int | None = None,
    positive_fraction: float | None = None,
) -> LSDLabelBatch:
    lsd = long_logp - short_logp
    bsz, seq_len = labels.shape
    valid = loss_mask.bool().clone()
    target_offsets = doc_offsets_full[:, 1 : seq_len + 1].to(labels.device)
    supervision_start = (
        int(params.short_window + params.min_remote_margin)
        if valid_offset_threshold is None
        else int(valid_offset_threshold)
    )
    valid &= target_offsets >= supervision_start
    if special_token_ids:
        special = torch.zeros_like(valid)
        for token_id in special_token_ids:
            special |= labels == int(token_id)
        valid &= ~special

    positive = torch.zeros_like(valid)
    target_fraction = float(params.lsd_top_fraction if positive_fraction is None else positive_fraction)
    for b in range(bsz):
        row_valid = valid[b]
        if not row_valid.any():
            continue
        nll = -long_logp[b]
        nll_threshold = _quantile(nll[row_valid], params.long_nll_max_quantile)
        row_valid = row_valid & (nll <= nll_threshold)
        valid[b] = row_valid
        if not row_valid.any():
            continue
        lsd_threshold = _quantile(lsd[b, row_valid], max(0.0, 1.0 - target_fraction))
        row_positive = row_valid & (lsd[b] >= lsd_threshold) & (lsd[b] > 0)
        positive[b] = row_positive
        valid[b] = row_valid

    source_target_rows = source_doc_rows_full[:, 1 : seq_len + 1].to(labels.device) if source_doc_rows_full is not None else None
    stats = _stats_from_labels(lsd, long_logp, short_logp, positive, valid, source_target_rows)
    return LSDLabelBatch(
        labels=positive.detach(),
        valid=valid.detach(),
        lsd=lsd.detach(),
        long_logp=long_logp.detach(),
        short_logp=short_logp.detach(),
        stats=stats,
    )


@torch.no_grad()
def mine_lsd_labels(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    tokenizer: Any,
    params: LGARParams,
    valid_offset_threshold: int | None = None,
    positive_fraction: float | None = None,
) -> LSDLabelBatch:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    input_doc_ids = batch["doc_ids_full"][:, :-1]
    special_ids = {
        int(x)
        for x in [tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id]
        if x is not None
    }
    long_logp, short_logp = compute_long_short_logp(model, input_ids, labels, input_doc_ids, params.short_window)
    return build_lsd_label_batch_from_scores(
        long_logp=long_logp,
        short_logp=short_logp,
        labels=labels,
        loss_mask=batch["loss_mask"],
        doc_offsets_full=batch["doc_offsets_full"],
        source_doc_rows_full=batch.get("source_doc_rows_full"),
        special_token_ids=special_ids,
        params=params,
        valid_offset_threshold=valid_offset_threshold,
        positive_fraction=positive_fraction,
    )


def select_topk_global_queries(scores: torch.Tensor, valid: torch.Tensor, budget_fraction: float) -> torch.Tensor:
    out = torch.zeros_like(valid, dtype=torch.bool)
    for b in range(scores.shape[0]):
        idx = torch.nonzero(valid[b], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        k = max(1, int(math.ceil(float(idx.numel()) * float(budget_fraction))))
        top_local = torch.topk(scores[b, idx].float(), k=min(k, idx.numel())).indices
        out[b, idx[top_local]] = True
    return out


def lsd_audit_examples(
    tokenizer: Any,
    batch: dict[str, torch.Tensor],
    label_batch: LSDLabelBatch,
    limit: int = 20,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    input_ids = batch["input_ids"].detach().cpu()
    labels = batch["labels"].detach().cpu()
    positives = torch.nonzero(label_batch.labels.detach().cpu(), as_tuple=False)
    valid_negatives = torch.nonzero((label_batch.valid & ~label_batch.labels).detach().cpu(), as_tuple=False)
    candidate_rows = torch.cat([positives[:limit], valid_negatives[: max(0, limit - positives.size(0))]], dim=0)
    for row in candidate_rows[:limit]:
        b = int(row[0].item())
        pred_pos = int(row[1].item())
        target_pos = pred_pos + 1
        ids = input_ids[b]
        full_start = max(0, target_pos - 4096)
        short_start = max(0, target_pos - 1024)
        remote_end = max(full_start, short_start)
        examples.append(
            {
                "batch": b,
                "pred_pos": pred_pos,
                "target_pos": target_pos,
                "label": bool(label_batch.labels[b, pred_pos].item()),
                "target_token": int(labels[b, pred_pos].item()),
                "target_text": tokenizer.decode([int(labels[b, pred_pos].item())], skip_special_tokens=False),
                "full_prefix_excerpt": tokenizer.decode(ids[full_start : pred_pos + 1].tolist(), skip_special_tokens=False)[-1600:],
                "short_window_excerpt": tokenizer.decode(ids[short_start : pred_pos + 1].tolist(), skip_special_tokens=False)[-1600:],
                "remote_prefix_excerpt": tokenizer.decode(ids[full_start:remote_end].tolist(), skip_special_tokens=False)[-1600:],
                "long_nll": float((-label_batch.long_logp[b, pred_pos]).item()),
                "short_nll": float((-label_batch.short_logp[b, pred_pos]).item()),
                "lsd": float(label_batch.lsd[b, pred_pos].item()),
            }
        )
    return examples
