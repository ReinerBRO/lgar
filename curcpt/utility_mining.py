from __future__ import annotations

from typing import Any

import torch

from lgar_cpt.mining import (
    compute_long_short_logp,
    mine_lsd_labels,
)

from .config import CUREParams


def compute_query_utility(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    doc_ids: torch.Tensor,
    local_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute query utility: u_q = NLL_local - NLL_full.

    Returns (utility, full_logp) both shaped [B, T].
    Positive utility means the prediction benefits from remote context.
    """
    full_logp, local_logp = compute_long_short_logp(
        model, input_ids, labels, doc_ids, local_window
    )
    utility = -local_logp - (-full_logp)
    return utility, full_logp


def mine_query_utility(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    tokenizer: Any,
    params: CUREParams,
    positive_fraction: float | None = None,
) -> LSDLabelBatch:
    """Mine query utility labels from a batch.

    Reuses LGAR mining infrastructure. The LSD signal is identical to
    query utility: u_q = NLL_local(x_{q+1}) - NLL_full(x_{q+1}).

    Stores both query_pos (i-1) and target_pos (i) in the audit output
    to avoid off-by-one bugs.
    """
    frac = positive_fraction or params.utility_top_fraction_training
    return mine_lsd_labels(
        model=model,
        batch=batch,
        tokenizer=tokenizer,
        params=params,
        valid_offset_threshold=params.min_remote_margin + params.local_window,
        positive_fraction=frac,
    )
