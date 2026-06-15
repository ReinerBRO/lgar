from __future__ import annotations

import torch

from curcpt.head_ablation import (
    _head_local_allowed,
    _make_head_ablation_hook,
    _remote_margin_valid_mask,
)
from lgar_cpt.mining import document_causal_attention_mask

NEG = -1.0e4


def test_head_local_allowed_blocks_cross_document_attention():
    doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    allowed = _head_local_allowed(doc_ids, local_window=8)
    assert bool(allowed[0, 4, 3])
    assert not bool(allowed[0, 4, 2])
    assert not bool(allowed[0, 2, 3])


def test_head_ablation_hook_keeps_other_heads_full_and_target_doc_local():
    doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    full_mask = document_causal_attention_mask(doc_ids)
    local_allowed = _head_local_allowed(doc_ids, local_window=2)
    hook = _make_head_ablation_hook(target_q_head=1, local_allowed=local_allowed, n_heads=3)

    _args, kwargs = hook(None, (), {"attention_mask": full_mask})
    mask = kwargs["attention_mask"]

    assert mask.shape == (1, 3, 6, 6)
    assert torch.equal(mask[:, 0], full_mask[:, 0])
    assert mask[0, 1, 4, 3] == 0.0
    assert mask[0, 1, 4, 2] == NEG
    assert mask[0, 1, 2, 3] == NEG


def test_remote_margin_valid_mask_uses_intra_document_offsets():
    doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    doc_offsets_full = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    valid = _remote_margin_valid_mask(
        doc_ids,
        doc_offsets_full,
        local_window=1,
        min_remote_margin=1,
    )
    assert valid.tolist() == [[False, True, True, False, False, True, True]]
