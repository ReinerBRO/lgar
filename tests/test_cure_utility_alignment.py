from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from curcpt.utility_mining import compute_query_utility


class _RemoteDepStub(nn.Module):
    """Stub LM that predicts the correct next token confidently ONLY when it can
    attend to key position 0 (the 'remote' token). Lets us verify utility is
    high exactly where a prediction depends on remote context — at the right
    query index (off-by-one contract)."""

    def __init__(self, vocab: int):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="eager")  # -> additive mask
        self._p = nn.Parameter(torch.zeros(1))
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab)
        BIG = 20.0
        for b in range(B):
            for i in range(T - 1):
                nxt = int(input_ids[b, i + 1].item())  # = labels[i] in packed data
                # additive mask [B,1,T,T]: 0.0 means query i may attend key 0
                can_see_key0 = (
                    attention_mask is None
                    or float(attention_mask[b, 0, i, 0].item()) == 0.0
                )
                if can_see_key0:
                    logits[b, i, nxt] = BIG
        return SimpleNamespace(logits=logits)


def test_query_utility_alignment_and_direction():
    vocab, T, w = 32, 8, 3
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
    labels = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 17]])  # labels[i]=input_ids[i+1]
    doc_ids = torch.zeros(1, T, dtype=torch.long)
    model = _RemoteDepStub(vocab)

    utility, full_logp = compute_query_utility(model, input_ids, labels, doc_ids, local_window=w)

    # shape aligned to labels (one utility value per query position)
    assert utility.shape == labels.shape == full_logp.shape

    # i in 0..2: local window still reaches key 0 -> full == local -> zero utility
    assert torch.allclose(utility[0, :3], torch.zeros(3), atol=1e-4)

    # i in 3..6: only the FULL mask reaches key 0 -> remote dependency -> positive
    # utility, located precisely at the query index (proves no off-by-one shift)
    assert (utility[0, 3:7] > 1.0).all(), f"utility misaligned: {utility[0]}"

    # full_logp is the log-prob of labels[i] at index i (confident ~0 everywhere it predicts)
    assert (full_logp[0, :7] > -0.5).all()


def test_query_utility_sign_convention():
    """Positive utility == prediction benefits from remote context (NLL_local > NLL_full)."""
    vocab, T, w = 16, 6, 2
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    labels = torch.tensor([[4, 5, 6, 7, 8, 8]])
    doc_ids = torch.zeros(1, T, dtype=torch.long)
    utility, _ = compute_query_utility(_RemoteDepStub(vocab), input_ids, labels, doc_ids, local_window=w)
    # remote-dependent positions (i >= w) must have utility > 0, never negative
    assert (utility[0, w:T - 1] > 0).all()
    assert (utility[0] >= -1e-4).all()
