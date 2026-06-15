from __future__ import annotations

import torch
import torch.nn as nn

from curcpt.forward import _build_rh_bottleneck_mask, _register_rh_bottleneck_mask_hooks

NEG = -1.0e4


def _single_doc(bsz, seq_len):
    return torch.zeros(bsz, seq_len, dtype=torch.long)


def test_mask_shape_and_format():
    doc = _single_doc(2, 8)
    mask = _build_rh_bottleneck_mask(doc, {3}, n_heads=4, local_window=4)
    assert mask.shape == (2, 4, 8, 8)
    # additive: entries are either 0 (allowed) or NEG (blocked)
    uniq = set(mask.unique().tolist())
    assert uniq.issubset({0.0, NEG})


def test_retrieval_head_is_full_causal():
    seq = 8
    doc = _single_doc(1, seq)
    mask = _build_rh_bottleneck_mask(doc, {1}, n_heads=4, local_window=3)
    rh = mask[0, 1]  # retrieval head -> full document-causal
    for q in range(seq):
        for k in range(seq):
            if k <= q:
                assert rh[q, k] == 0.0, f"rh should attend ({q},{k})"
            else:
                assert rh[q, k] == NEG, f"rh must not see future ({q},{k})"


def test_non_retrieval_head_is_local_only():
    seq, w = 8, 3
    doc = _single_doc(1, seq)
    mask = _build_rh_bottleneck_mask(doc, {1}, n_heads=4, local_window=w)
    nh = mask[0, 0]  # non-retrieval head -> local window only
    for q in range(seq):
        for k in range(seq):
            in_window = (k <= q) and (k >= q - w + 1)
            if in_window:
                assert nh[q, k] == 0.0, f"local head should attend ({q},{k})"
            else:
                assert nh[q, k] == NEG, f"local head blocked outside window ({q},{k})"


def test_empty_retrieval_set_makes_all_heads_local():
    seq, w = 6, 2
    doc = _single_doc(1, seq)
    mask = _build_rh_bottleneck_mask(doc, set(), n_heads=3, local_window=w)
    for head in range(3):
        for q in range(seq):
            for k in range(seq):
                in_window = (k <= q) and (k >= q - w + 1)
                expected = 0.0 if in_window else NEG
                assert mask[0, head, q, k] == expected


class _DummySelfAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.last_attention_mask = None

    def forward(self, x, attention_mask=None):
        self.last_attention_mask = attention_mask
        return x


class _DummyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _DummySelfAttn()


class _DummyBackbone(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_DummyLayer() for _ in range(layers))


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _DummyBackbone(3)
        self.config = type("Config", (), {"num_attention_heads": 2})()


def test_rh_bottleneck_hooks_cover_non_routed_layers_as_local():
    model = _DummyModel()
    doc = _single_doc(1, 5)
    handles = _register_rh_bottleneck_mask_hooks(
        model, doc, retrieval_heads={(1, 0)}, local_window=2
    )
    try:
        x = torch.zeros(1)
        for layer in model.model.layers:
            layer.self_attn(x, attention_mask=None)

        # Layer 0 has no retrieval heads, so every head must be local-only.
        layer0_mask = model.model.layers[0].self_attn.last_attention_mask
        assert layer0_mask.shape == (1, 2, 5, 5)
        assert layer0_mask[0, 0, 4, 2] == NEG
        assert layer0_mask[0, 1, 4, 2] == NEG

        # Layer 1 head 0 is retrieval/full, head 1 remains local.
        layer1_mask = model.model.layers[1].self_attn.last_attention_mask
        assert layer1_mask[0, 0, 4, 2] == 0.0
        assert layer1_mask[0, 1, 4, 2] == NEG
    finally:
        for handle in handles:
            handle.remove()


def test_rh_bottleneck_routed_scope_leaves_non_routed_layers_unhooked():
    model = _DummyModel()
    doc = _single_doc(1, 5)
    handles = _register_rh_bottleneck_mask_hooks(
        model, doc, retrieval_heads={(1, 0)}, local_window=2, scope="routed_layers"
    )
    try:
        x = torch.zeros(1)
        for layer in model.model.layers:
            layer.self_attn(x, attention_mask=None)

        assert model.model.layers[0].self_attn.last_attention_mask is None
        layer1_mask = model.model.layers[1].self_attn.last_attention_mask
        assert layer1_mask[0, 0, 4, 2] == 0.0
        assert layer1_mask[0, 1, 4, 2] == NEG
    finally:
        for handle in handles:
            handle.remove()


def test_document_boundaries_block_cross_doc():
    # two docs of length 4 packed in one sequence
    doc = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    mask = _build_rh_bottleneck_mask(doc, {0}, n_heads=2, local_window=8)
    rh = mask[0, 0]
    # query in doc 1 (pos 5) must not attend to doc 0 keys (pos 0..3)
    for k in range(4):
        assert rh[5, k] == NEG, f"cross-doc attention leaked at (5,{k})"
    # but can attend within its own doc, causally
    assert rh[5, 4] == 0.0 and rh[5, 5] == 0.0


def test_padding_query_attends_only_self():
    # pos 3 is padding (doc_id = -1)
    doc = torch.tensor([[0, 0, 0, -1]])
    mask = _build_rh_bottleneck_mask(doc, {0}, n_heads=2, local_window=8)
    for head in range(2):
        row = mask[0, head, 3]
        assert row[3] == 0.0, "padding query must attend to itself"
        assert (row[:3] == NEG).all(), "padding query must not attend elsewhere"
