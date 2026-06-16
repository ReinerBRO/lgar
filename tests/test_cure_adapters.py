from __future__ import annotations

import torch

from curcpt.adapters import HeadAdapterSet, HeadSliceLoRA


def test_head_slice_lora_noop_init_has_live_gradient_path():
    lora = HeadSliceLoRA(head_dim=64, rank=8, alpha=16.0)
    x = torch.randn(2, 10, 64, requires_grad=True)
    out = lora(x)
    assert torch.allclose(out, torch.zeros_like(out)), "LoRA should start as a no-op"
    assert out.shape == x.shape
    assert not torch.allclose(lora.lora_A, torch.zeros_like(lora.lora_A))
    assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))
    (out + x).sum().backward()
    assert lora.lora_B.grad is not None
    assert float(lora.lora_B.grad.abs().sum().item()) > 0.0


def test_head_adapter_set():
    heads = {(0, 1), (1, 3), (2, 0)}
    adapter = HeadAdapterSet(heads, head_dim=64, rank=8)
    assert adapter.num_heads() == 3
    assert adapter.has_adapter(0, 1)
    assert not adapter.has_adapter(0, 2)
    assert adapter.get_q_lora(0, 1) is not None
    assert adapter.get_q_lora(5, 0) is None
    assert adapter.num_parameters() > 0


def test_o_side_adapter_can_write_hidden_size_residual():
    adapter = HeadAdapterSet({(0, 1)}, head_dim=8, rank=2, hidden_size=32)
    q = adapter.get_q_lora(0, 1)
    o = adapter.get_o_lora(0, 1)
    x = torch.randn(2, 3, 8)

    assert q is not None and q(x).shape == (2, 3, 8)
    assert o is not None and o(x).shape == (2, 3, 32)


def test_adapter_params_only():
    heads = {(10, 2), (11, 5)}
    adapter = HeadAdapterSet(heads, head_dim=64, rank=4)
    params = adapter.adapter_parameters()
    assert any(not torch.allclose(p, torch.zeros_like(p)) for p in params)
    assert any(torch.allclose(p, torch.zeros_like(p)) for p in params)


def test_multiple_retrieval_heads_independent():
    adapter = HeadAdapterSet({(0, 0), (0, 1)}, head_dim=32, rank=4)
    q0 = adapter.get_q_lora(0, 0)
    q1 = adapter.get_q_lora(0, 1)
    assert q0 is not q1
    # Modify one, other should be unaffected
    q0.lora_A.data.fill_(1.0)
    assert not torch.allclose(q1.lora_A.data, q0.lora_A.data)
