from __future__ import annotations

import torch
import torch.nn as nn

from curcpt.losses import (
    apply_isolated_adapter_grads,
    apply_isolated_grads,
    masked_cross_entropy,
    masked_kl_divergence,
)


class _TinyModel(nn.Module):
    """Stand-in: a 'base' param + 'adapter' params feeding two logit paths."""

    def __init__(self, vocab=12, dim=8):
        super().__init__()
        self.base = nn.Linear(dim, vocab)        # base-model proxy
        self.adapter = nn.Linear(dim, vocab, bias=False)  # adapter proxy

    def forward(self, x):
        logits_full = self.base(x)               # full path: base only
        logits_rh = self.base(x) + self.adapter(x)  # RH path: base + adapter
        return logits_full, logits_rh


class _TinyFullAdapterModel(nn.Module):
    """Full path includes adapter params, matching CURE-v2 deployment path."""

    def __init__(self, vocab=12, dim=8):
        super().__init__()
        self.base = nn.Linear(dim, vocab)
        self.adapter = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        return self.base(x) + self.adapter(x)


def _losses(model, x, labels, hu_mask):
    logits_full, logits_rh = model(x)
    valid = torch.ones_like(labels, dtype=torch.bool)
    l_ce = masked_cross_entropy(logits_full, labels, valid)
    l_rh = 0.1 * masked_cross_entropy(logits_rh, labels, hu_mask) + 0.05 * masked_kl_divergence(
        logits_full.detach(), logits_rh, hu_mask
    )
    return l_ce, l_rh


def test_l_rh_does_not_touch_base_grads():
    """L_RH gradient must reach adapters only; base grad must come from L_CE alone."""
    torch.manual_seed(0)
    model = _TinyModel()
    x = torch.randn(2, 5, 8)
    labels = torch.randint(0, 12, (2, 5))
    hu = torch.zeros(2, 5, dtype=torch.bool)
    hu[0, 1] = hu[1, 3] = True
    adapter_params = list(model.adapter.parameters())

    # Reference: base grad from L_CE ONLY
    model.zero_grad()
    l_ce_ref, _ = _losses(model, x, labels, hu)
    l_ce_ref.backward()
    base_grad_ce_only = model.base.weight.grad.clone()

    # Isolated path: L_CE into all params + L_RH into adapters only
    model.zero_grad()
    l_ce, l_rh = _losses(model, x, labels, hu)
    apply_isolated_grads(l_ce, l_rh, adapter_params)

    # Base grad must be IDENTICAL to the CE-only reference (no L_RH leakage)
    assert torch.allclose(model.base.weight.grad, base_grad_ce_only, atol=1e-6), \
        "L_RH leaked gradient into base params"
    # Adapters must have received gradient (from L_RH, since L_CE path excludes them)
    assert model.adapter.weight.grad is not None
    assert model.adapter.weight.grad.abs().sum() > 0, "adapters got no L_RH gradient"


def test_isolation_holds_when_no_high_utility():
    """Empty high-utility mask: base still gets L_CE grad, adapters get ~zero, no crash."""
    torch.manual_seed(1)
    model = _TinyModel()
    x = torch.randn(2, 4, 8)
    labels = torch.randint(0, 12, (2, 4))
    hu = torch.zeros(2, 4, dtype=torch.bool)  # nothing high-utility
    adapter_params = list(model.adapter.parameters())

    model.zero_grad()
    l_ce, l_rh = _losses(model, x, labels, hu)
    apply_isolated_grads(l_ce, l_rh, adapter_params)  # must not raise

    assert model.base.weight.grad is not None and model.base.weight.grad.abs().sum() > 0
    # adapter grad is the grad-connected zero -> all zeros (or None coerced)
    g = model.adapter.weight.grad
    assert g is None or torch.allclose(g, torch.zeros_like(g), atol=1e-6)


def test_full_path_auxiliary_updates_only_adapter_extra_grads():
    """Full-path high-utility aux loss must not add extra gradient to base params."""
    torch.manual_seed(2)
    model = _TinyFullAdapterModel()
    x = torch.randn(2, 5, 8)
    labels = torch.randint(0, 12, (2, 5))
    valid = torch.ones(2, 5, dtype=torch.bool)
    hu = torch.zeros(2, 5, dtype=torch.bool)
    hu[0, 2] = hu[1, 4] = True
    adapter_params = list(model.adapter.parameters())

    model.zero_grad()
    logits_ref = model(x)
    l_ce_ref = masked_cross_entropy(logits_ref, labels, valid)
    l_ce_ref.backward()
    base_grad_ce_only = model.base.weight.grad.clone()
    adapter_grad_ce_only = model.adapter.weight.grad.clone()

    model.zero_grad()
    logits = model(x)
    l_ce = masked_cross_entropy(logits, labels, valid)
    l_full_hu = 0.5 * masked_cross_entropy(logits, labels, hu)
    metrics = apply_isolated_adapter_grads(l_ce, {"full_hu": l_full_hu}, adapter_params)

    assert torch.allclose(model.base.weight.grad, base_grad_ce_only, atol=1e-6), \
        "full-path auxiliary loss leaked extra gradient into base params"
    assert model.adapter.weight.grad is not None
    assert not torch.allclose(model.adapter.weight.grad, adapter_grad_ce_only, atol=1e-6), \
        "adapter did not receive full-path auxiliary gradient"
    assert metrics["grad/adapter_full_hu_norm"] > 0.0
