from __future__ import annotations

import torch
import torch.nn.functional as F

from curcpt.losses import (
    cure_nonhu_logp_consistency_loss,
    cure_full_hu_loss,
    masked_cross_entropy,
    masked_kl_divergence,
    cure_rh_loss,
)


def test_masked_ce_matches_full_when_all_selected():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 16)
    labels = torch.randint(0, 16, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.bool)
    got = masked_cross_entropy(logits, labels, mask)
    want = F.cross_entropy(logits.float().reshape(-1, 16), labels.reshape(-1))
    assert torch.allclose(got, want, atol=1e-5)


def test_masked_ce_sparse_gather_equals_index_then_mean():
    """Gather-before-softmax must equal the old full-softmax-then-index path."""
    torch.manual_seed(1)
    logits = torch.randn(3, 7, 32)
    labels = torch.randint(0, 32, (3, 7))
    mask = torch.zeros(3, 7, dtype=torch.bool)
    mask[0, 1] = mask[1, 4] = mask[2, 6] = True
    got = masked_cross_entropy(logits, labels, mask)
    # reference: per-position CE then select
    per = F.cross_entropy(
        logits.float().reshape(-1, 32), labels.reshape(-1), reduction="none"
    ).view(3, 7)
    want = per[mask].mean()
    assert torch.allclose(got, want, atol=1e-5)


def test_masked_kl_teacher_detached_and_sparse():
    """KL gathers masked rows; teacher carries no grad, student does."""
    torch.manual_seed(2)
    teacher = torch.randn(2, 6, 24, requires_grad=True)
    student = torch.randn(2, 6, 24, requires_grad=True)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, 2] = mask[1, 5] = True
    kl = masked_kl_divergence(teacher, student, mask)
    kl.backward()
    # teacher detached inside the loss -> no grad reaches the input teacher tensor
    assert teacher.grad is None
    # student is the trained path -> grad flows
    assert student.grad is not None and student.grad.abs().sum() > 0


def test_masked_ce_empty_mask_is_grad_connected_zero():
    """Empty mask must yield a 0 that still has grad_fn (the step-1 crash fix)."""
    logits = torch.randn(2, 4, 12, requires_grad=True)
    labels = torch.randint(0, 12, (2, 4))
    mask = torch.zeros(2, 4, dtype=torch.bool)
    loss = masked_cross_entropy(logits, labels, mask)
    assert loss.item() == 0.0
    assert loss.requires_grad and loss.grad_fn is not None
    loss.backward()  # must not raise "does not require grad"
    assert logits.grad is not None


def test_masked_kl_empty_mask_is_grad_connected_zero():
    student = torch.randn(2, 4, 12, requires_grad=True)
    teacher = torch.randn(2, 4, 12)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    loss = masked_kl_divergence(teacher, student, mask)
    assert loss.item() == 0.0
    assert loss.requires_grad and loss.grad_fn is not None
    loss.backward()
    assert student.grad is not None


def test_cure_rh_loss_empty_high_utility_keeps_graph():
    """cure_rh_loss with no high-utility positions returns a differentiable total."""
    torch.manual_seed(3)
    logits_full = torch.randn(2, 5, 16)
    logits_rh = torch.randn(2, 5, 16, requires_grad=True)
    labels = torch.randint(0, 16, (2, 5))
    high_utility = torch.zeros(2, 5, dtype=torch.bool)  # nothing selected
    valid = torch.ones(2, 5, dtype=torch.bool)
    total, metrics = cure_rh_loss(
        logits_full, logits_rh, labels, high_utility, valid,
        lambda_rh_ce=0.1, lambda_rh_kd=0.05, lambda_cov=0.005,
        step_tokens=0, cov_warmup_tokens=10_000_000,
    )
    assert total.requires_grad and total.grad_fn is not None
    torch.autograd.grad(total, logits_rh, allow_unused=True)  # must not raise
    assert metrics["loss/rh_ce"] == 0.0 and metrics["loss/rh_kd"] == 0.0


def test_cure_full_hu_loss_selects_high_utility_valid_tokens():
    torch.manual_seed(4)
    logits = torch.randn(2, 5, 16, requires_grad=True)
    labels = torch.randint(0, 16, (2, 5))
    high = torch.zeros(2, 5, dtype=torch.bool)
    valid = torch.ones(2, 5, dtype=torch.bool)
    high[0, 1] = True
    high[1, 2] = True
    valid[1, 2] = False

    total, metrics = cure_full_hu_loss(
        logits, labels, high, valid, lambda_full_hu_ce=0.5
    )
    want = 0.5 * masked_cross_entropy(logits, labels, high & valid)
    assert torch.allclose(total, want, atol=1e-6)
    assert metrics["utility/full_hu_count"] == 1.0
    total.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_cure_nonhu_logp_consistency_selects_non_high_valid_tokens():
    torch.manual_seed(5)
    active = torch.randn(2, 5, 16, requires_grad=True)
    labels = torch.randint(0, 16, (2, 5))
    teacher = active.detach().clone()
    teacher[0, 2, labels[0, 2]] += 0.5
    high = torch.zeros(2, 5, dtype=torch.bool)
    valid = torch.zeros(2, 5, dtype=torch.bool)
    valid[0, 1] = True
    valid[0, 2] = True
    high[0, 1] = True

    total, metrics = cure_nonhu_logp_consistency_loss(
        active, teacher, labels, high, valid, lambda_nonhu_logp=0.25
    )
    assert metrics["utility/nonhu_count"] == 1.0
    assert total.requires_grad and total.grad_fn is not None
    total.backward()
    assert active.grad is not None and active.grad.abs().sum() > 0
