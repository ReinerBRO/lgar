from __future__ import annotations

import pytest
import torch

from curcpt.losses import select_cure_main_loss_mask


def _masks():
    loss = torch.tensor([[True, True, True, False], [True, True, False, False]])
    valid = torch.tensor([[False, True, True, True], [True, False, False, True]])
    high = torch.tensor([[False, True, False, True], [True, True, False, False]])
    return loss, valid, high


def test_select_cure_main_loss_mask_all():
    loss, valid, high = _masks()
    got = select_cure_main_loss_mask(loss, valid, high, "all")
    assert torch.equal(got, loss)


def test_select_cure_main_loss_mask_valid_remote():
    loss, valid, high = _masks()
    got = select_cure_main_loss_mask(loss, valid, high, "valid_remote")
    assert torch.equal(got, valid & loss)


def test_select_cure_main_loss_mask_high_utility():
    loss, valid, high = _masks()
    got = select_cure_main_loss_mask(loss, valid, high, "high_utility")
    assert torch.equal(got, high & valid & loss)


def test_select_cure_main_loss_mask_none():
    loss, valid, high = _masks()
    got = select_cure_main_loss_mask(loss, valid, high, "none")
    assert got.dtype == torch.bool
    assert got.shape == loss.shape
    assert not got.any()


def test_select_cure_main_loss_mask_rejects_unknown_mode():
    loss, valid, high = _masks()
    with pytest.raises(ValueError, match="unsupported cure_main_loss_mask"):
        select_cure_main_loss_mask(loss, valid, high, "remoteish")
