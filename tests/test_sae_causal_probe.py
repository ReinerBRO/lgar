from __future__ import annotations

import torch
import torch.nn as nn

from curcpt.sae import SparseAutoencoder
from curcpt.sae_causal_probe import (
    _choose_random_feature_ids,
    _register_resid_mid_feature_ablation_hook,
)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.Identity()

    def forward(self, x):
        residual = x
        hidden = self.post_attention_layernorm(x)
        return residual + hidden


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_random_control_features_exclude_selected_features():
    sae = SparseAutoencoder(input_dim=4, feature_dim=16)
    selected = (1, 5, 9)
    random_ids = _choose_random_feature_ids(sae, exclude=selected, count=len(selected), seed=7)

    assert len(random_ids) == len(selected)
    assert len(set(random_ids)) == len(random_ids)
    assert set(random_ids).isdisjoint(selected)


def test_resid_mid_feature_ablation_hook_subtracts_decoder_direction():
    model = _Model()
    sae = SparseAutoencoder(input_dim=3, feature_dim=4)
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.zero_()
        sae.encoder.weight[2, 0] = 1.0
        sae.decoder.weight.zero_()
        sae.decoder.bias.zero_()
        sae.input_scale.fill_(1.0)
        sae.decoder.weight[:, 2] = torch.tensor([0.5, 1.0, -2.0])

    x = torch.tensor([[[2.0, 3.0, 4.0]]])
    handles = _register_resid_mid_feature_ablation_hook(model, 0, sae, (2,), scale=1.0)
    try:
        out = model(x)
    finally:
        for handle in handles:
            handle.remove()

    expected = torch.tensor([[[2.0, 2.0, 16.0]]])
    assert torch.allclose(out, expected)
