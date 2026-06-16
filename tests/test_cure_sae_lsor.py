from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from curcpt.adapters import HeadAdapterSet
from curcpt.forward import cure_forward
from curcpt.sae import (
    EvidenceFeatureSet,
    SparseAutoencoder,
    load_evidence_feature_set,
    lsor_loss,
    sae_decoder_feature_contribution,
    sae_feature_losses,
)
from curcpt.train import _validate_sae_layer_covers_retrieval_heads


class _DummyAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.o_proj.weight.zero_()

    def forward(self, x, attention_mask=None):
        self.q_proj(x)
        return self.o_proj(x)


class _DummyLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = _DummyAttention(hidden_size)
        self.post_attention_layernorm = nn.Identity()

    def forward(self, x, attention_mask=None):
        return self.post_attention_layernorm(x + self.self_attn(x, attention_mask=attention_mask))


class _DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_DummyLayer(hidden_size)])


class _DummyModel(nn.Module):
    def __init__(self, hidden_size: int = 4, n_heads: int = 2, vocab_size: int = 11) -> None:
        super().__init__()
        self.config = type(
            "Config",
            (),
            {"hidden_size": hidden_size, "num_attention_heads": n_heads},
        )()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = _DummyBackbone(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x, attention_mask=attention_mask)
        return {"logits": self.lm_head(x)}


def test_evidence_features_must_be_validated(tmp_path):
    path = tmp_path / "features.json"
    path.write_text(
        json.dumps(
            {
                "layer": 3,
                "hook_point": "resid_mid",
                "feature_ids": [1, 2],
                "validation": {"passed": False},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="causally validated"):
        load_evidence_feature_set(path, require_validated=True)

    features = load_evidence_feature_set(path, require_validated=False)
    assert features.layer == 3
    assert features.feature_ids == (1, 2)


def test_validated_features_require_answer_level_causal_checks(tmp_path):
    path = tmp_path / "features.json"
    path.write_text(
        json.dumps(
            {
                "layer": 3,
                "hook_point": "resid_mid",
                "feature_ids": [1, 2],
                "validation_passed": True,
                "validation": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="answer-level causal validation"):
        load_evidence_feature_set(path, require_validated=True)

    path.write_text(
        json.dumps(
            {
                "layer": 3,
                "hook_point": "resid_mid",
                "feature_ids": [1, 2],
                "validation_passed": True,
                "validation": {
                    "passed": True,
                    "selected_ablation_delta_logp": 0.2,
                    "masked_head_answer_delta_logp": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )
    features = load_evidence_feature_set(path, require_validated=True)
    assert features.validation_passed


def test_sae_feature_loss_uses_high_utility_target_tokens_only():
    torch.manual_seed(0)
    sae = SparseAutoencoder(input_dim=4, feature_dim=3)
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.zero_()
        sae.encoder.weight[1, 0] = 1.0

    residual = torch.zeros(1, 4, 4, requires_grad=True)
    residual.data[0, 0, 0] = 1.0
    residual.data[0, 1, 0] = 1.0
    target_full = torch.ones(1, 4, 1)
    target_neg = torch.zeros(1, 4, 1)
    target_mask = torch.tensor([[True, True, True, True]])
    valid = torch.tensor([[True, True, True, True]])
    high = torch.tensor([[True, False, False, False]])
    features = EvidenceFeatureSet(
        layer=2,
        hook_point="resid_mid",
        feature_ids=(1,),
        sae_checkpoint=None,
        validation_passed=True,
        scores={},
        config={},
    )

    losses, metrics = sae_feature_losses(
        {2: residual},
        sae,
        features,
        target_full,
        target_neg,
        target_mask,
        valid,
        high,
        beta_margin=0.5,
        beta_match=0.25,
        gamma=2.0,
    )
    total = sum(losses.values())
    total.backward()

    assert metrics["sae/target_token_count"] == 1.0
    assert residual.grad is not None
    assert residual.grad[0, 0].abs().sum() > 0
    assert residual.grad[0, 1:].abs().sum() == 0


def test_lsor_loss_backprops_to_adapter_write_not_residual_context():
    torch.manual_seed(1)
    residual = torch.randn(1, 6, 4, requires_grad=True)
    dz = torch.randn(1, 6, 4, requires_grad=True)
    doc_ids = torch.zeros(1, 6, dtype=torch.long)
    selected = torch.zeros(1, 6, dtype=torch.bool)
    selected[0, 4] = True

    loss, metrics = lsor_loss(
        {0: dz},
        {0: residual},
        doc_ids,
        selected,
        top_k=2,
        window=4,
        max_context_tokens=16,
        lambda_lsor=0.1,
    )
    loss.backward()

    assert metrics["lsor/active_layers"] == 1.0
    assert dz.grad is not None and dz.grad.abs().sum() > 0
    assert residual.grad is None


def test_sae_decoder_feature_contribution_uses_selected_decoder_vectors():
    sae = SparseAutoencoder(input_dim=3, feature_dim=4)
    with torch.no_grad():
        sae.decoder.weight.zero_()
        sae.decoder.bias.zero_()
        sae.input_scale.copy_(torch.tensor([2.0, 3.0, 5.0]))
        sae.decoder.weight[:, 1] = torch.tensor([1.0, 0.0, 2.0])
        sae.decoder.weight[:, 3] = torch.tensor([0.0, 4.0, 0.0])

    values = torch.tensor([[[2.0, 0.5]]])
    contribution = sae_decoder_feature_contribution(sae, values, (1, 3))

    expected = torch.tensor([[[4.0, 6.0, 20.0]]])
    assert torch.allclose(contribution, expected)


def test_sae_loss_backprops_to_o_side_residual_write_adapter():
    torch.manual_seed(4)
    model = _DummyModel(hidden_size=4, n_heads=2)
    with torch.no_grad():
        model.embed.weight.fill_(1.0)
    adapters = HeadAdapterSet({(0, 0)}, head_dim=2, rank=1, alpha=1.0, hidden_size=4)
    input_ids = torch.tensor([[1, 2, 3]])
    doc_ids = torch.zeros(1, 3, dtype=torch.long)

    sae = SparseAutoencoder(input_dim=4, feature_dim=2)
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.zero_()
        sae.encoder.weight[0, 0] = 1.0
    features = EvidenceFeatureSet(
        layer=0,
        hook_point="resid_mid",
        feature_ids=(0,),
        sae_checkpoint=None,
        validation_passed=True,
        scores={},
        config={},
    )

    out = cure_forward(
        model,
        input_ids,
        doc_ids,
        full_attention_mask=None,
        retrieval_heads={(0, 0)},
        adapters=adapters,
        local_window=2,
        compute_rh_bottleneck=False,
        capture_resid_mid_layers={0},
    )
    target_full = torch.ones(1, 3, 1) * 5.0
    target_neg = torch.zeros(1, 3, 1)
    target_mask = torch.ones(1, 3, dtype=torch.bool)
    valid = torch.ones(1, 3, dtype=torch.bool)
    high = torch.ones(1, 3, dtype=torch.bool)
    losses, _metrics = sae_feature_losses(
        out.resid_mid_by_layer or {},
        sae,
        features,
        target_full,
        target_neg,
        target_mask,
        valid,
        high,
        beta_margin=0.0,
        beta_match=1.0,
        gamma=0.0,
    )
    sum(losses.values()).backward()

    o_lora = adapters.get_o_lora(0, 0)
    assert o_lora is not None
    assert o_lora.lora_B.grad is not None
    assert float(o_lora.lora_B.grad.abs().sum().item()) > 0.0


def test_sae_layer_must_cover_selected_retrieval_heads():
    _validate_sae_layer_covers_retrieval_heads(3, {(1, 0), (3, 2)})
    with pytest.raises(ValueError, match="SAE layer must be at or after"):
        _validate_sae_layer_covers_retrieval_heads(2, {(1, 0), (3, 2)})
