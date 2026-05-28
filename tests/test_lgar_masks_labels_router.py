from __future__ import annotations

import math
import types
import unittest

import torch
import torch.nn as nn

from lgar_cpt.config import LGARParams, routed_layer_indices
from lgar_cpt.hard_lgar import qwen_lgar_forward
from lgar_cpt.mining import (
    build_lsd_label_batch_from_scores,
    document_causal_attention_mask,
    lgar_routed_attention_mask,
    local_document_attention_mask,
)
from lgar_cpt.router import SharedQueryRouter, router_aux_loss
from lgar_cpt.train_stage0 import (
    _aligned_routing_params,
    _effective_routed_target_budget,
    _effective_routing_supervision_start,
)


class LGARMaskLabelRouterTests(unittest.TestCase):
    def test_document_causal_mask_blocks_future_and_cross_doc(self) -> None:
        doc_ids = torch.tensor([[0, 0, 1, 1]])
        allowed = document_causal_attention_mask(doc_ids)[0, 0] == 0
        self.assertTrue(bool(allowed[1, 0]))
        self.assertFalse(bool(allowed[0, 1]))
        self.assertFalse(bool(allowed[2, 1]))
        self.assertTrue(bool(allowed[3, 2]))

    def test_local_mask_blocks_old_context(self) -> None:
        doc_ids = torch.tensor([[0, 0, 0, 0, 0]])
        allowed = local_document_attention_mask(doc_ids, window=2)[0, 0] == 0
        self.assertFalse(bool(allowed[4, 1]))
        self.assertTrue(bool(allowed[4, 3]))
        self.assertTrue(bool(allowed[4, 4]))

    def test_routed_mask_keeps_low_queries_local_and_high_queries_global(self) -> None:
        doc_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
        global_queries = torch.tensor([[False, False, False, False, False, True]])
        allowed = lgar_routed_attention_mask(doc_ids, global_queries, local_window=2)[0, 0] == 0
        self.assertFalse(bool(allowed[4, 1]))
        self.assertTrue(bool(allowed[4, 3]))
        self.assertTrue(bool(allowed[5, 0]))
        self.assertFalse(bool(allowed[2, 5]))

    def test_lsd_labels_are_on_query_position_for_next_token(self) -> None:
        params = LGARParams(seq_len=10, short_window=4, min_remote_margin=1, lsd_top_fraction=0.20)
        long_logp = torch.full((1, 10), -1.0)
        short_logp = long_logp.clone()
        short_logp[0, 6] = -2.2
        labels = torch.arange(10).view(1, 10)
        loss_mask = torch.ones((1, 10), dtype=torch.bool)
        offsets_full = torch.arange(11).view(1, 11)
        source_rows = torch.zeros((1, 11), dtype=torch.long)
        batch = build_lsd_label_batch_from_scores(
            long_logp=long_logp,
            short_logp=short_logp,
            labels=labels,
            loss_mask=loss_mask,
            doc_offsets_full=offsets_full,
            source_doc_rows_full=source_rows,
            special_token_ids=set(),
            params=params,
        )
        self.assertTrue(bool(batch.labels[0, 6]))
        self.assertFalse(bool(batch.labels[0, 5]))

    def test_long_nll_filter_drops_noisy_high_lsd_token(self) -> None:
        params = LGARParams(seq_len=10, short_window=4, min_remote_margin=1, lsd_top_fraction=0.50, long_nll_max_quantile=0.70)
        long_logp = torch.full((1, 10), -1.0)
        short_logp = long_logp.clone()
        long_logp[0, 8] = -10.0
        short_logp[0, 8] = -20.0
        labels = torch.arange(10).view(1, 10)
        loss_mask = torch.ones((1, 10), dtype=torch.bool)
        offsets_full = torch.arange(11).view(1, 11)
        source_rows = torch.zeros((1, 11), dtype=torch.long)
        batch = build_lsd_label_batch_from_scores(
            long_logp=long_logp,
            short_logp=short_logp,
            labels=labels,
            loss_mask=loss_mask,
            doc_offsets_full=offsets_full,
            source_doc_rows_full=source_rows,
            special_token_ids=set(),
            params=params,
        )
        self.assertFalse(bool(batch.labels[0, 8]))

    def test_router_aux_metrics_are_finite_with_positive_and_negative_labels(self) -> None:
        params = LGARParams(router_target_budget=0.25, routed_layer_fraction=0.5)
        router = SharedQueryRouter(hidden_size=8, hidden_dim=4)
        hidden_states = tuple(torch.randn(2, 6, 8) for _ in range(5))
        labels = torch.zeros((2, 6), dtype=torch.bool)
        labels[0, 2] = True
        labels[1, 4] = True
        valid = torch.ones((2, 6), dtype=torch.bool)
        cfg = types.SimpleNamespace(num_hidden_layers=4)
        loss, metrics, scores = router_aux_loss(router, hidden_states, labels, valid, cfg, params)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(scores.shape, labels.shape)
        self.assertTrue(math.isfinite(metrics["router/loss_bce"]))

    def test_routed_layer_indices_select_top_fraction(self) -> None:
        self.assertEqual(routed_layer_indices(24, 1.0 / 3.0), list(range(16, 24)))

    def test_train_routed_budget_uses_stage1_final_budget_unless_overridden(self) -> None:
        params = LGARParams(final_global_budget=0.25)
        self.assertEqual(_effective_routed_target_budget("lgar_routed", params, None), 0.25)
        self.assertEqual(_effective_routed_target_budget("lgar_routed", params, 0.75), 0.75)
        self.assertIsNone(_effective_routed_target_budget("router_aux", params, None))

    def test_routed_supervision_starts_no_later_than_local_window(self) -> None:
        params = LGARParams(short_window=1024, min_remote_margin=256, local_window=1024)
        self.assertEqual(_effective_routing_supervision_start(params), 1024)
        widened = LGARParams(short_window=1024, min_remote_margin=256, local_window=2048)
        self.assertEqual(_effective_routing_supervision_start(widened), 1280)

    def test_routing_params_align_label_and_router_budget_to_final_budget(self) -> None:
        params = LGARParams(lsd_top_fraction=0.05, router_target_budget=0.10, final_global_budget=0.25)
        aligned = _aligned_routing_params("lgar_routed", params)
        self.assertEqual(aligned.lsd_top_fraction, 0.25)
        self.assertEqual(aligned.router_target_budget, 0.25)
        self.assertEqual(_aligned_routing_params("longce", params), params)

    def test_hard_lgar_applies_routed_masks_only_to_upper_layers(self) -> None:
        class FakeLayer(nn.Module):
            def __init__(self, idx: int) -> None:
                super().__init__()
                self.idx = idx
                self.last_mask: torch.Tensor | None = None

            def forward(self, hidden_states, attention_mask=None, **kwargs):
                del kwargs
                self.last_mask = attention_mask.detach().clone()
                return hidden_states + float(self.idx + 1)

        class FakeBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(16, 4)
                self.layers = nn.ModuleList([FakeLayer(i) for i in range(6)])
                self.norm = nn.Identity()

            def rotary_emb(self, hidden_states, position_ids):
                return hidden_states, hidden_states

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = types.SimpleNamespace(num_hidden_layers=6)
                self.model = FakeBackbone()
                self.lm_head = nn.Linear(4, 16)

        model = FakeModel()
        params = LGARParams(routed_layer_fraction=1 / 3, local_window=2, final_global_budget=0.25, router_hidden_dim=4)
        router = SharedQueryRouter(hidden_size=4, hidden_dim=4)
        input_ids = torch.arange(6).view(1, 6)
        doc_ids = torch.zeros((1, 6), dtype=torch.long)
        full_mask = document_causal_attention_mask(doc_ids)
        out = qwen_lgar_forward(
            model=model,
            input_ids=input_ids,
            full_attention_mask=full_mask,
            doc_ids=doc_ids,
            router=router,
            params=params,
            routing_valid_mask=torch.ones((1, 6), dtype=torch.bool),
            mode="routed",
            target_budget=0.25,
        )
        self.assertEqual(out.logits.shape[:2], (1, 6))
        self.assertTrue(torch.equal(model.model.layers[0].last_mask, full_mask))
        self.assertTrue(torch.equal(model.model.layers[3].last_mask, full_mask))
        routed_mask_layer4 = model.model.layers[4].last_mask
        routed_mask_layer5 = model.model.layers[5].last_mask
        self.assertIsNotNone(routed_mask_layer4)
        self.assertIsNotNone(routed_mask_layer5)
        self.assertFalse(torch.equal(routed_mask_layer4, full_mask))
        self.assertFalse(torch.equal(routed_mask_layer5, full_mask))

    def test_hard_lgar_can_force_last_query_global(self) -> None:
        class FakeLayer(nn.Module):
            def __init__(self, idx: int) -> None:
                super().__init__()
                self.idx = idx
                self.last_mask: torch.Tensor | None = None

            def forward(self, hidden_states, attention_mask=None, **kwargs):
                del kwargs
                self.last_mask = attention_mask.detach().clone()
                return hidden_states

        class FakeBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(16, 4)
                self.layers = nn.ModuleList([FakeLayer(i) for i in range(6)])
                self.norm = nn.Identity()

            def rotary_emb(self, hidden_states, position_ids):
                return hidden_states, hidden_states

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = types.SimpleNamespace(num_hidden_layers=6)
                self.model = FakeBackbone()
                self.lm_head = nn.Linear(4, 16)

        class FakeRouter(nn.Module):
            def forward(self, hidden_states):
                del hidden_states
                return torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])

        model = FakeModel()
        params = LGARParams(routed_layer_fraction=1 / 3, local_window=2, final_global_budget=0.25, router_hidden_dim=4)
        router = FakeRouter()
        input_ids = torch.arange(6).view(1, 6)
        doc_ids = torch.zeros((1, 6), dtype=torch.long)
        full_mask = document_causal_attention_mask(doc_ids)
        force_global = torch.zeros((1, 6), dtype=torch.bool)
        force_global[0, -1] = True
        qwen_lgar_forward(
            model=model,
            input_ids=input_ids,
            full_attention_mask=full_mask,
            doc_ids=doc_ids,
            router=router,
            params=params,
            routing_valid_mask=torch.ones((1, 6), dtype=torch.bool),
            force_global_query_mask=force_global,
            mode="routed",
            target_budget=0.25,
        )
        routed_mask_layer5 = model.model.layers[5].last_mask
        self.assertIsNotNone(routed_mask_layer5)
        self.assertTrue(bool(torch.equal(routed_mask_layer5[0, 0, -1], full_mask[0, 0, -1])))
        self.assertFalse(bool(torch.equal(routed_mask_layer5[0, 0, 4], full_mask[0, 0, 4])))


if __name__ == "__main__":
    unittest.main()
