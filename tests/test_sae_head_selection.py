from __future__ import annotations

import torch

from curcpt.sae_head_selection import (
    HeadWriteStats,
    build_selection_payload,
    head_feature_projection_weight,
    parse_layer_spec,
    select_top_heads,
)


def test_head_projection_scores_o_side_write_into_evidence_direction():
    decoder_vectors = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    o_proj_weight = torch.zeros(4, 4)
    o_proj_weight[0, 0] = 2.0
    o_proj_weight[3, 2] = 5.0

    head0_proj = head_feature_projection_weight(o_proj_weight, decoder_vectors, head_id=0, head_dim=2)
    head1_proj = head_feature_projection_weight(o_proj_weight, decoder_vectors, head_id=1, head_dim=2)

    head0_output = torch.tensor([[[1.0, 0.0]]])
    head1_output = torch.tensor([[[1.0, 0.0]]])
    head0_score = torch.einsum("btd,kd->btk", head0_output, head0_proj).mean()
    head1_score = torch.einsum("btd,kd->btk", head1_output, head1_proj).mean()

    assert head0_score.item() == 2.0
    assert head1_score.item() == 0.0


def test_head_write_stats_rank_specific_high_utility_write():
    high = torch.tensor([[True, True, False, False]])
    nonhigh = torch.tensor([[False, False, True, True]])
    stats = HeadWriteStats()

    stats.update(torch.tensor([[2.0, 4.0, 1.0, 1.0]]), high, nonhigh)
    metrics = stats.as_metrics()

    assert metrics["high_mean"] == 3.0
    assert metrics["nonhigh_mean"] == 1.0
    assert metrics["score"] == 2.0


def test_select_top_heads_obeys_max_heads_per_layer():
    scores = {
        "20_0": {"score": 9.0, "high_mean": 9.0},
        "20_1": {"score": 8.0, "high_mean": 8.0},
        "21_0": {"score": 7.0, "high_mean": 7.0},
        "19_0": {"score": 6.0, "high_mean": 6.0},
    }

    selected = select_top_heads(scores, top_k=3, max_heads_per_layer=1)

    assert selected == [[20, 0], [21, 0], [19, 0]]


def test_layer_spec_and_payload_are_cure_compatible():
    assert parse_layer_spec("16-18,21", max_layers=24) == [16, 17, 18, 21]
    payload = build_selection_payload(
        retrieval_heads=[[21, 3]],
        head_scores={"21_3": {"score": 1.0}},
        metadata={"sae_layer": 21},
    )

    assert payload["retrieval_heads"] == [[21, 3]]
    assert payload["num_selected"] == 1
    assert payload["source"] == "sae_guided_evidence_write_selection"
