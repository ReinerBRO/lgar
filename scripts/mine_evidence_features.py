#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curcpt.sae import load_sparse_autoencoder
from scripts.precompute_sae_feature_targets import _device_from_arg


def _activation_path(activation_dir: str | None, activation_file: str | None, layer: int) -> Path:
    if activation_file:
        return Path(activation_file)
    if not activation_dir:
        raise ValueError("provide --activation-file or --activation-dir")
    return Path(activation_dir) / f"layer{int(layer)}_activations.npz"


@torch.no_grad()
def _encode_views(
    sae,
    payload,
    neg_views: list[str],
    device: torch.device,
    chunk_size: int,
):
    n = int(payload["full"].shape[0])
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        full = sae.encode(torch.as_tensor(payload["full"][start:end], device=device)).detach().float().cpu()
        neg_chunks = []
        for view in neg_views:
            if view not in payload:
                raise ValueError(f"activation file has no negative view {view!r}")
            neg_chunks.append(
                sae.encode(torch.as_tensor(payload[view][start:end], device=device)).detach().float().cpu()
            )
        neg = torch.stack(neg_chunks, dim=0).amax(dim=0)
        yield full, neg


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return 0.5
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    pos_ranks = ranks[: pos.size].sum()
    return float((pos_ranks - pos.size * (pos.size + 1) / 2.0) / max(1.0, pos.size * neg.size))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine full-vs-negative SAE evidence-tracking features.")
    parser.add_argument("--activation-dir", default=None)
    parser.add_argument("--activation-file", default=None)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sae-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--negative-views", default="short,masked_head")
    parser.add_argument("--top-k-features", type=int, default=32)
    parser.add_argument("--candidate-multiplier", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.01)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--min-diff", type=float, default=0.0)
    parser.add_argument("--max-token-concentration", type=float, default=0.50)
    parser.add_argument("--max-position-bin-concentration", type=float, default=0.50)
    parser.add_argument("--position-bins", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = _device_from_arg(args.device)
    path = _activation_path(args.activation_dir, args.activation_file, int(args.layer))
    payload = np.load(path)
    if "full" not in payload:
        raise ValueError(f"{path} has no full view")
    neg_views = [view.strip() for view in str(args.negative_views).split(",") if view.strip()]
    sae = load_sparse_autoencoder(args.sae_checkpoint, device=device)
    feature_dim = int(sae.encoder.out_features)

    n = int(payload["full"].shape[0])
    sum_diff = torch.zeros(feature_dim)
    sum_full = torch.zeros(feature_dim)
    sum_neg = torch.zeros(feature_dim)
    coverage_count = torch.zeros(feature_dim)
    for full, neg in _encode_views(sae, payload, neg_views, device, int(args.chunk_size)):
        diff = full - neg
        sum_diff += diff.sum(dim=0)
        sum_full += full.sum(dim=0)
        sum_neg += neg.sum(dim=0)
        coverage_count += (full > float(args.activation_threshold)).float().sum(dim=0)

    mean_diff = sum_diff / max(1, n)
    mean_full = sum_full / max(1, n)
    mean_neg = sum_neg / max(1, n)
    coverage = coverage_count / max(1, n)
    prelim = mean_diff * coverage
    valid = (coverage >= float(args.min_coverage)) & (mean_diff >= float(args.min_diff))
    if not valid.any():
        raise ValueError("no SAE features passed min coverage/diff filters")

    candidate_count = min(
        int(feature_dim),
        max(int(args.top_k_features), int(args.top_k_features) * int(args.candidate_multiplier)),
    )
    prelim_masked = prelim.clone()
    prelim_masked[~valid] = -float("inf")
    candidates = torch.topk(prelim_masked, k=candidate_count).indices.cpu().numpy().astype(np.int64)

    full_candidate_chunks: list[np.ndarray] = []
    neg_candidate_chunks: list[np.ndarray] = []
    for full, neg in _encode_views(sae, payload, neg_views, device, int(args.chunk_size)):
        full_candidate_chunks.append(full[:, candidates].numpy())
        neg_candidate_chunks.append(neg[:, candidates].numpy())
    full_candidates = np.concatenate(full_candidate_chunks, axis=0)
    neg_candidates = np.concatenate(neg_candidate_chunks, axis=0)

    records = []
    target_token_id = np.asarray(payload["target_token_id"]) if "target_token_id" in payload else None
    token_position = np.asarray(payload["token_position"]) if "token_position" in payload else None
    if token_position is not None and token_position.size:
        max_pos = max(1, int(token_position.max()) + 1)
        position_bins = np.minimum(
            np.asarray(token_position, dtype=np.int64) * int(args.position_bins) // max_pos,
            int(args.position_bins) - 1,
        )
    else:
        position_bins = None
    for local_idx, feature_id in enumerate(candidates.tolist()):
        active = full_candidates[:, local_idx] > float(args.activation_threshold)
        if target_token_id is not None and active.any():
            _, counts = np.unique(target_token_id[active], return_counts=True)
            token_concentration = float(counts.max() / max(1, int(active.sum())))
        else:
            token_concentration = 0.0
        if position_bins is not None and active.any():
            _, counts = np.unique(position_bins[active], return_counts=True)
            position_concentration = float(counts.max() / max(1, int(active.sum())))
        else:
            position_concentration = 0.0
        if token_concentration > float(args.max_token_concentration):
            continue
        if position_concentration > float(args.max_position_bin_concentration):
            continue
        auc = _auroc(full_candidates[:, local_idx], neg_candidates[:, local_idx])
        score = float(mean_diff[feature_id].item()) * auc * float(coverage[feature_id].item())
        records.append(
            {
                "feature_id": int(feature_id),
                "score": score,
                "auroc": auc,
                "coverage": float(coverage[feature_id].item()),
                "full_activation_mean": float(mean_full[feature_id].item()),
                "negative_activation_mean": float(mean_neg[feature_id].item()),
                "mean_diff": float(mean_diff[feature_id].item()),
                "token_concentration": token_concentration,
                "position_bin_concentration": position_concentration,
            }
        )
    records.sort(key=lambda item: item["score"], reverse=True)
    selected = records[: int(args.top_k_features)]
    if len(selected) < int(args.top_k_features):
        raise ValueError(
            f"only {len(selected)} features survived confound filters; "
            f"requested {int(args.top_k_features)}"
        )
    output_payload = {
        "layer": int(args.layer),
        "hook_point": "resid_mid",
        "feature_ids": [int(item["feature_id"]) for item in selected],
        "scores": {str(item["feature_id"]): item for item in selected},
        "mining_config": {
            "activation_file": str(path),
            "sae_checkpoint": str(args.sae_checkpoint),
            "negative_views": neg_views,
            "top_k_features": int(args.top_k_features),
            "min_coverage": float(args.min_coverage),
            "activation_threshold": float(args.activation_threshold),
            "min_diff": float(args.min_diff),
            "max_token_concentration": float(args.max_token_concentration),
            "max_position_bin_concentration": float(args.max_position_bin_concentration),
            "position_bins": int(args.position_bins),
            "num_tokens": int(n),
        },
        "sae_checkpoint": str(args.sae_checkpoint),
        "validation_passed": False,
        "validation": {
            "passed": False,
            "reason": "mined features require causal validation before SAE-CURE training",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "mine_evidence_features_done",
                "output": str(output),
                "layer": int(args.layer),
                "num_features": len(output_payload["feature_ids"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
