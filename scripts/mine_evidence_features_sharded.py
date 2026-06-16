#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curcpt.sae import load_sparse_autoencoder


def _parse_devices(value: str | None, count: int) -> list[str]:
    if value:
        devices = [item.strip() for item in value.split(",") if item.strip()]
    elif torch.cuda.is_available():
        devices = [str(idx) for idx in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    if not devices:
        devices = ["cpu"]
    return [devices[idx % len(devices)] for idx in range(count)]


def _device_from_id(device_id: str) -> torch.device:
    if device_id == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(device_id))
        return torch.device(f"cuda:{int(device_id)}")
    return torch.device("cpu")


def _resolve_activation_files(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.activation_files:
        for item in args.activation_files:
            paths.extend(Path(part) for part in str(item).split(",") if part.strip())
    if args.activation_glob:
        paths.extend(Path(path) for path in glob.glob(str(args.activation_glob)))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError("provide --activation-files or --activation-glob")
    return unique


def _negative_views(value: str) -> list[str]:
    views = [view.strip() for view in str(value).split(",") if view.strip()]
    if not views:
        raise ValueError("--negative-views must not be empty")
    return views


def _materialized_array_path(
    materialize_dir: str | None,
    worker_idx: int | None,
    key: str,
) -> Path | None:
    if not materialize_dir:
        return None
    if worker_idx is None:
        raise ValueError("worker_idx is required with --materialize-dir")
    safe_key = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in key)
    return Path(materialize_dir) / f"shard{int(worker_idx):02d}_{safe_key}.npy"


def _load_activation_arrays(
    path: Path,
    neg_views: list[str],
    materialize_dir: str | None = None,
    worker_idx: int | None = None,
) -> dict[str, np.ndarray]:
    keys = ["full", *neg_views, "target_token_id", "token_position"]
    if materialize_dir:
        Path(materialize_dir).mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        with np.load(path) as payload:
            if "full" not in payload:
                raise ValueError(f"{path} has no full view")
            for view in neg_views:
                if view not in payload:
                    raise ValueError(f"{path} has no negative view {view!r}")
            for key in keys:
                if key not in payload:
                    continue
                out = _materialized_array_path(materialize_dir, worker_idx, key)
                assert out is not None
                if not out.exists() or out.stat().st_size == 0:
                    tmp = out.with_name(out.name + ".tmp")
                    if tmp.exists():
                        tmp.unlink()
                    arr = np.asarray(payload[key])
                    with tmp.open("wb") as handle:
                        np.save(handle, arr)
                    tmp.replace(out)
                    del arr
                arrays[key] = np.load(out, mmap_mode="r")
        return arrays

    with np.load(path) as payload:
        if "full" not in payload:
            raise ValueError(f"{path} has no full view")
        arrays: dict[str, np.ndarray] = {"full": np.asarray(payload["full"])}
        for view in neg_views:
            if view not in payload:
                raise ValueError(f"{path} has no negative view {view!r}")
            arrays[view] = np.asarray(payload[view])
        if "target_token_id" in payload:
            arrays["target_token_id"] = np.asarray(payload["target_token_id"])
        if "token_position" in payload:
            arrays["token_position"] = np.asarray(payload["token_position"])
    return arrays


@torch.no_grad()
def _encode_pair_chunks(
    sae: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    neg_views: list[str],
    device: torch.device,
    chunk_size: int,
):
    full_arr = arrays["full"]
    n = int(full_arr.shape[0])
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        full = sae.encode(torch.as_tensor(full_arr[start:end], device=device)).detach().float().cpu()
        neg = None
        for view in neg_views:
            encoded = sae.encode(torch.as_tensor(arrays[view][start:end], device=device)).detach().float().cpu()
            neg = encoded if neg is None else torch.maximum(neg, encoded)
        if neg is None:
            raise ValueError("negative views must not be empty")
        yield start, end, full, neg


def _stats_worker(
    worker_idx: int,
    path: str,
    device_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    device = _device_from_id(device_id)
    neg_views = list(config["negative_views"])
    arrays = _load_activation_arrays(
        Path(path),
        neg_views,
        materialize_dir=config.get("materialize_dir"),
        worker_idx=worker_idx,
    )
    sae = load_sparse_autoencoder(config["sae_checkpoint"], device=device)
    feature_dim = int(sae.encoder.out_features)
    n = int(arrays["full"].shape[0])

    sum_diff = np.zeros(feature_dim, dtype=np.float64)
    sum_full = np.zeros(feature_dim, dtype=np.float64)
    sum_neg = np.zeros(feature_dim, dtype=np.float64)
    coverage_count = np.zeros(feature_dim, dtype=np.float64)
    threshold = float(config["activation_threshold"])

    for _start, _end, full, neg in _encode_pair_chunks(
        sae,
        arrays,
        neg_views,
        device,
        int(config["chunk_size"]),
    ):
        diff = full - neg
        sum_diff += diff.sum(dim=0).numpy().astype(np.float64, copy=False)
        sum_full += full.sum(dim=0).numpy().astype(np.float64, copy=False)
        sum_neg += neg.sum(dim=0).numpy().astype(np.float64, copy=False)
        coverage_count += (full > threshold).sum(dim=0).numpy().astype(np.float64, copy=False)

    return {
        "worker": int(worker_idx),
        "path": str(path),
        "device": str(device),
        "n": int(n),
        "sum_diff": sum_diff,
        "sum_full": sum_full,
        "sum_neg": sum_neg,
        "coverage_count": coverage_count,
    }


def _candidate_worker(
    worker_idx: int,
    path: str,
    device_id: str,
    candidates: np.ndarray,
    scratch_dir: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    device = _device_from_id(device_id)
    neg_views = list(config["negative_views"])
    arrays = _load_activation_arrays(
        Path(path),
        neg_views,
        materialize_dir=config.get("materialize_dir"),
        worker_idx=worker_idx,
    )
    sae = load_sparse_autoencoder(config["sae_checkpoint"], device=device)
    n = int(arrays["full"].shape[0])
    candidate_count = int(candidates.shape[0])
    dtype = np.float16 if config["candidate_dtype"] == "float16" else np.float32

    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    full_path = scratch / f"candidate_full_shard{worker_idx:02d}.npy"
    neg_path = scratch / f"candidate_neg_shard{worker_idx:02d}.npy"
    target_path = scratch / f"target_token_id_shard{worker_idx:02d}.npy"
    position_path = scratch / f"token_position_shard{worker_idx:02d}.npy"

    full_out = np.lib.format.open_memmap(
        full_path,
        mode="w+",
        dtype=dtype,
        shape=(candidate_count, n),
    )
    neg_out = np.lib.format.open_memmap(
        neg_path,
        mode="w+",
        dtype=dtype,
        shape=(candidate_count, n),
    )
    for start, end, full, neg in _encode_pair_chunks(
        sae,
        arrays,
        neg_views,
        device,
        int(config["chunk_size"]),
    ):
        full_out[:, start:end] = full[:, candidates].numpy().T.astype(dtype, copy=False)
        neg_out[:, start:end] = neg[:, candidates].numpy().T.astype(dtype, copy=False)
    full_out.flush()
    neg_out.flush()
    del full_out
    del neg_out

    meta: dict[str, Any] = {
        "worker": int(worker_idx),
        "path": str(path),
        "device": str(device),
        "n": int(n),
        "full_file": str(full_path),
        "neg_file": str(neg_path),
    }
    if "target_token_id" in arrays:
        np.save(target_path, arrays["target_token_id"])
        meta["target_token_id_file"] = str(target_path)
    if "token_position" in arrays:
        np.save(position_path, arrays["token_position"])
        meta["token_position_file"] = str(position_path)
    return meta


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


def _score_candidate_worker(
    local_idx: int,
    feature_id: int,
    candidate_parts: list[dict[str, Any]],
    stats: dict[str, list[float]],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    threshold = float(config["activation_threshold"])
    max_token_id = int(config["max_token_id"])
    max_position = int(config["max_position"])
    position_bins_count = int(config["position_bins"])

    pos_values: list[np.ndarray] = []
    neg_values: list[np.ndarray] = []
    token_counts = np.zeros(max_token_id + 1, dtype=np.int64) if max_token_id >= 0 else None
    position_counts = np.zeros(position_bins_count, dtype=np.int64) if max_position > 0 else None
    active_total = 0

    for part in candidate_parts:
        full_map = np.load(part["full_file"], mmap_mode="r")
        neg_map = np.load(part["neg_file"], mmap_mode="r")
        full_col = np.asarray(full_map[local_idx], dtype=np.float32)
        neg_col = np.asarray(neg_map[local_idx], dtype=np.float32)
        pos_values.append(full_col)
        neg_values.append(neg_col)

        active = full_col > threshold
        active_count = int(active.sum())
        active_total += active_count
        if active_count and token_counts is not None and "target_token_id_file" in part:
            target_token_id = np.load(part["target_token_id_file"], mmap_mode="r")
            token_counts += np.bincount(
                np.asarray(target_token_id[active], dtype=np.int64),
                minlength=max_token_id + 1,
            )
        if active_count and position_counts is not None and "token_position_file" in part:
            token_position = np.load(part["token_position_file"], mmap_mode="r")
            bins = np.minimum(
                np.asarray(token_position[active], dtype=np.int64) * position_bins_count // max_position,
                position_bins_count - 1,
            )
            position_counts += np.bincount(bins, minlength=position_bins_count)

    token_concentration = 0.0
    if active_total and token_counts is not None:
        token_concentration = float(token_counts.max() / max(1, active_total))
    position_concentration = 0.0
    if active_total and position_counts is not None:
        position_concentration = float(position_counts.max() / max(1, active_total))
    if token_concentration > float(config["max_token_concentration"]):
        return None
    if position_concentration > float(config["max_position_bin_concentration"]):
        return None

    pos = np.concatenate(pos_values, axis=0)
    neg = np.concatenate(neg_values, axis=0)
    auc = _auroc(pos, neg)
    mean_diff = float(stats["mean_diff"][feature_id])
    coverage = float(stats["coverage"][feature_id])
    score = mean_diff * auc * coverage
    return {
        "feature_id": int(feature_id),
        "score": score,
        "auroc": auc,
        "coverage": coverage,
        "full_activation_mean": float(stats["mean_full"][feature_id]),
        "negative_activation_mean": float(stats["mean_neg"][feature_id]),
        "mean_diff": mean_diff,
        "token_concentration": token_concentration,
        "position_bin_concentration": position_concentration,
    }


def _run_pool(callables: list[tuple[Any, tuple[Any, ...]]], max_workers: int) -> list[Any]:
    ctx = torch.multiprocessing.get_context("spawn")
    results: list[Any] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = [pool.submit(fn, *args) for fn, args in callables]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if isinstance(result, dict) and "worker" in result:
                print(
                    json.dumps(
                        {
                            "event": "mine_sharded_worker_done",
                            "worker": int(result["worker"]),
                            "n": int(result.get("n", 0)),
                            "path": result.get("path"),
                            "device": result.get("device"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine SAE evidence features from sharded activation files in parallel."
    )
    parser.add_argument("--activation-files", nargs="*", default=None)
    parser.add_argument("--activation-glob", default=None)
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
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--stats-workers", type=int, default=0)
    parser.add_argument("--candidate-workers", type=int, default=0)
    parser.add_argument("--score-workers", type=int, default=8)
    parser.add_argument("--candidate-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--materialize-dir", default=None)
    args = parser.parse_args()

    activation_files = _resolve_activation_files(args)
    neg_views = _negative_views(args.negative_views)
    output = Path(args.output)
    scratch = Path(args.scratch_dir) if args.scratch_dir else output.parent / f"{output.stem}_sharded_cache"
    scratch.mkdir(parents=True, exist_ok=True)

    devices = _parse_devices(args.devices, len(activation_files))
    common_config = {
        "sae_checkpoint": str(args.sae_checkpoint),
        "negative_views": neg_views,
        "activation_threshold": float(args.activation_threshold),
        "chunk_size": int(args.chunk_size),
        "candidate_dtype": str(args.candidate_dtype),
        "materialize_dir": str(args.materialize_dir) if args.materialize_dir else None,
    }

    print(
        json.dumps(
            {
                "event": "mine_sharded_stats_start",
                "files": [str(path) for path in activation_files],
                "devices": devices,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    stats_jobs = [
        (_stats_worker, (idx, str(path), devices[idx], common_config))
        for idx, path in enumerate(activation_files)
    ]
    stats_workers = int(args.stats_workers) or len(stats_jobs)
    stats_parts = _run_pool(stats_jobs, max_workers=max(1, stats_workers))
    stats_parts.sort(key=lambda item: int(item["worker"]))

    n_total = int(sum(int(part["n"]) for part in stats_parts))
    sum_diff = np.sum([part["sum_diff"] for part in stats_parts], axis=0)
    sum_full = np.sum([part["sum_full"] for part in stats_parts], axis=0)
    sum_neg = np.sum([part["sum_neg"] for part in stats_parts], axis=0)
    coverage_count = np.sum([part["coverage_count"] for part in stats_parts], axis=0)
    mean_diff = sum_diff / max(1, n_total)
    mean_full = sum_full / max(1, n_total)
    mean_neg = sum_neg / max(1, n_total)
    coverage = coverage_count / max(1, n_total)
    prelim = mean_diff * coverage
    valid = (coverage >= float(args.min_coverage)) & (mean_diff >= float(args.min_diff))
    if not bool(valid.any()):
        raise ValueError("no SAE features passed min coverage/diff filters")

    feature_dim = int(mean_diff.shape[0])
    candidate_count = min(
        feature_dim,
        max(int(args.top_k_features), int(args.top_k_features) * int(args.candidate_multiplier)),
    )
    prelim_masked = np.where(valid, prelim, -np.inf)
    candidate_indices = np.argpartition(-prelim_masked, candidate_count - 1)[:candidate_count]
    candidate_indices = candidate_indices[np.argsort(-prelim_masked[candidate_indices])].astype(np.int64)

    print(
        json.dumps(
            {
                "event": "mine_sharded_candidates_start",
                "num_tokens": int(n_total),
                "candidate_count": int(candidate_count),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    candidate_jobs = [
        (
            _candidate_worker,
            (idx, str(path), devices[idx], candidate_indices, str(scratch), common_config),
        )
        for idx, path in enumerate(activation_files)
    ]
    candidate_workers = int(args.candidate_workers) or len(candidate_jobs)
    candidate_parts = _run_pool(candidate_jobs, max_workers=max(1, candidate_workers))
    candidate_parts.sort(key=lambda item: int(item["worker"]))

    max_token_id = -1
    max_position = 0
    for part in candidate_parts:
        if "target_token_id_file" in part:
            target = np.load(part["target_token_id_file"], mmap_mode="r")
            if target.size:
                max_token_id = max(max_token_id, int(target.max()))
        if "token_position_file" in part:
            token_position = np.load(part["token_position_file"], mmap_mode="r")
            if token_position.size:
                max_position = max(max_position, int(token_position.max()) + 1)

    score_config = {
        "activation_threshold": float(args.activation_threshold),
        "max_token_concentration": float(args.max_token_concentration),
        "max_position_bin_concentration": float(args.max_position_bin_concentration),
        "position_bins": int(args.position_bins),
        "max_token_id": int(max_token_id),
        "max_position": int(max_position),
    }
    stats_payload = {
        "mean_diff": mean_diff.tolist(),
        "mean_full": mean_full.tolist(),
        "mean_neg": mean_neg.tolist(),
        "coverage": coverage.tolist(),
    }

    print(
        json.dumps(
            {
                "event": "mine_sharded_score_start",
                "score_workers": int(args.score_workers),
                "candidate_count": int(candidate_count),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    score_jobs = [
        (
            _score_candidate_worker,
            (local_idx, int(feature_id), candidate_parts, stats_payload, score_config),
        )
        for local_idx, feature_id in enumerate(candidate_indices.tolist())
    ]
    records = [
        record
        for record in _run_pool(score_jobs, max_workers=max(1, int(args.score_workers)))
        if record is not None
    ]
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
            "activation_files": [str(path) for path in activation_files],
            "sae_checkpoint": str(args.sae_checkpoint),
            "negative_views": neg_views,
            "top_k_features": int(args.top_k_features),
            "candidate_multiplier": int(args.candidate_multiplier),
            "min_coverage": float(args.min_coverage),
            "activation_threshold": float(args.activation_threshold),
            "min_diff": float(args.min_diff),
            "max_token_concentration": float(args.max_token_concentration),
            "max_position_bin_concentration": float(args.max_position_bin_concentration),
            "position_bins": int(args.position_bins),
            "num_tokens": int(n_total),
            "num_shards": int(len(activation_files)),
            "devices": devices,
            "scratch_dir": str(scratch),
        },
        "sae_checkpoint": str(args.sae_checkpoint),
        "validation_passed": False,
        "validation": {
            "passed": False,
            "reason": "mined features require causal validation before SAE-CURE training",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "mine_evidence_features_sharded_done",
                "output": str(output),
                "layer": int(args.layer),
                "num_features": len(output_payload["feature_ids"]),
                "num_tokens": int(n_total),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
