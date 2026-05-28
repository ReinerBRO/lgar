from __future__ import annotations

import argparse
import json
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .config import LGARParams, Paths, TrainParams
from .data import prepare_qwen_fineweb_cache
from .train_stage0 import _summarize_run, _train_one
from .utils import ensure_dir, set_seed, write_json


def _guard_stage0(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = summary.get("decision_checks", {})
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise SystemExit(f"Stage1 blocked: failed Stage0 checks: {failed}")
    if not summary.get("hard_lgar_available", False):
        raise SystemExit("Stage1 blocked: hard L-GAR is not available in Stage0 summary.")


def _distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
    return rank, world_size, local_rank, device


def _is_rank0(rank: int) -> bool:
    return int(rank) == 0


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _summarize_existing_run(run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    run = {
        "run_dir": str(run_dir),
        "metrics": data.get("metrics", []),
        "extra": data.get("extra", {}),
        "eval": data.get("eval", {}),
    }
    return _summarize_run(run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L-GAR Stage1 four-run CPT screen.")
    defaults = Paths()
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--raw-data-dir", default=defaults.raw_data_dir)
    parser.add_argument("--cache-dir", default=defaults.cache_dir)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--stage0-summary", default=str(Path(defaults.output_dir) / "reports/stage0_summary.json"))
    parser.add_argument("--tokens-per-run", type=int, default=150_000_000)
    parser.add_argument("--target-cache-tokens", type=int, default=220_000_000)
    parser.add_argument("--max-shards", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--short-window", type=int, default=1024)
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--lsd-top-fraction", type=float, default=0.10)
    parser.add_argument("--long-nll-max-quantile", type=float, default=0.70)
    parser.add_argument("--router-target-budget", type=float, default=0.10)
    parser.add_argument("--final-global-budget", type=float, default=0.25)
    parser.add_argument("--routed-layer-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--lambda-router", type=float, default=0.02)
    parser.add_argument("--lambda-budget", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--short-limit", type=int, default=16)
    parser.add_argument("--long-limit", type=int, default=8)
    parser.add_argument("--long-lengths", default="4096,8192")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--allow-without-stage0", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--resume-if-available", action="store_true")
    parser.add_argument("--offline-signal-dir", default=None)
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Run selection as NAME=method. May be repeated. Defaults to all Stage1 runs.",
    )
    args = parser.parse_args()

    rank, world_size, local_rank, device = _distributed_context()
    if not args.allow_without_stage0:
        _guard_stage0(Path(args.stage0_summary))
    set_seed(args.seed)
    paths = Paths(
        model_path=args.model_path,
        raw_data_dir=args.raw_data_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    if _is_rank0(rank) and not (Path(paths.cache_dir) / "cache_info.json").exists():
        prepare_qwen_fineweb_cache(
            raw_data_dir=paths.raw_data_dir,
            model_path=paths.model_path,
            cache_dir=paths.cache_dir,
            target_tokens=args.target_cache_tokens,
            max_shards=args.max_shards,
        )
    _barrier()
    params = LGARParams(
        seq_len=args.seq_len,
        short_window=args.short_window,
        local_window=args.local_window,
        lsd_top_fraction=args.lsd_top_fraction,
        long_nll_max_quantile=args.long_nll_max_quantile,
        router_target_budget=args.router_target_budget,
        final_global_budget=args.final_global_budget,
        routed_layer_fraction=args.routed_layer_fraction,
        lambda_router_final=args.lambda_router,
        lambda_budget=args.lambda_budget,
    )
    steps = max(1, math.ceil(args.tokens_per_run / max(1, args.seq_len * args.batch_size * world_size)))
    train = TrainParams(
        batch_size=args.batch_size,
        steps=steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    if args.run:
        runs = []
        for item in args.run:
            if "=" not in item:
                raise SystemExit(f"--run must be NAME=method, got {item!r}")
            name, method = item.split("=", 1)
            runs.append((name, method))
    else:
        runs = [
            ("Qwen_CE_CPT", "ce"),
            ("Qwen_LongCE_CPT", "longce"),
            ("Qwen_RouterAux_CPT", "router_aux"),
            ("Qwen_LGAR_CPT", "lgar_routed"),
        ]
    long_lengths = [int(item.strip()) for item in str(args.long_lengths).split(",") if item.strip()]
    summary = {}
    if _is_rank0(rank):
        print(
            json.dumps(
                {
                    "event": "stage1_start",
                    "world_size": int(world_size),
                    "local_rank": int(local_rank),
                    "steps_per_run": int(steps),
                    "tokens_per_run": int(args.tokens_per_run),
                    "runs": [name for name, _ in runs],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for name, method in runs:
        run_dir = Path(paths.output_dir) / "runs" / "stage1" / name
        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "checkpoint.pt"
        if args.skip_completed and metrics_path.exists() and checkpoint_path.exists():
            if _is_rank0(rank):
                summary[name] = _summarize_existing_run(run_dir)
                print(
                    json.dumps(
                        {
                            "event": "run_skipped_completed",
                            "run": name,
                            "metrics": str(metrics_path),
                            "checkpoint": str(checkpoint_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            _barrier()
            continue
        result = _train_one(
            name,
            method,
            paths,
            params,
            train,
            device,
            stage_name="stage1",
            rank=rank,
            world_size=world_size,
            final_eval=True,
            final_short_limit=args.short_limit,
            final_long_limit=args.long_limit,
            final_long_lengths=long_lengths,
            save_interval=args.save_interval,
            resume_if_available=args.resume_if_available,
            offline_signal_dir=args.offline_signal_dir,
        )
        if _is_rank0(rank):
            summary[name] = _summarize_run(result)
            print(
                json.dumps(
                    {
                        "event": "run_complete",
                        "run": name,
                        "tokens_seen": summary[name]["tokens_seen"],
                        "ce_last_quarter": summary[name]["ce_last_quarter"],
                        "actual_budget_last_quarter": summary[name]["actual_budget_last_quarter"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _barrier()
    if _is_rank0(rank):
        out_dir = ensure_dir(Path(paths.output_dir) / "reports")
        out_name = "stage1_summary.json" if len(summary) > 1 else f"stage1_summary_{next(iter(summary))}.json"
        write_json(out_dir / out_name, summary)
        print(json.dumps({"stage1_summary": str(out_dir / out_name), "runs": list(summary), "world_size": int(world_size)}, indent=2))
    _barrier()
    _destroy_process_group()


if __name__ == "__main__":
    main()
