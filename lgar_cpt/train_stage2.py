from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .config import LGARParams, Paths, TrainParams
from .data import prepare_qwen_fineweb_cache
from .train_stage0 import _summarize_run, _train_one
from .train_stage1 import _barrier, _destroy_process_group, _distributed_context, _is_rank0, _summarize_existing_run
from .utils import ensure_dir, set_seed, write_json


DEFAULT_STAGE2_RUNS = [
    ("Qwen0.5B_CE_CPT_2B", "ce"),
    ("Qwen0.5B_LGAR_CPT_2B", "lgar_routed"),
]


def _guard_stage1(review_path: Path) -> None:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("status") != "pass" or review.get("blocking_checks"):
        raise SystemExit(
            f"Stage2 blocked: Stage1 review did not pass: "
            f"status={review.get('status')!r} blocking={review.get('blocking_checks')!r}"
        )


def _cache_satisfies_target(cache_dir: Path, target_tokens: int) -> bool:
    info_path = cache_dir / "cache_info.json"
    if not info_path.exists():
        return False
    required = [
        cache_dir / "tokens.npy",
        cache_dir / "doc_offsets.npy",
        cache_dir / "doc_lengths.npy",
        cache_dir / "docs.jsonl",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return int(info.get("actual_tokens", 0)) >= int(target_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L-GAR Stage2 two-run 2B-token CPT scale-up.")
    defaults = Paths()
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--raw-data-dir", default=defaults.raw_data_dir)
    parser.add_argument("--cache-dir", default=defaults.cache_dir)
    parser.add_argument("--output-dir", default=str(Path(defaults.output_dir) / "stage2_2k"))
    parser.add_argument("--stage1-review", default=str(Path(defaults.output_dir) / "reports/stage1_review.json"))
    parser.add_argument("--tokens-per-run", type=int, default=2_000_000_000)
    parser.add_argument("--target-cache-tokens", type=int, default=2_200_000_000)
    parser.add_argument("--max-shards", type=int, default=80)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--short-window", type=int, default=1024)
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--lsd-top-fraction", type=float, default=0.05)
    parser.add_argument("--long-nll-max-quantile", type=float, default=0.60)
    parser.add_argument("--router-target-budget", type=float, default=0.10)
    parser.add_argument("--final-global-budget", type=float, default=0.25)
    parser.add_argument("--routed-layer-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--lambda-router", type=float, default=0.02)
    parser.add_argument("--lambda-budget", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=800)
    parser.add_argument("--short-limit", type=int, default=16)
    parser.add_argument("--long-limit", type=int, default=8)
    parser.add_argument("--long-lengths", default="1024,1536,2048")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--allow-without-stage1", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--save-interval", type=int, default=800)
    parser.add_argument("--resume-if-available", action="store_true")
    parser.add_argument("--offline-signal-dir", default=None)
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Run selection as NAME=method. May be repeated. Defaults to CE and best Stage1 L-GAR.",
    )
    args = parser.parse_args()

    rank, world_size, local_rank, device = _distributed_context()
    if not args.allow_without_stage1:
        _guard_stage1(Path(args.stage1_review))
    set_seed(args.seed)
    paths = Paths(
        model_path=args.model_path,
        raw_data_dir=args.raw_data_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    cache_dir = Path(paths.cache_dir)
    if _is_rank0(rank) and not _cache_satisfies_target(cache_dir, args.target_cache_tokens):
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
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if args.run:
        runs = []
        for item in args.run:
            if "=" not in item:
                raise SystemExit(f"--run must be NAME=method, got {item!r}")
            name, method = item.split("=", 1)
            runs.append((name, method))
    else:
        runs = DEFAULT_STAGE2_RUNS
    long_lengths = [int(item.strip()) for item in str(args.long_lengths).split(",") if item.strip()]

    summary = {}
    if _is_rank0(rank):
        print(
            json.dumps(
                {
                    "event": "stage2_start",
                    "world_size": int(world_size),
                    "local_rank": int(local_rank),
                    "steps_per_run": int(steps),
                    "tokens_per_run": int(args.tokens_per_run),
                    "runs": [name for name, _ in runs],
                    "seq_len": int(args.seq_len),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for name, method in runs:
        run_dir = Path(paths.output_dir) / "runs" / "stage2" / name
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
            stage_name="stage2",
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
        out_name = "stage2_summary.json" if len(summary) > 1 else f"stage2_summary_{next(iter(summary))}.json"
        write_json(out_dir / out_name, summary)
        print(json.dumps({"stage2_summary": str(out_dir / out_name), "runs": list(summary), "world_size": int(world_size)}, indent=2))
    _barrier()
    _destroy_process_group()


if __name__ == "__main__":
    main()
