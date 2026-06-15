from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .ruler13_generate import (
    OFFICIAL_RULER13_TASKS,
    _env_rank,
    _generate_full,
    _generate_routed,
    _load_model_bundle,
    _load_official_task_configs,
    _run_task,
)
from .utils import write_json


def _split_values(text: str) -> list[str]:
    return [x for x in str(text).replace(",", " ").split() if x]


def _parse_seq_lens(text: str) -> list[int]:
    return [int(x) for x in _split_values(text)]


def _parse_batch_map(text: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for item in _split_values(text):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[int(key)] = int(value)
    return out


def _folder_complete(ruler_dir: Path, folder: str, tasks: list[str], max_samples: int | None) -> bool:
    if max_samples is None:
        return all((ruler_dir / "generated" / folder / task / "validation.jsonl").exists() for task in tasks)
    for task in tasks:
        path = ruler_dir / "generated" / folder / task / "validation.jsonl"
        if not path.exists():
            return False
        with path.open("r", encoding="utf-8") as handle:
            if sum(1 for _ in handle) < int(max_samples):
                return False
    return True


def _resolve_folder(
    ruler_dir: Path,
    seq_len: int,
    max_samples: int | None,
    tasks: list[str],
    reuse_plain: bool,
) -> str:
    if max_samples is None:
        suffixed = f"official{seq_len}"
    else:
        suffixed = f"official{seq_len}_n{max_samples}"
    plain = f"official{seq_len}"
    if _folder_complete(ruler_dir, suffixed, tasks, max_samples):
        return suffixed
    if reuse_plain and _folder_complete(ruler_dir, plain, tasks, max_samples):
        return plain
    return suffixed


def _build_generator(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    lgar_params: Any,
    router: Any,
    retrieval_heads: set[tuple[int, int]],
    device: torch.device,
) -> Any:
    if args.mode == "full":
        def generator(prompts: list[str], seq_len: int, max_new_tokens: int) -> list[str]:
            return _generate_full(model, tokenizer, prompts, seq_len, max_new_tokens, device)

        return generator

    if args.mode in {"rh_bottleneck", "rh_layer_bottleneck"}:
        from curcpt.rh_bottleneck import greedy_generate

        def generator(prompts: list[str], seq_len: int, max_new_tokens: int) -> list[str]:
            return greedy_generate(
                model,
                tokenizer,
                prompts,
                seq_len,
                max_new_tokens,
                device,
                eval_mode=args.mode,
                retrieval_heads=retrieval_heads,
                local_window=int(lgar_params.local_window),
            )

        return generator

    if router is None:
        raise SystemExit("routed generation requires router weights")

    def generator(prompts: list[str], seq_len: int, max_new_tokens: int) -> list[str]:
        return _generate_routed(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            seq_len=seq_len,
            max_new_tokens=max_new_tokens,
            device=device,
            params=lgar_params,
            router=router,
            mode=args.mode,
            target_budget=args.target_budget,
            force_last_query_global=bool(args.force_last_query_global),
        )

    return generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RULER predictions for multiple lengths after one model load.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--ruler-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seq-lens", required=True)
    parser.add_argument("--tasks", default=",".join(OFFICIAL_RULER13_TASKS))
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-size-map", default="")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--reuse-plain-folders", action="store_true")
    parser.add_argument("--mode", choices=["full", "router_aux", "routed", "rh_bottleneck", "rh_layer_bottleneck"], default="full")
    parser.add_argument("--target-budget", type=float, default=None)
    parser.add_argument("--force-last-query-global", action="store_true")
    parser.add_argument("--append-answer-prefix", action="store_true")
    parser.add_argument("--retrieval-heads", default=None)
    parser.add_argument("--retrieval-heads-json", default=None)
    parser.add_argument("--reference-checkpoint-path", default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--short-window", type=int, default=None)
    parser.add_argument("--local-window", type=int, default=None)
    parser.add_argument("--router-target-budget", type=float, default=None)
    parser.add_argument("--final-global-budget", type=float, default=None)
    parser.add_argument("--routed-layer-fraction", type=float, default=None)
    parser.add_argument("--router-hidden-dim", type=int, default=None)
    args = parser.parse_args()

    rank, world_size, local_rank, device = _env_rank()
    ruler_dir = Path(args.ruler_dir)
    output_root = Path(args.output_root)
    seq_lens = _parse_seq_lens(args.seq_lens)
    batch_map = _parse_batch_map(args.batch_size_map)
    requested_tasks = [task.strip() for task in str(args.tasks).split(",") if task.strip()]
    task_configs = _load_official_task_configs(ruler_dir)
    for task in requested_tasks:
        if task not in task_configs:
            raise SystemExit(f"unknown RULER task: {task}")

    model, tokenizer, lgar_params, router, retrieval_heads = _load_model_bundle(args, device)
    generator = _build_generator(args, model, tokenizer, lgar_params, router, retrieval_heads, device)

    for seq_len in seq_lens:
        folder = _resolve_folder(
            ruler_dir=ruler_dir,
            seq_len=int(seq_len),
            max_samples=args.max_samples,
            tasks=requested_tasks,
            reuse_plain=bool(args.reuse_plain_folders),
        )
        data_dir = ruler_dir / "generated" / folder
        save_dir = output_root / args.label / folder / "pred"
        summary_csv = save_dir / "summary.csv"
        if summary_csv.exists() and summary_csv.stat().st_size > 0 and not args.force_eval:
            if rank == 0:
                print(json.dumps({"event": "skip_length", "label": args.label, "seq_len": seq_len, "folder": folder}), flush=True)
            continue

        batch_size = int(batch_map.get(int(seq_len), int(args.batch_size)))
        save_dir.mkdir(parents=True, exist_ok=True)
        for task_name in requested_tasks:
            task_rows = []
            data_path = data_dir / task_name / f"{args.subset}.jsonl"
            with data_path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    if args.max_samples is not None and idx >= int(args.max_samples):
                        break
                    task_rows.append(json.loads(line))
            shard_rows = task_rows[rank::world_size]
            output_path = save_dir / f"{task_name}-{rank}.jsonl"
            summary = _run_task(
                task_name=task_name,
                rows=shard_rows,
                output_path=output_path,
                batch_size=batch_size,
                generator_fn=generator,
                max_new_tokens=int(task_configs[task_name]["tokens_to_generate"]),
                seq_len=int(seq_len),
                append_answer_prefix=bool(args.append_answer_prefix),
                rank=rank,
            )
            summary.update({"label": args.label, "seq_len": int(seq_len), "folder": folder})
            print(json.dumps(summary, sort_keys=True), flush=True)

        meta = {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "device": str(device),
            "mode": args.mode,
            "checkpoint_path": args.checkpoint_path,
            "data_dir": str(data_dir),
            "save_dir": str(save_dir),
            "append_answer_prefix": bool(args.append_answer_prefix),
            "batch_size": batch_size,
            "label": args.label,
            "seq_len": int(seq_len),
            "folder": folder,
            "lgar_params": asdict(lgar_params),
        }
        write_json(save_dir / f"rank{rank:02d}.meta.json", meta)


if __name__ == "__main__":
    main()
