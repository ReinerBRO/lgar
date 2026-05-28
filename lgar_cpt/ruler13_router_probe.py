from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from .hard_lgar import qwen_lgar_forward
from .mining import attention_mask_format_for_model, document_causal_attention_mask, select_topk_global_queries
from .ruler13_generate import (
    OFFICIAL_RULER13_TASKS,
    _env_rank,
    _load_model_bundle,
    _load_official_task_configs,
    _pack_routed_batch,
    _slice_prompt_ids,
    _task_prompt,
)
from .utils import ensure_dir, read_jsonl, write_json


@torch.no_grad()
def _probe_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    rows: list[dict[str, Any]],
    seq_len: int,
    device: torch.device,
    params: Any,
    router: torch.nn.Module,
    target_budget: float | None,
    task_name: str,
) -> list[dict[str, Any]]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise SystemExit("tokenizer must expose pad_token_id or eos_token_id")
    sequences = [_slice_prompt_ids(tokenizer, prompt, seq_len) for prompt in prompts]
    input_ids, doc_ids, lengths = _pack_routed_batch(sequences, int(pad_token_id), device)
    mask_format = attention_mask_format_for_model(model)
    forward = qwen_lgar_forward(
        model=model,
        input_ids=input_ids,
        full_attention_mask=document_causal_attention_mask(doc_ids, mask_format=mask_format),
        doc_ids=doc_ids,
        router=router,
        params=params,
        routing_valid_mask=doc_ids >= 0,
        mode="router_aux",
        target_budget=target_budget,
    )
    if forward.router_scores is None:
        raise RuntimeError("router probe requires router scores")
    scores = forward.router_scores.detach()
    budget = float(params.final_global_budget if target_budget is None else target_budget)
    selected = select_topk_global_queries(scores, doc_ids >= 0, budget)

    out: list[dict[str, Any]] = []
    for row_idx, (row, length) in enumerate(zip(rows, lengths)):
        valid_scores = scores[row_idx, : int(length)].float()
        if valid_scores.numel() == 0:
            continue
        last_idx = int(length) - 1
        last_score = valid_scores[last_idx]
        rank = int((valid_scores > last_score).sum().item()) + 1
        out.append(
            {
                "task": task_name,
                "index": int(row["index"]),
                "prompt_tokens": int(length),
                "last_selected": bool(selected[row_idx, last_idx].item()),
                "last_score": float(last_score.item()),
                "last_rank": rank,
                "last_rank_frac": float(rank / max(1, int(length))),
                "score_mean": float(valid_scores.mean().item()),
                "score_p90": float(torch.quantile(valid_scores, 0.90).item()),
                "score_p95": float(torch.quantile(valid_scores, 0.95).item()),
            }
        )
    return out


def _summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"count": 0}
    selected = [float(item["last_selected"]) for item in items]
    rank_frac = [float(item["last_rank_frac"]) for item in items]
    scores = [float(item["last_score"]) for item in items]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_task.setdefault(str(item["task"]), []).append(item)
    return {
        "count": len(items),
        "last_selected_rate": sum(selected) / len(selected),
        "last_rank_frac_mean": sum(rank_frac) / len(rank_frac),
        "last_score_mean": sum(scores) / len(scores),
        "tasks": {
            task: {
                "count": len(rows),
                "last_selected_rate": sum(float(row["last_selected"]) for row in rows) / len(rows),
                "last_rank_frac_mean": sum(float(row["last_rank_frac"]) for row in rows) / len(rows),
                "last_score_mean": sum(float(row["last_score"]) for row in rows) / len(rows),
            }
            for task, rows in sorted(by_task.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether RULER answer-query positions are selected by the LGAR router.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--ruler-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--tasks", default=",".join(OFFICIAL_RULER13_TASKS))
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-budget", type=float, default=None)
    parser.add_argument("--append-answer-prefix", action="store_true")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--short-window", type=int, default=None)
    parser.add_argument("--local-window", type=int, default=None)
    parser.add_argument("--router-target-budget", type=float, default=None)
    parser.add_argument("--final-global-budget", type=float, default=None)
    parser.add_argument("--routed-layer-fraction", type=float, default=None)
    parser.add_argument("--router-hidden-dim", type=int, default=None)
    parser.set_defaults(mode="router_aux")
    args = parser.parse_args()

    rank, world_size, local_rank, device = _env_rank()
    del local_rank
    ruler_dir = Path(args.ruler_dir)
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)
    task_configs = _load_official_task_configs(ruler_dir)
    requested_tasks = [task.strip() for task in str(args.tasks).split(",") if task.strip()]
    for task in requested_tasks:
        if task not in task_configs:
            raise SystemExit(f"unknown RULER task: {task}")

    model, tokenizer, params, router = _load_model_bundle(args, device)
    if router is None:
        raise SystemExit("router probe requires router weights")
    seq_len = int(params.seq_len if args.seq_len is None else args.seq_len)
    output_path = save_dir / f"router_probe_rank{rank}.jsonl"
    if output_path.exists():
        output_path.unlink()

    all_items: list[dict[str, Any]] = []
    start = time.time()
    with output_path.open("w", encoding="utf-8") as fout:
        for task_name in requested_tasks:
            task_rows = read_jsonl(data_dir / task_name / f"{args.subset}.jsonl", limit=args.max_samples)
            shard_rows = task_rows[rank::world_size]
            for offset in range(0, len(shard_rows), int(args.batch_size)):
                batch_rows = shard_rows[offset : offset + int(args.batch_size)]
                prompts = [_task_prompt(row, append_answer_prefix=bool(args.append_answer_prefix)) for row in batch_rows]
                items = _probe_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    rows=batch_rows,
                    seq_len=seq_len,
                    device=device,
                    params=params,
                    router=router,
                    target_budget=args.target_budget,
                    task_name=task_name,
                )
                for item in items:
                    fout.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                all_items.extend(items)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "task": task_name,
                        "items": len([item for item in all_items if item["task"] == task_name]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = _summarize(all_items)
    summary.update(
        {
            "rank": rank,
            "world_size": world_size,
            "seconds": round(time.time() - start, 3),
            "seq_len": seq_len,
            "budget": float(params.final_global_budget if args.target_budget is None else args.target_budget),
            "pid": os.getpid(),
        }
    )
    write_json(save_dir / f"router_probe_rank{rank}_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
