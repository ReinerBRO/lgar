from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lgar_cpt.mining import attention_mask_format_for_model, document_causal_attention_mask
from lgar_cpt.modeling import unwrap_logits
from lgar_cpt.ruler13_generate import OFFICIAL_RULER13_TASKS
from lgar_cpt.utils import ensure_dir, read_jsonl, write_json

from .forward import _register_adapter_hooks
from .model_eval_utils import load_model_tokenizer_for_eval


@dataclass(frozen=True)
class ScoreItem:
    task: str
    index: int
    item_index: int
    output_text: str
    ids: list[int]
    target_mask: list[bool]
    prompt_tokens: int
    target_tokens: int
    answer_position: int | None


def _parse_checkpoint_args(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--checkpoint must be NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    if not out:
        raise SystemExit("at least one --checkpoint NAME=PATH is required")
    return out


def _maybe_space(prefix: str, target: str) -> str:
    if not target:
        return ""
    if prefix.endswith((" ", "\n", "\t")) or target.startswith((" ", "\n", "\t")):
        return ""
    return " "


def build_score_items(
    tokenizer: Any,
    task: str,
    row: dict[str, Any],
    *,
    seq_len: int,
    item_limit: int | None = None,
) -> list[ScoreItem]:
    prompt = str(row.get("input", "")) + str(row.get("answer_prefix", ""))
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    outputs = row.get("outputs", [row.get("output", "")])
    if item_limit is not None:
        outputs = list(outputs)[: int(item_limit)]

    items: list[ScoreItem] = []
    for item_idx, output in enumerate(outputs):
        output_text = str(output)
        target_ids = tokenizer.encode(
            _maybe_space(prompt, output_text) + output_text,
            add_special_tokens=False,
        )
        if not target_ids:
            continue
        full_ids = prompt_ids + target_ids
        target_start = len(prompt_ids)
        if len(full_ids) > int(seq_len):
            crop = len(full_ids) - int(seq_len)
            full_ids = full_ids[crop:]
            target_start -= crop

        # Labels are ids[1:]. A label position i scores token ids[i + 1].
        label_len = max(0, len(full_ids) - 1)
        target_mask = [
            (pos + 1) >= max(0, target_start)
            for pos in range(label_len)
        ]
        if not any(target_mask):
            continue
        items.append(
            ScoreItem(
                task=task,
                index=int(row.get("index", len(items))),
                item_index=item_idx,
                output_text=output_text,
                ids=[int(x) for x in full_ids],
                target_mask=target_mask,
                prompt_tokens=len(prompt_ids),
                target_tokens=sum(1 for x in target_mask if x),
                answer_position=(
                    int(row["token_position_answer"])
                    if row.get("token_position_answer") is not None
                    else None
                ),
            )
        )
    return items


def _pack_items(
    items: list[ScoreItem],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(item.ids) - 1 for item in items)
    input_ids = torch.full((len(items), max_len), int(pad_token_id), dtype=torch.long, device=device)
    labels = torch.full((len(items), max_len), int(pad_token_id), dtype=torch.long, device=device)
    doc_ids = torch.full((len(items), max_len), -1, dtype=torch.long, device=device)
    target_mask = torch.zeros((len(items), max_len), dtype=torch.bool, device=device)

    for row_idx, item in enumerate(items):
        input_row = torch.tensor(item.ids[:-1], dtype=torch.long, device=device)
        label_row = torch.tensor(item.ids[1:], dtype=torch.long, device=device)
        length = int(input_row.numel())
        input_ids[row_idx, :length] = input_row
        labels[row_idx, :length] = label_row
        doc_ids[row_idx, :length] = 0
        target_mask[row_idx, : len(item.target_mask)] = torch.tensor(
            item.target_mask, dtype=torch.bool, device=device
        )
    return input_ids, labels, doc_ids, target_mask


@torch.no_grad()
def _score_items(
    model: torch.nn.Module,
    items: list[ScoreItem],
    *,
    pad_token_id: int,
    device: torch.device,
) -> list[dict[str, float]]:
    input_ids, labels, doc_ids, target_mask = _pack_items(items, pad_token_id, device)
    mask_format = attention_mask_format_for_model(model)
    attention_mask = document_causal_attention_mask(doc_ids, mask_format=mask_format).to(device)
    logits = unwrap_logits(model(input_ids, attention_mask=attention_mask, use_cache=False))

    out: list[dict[str, float]] = []
    for row_idx in range(len(items)):
        mask = target_mask[row_idx]
        row_logits = logits[row_idx, mask].float()
        row_labels = labels[row_idx, mask]
        if row_logits.numel() == 0:
            out.append({
                "target_nll": float("nan"),
                "first_token_nll": float("nan"),
                "target_top1_acc": float("nan"),
                "target_top5_acc": float("nan"),
                "gold_margin": float("nan"),
            })
            continue

        logp = F.log_softmax(row_logits, dim=-1)
        gold_logp = logp.gather(-1, row_labels[:, None]).squeeze(-1)
        topk = row_logits.topk(k=min(5, row_logits.size(-1)), dim=-1)
        top1 = topk.indices[:, 0] == row_labels
        top5 = (topk.indices == row_labels[:, None]).any(dim=-1)
        gold_logits = row_logits.gather(-1, row_labels[:, None]).squeeze(-1)
        if topk.values.size(1) > 1:
            best_other = torch.where(top1, topk.values[:, 1], topk.values[:, 0])
            margin = gold_logits - best_other
        else:
            margin = gold_logits.new_full(gold_logits.shape, float("nan"))

        out.append({
            "target_nll": float((-gold_logp).mean().item()),
            "first_token_nll": float((-gold_logp[0]).item()),
            "target_top1_acc": float(top1.float().mean().item()),
            "target_top5_acc": float(top5.float().mean().item()),
            "gold_margin": float(margin.mean().item()),
        })
    return out


def _weighted_mean(records: list[dict[str, Any]], key: str, weight_key: str = "target_tokens") -> float:
    total = 0.0
    weight = 0.0
    for record in records:
        value = float(record.get(key, float("nan")))
        w = float(record.get(weight_key, 1.0))
        if math.isnan(value) or w <= 0:
            continue
        total += value * w
        weight += w
    return total / weight if weight else float("nan")


def _plain_mean(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if key in record and not math.isnan(float(record[key]))]
    return sum(values) / len(values) if values else float("nan")


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(str(record["task"]), []).append(record)

    def one(rows: list[dict[str, Any]]) -> dict[str, Any]:
        unique_rows = {(str(row["task"]), int(row["index"])) for row in rows}
        return {
            "items": len(rows),
            "rows": len(unique_rows),
            "target_tokens": int(sum(int(row["target_tokens"]) for row in rows)),
            "target_nll": _weighted_mean(rows, "target_nll"),
            "first_token_nll": _plain_mean(rows, "first_token_nll"),
            "target_top1_acc": _weighted_mean(rows, "target_top1_acc"),
            "target_top5_acc": _weighted_mean(rows, "target_top5_acc"),
            "gold_margin": _weighted_mean(rows, "gold_margin"),
            "adapter_active_minus_off_nll": _weighted_mean(rows, "adapter_active_minus_off_nll"),
            "adapter_logit_l2_proxy": _weighted_mean(rows, "adapter_logit_l2_proxy"),
            "adapter_top1_flip_rate": _weighted_mean(rows, "adapter_top1_flip_rate"),
            "answer_position_mean": _plain_mean(
                [row for row in rows if row.get("answer_position") is not None],
                "answer_position",
            ),
        }

    return {
        "overall": one(records),
        "by_task": {task: one(rows) for task, rows in sorted(by_task.items())},
    }


def _delta_vs_reference(summaries: dict[str, Any], reference_name: str) -> dict[str, Any]:
    reference = summaries[reference_name]
    out: dict[str, Any] = {}
    for name, summary in summaries.items():
        if name == reference_name:
            continue
        out[name] = {"overall": {}, "by_task": {}}
        for key in ("target_nll", "first_token_nll", "target_top1_acc", "target_top5_acc", "gold_margin"):
            out[name]["overall"][key] = (
                float(summary["overall"][key]) - float(reference["overall"][key])
            )
        for task, task_summary in summary["by_task"].items():
            if task not in reference["by_task"]:
                continue
            out[name]["by_task"][task] = {}
            for key in ("target_nll", "target_top1_acc", "target_top5_acc", "gold_margin"):
                out[name]["by_task"][task][key] = (
                    float(task_summary[key]) - float(reference["by_task"][task][key])
                )
    return out


def _dist_context() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank, device


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _evaluate_checkpoint(
    name: str,
    checkpoint_path: str,
    model_path: str,
    task_items: dict[str, list[ScoreItem]],
    *,
    batch_size: int,
    device: torch.device,
    dtype: str,
    attn_implementation: str,
) -> list[dict[str, Any]]:
    model, tokenizer, adapters, handles, _checkpoint = load_model_tokenizer_for_eval(
        model_path,
        checkpoint_path,
        device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise SystemExit("tokenizer must have pad_token_id or eos_token_id")

    records: list[dict[str, Any]] = []
    try:
        for task, items in task_items.items():
            for offset in range(0, len(items), int(batch_size)):
                batch = items[offset : offset + int(batch_size)]
                active_scores = _score_items(model, batch, pad_token_id=int(pad_token_id), device=device)

                off_scores = None
                if adapters is not None:
                    for handle in handles:
                        handle.remove()
                    off_scores = _score_items(model, batch, pad_token_id=int(pad_token_id), device=device)
                    handles = _register_adapter_hooks(model, adapters)

                for score_idx, (item, active) in enumerate(zip(batch, active_scores)):
                    record: dict[str, Any] = {
                        "model": name,
                        "task": item.task,
                        "index": item.index,
                        "item_index": item.item_index,
                        "output_text": item.output_text,
                        "prompt_tokens": item.prompt_tokens,
                        "target_tokens": item.target_tokens,
                        "answer_position": item.answer_position,
                        **active,
                    }
                    if off_scores is not None:
                        off = off_scores[score_idx]
                        record["adapter_off_target_nll"] = off["target_nll"]
                        record["adapter_active_minus_off_nll"] = active["target_nll"] - off["target_nll"]
                        record["adapter_top1_flip_rate"] = abs(active["target_top1_acc"] - off["target_top1_acc"])
                        record["adapter_logit_l2_proxy"] = abs(active["gold_margin"] - off["gold_margin"])
                    else:
                        record["adapter_active_minus_off_nll"] = float("nan")
                        record["adapter_top1_flip_rate"] = float("nan")
                        record["adapter_logit_l2_proxy"] = float("nan")
                    records.append(record)
            print(json.dumps({"event": "task_done", "model": name, "task": task, "items": len(items)}, sort_keys=True), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-forced RULER gold-answer probe for CURE checkpoints.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--reference-name", default=None)
    parser.add_argument("--tasks", default=",".join(OFFICIAL_RULER13_TASKS))
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-output-items", type=int, default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    rank, world_size, local_rank, device = _dist_context()
    checkpoints = _parse_checkpoint_args(args.checkpoint)
    reference_name = args.reference_name or next(iter(checkpoints))
    if reference_name not in checkpoints:
        raise SystemExit(f"reference {reference_name!r} missing from checkpoints")

    # Use the first checkpoint only to get the tokenizer. Model weights are loaded below.
    _tmp_model, tokenizer, _tmp_adapters, tmp_handles, _tmp_ckpt = load_model_tokenizer_for_eval(
        args.model_path,
        checkpoints[reference_name],
        device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    for handle in tmp_handles:
        handle.remove()
    del _tmp_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    requested_tasks = [task.strip() for task in str(args.tasks).split(",") if task.strip()]
    data_dir = Path(args.data_dir)
    task_items: dict[str, list[ScoreItem]] = {}
    for task in requested_tasks:
        rows = read_jsonl(data_dir / task / "validation.jsonl", limit=int(args.max_samples))
        shard_rows = rows[rank::world_size]
        items: list[ScoreItem] = []
        for row in shard_rows:
            items.extend(
                build_score_items(
                    tokenizer,
                    task,
                    row,
                    seq_len=int(args.seq_len),
                    item_limit=args.max_output_items,
                )
            )
        task_items[task] = items

    output = Path(args.output)
    rank_dir = ensure_dir(output.parent / f"{output.stem}_rank_parts")
    for name, checkpoint_path in checkpoints.items():
        print(json.dumps({"event": "model_start", "model": name, "rank": rank, "world_size": world_size}, sort_keys=True), flush=True)
        records = _evaluate_checkpoint(
            name,
            checkpoint_path,
            args.model_path,
            task_items,
            batch_size=int(args.batch_size),
            device=device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        write_json(rank_dir / f"{name}.rank{rank:02d}.json", {"model": name, "rank": rank, "records": records})
        print(json.dumps({"event": "model_done", "model": name, "records": len(records), "rank": rank}, sort_keys=True), flush=True)

    _barrier()

    if rank == 0:
        all_records_by_model: dict[str, list[dict[str, Any]]] = {}
        summaries: dict[str, Any] = {}
        for name in checkpoints:
            merged: list[dict[str, Any]] = []
            for part in sorted(rank_dir.glob(f"{name}.rank*.json")):
                merged.extend(json.loads(part.read_text(encoding="utf-8"))["records"])
            all_records_by_model[name] = merged
            summaries[name] = _summarize(merged)

        result = {
            "protocol": {
                "data_dir": str(data_dir),
                "tasks": requested_tasks,
                "max_samples_per_task": int(args.max_samples),
                "seq_len": int(args.seq_len),
                "target_scoring": "teacher-forced gold output item after input+answer_prefix; multi-output rows score each output item separately.",
                "nll_delta": "negative vs reference means better gold-answer likelihood.",
            },
            "reference_name": reference_name,
            "checkpoints": checkpoints,
            "summaries": summaries,
            "deltas_vs_reference": _delta_vs_reference(summaries, reference_name),
        }
        write_json(output, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    _barrier()


if __name__ == "__main__":
    main()
