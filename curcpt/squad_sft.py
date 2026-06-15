from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from lgar_cpt.modeling import load_qwen_causal_lm, load_tokenizer
from lgar_cpt.utils import ensure_dir, set_seed, write_json

from .forward import _register_adapter_hooks
from .model_eval_utils import load_adapters_from_checkpoint


PROMPT_TEMPLATE = (
    "You will answer a question based on the following book snippet:\n\n"
    "{context}\n\n"
    "Use the information provided in the book snippet to answer the question. "
    "Your answer should be short and based on either explicitly stated facts or strong, logical inferences.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


@dataclass(frozen=True)
class SFTExample:
    prompt_ids: list[int]
    answer_ids: list[int]


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


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _sync_module_grads(module: torch.nn.Module | None, world_size: int) -> None:
    if module is None or world_size <= 1:
        return
    for param in module.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)


def _lr_schedule(step: int, total_steps: int, warmup_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _extract_question(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    for msg in messages:
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip()
    return str(row.get("question", "")).strip()


def _extract_answer(row: dict[str, Any]) -> str:
    answers = row.get("answers") or []
    if not answers:
        return ""
    answer = answers[0]
    if isinstance(answer, dict):
        return str(answer.get("text", "")).strip()
    return str(answer).strip()


def _load_squad_examples(
    data_path: str | Path,
    tokenizer: Any,
    seq_len: int,
    max_examples: int | None,
    seed: int,
) -> tuple[list[SFTExample], dict[str, Any]]:
    rows = json.loads(Path(data_path).read_text(encoding="utf-8"))
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    if max_examples is not None and max_examples > 0:
        order = order[: int(max_examples)]

    examples: list[SFTExample] = []
    skipped_empty = 0
    skipped_long = 0
    max_prompt_tokens = 0
    max_answer_tokens = 0
    for idx in order:
        row = rows[idx]
        context = str(row.get("document", "")).strip()
        question = _extract_question(row)
        answer = _extract_answer(row)
        if not context or not question or not answer:
            skipped_empty += 1
            continue
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_text = answer if answer.startswith((" ", "\n")) else " " + answer
        answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [int(tokenizer.eos_token_id)]
        if len(prompt_ids) + len(answer_ids) < 2:
            skipped_empty += 1
            continue
        if len(prompt_ids) + len(answer_ids) > int(seq_len):
            skipped_long += 1
            continue
        max_prompt_tokens = max(max_prompt_tokens, len(prompt_ids))
        max_answer_tokens = max(max_answer_tokens, len(answer_ids))
        examples.append(SFTExample(prompt_ids=prompt_ids, answer_ids=answer_ids))

    stats = {
        "source_rows": len(rows),
        "requested_examples": max_examples,
        "usable_examples": len(examples),
        "skipped_empty": skipped_empty,
        "skipped_long": skipped_long,
        "max_prompt_tokens": max_prompt_tokens,
        "max_answer_tokens": max_answer_tokens,
    }
    if not examples:
        raise ValueError(f"no usable SQuAD examples after filtering: {stats}")
    return examples, stats


def _make_batch(
    examples: list[SFTExample],
    indices: list[int],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs: list[list[int]] = []
    label_masks: list[list[bool]] = []
    for idx in indices:
        ex = examples[idx % len(examples)]
        full_ids = ex.prompt_ids + ex.answer_ids
        input_ids = full_ids[:-1]
        labels = full_ids[1:]
        mask = [False] * len(labels)
        first_answer_label = max(0, len(ex.prompt_ids) - 1)
        for pos in range(first_answer_label, len(mask)):
            mask[pos] = True
        seqs.append(input_ids)
        label_masks.append(mask)

    max_len = max(len(seq) for seq in seqs)
    input_batch = torch.full((len(seqs), max_len), int(pad_token_id), dtype=torch.long)
    label_batch = torch.full((len(seqs), max_len), -100, dtype=torch.long)
    loss_mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for row, (seq, mask) in enumerate(zip(seqs, label_masks)):
        length = len(seq)
        input_batch[row, :length] = torch.tensor(seq, dtype=torch.long)
        labels = seqs[row][1:] + [pad_token_id]
        # Replace labels with the already shifted full sequence labels.
        ex = examples[indices[row] % len(examples)]
        full_ids = ex.prompt_ids + ex.answer_ids
        shifted_labels = full_ids[1:]
        label_batch[row, :length] = torch.tensor(shifted_labels, dtype=torch.long)
        loss_mask[row, :length] = torch.tensor(mask, dtype=torch.bool)
    label_batch[~loss_mask] = -100
    return input_batch.to(device), label_batch.to(device), loss_mask.to(device)


def _sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


def train_squad_sft(args: argparse.Namespace) -> dict[str, Any]:
    rank, world_size, local_rank, device = _distributed_context()
    set_seed(int(args.seed) + rank * 1009)

    tokenizer = load_tokenizer(args.model_path)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must have pad_token_id or eos_token_id")

    examples, data_stats = _load_squad_examples(
        args.squad_path,
        tokenizer,
        int(args.seq_len),
        int(args.max_train_examples) if args.max_train_examples else None,
        int(args.seed),
    )

    model = load_qwen_causal_lm(
        args.model_path,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=bool(args.gradient_checkpointing),
    ).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)

    adapters = load_adapters_from_checkpoint(checkpoint, model, device)
    handles = _register_adapter_hooks(model, adapters) if adapters is not None else []
    if adapters is not None:
        adapters.train()
        for param in adapters.parameters():
            param.requires_grad_(not bool(args.freeze_adapters))

    if bool(args.freeze_base_model):
        for param in model.parameters():
            param.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if adapters is not None:
        trainable_params.extend([p for p in adapters.parameters() if p.requires_grad])
    if not trainable_params:
        raise ValueError("no trainable parameters")

    ddp_model: torch.nn.Module = model
    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    global_batch = int(args.batch_size) * world_size
    steps_per_epoch = math.ceil(len(examples) / max(1, global_batch))
    total_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(args.epochs) * steps_per_epoch
    warmup_steps = max(1, int(total_steps * float(args.warmup_fraction)))
    min_lr = float(args.min_lr)

    indices = list(range(len(examples)))
    rng = random.Random(int(args.seed))
    metrics: list[dict[str, Any]] = []
    wall_start = time.monotonic()
    tokens_seen = 0

    model.train()
    for step in range(total_steps):
        if step % steps_per_epoch == 0:
            rng.shuffle(indices)
        lr_now = _lr_schedule(step, total_steps, warmup_steps, float(args.lr), min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        base = (step % steps_per_epoch) * global_batch + rank * int(args.batch_size)
        batch_indices = [indices[(base + offset) % len(indices)] for offset in range(int(args.batch_size))]
        input_ids, labels, loss_mask = _make_batch(examples, batch_indices, int(pad_token_id), device)

        optimizer.zero_grad(set_to_none=True)
        out = ddp_model(input_ids=input_ids, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        loss = _sft_loss(logits, labels)
        loss.backward()
        _sync_module_grads(adapters, world_size)
        torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))
        optimizer.step()

        tokens_seen += int(input_ids.numel())
        if rank == 0 and ((step + 1) % int(args.log_interval) == 0 or step == 0):
            entry = {
                "step": step + 1,
                "total_steps": total_steps,
                "loss/sft": float(loss.detach().item()),
                "lr": lr_now,
                "tokens_seen_rank0": tokens_seen,
                "answer_tokens": int(loss_mask.sum().item()),
                "batch_tokens": int(input_ids.numel()),
                "wall_elapsed": time.monotonic() - wall_start,
            }
            metrics.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)

    _barrier()
    summary = {
        "run_name": args.run_name,
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "squad_path": str(Path(args.squad_path).resolve()),
        "seq_len": int(args.seq_len),
        "batch_size_per_rank": int(args.batch_size),
        "world_size": world_size,
        "epochs": int(args.epochs),
        "total_steps": total_steps,
        "tokens_seen_rank0": tokens_seen,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "freeze_base_model": bool(args.freeze_base_model),
        "freeze_adapters": bool(args.freeze_adapters),
        "has_adapters": adapters is not None,
        "adapter_parameters": adapters.num_parameters() if adapters is not None else 0,
        "trainable_parameters": sum(p.numel() for p in trainable_params),
        "data": data_stats,
        "wall_seconds": time.monotonic() - wall_start,
        "final_metrics": metrics[-1] if metrics else {},
    }

    if rank == 0:
        run_dir = ensure_dir(Path(args.output_dir) / args.run_name)
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "metrics.json", metrics)
        state_model = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        ckpt: dict[str, Any] = {
            "model": state_model,
            "optimizer": optimizer.state_dict(),
            "summary": summary,
        }
        if adapters is not None:
            ckpt["adapters"] = adapters.state_dict()
        torch.save(ckpt, run_dir / "checkpoint.pt")

    for handle in handles:
        handle.remove()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Short SQuAD answer-only SFT for CURE/CE checkpoints.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-path", default="/data/user/xnie012/cache/Models/Qwen2.5-0.5B")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--squad-path", default="/data/user/xnie012/cache/data/chatqa/squad1.1/train.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--freeze-base-model", action="store_true")
    parser.add_argument("--freeze-adapters", action="store_true")
    args = parser.parse_args()
    train_squad_sft(args)


if __name__ == "__main__":
    main()
