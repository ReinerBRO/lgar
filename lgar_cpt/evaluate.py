from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import Dataset

from .config import LGARParams, Paths, TrainParams
from .data import PackedFineWebDataset
from .hard_lgar import qwen_lgar_forward
from .router import SharedQueryRouter
from .mining import (
    attention_mask_format_for_model,
    ce_from_logits,
    compute_long_short_logp,
    document_causal_attention_mask,
    gather_logprob,
)
from .modeling import load_qwen_causal_lm, load_tokenizer, unwrap_logits
from .utils import ensure_dir, safe_mean, write_json


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


def _destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    return 0, 1


def _all_reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _eval_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    params: LGARParams,
    router: torch.nn.Module | None = None,
    mode: str = "full",
    target_budget: float | None = None,
    routing_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = document_causal_attention_mask(doc_ids, mask_format=attention_mask_format_for_model(model))
    return qwen_lgar_forward(
        model=model,
        input_ids=input_ids,
        full_attention_mask=mask,
        doc_ids=doc_ids,
        router=router,
        params=params,
        routing_valid_mask=routing_valid_mask,
        mode=mode,
        target_budget=target_budget,
    ).logits


def to_torch_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value.dtype == bool:
            out[key] = torch.as_tensor(value, dtype=torch.bool, device=device)
        elif value.dtype.kind == "f":
            out[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
        else:
            out[key] = torch.as_tensor(value, dtype=torch.long, device=device)
    return out


def normal_and_lsd_ce(
    model: torch.nn.Module,
    dataset: PackedFineWebDataset,
    params: LGARParams,
    batches: int,
    batch_size: int,
    device: torch.device,
    router: torch.nn.Module | None = None,
    mode: str = "full",
    target_budget: float | None = None,
) -> dict[str, float]:
    rank, world_size = _rank_world()
    ce_sum = 0.0
    ce_count = 0.0
    high_lsd_sum = 0.0
    high_lsd_count = 0.0
    long_1024_sum = 0.0
    long_1024_count = 0.0
    long_4096_sum = 0.0
    long_4096_count = 0.0
    with torch.no_grad():
        for batch_idx in range(batches):
            if batch_idx % world_size != rank:
                continue
            print(
                json.dumps(
                    {
                        "event": "final_eval_ntp_start",
                        "batch_idx": int(batch_idx),
                        "batches": int(batches),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            batch = to_torch_batch(dataset.sample(batch_size), device)
            input_doc = batch["doc_ids_full"][:, :-1]
            logits = _eval_logits(
                model,
                batch["input_ids"],
                input_doc,
                params,
                router=router,
                mode=mode,
                target_budget=target_budget,
                routing_valid_mask=batch["loss_mask"],
            )
            eval_logp = gather_logprob(logits, batch["labels"])
            ce_sum += float(ce_from_logits(logits, batch["labels"], batch["loss_mask"]).item())
            ce_count += 1.0
            full_logp, short_logp = compute_long_short_logp(
                model,
                batch["input_ids"],
                batch["labels"],
                input_doc,
                params.short_window,
            )
            lsd = full_logp - short_logp
            valid = batch["loss_mask"].bool()
            if valid.any():
                thresh = torch.quantile(lsd[valid].float(), 0.90)
                high = valid & (lsd >= thresh)
                if high.any():
                    high_lsd_sum += float((-eval_logp[high]).mean().item())
                    high_lsd_count += 1.0
            target_offsets = batch["doc_offsets_full"][:, 1 : batch["input_ids"].size(1) + 1].to(device)
            long_1024 = valid & (target_offsets >= 1024)
            long_4096 = valid & (target_offsets >= 4096)
            if long_1024.any():
                long_1024_sum += float((-eval_logp[long_1024]).mean().item())
                long_1024_count += 1.0
            if long_4096.any():
                long_4096_sum += float((-eval_logp[long_4096]).mean().item())
                long_4096_count += 1.0
            print(
                json.dumps(
                    {
                        "event": "final_eval_ntp_done",
                        "batch_idx": int(batch_idx),
                        "ce_count": float(ce_count),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    ce_sum = _all_reduce_sum(ce_sum, device)
    ce_count = _all_reduce_sum(ce_count, device)
    high_lsd_sum = _all_reduce_sum(high_lsd_sum, device)
    high_lsd_count = _all_reduce_sum(high_lsd_count, device)
    long_1024_sum = _all_reduce_sum(long_1024_sum, device)
    long_1024_count = _all_reduce_sum(long_1024_count, device)
    long_4096_sum = _all_reduce_sum(long_4096_sum, device)
    long_4096_count = _all_reduce_sum(long_4096_count, device)
    return {
        "normal_val_ce": float(ce_sum / ce_count) if ce_count else float("nan"),
        "high_lsd_ce": float(high_lsd_sum / high_lsd_count) if high_lsd_count else float("nan"),
        "long_1024_target_ce": float(long_1024_sum / long_1024_count) if long_1024_count else float("nan"),
        "long_4096_target_ce": float(long_4096_sum / long_4096_count) if long_4096_count else float("nan"),
    }


def _encode(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


@torch.no_grad()
def continuation_nll(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    continuation: str,
    max_context: int,
    device: torch.device,
    params: LGARParams | None = None,
    router: torch.nn.Module | None = None,
    mode: str = "full",
    target_budget: float | None = None,
) -> tuple[float, int]:
    prompt_ids = _encode(tokenizer, prompt)
    cont_ids = _encode(tokenizer, continuation)
    if not cont_ids:
        return float("inf"), 0
    max_prompt = max(1, max_context - len(cont_ids))
    prompt_ids = prompt_ids[-max_prompt:]
    ids = prompt_ids + cont_ids
    if len(ids) < 2:
        return float("inf"), 0
    input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).view(1, -1)
    labels = torch.tensor(ids[1:], dtype=torch.long, device=device).view(1, -1)
    if params is None:
        outputs = model(input_ids, use_cache=False)
        logits = unwrap_logits(outputs)
    else:
        doc_ids = torch.zeros_like(input_ids)
        logits = _eval_logits(
            model,
            input_ids,
            doc_ids,
            params,
            router=router,
            mode=mode,
            target_budget=target_budget,
            routing_valid_mask=torch.ones_like(input_ids, dtype=torch.bool),
        )
    logp = gather_logprob(logits, labels)[0]
    start = max(0, len(prompt_ids) - 1)
    answer = logp[start:]
    return float((-answer).mean().item()), int(answer.numel())


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _ranked_choice_accuracy(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    max_context: int,
    device: torch.device,
    params: LGARParams | None = None,
    router: torch.nn.Module | None = None,
    mode: str = "full",
    target_budget: float | None = None,
) -> dict[str, float]:
    rank, world_size = _rank_world()
    hits_sum = 0.0
    nll_sum = 0.0
    count = 0.0
    for row in cases[rank::world_size]:
        prompt = row["prompt"]
        target = str(row["target"])
        choices = [str(x) for x in row.get("choices", [])]
        if target not in choices:
            choices = [target] + choices
        seen = set()
        uniq_choices = []
        for item in choices:
            if item and item not in seen:
                uniq_choices.append(item)
                seen.add(item)
        scored = []
        for choice in uniq_choices:
            variants = [choice, " " + choice] if not choice.startswith(" ") else [choice]
            best = min(
                continuation_nll(
                    model,
                    tokenizer,
                    prompt,
                    v,
                    max_context,
                    device,
                    params=params,
                    router=router,
                    mode=mode,
                    target_budget=target_budget,
                )[0]
                for v in variants
            )
            scored.append((best, choice))
        scored.sort(key=lambda x: x[0])
        hits_sum += float(_normalize(scored[0][1]) == _normalize(target))
        target_nll = next(score for score, choice in scored if choice == target)
        nll_sum += float(target_nll)
        count += 1.0
    hits_sum = _all_reduce_sum(hits_sum, device)
    nll_sum = _all_reduce_sum(nll_sum, device)
    count = _all_reduce_sum(count, device)
    return {
        "accuracy": float(hits_sum / count) if count else float("nan"),
        "target_nll": float(nll_sum / count) if count else float("nan"),
        "num_cases": float(count),
    }


def _pad_to_tokens(tokenizer: Any, prefix: str, suffix: str, target_tokens: int) -> str:
    filler = (
        "The grass is green. The sky is blue. The sun is yellow. "
        "Here we go. There and back again. "
    )
    prefix_ids = _encode(tokenizer, prefix)
    suffix_ids = _encode(tokenizer, suffix)
    if len(prefix_ids) + len(suffix_ids) >= target_tokens:
        # Keep the suffix/question intact and trim overflow from the prefix side.
        keep_prefix = max(1, target_tokens - len(suffix_ids))
        return tokenizer.decode(prefix_ids[:keep_prefix] + suffix_ids)
    filler_ids = _encode(tokenizer, filler)
    if not filler_ids:
        return prefix + suffix
    filler_budget = target_tokens - len(prefix_ids) - len(suffix_ids)
    repeats = (filler_budget + len(filler_ids) - 1) // len(filler_ids)
    padded_ids = prefix_ids + (filler_ids * repeats)[:filler_budget] + suffix_ids
    return tokenizer.decode(padded_ids)


def build_ruler_mini_cases(tokenizer: Any, length_tokens: int, limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + length_tokens)
    cases: list[dict[str, Any]] = []
    for i in range(limit):
        key = f"key-{i}-{rng.randint(100, 999)}"
        values = [str(rng.randint(1000000, 9999999)) for _ in range(4)]
        target = values[rng.randrange(len(values))]
        facts = " ".join(f"One special magic number for {key}-{j} is {value}." for j, value in enumerate(values))
        ask_j = values.index(target)
        question = f"\nQuestion: What is the special magic number for {key}-{ask_j}?\nAnswer:"
        prompt = _pad_to_tokens(tokenizer, facts + "\n", question, length_tokens)
        cases.append({"prompt": prompt, "target": target, "choices": values, "task": "ruler_multikey"})
    for i in range(limit):
        base_value = str(rng.randint(10000, 99999))
        vars_ = [f"V{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}{i}" for _ in range(5)]
        facts = f"VAR {vars_[0]} = {base_value}. " + " ".join(f"VAR {vars_[j]} = VAR {vars_[j-1]}." for j in range(1, len(vars_)))
        question = f"\nQuestion: Which variable is assigned the value {base_value} at the end of the chain?\nAnswer:"
        prompt = _pad_to_tokens(tokenizer, facts + "\n", question, length_tokens)
        cases.append({"prompt": prompt, "target": vars_[-1], "choices": vars_, "task": "ruler_vt"})
    return cases


def build_nolima_style_cases(tokenizer: Any, length_tokens: int, limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 17 + length_tokens)
    names = ["Yuki", "Stuart", "Katie", "Veronica", "Gary", "Megan", "Calvin", "Mandy"]
    places = ["Helsinki", "Frankfurt", "Dresden", "Kyoto", "Lisbon", "Nairobi", "Boston", "Seoul"]
    cases = []
    for i in range(limit):
        name = rng.choice(names)
        place = rng.choice(places)
        distractors = [x for x in names if x != name]
        fact = f"Actually, {name} lives next to {place}."
        question = (
            "\nUse the information above to answer the question. "
            f"Which character has been to {place}?\nAnswer:"
        )
        prompt = _pad_to_tokens(tokenizer, fact + "\n", question, length_tokens)
        cases.append({"prompt": prompt, "target": name, "choices": [name] + distractors[:3], "task": "nolima_style"})
    return cases


def eval_long_dependency(
    model: torch.nn.Module,
    tokenizer: Any,
    max_context: int,
    device: torch.device,
    limit: int = 8,
    seed: int = 1337,
    lengths: list[int] | tuple[int, ...] = (4096, 8192),
    params: LGARParams | None = None,
    router: torch.nn.Module | None = None,
    mode: str = "full",
    target_budget: float | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    rank, _world_size = _rank_world()
    for length in lengths:
        if rank == 0:
            print(
                json.dumps(
                    {"event": "final_eval_long_start", "length": int(length), "limit": int(limit)},
                    sort_keys=True,
                ),
                flush=True,
            )
        ruler = build_ruler_mini_cases(tokenizer, length, limit, seed)
        ruler_scores = _ranked_choice_accuracy(
            model,
            tokenizer,
            ruler,
            max_context,
            device,
            params=params,
            router=router,
            mode=mode,
            target_budget=target_budget,
        )
        metrics[f"ruler_mini_{length}_accuracy"] = ruler_scores["accuracy"]
        metrics[f"ruler_mini_{length}_target_nll"] = ruler_scores["target_nll"]
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "final_eval_long_done",
                        "benchmark": "ruler_mini",
                        "length": int(length),
                        "accuracy": float(ruler_scores["accuracy"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        nolima = build_nolima_style_cases(tokenizer, length, limit, seed)
        nolima_scores = _ranked_choice_accuracy(
            model,
            tokenizer,
            nolima,
            max_context,
            device,
            params=params,
            router=router,
            mode=mode,
            target_budget=target_budget,
        )
        metrics[f"nolima_style_{length}_accuracy"] = nolima_scores["accuracy"]
        metrics[f"nolima_style_{length}_target_nll"] = nolima_scores["target_nll"]
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "final_eval_long_done",
                        "benchmark": "nolima_style",
                        "length": int(length),
                        "accuracy": float(nolima_scores["accuracy"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return metrics


def _mc_cases_from_arrow(root: Path, limit: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    hs = Dataset.from_file(str(root / "hellaswag" / "hellaswag-validation.arrow"))
    for row in hs.select(range(min(limit, len(hs)))):
        cases.append({"prompt": row["ctx"], "target": row["endings"][int(row["label"])], "choices": row["endings"], "task": "hellaswag"})
    piqa = Dataset.from_file(str(root / "piqa" / "piqa-validation.arrow"))
    for row in piqa.select(range(min(limit, len(piqa)))):
        choices = [row["sol1"], row["sol2"]]
        cases.append({"prompt": row["goal"] + "\nAnswer:", "target": choices[int(row["label"])], "choices": choices, "task": "piqa"})
    arc = Dataset.from_file(str(root / "arc_easy" / "ai2_arc-validation.arrow"))
    for row in arc.select(range(min(limit, len(arc)))):
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        idx = labels.index(row["answerKey"])
        cases.append({"prompt": row["question"] + "\nAnswer:", "target": texts[idx], "choices": texts, "task": "arc_easy"})
    return cases


def eval_short_mc(
    model: torch.nn.Module,
    tokenizer: Any,
    short_mc_dir: str | Path,
    max_context: int,
    device: torch.device,
    limit: int = 16,
    params: LGARParams | None = None,
    router: torch.nn.Module | None = None,
    mode: str = "full",
    target_budget: float | None = None,
) -> dict[str, float]:
    cases = _mc_cases_from_arrow(Path(short_mc_dir), limit)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in cases:
        by_task.setdefault(row["task"], []).append(row)
    metrics: dict[str, float] = {}
    accs = []
    rank, _world_size = _rank_world()
    for task, rows in by_task.items():
        if rank == 0:
            print(
                json.dumps(
                    {"event": "final_eval_short_start", "task": task, "num_cases": int(len(rows))},
                    sort_keys=True,
                ),
                flush=True,
            )
        score = _ranked_choice_accuracy(
            model,
            tokenizer,
            rows,
            max_context,
            device,
            params=params,
            router=router,
            mode=mode,
            target_budget=target_budget,
        )
        metrics[f"short_mc_{task}_accuracy"] = score["accuracy"]
        metrics[f"short_mc_{task}_target_nll"] = score["target_nll"]
        accs.append(score["accuracy"])
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "final_eval_short_done",
                        "task": task,
                        "accuracy": float(score["accuracy"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    metrics["short_mc_avg_accuracy"] = safe_mean(accs)
    return metrics


def default_long_lengths_for_seq_len(seq_len: int) -> list[int]:
    seq_len = int(seq_len)
    if seq_len <= 2048:
        return [1024, 1536, 2048]
    if seq_len <= 4096:
        return [2048, 3072, 4096]
    return [4096, 8192]


def evaluate_model(
    model: torch.nn.Module,
    tokenizer: Any,
    paths: Paths,
    lgar_params: LGARParams,
    train_params: TrainParams,
    batches: int,
    short_limit: int,
    long_limit: int,
    long_lengths: list[int] | tuple[int, ...] | None,
    device: torch.device,
    router: torch.nn.Module | None = None,
    mode: str = "full",
    target_budget: float | None = None,
) -> dict[str, float]:
    if long_lengths is None:
        long_lengths = default_long_lengths_for_seq_len(lgar_params.seq_len)
    val_data = PackedFineWebDataset(
        paths.cache_dir,
        seq_len=lgar_params.seq_len,
        pad_token_id=int(tokenizer.pad_token_id),
        split="val",
        seed=train_params.seed,
    )
    metrics = normal_and_lsd_ce(
        model,
        val_data,
        lgar_params,
        batches,
        train_params.batch_size,
        device,
        router=router,
        mode=mode,
        target_budget=target_budget,
    )
    metrics.update(
        eval_long_dependency(
            model,
            tokenizer,
            lgar_params.seq_len,
            device,
            limit=long_limit,
            seed=train_params.seed,
            lengths=long_lengths,
            params=lgar_params,
            router=router,
            mode=mode,
            target_budget=target_budget,
        )
    )
    metrics.update(
        eval_short_mc(
            model,
            tokenizer,
            paths.short_mc_dir,
            lgar_params.seq_len,
            device,
            limit=short_limit,
            params=lgar_params,
            router=router,
            mode=mode,
            target_budget=target_budget,
        )
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen/L-GAR-CPT model.")
    defaults = Paths()
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--cache-dir", default=defaults.cache_dir)
    parser.add_argument("--output", default=None)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--short-limit", type=int, default=16)
    parser.add_argument("--long-limit", type=int, default=8)
    parser.add_argument("--long-lengths", default="4096,8192")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--short-window", type=int, default=1024)
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--router-target-budget", type=float, default=0.10)
    parser.add_argument("--final-global-budget", type=float, default=0.25)
    parser.add_argument("--routed-layer-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--router-hidden-dim", type=int, default=512)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--router-path", default=None)
    parser.add_argument("--mode", choices=["full", "router_aux", "routed"], default="full")
    parser.add_argument("--target-budget", type=float, default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    rank, _world_size, _local_rank, device = _distributed_context()
    paths = Paths(model_path=args.model_path, cache_dir=args.cache_dir)
    lgar_params = LGARParams(
        seq_len=args.seq_len,
        short_window=args.short_window,
        local_window=args.local_window,
        router_target_budget=args.router_target_budget,
        final_global_budget=args.final_global_budget,
        routed_layer_fraction=args.routed_layer_fraction,
        router_hidden_dim=args.router_hidden_dim,
    )
    train_params = TrainParams(attn_implementation=args.attn_implementation, dtype=args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = load_qwen_causal_lm(args.model_path, dtype_name=args.dtype, attn_implementation=args.attn_implementation)
    model.to(device)
    model.eval()
    router = None
    if args.mode in {"router_aux", "routed"}:
        router = SharedQueryRouter(int(model.config.hidden_size), lgar_params.router_hidden_dim).to(device)
        router.eval()
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if router is not None:
            router_state = checkpoint.get("router")
            if router_state is None and args.router_path:
                router_state = torch.load(args.router_path, map_location="cpu").get("router")
            if router_state is None:
                raise SystemExit("Checkpoint/router state missing for routed evaluation.")
            router.load_state_dict(router_state)
    long_lengths = [int(item.strip()) for item in str(args.long_lengths).split(",") if item.strip()]
    metrics = evaluate_model(
        model,
        tokenizer,
        paths,
        lgar_params,
        train_params,
        batches=args.eval_batches,
        short_limit=args.short_limit,
        long_limit=args.long_limit,
        long_lengths=long_lengths,
        device=device,
        router=router,
        mode=args.mode,
        target_budget=args.target_budget,
    )
    if _is_rank0(rank):
        print(json.dumps(metrics, indent=2, sort_keys=True))
        if args.output:
            ensure_dir(Path(args.output).parent)
            write_json(args.output, metrics)
    _destroy_process_group()


if __name__ == "__main__":
    main()
