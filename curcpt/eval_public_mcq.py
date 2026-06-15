from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from datasets import Dataset

from .model_eval_utils import load_model_tokenizer_for_eval
from .rh_bottleneck import resolve_retrieval_heads, rh_bottleneck_logits


DATA_ROOT = Path(os.environ.get("CURCPT_PUBLIC_DATA_ROOT", "/data/user/xnie012/cache/data"))
ARROW_DATA_ROOT = Path(os.environ.get("CURCPT_LM_EVAL_DATASET_ROOT", "/data/user/xnie012/cache/hf_lm_eval_c400/datasets"))
DEFAULT_TASKS = ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "openbookqa")
DEFAULT_CONDITIONS = ("clean", "irrelevant_demo")


@dataclass
class MCQExample:
    example_id: str
    prompt: str
    choices: list[str]
    answer_index: int
    demo_text: str


@dataclass
class CandidateItem:
    example_index: int
    choice_index: int
    token_ids: list[int]
    prompt_len: int


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in Dataset.from_parquet(str(path))]


def _read_arrow(path: Path) -> list[dict[str, Any]]:
    import pyarrow.ipc as ipc

    with ipc.open_stream(path) as reader:
        return reader.read_all().to_pylist()


def _resolve_arrow(pattern: str) -> Path:
    matches = sorted(ARROW_DATA_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no local MCQ arrow file matched {pattern} under {ARROW_DATA_ROOT}")
    return matches[0]


def _task_rows(task: str) -> list[dict[str, Any]]:
    if task == "piqa":
        return _read_arrow(_resolve_arrow("baber___piqa/default/0.0.0/*/piqa-validation.arrow"))
    if task == "arc_easy":
        return _load_parquet(DATA_ROOT / "arc_easy" / "ARC-Easy" / "validation-00000-of-00001.parquet")
    if task == "arc_challenge":
        return _load_parquet(DATA_ROOT / "arc_challenge" / "ARC-Challenge" / "validation-00000-of-00001.parquet")
    if task == "openbookqa":
        return _read_arrow(_resolve_arrow("allenai___openbookqa/main/0.0.0/*/openbookqa-validation.arrow"))
    if task == "hellaswag":
        return _load_parquet(DATA_ROOT / "hellaswag" / "data" / "validation-00000-of-00001.parquet")
    if task == "winogrande":
        return _load_parquet(DATA_ROOT / "winogrande" / "winogrande_xl" / "validation-00000-of-00001.parquet")
    raise ValueError(f"unsupported task or missing local loader: {task}")


def _normalize_space(text: Any) -> str:
    return " ".join(str(text).strip().split())


def _build_examples(task: str, rows: list[dict[str, Any]]) -> list[MCQExample]:
    examples: list[MCQExample] = []
    if task == "piqa":
        for idx, row in enumerate(rows):
            goal = _normalize_space(row["goal"])
            sol1 = _normalize_space(row["sol1"])
            sol2 = _normalize_space(row["sol2"])
            label = int(row["label"])
            prompt = f"Question: {goal}\nAnswer:"
            choices = [f" {sol1}", f" {sol2}"]
            demo_text = f"Question: {goal}\nAnswer:{choices[label]}\n"
            examples.append(MCQExample(f"piqa-{idx}", prompt, choices, label, demo_text))
        return examples

    if task in {"arc_easy", "arc_challenge"}:
        for row in rows:
            question = _normalize_space(row["question"])
            labels = list(row["choices"]["label"])
            texts = [_normalize_space(text) for text in row["choices"]["text"]]
            answer_index = {label: idx for idx, label in enumerate(labels)}[row["answerKey"]]
            prompt = f"Question: {question}\nAnswer:"
            choices = [f" {text}" for text in texts]
            demo_text = f"Question: {question}\nAnswer:{choices[answer_index]}\n"
            examples.append(MCQExample(str(row["id"]), prompt, choices, answer_index, demo_text))
        return examples

    if task == "openbookqa":
        for row in rows:
            question = _normalize_space(row["question_stem"])
            labels = list(row["choices"]["label"])
            texts = [_normalize_space(text) for text in row["choices"]["text"]]
            answer_index = {label: idx for idx, label in enumerate(labels)}[row["answerKey"]]
            prompt = f"Question: {question}\nAnswer:"
            choices = [f" {text}" for text in texts]
            demo_text = f"Question: {question}\nAnswer:{choices[answer_index]}\n"
            examples.append(MCQExample(str(row["id"]), prompt, choices, answer_index, demo_text))
        return examples

    if task == "hellaswag":
        for row in rows:
            ctx = _normalize_space(row["ctx"])
            endings = [f" {_normalize_space(ending)}" for ending in row["endings"]]
            label = int(row["label"])
            prompt = f"Context: {ctx}\nContinuation:"
            demo_text = f"Context: {ctx}\nContinuation:{endings[label]}\n"
            examples.append(MCQExample(str(row["ind"]), prompt, endings, label, demo_text))
        return examples

    if task == "winogrande":
        for idx, row in enumerate(rows):
            sentence = str(row["sentence"])
            if "_" not in sentence:
                continue
            prefix, suffix = sentence.split("_", 1)
            option1 = _normalize_space(row["option1"])
            option2 = _normalize_space(row["option2"])
            label = int(row["answer"]) - 1
            prompt = f"Complete the sentence:\n{prefix}"
            choices = [f"{option1}{suffix}", f"{option2}{suffix}"]
            demo_text = f"Complete the sentence:\n{prefix}{choices[label]}\n"
            examples.append(MCQExample(f"winogrande-{idx}", prompt, choices, label, demo_text))
        return examples

    raise ValueError(f"unsupported task: {task}")


def _distractor_indices(index: int, total: int, count: int, seed: int) -> Iterable[int]:
    produced = 0
    cursor = (index + seed + 1) % total
    stride = max((seed * 2) + 1, 3)
    while produced < count:
        if cursor != index:
            yield cursor
            produced += 1
        cursor = (cursor + stride) % total


def _compose_prompt(
    example_index: int,
    example: MCQExample,
    examples: list[MCQExample],
    condition: str,
    num_distractors: int,
) -> str:
    if condition == "clean":
        return example.prompt
    if condition not in {"irrelevant_demo", "fewshot", "icl"}:
        raise ValueError(f"unsupported condition: {condition}")
    parts = [examples[i].demo_text for i in _distractor_indices(example_index, len(examples), num_distractors, 39)]
    parts.append(example.prompt)
    return "\n".join(parts)


def _truncate_ids(prefix_ids: list[int], choice_ids: list[int], seq_len: int, max_prefix_tokens: int) -> tuple[list[int], int]:
    prefix_ids = prefix_ids[:max_prefix_tokens]
    max_prompt_tokens = max(seq_len - len(choice_ids), 1)
    if len(prefix_ids) > max_prompt_tokens:
        prefix_ids = prefix_ids[-max_prompt_tokens:]
    token_ids = prefix_ids + choice_ids
    if len(token_ids) > seq_len:
        token_ids = token_ids[-seq_len:]
        prompt_len = max(len(token_ids) - len(choice_ids), 0)
    else:
        prompt_len = len(prefix_ids)
    return token_ids, prompt_len


def _build_candidates(
    tokenizer: Any,
    example_index: int,
    example: MCQExample,
    examples: list[MCQExample],
    condition: str,
    seq_len: int,
    num_distractors: int,
    max_prefix_tokens: int,
) -> list[CandidateItem]:
    prompt_text = _compose_prompt(example_index, example, examples, condition, num_distractors)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    items = []
    for choice_index, choice in enumerate(example.choices):
        choice_ids = tokenizer.encode(choice, add_special_tokens=False)
        token_ids, prompt_len = _truncate_ids(prompt_ids, choice_ids, seq_len, max_prefix_tokens)
        items.append(CandidateItem(example_index, choice_index, token_ids, prompt_len))
    return items


def _select_examples(examples: list[MCQExample], max_examples: int) -> list[MCQExample]:
    if max_examples <= 0 or len(examples) <= max_examples:
        return examples
    if max_examples == 1:
        return [examples[0]]
    last_index = len(examples) - 1
    selected = []
    seen = set()
    for idx in range(max_examples):
        selected_index = round(idx * last_index / (max_examples - 1))
        if selected_index not in seen:
            selected.append(selected_index)
            seen.add(selected_index)
    cursor = 0
    while len(selected) < max_examples:
        if cursor not in seen:
            selected.append(cursor)
            seen.add(cursor)
        cursor += 1
    return [examples[idx] for idx in sorted(selected)]


def _score_candidates(
    model: torch.nn.Module,
    items: list[CandidateItem],
    device: torch.device,
    pad_token_id: int,
    eval_mode: str,
    retrieval_heads: set[tuple[int, int]],
    local_window: int,
) -> tuple[list[float], list[float], list[int]]:
    max_len = max(len(item.token_ids) for item in items)
    input_ids = torch.full((len(items), max_len), pad_token_id, dtype=torch.long)
    doc_ids = torch.full((len(items), max_len), -1, dtype=torch.long)
    target_mask = torch.zeros((len(items), max_len - 1), dtype=torch.bool)
    for row_idx, item in enumerate(items):
        length = len(item.token_ids)
        input_ids[row_idx, :length] = torch.tensor(item.token_ids, dtype=torch.long)
        doc_ids[row_idx, :length] = 0
        if item.prompt_len > 0:
            target_mask[row_idx, max(item.prompt_len - 1, 0) : length - 1] = True

    input_ids = input_ids.to(device, non_blocking=True)
    doc_ids = doc_ids.to(device, non_blocking=True)
    target_mask = target_mask.to(device, non_blocking=True)
    with torch.no_grad():
        if eval_mode in {"rh_bottleneck", "rh_layer_bottleneck"}:
            logits = rh_bottleneck_logits(
                model,
                input_ids,
                doc_ids,
                retrieval_heads,
                int(local_window),
                bottleneck_scope="routed_layers" if eval_mode == "rh_layer_bottleneck" else "all_layers",
            )
        else:
            logits = model(input_ids, use_cache=False).logits
        log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)

    token_logp = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    masked = token_logp * target_mask
    logp_sum = masked.sum(dim=1)
    token_count = target_mask.sum(dim=1).clamp(min=1)
    logp_avg = logp_sum / token_count
    return (
        [float(x) for x in logp_sum.cpu().tolist()],
        [float(x) for x in logp_avg.cpu().tolist()],
        [int(x) for x in token_count.cpu().tolist()],
    )


def _evaluate_condition(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[MCQExample],
    condition: str,
    seq_len: int,
    num_distractors: int,
    max_prefix_tokens: int,
    batch_size: int,
    device: torch.device,
    eval_mode: str,
    retrieval_heads: set[tuple[int, int]],
    local_window: int,
) -> dict[str, Any]:
    candidate_items: list[CandidateItem] = []
    for example_index, example in enumerate(examples):
        candidate_items.extend(
            _build_candidates(tokenizer, example_index, example, examples, condition, seq_len, num_distractors, max_prefix_tokens)
        )

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must provide pad_token_id or eos_token_id")

    scores_sum: list[list[float]] = [[] for _ in examples]
    scores_avg: list[list[float]] = [[] for _ in examples]
    total_choice_tokens = 0
    for start in range(0, len(candidate_items), batch_size):
        batch = candidate_items[start : start + batch_size]
        batch_sum, batch_avg, token_counts = _score_candidates(
            model,
            batch,
            device,
            int(pad_token_id),
            eval_mode,
            retrieval_heads,
            int(local_window),
        )
        for item, score_sum, score_avg, token_count in zip(batch, batch_sum, batch_avg, token_counts):
            scores_sum[item.example_index].append(score_sum)
            scores_avg[item.example_index].append(score_avg)
            total_choice_tokens += token_count

    correct_sum = 0
    correct_avg = 0
    margin_sum = 0.0
    margin_avg = 0.0
    for idx, example in enumerate(examples):
        pred_sum = max(range(len(scores_sum[idx])), key=scores_sum[idx].__getitem__)
        pred_avg = max(range(len(scores_avg[idx])), key=scores_avg[idx].__getitem__)
        correct_sum += int(pred_sum == example.answer_index)
        correct_avg += int(pred_avg == example.answer_index)
        answer_sum = scores_sum[idx][example.answer_index]
        answer_avg = scores_avg[idx][example.answer_index]
        best_wrong_sum = max(score for j, score in enumerate(scores_sum[idx]) if j != example.answer_index)
        best_wrong_avg = max(score for j, score in enumerate(scores_avg[idx]) if j != example.answer_index)
        margin_sum += answer_sum - best_wrong_sum
        margin_avg += answer_avg - best_wrong_avg

    num_examples = len(examples)
    num_choices = sum(len(example.choices) for example in examples)
    return {
        "num_examples": num_examples,
        "num_choices": num_choices,
        "accuracy_sum": correct_sum / max(num_examples, 1),
        "accuracy_avg": correct_avg / max(num_examples, 1),
        "avg_margin_sum": margin_sum / max(num_examples, 1),
        "avg_margin_avg": margin_avg / max(num_examples, 1),
        "avg_choice_tokens": total_choice_tokens / max(num_choices, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CURE checkpoints on standard public MCQ tasks.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-distractors", type=int, default=2)
    parser.add_argument("--max-prefix-tokens", type=int, default=1800)
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--eval-mode", choices=["full", "rh_bottleneck", "rh_layer_bottleneck"], default="full")
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--retrieval-heads", default=None, help="Comma-separated L:H heads for rh_bottleneck eval.")
    parser.add_argument("--retrieval-heads-json", default=None, help="Ablation JSON containing retrieval_heads.")
    parser.add_argument("--reference-checkpoint-path", default=None, help="Checkpoint to read retrieval heads from when this checkpoint has none.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else int(device.index))
    model, tokenizer, adapters, handles, _checkpoint = load_model_tokenizer_for_eval(
        args.model_path,
        args.checkpoint_path,
        device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    retrieval_heads = resolve_retrieval_heads(
        _checkpoint,
        explicit_heads=args.retrieval_heads,
        heads_json=args.retrieval_heads_json,
        reference_checkpoint_path=args.reference_checkpoint_path,
    )
    if args.eval_mode in {"rh_bottleneck", "rh_layer_bottleneck"} and not retrieval_heads:
        raise SystemExit("rh_bottleneck eval requires retrieval heads")

    output: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint_path).resolve()),
        "adapters_active": adapters is not None,
        "num_retrieval_heads": adapters.num_heads() if adapters is not None else 0,
        "eval_mode": args.eval_mode,
        "rh_bottleneck_retrieval_heads": sorted([list(head) for head in retrieval_heads]),
        "seq_len": int(args.seq_len),
        "max_examples": int(args.max_examples),
        "tasks": {},
    }
    try:
        for task in args.tasks:
            rows = _task_rows(task)
            examples = _select_examples(_build_examples(task, rows), int(args.max_examples))
            task_result: dict[str, Any] = {"num_examples": len(examples), "conditions": {}}
            if len(examples) < int(args.max_examples):
                task_result["warning"] = f"only {len(examples)} examples available"
            for condition in args.conditions:
                task_result["conditions"][condition] = _evaluate_condition(
                    model,
                    tokenizer,
                    examples,
                    condition,
                    int(args.seq_len),
                    int(args.num_distractors),
                    int(args.max_prefix_tokens),
                    int(args.batch_size),
                    device,
                    args.eval_mode,
                    retrieval_heads,
                    int(args.local_window),
                )
            clean = task_result["conditions"].get("clean")
            distract = task_result["conditions"].get("irrelevant_demo")
            if clean and distract:
                clean_acc = clean["accuracy_avg"]
                distract_acc = distract["accuracy_avg"]
                task_result["delta"] = {
                    "accuracy_avg_abs": distract_acc - clean_acc,
                    "accuracy_avg_rel_pct": 0.0 if math.isclose(clean_acc, 0.0) else (distract_acc - clean_acc) / clean_acc * 100.0,
                    "margin_avg_abs": distract["avg_margin_avg"] - clean["avg_margin_avg"],
                }
            output["tasks"][task] = task_result
    finally:
        for handle in handles:
            handle.remove()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
