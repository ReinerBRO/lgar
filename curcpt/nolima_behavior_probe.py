from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from lgar_cpt.mining import attention_mask_format_for_model, document_causal_attention_mask
from lgar_cpt.modeling import unwrap_logits

from .eval_nolima import _load_prefix_nolima, _sample_stride, _truncate_prompt_ids
from .forward import _register_adapter_hooks
from .model_eval_utils import load_model_tokenizer_for_eval


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    variant: str
    context_length: int
    book: str
    book_index: int
    test_name: str
    question_type: str
    depth_percent: float
    question: str
    prompt_ids: list[int]
    gold_answers: list[str]
    candidates: list[str]
    prompt_tokens: int


@dataclass(frozen=True)
class CandidateRequest:
    case_index: int
    candidate: str
    prompt_ids: list[int]


def _unique_texts(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text.casefold())
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _normalize(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _maybe_space(prompt: str, target: str) -> str:
    if not prompt or not target:
        return ""
    if prompt.endswith((" ", "\n", "\t")) or target.startswith((" ", "\n", "\t")):
        return ""
    return " "


def _split_mc_question(question: str) -> tuple[str, list[str]]:
    first_q = question.find("?")
    if first_q < 0:
        return question, []
    stem = question[: first_q + 1].strip()
    tail = question[first_q + 1 :].strip()
    tail = re.sub(r"\?+$", "", tail).strip()
    tail = re.sub(r"\s+", " ", tail)
    tail = re.sub(r",\s+or\s+", ", ", tail, flags=re.IGNORECASE)
    tail = re.sub(r"\s+or\s+", ", ", tail, flags=re.IGNORECASE)
    options = [part.strip(" \t\n\r.,;:") for part in tail.split(",") if part.strip(" \t\n\r.,;:")]
    return stem, options


def _candidate_set(test: dict[str, Any], question: str, gold_answers: list[str], variant: str) -> list[str]:
    candidates: list[str] = []
    if variant == "mc":
        _stem, options = _split_mc_question(question)
        candidates.extend(options)
        if candidates:
            candidates.extend(gold_answers)
            return _unique_texts(candidates)
    char_set = test.get("character_set")
    if isinstance(char_set, list):
        candidates.extend(str(x) for x in char_set)
    candidates.extend(gold_answers)
    return _unique_texts(candidates)


def _build_cases(args: argparse.Namespace, tokenizer: Any) -> list[ProbeCase]:
    prefix_nolima = _load_prefix_nolima(args.prefix_scripts_dir)
    needle_set = json.loads(Path(args.needle_set_path).read_text(encoding="utf-8"))
    tests = prefix_nolima._build_tests(needle_set, max_experiments=args.max_experiments)
    haystack_paths = sorted(path for path in Path(args.haystack_dir).glob("*.txt") if not path.name.startswith("._"))
    if args.max_books is not None:
        haystack_paths = haystack_paths[: int(args.max_books)]
    depth_values = np.linspace(
        float(args.document_depth_percent_min),
        float(args.document_depth_percent_max),
        int(args.document_depth_percent_intervals),
    )
    sample_stride = _sample_stride(float(args.sample_fraction))
    sample_offset = int(args.sample_offset) % sample_stride
    max_prompt_tokens = int(args.seq_len) - int(args.max_candidate_tokens)

    cases: list[ProbeCase] = []
    candidate_index = 0
    for book_index, haystack_path in enumerate(haystack_paths):
        haystack = prefix_nolima.BookHaystack(str(haystack_path))
        for test in tests:
            rng = np.random.default_rng(int(test["seed"]) + book_index)
            for depth_percent in depth_values:
                current_index = candidate_index
                candidate_index += 1
                if current_index % sample_stride != sample_offset:
                    continue

                needle = str(test["needle"])
                question = str(test["retrieval_question"])
                gold_answers = [str(x) for x in list(test["gold_answers"])]
                if "{CHAR}" in needle:
                    selected_character = str(rng.choice(test["character_set"]))
                    needle = needle.replace("{CHAR}", selected_character)
                    question = question.replace("{CHAR}", selected_character)
                    gold_answers = [selected_character]

                placement = haystack.generate_w_needle_placement(
                    needle=needle,
                    encoding_func=lambda text: tokenizer.encode(text, add_special_tokens=False),
                    decoding_func=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
                    context_length=int(args.context_length),
                    depth=float(depth_percent / 100.0),
                    distractor=test["distractor"],
                    rng=rng,
                )
                prompt = prefix_nolima.DEFAULT_TASK_TEMPLATE.format(
                    haystack=placement["text"],
                    question=question,
                )
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(prompt_ids) > max_prompt_tokens:
                    prompt_ids = _truncate_prompt_ids(prompt_ids, max_prompt_tokens)

                candidates = _candidate_set(test, question, gold_answers, args.variant)
                if not candidates:
                    candidates = _unique_texts(gold_answers)
                if not any(_normalize(c) in {_normalize(g) for g in gold_answers} for c in candidates):
                    candidates = _unique_texts(candidates + gold_answers)

                cases.append(
                    ProbeCase(
                        case_id=f"{args.variant}|cl{args.context_length}|book{book_index:04d}|{haystack_path.stem}|{test['test_name']}|d{float(depth_percent):.3f}",
                        variant=str(args.variant),
                        context_length=int(args.context_length),
                        book=haystack_path.name,
                        book_index=book_index,
                        test_name=str(test["test_name"]),
                        question_type=str(test.get("question_type", "")),
                        depth_percent=float(depth_percent),
                        question=question,
                        prompt_ids=[int(x) for x in prompt_ids],
                        gold_answers=gold_answers,
                        candidates=candidates,
                        prompt_tokens=len(prompt_ids),
                    )
                )
    return cases


def _pack_candidate_batch(
    tokenizer: Any,
    requests: list[CandidateRequest],
    pad_token_id: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    rows: list[tuple[list[int], list[int], list[bool], int]] = []
    for req in requests:
        prompt = tokenizer.decode(req.prompt_ids, skip_special_tokens=False)
        target_ids = tokenizer.encode(_maybe_space(prompt, req.candidate) + req.candidate, add_special_tokens=False)
        if not target_ids:
            rows.append(([pad_token_id], [pad_token_id], [False], 0))
            continue
        full_ids = list(req.prompt_ids) + [int(x) for x in target_ids]
        target_start = len(req.prompt_ids)
        if len(full_ids) > seq_len:
            crop = len(full_ids) - int(seq_len)
            full_ids = full_ids[crop:]
            target_start -= crop
        label_len = max(0, len(full_ids) - 1)
        target_mask = [(pos + 1) >= max(0, target_start) for pos in range(label_len)]
        if not any(target_mask):
            target_mask = [False] * label_len
        rows.append((full_ids[:-1], full_ids[1:], target_mask, len(target_ids)))

    max_len = max(max(1, len(row[0])) for row in rows)
    input_ids = torch.full((len(rows), max_len), pad_token_id, dtype=torch.long, device=device)
    labels = torch.full((len(rows), max_len), pad_token_id, dtype=torch.long, device=device)
    doc_ids = torch.full((len(rows), max_len), -1, dtype=torch.long, device=device)
    target_mask_t = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
    token_counts: list[int] = []
    for idx, (inputs, row_labels, target_mask, token_count) in enumerate(rows):
        length = len(inputs)
        input_ids[idx, :length] = torch.tensor(inputs, dtype=torch.long, device=device)
        labels[idx, :length] = torch.tensor(row_labels, dtype=torch.long, device=device)
        doc_ids[idx, :length] = 0
        target_mask_t[idx, : len(target_mask)] = torch.tensor(target_mask, dtype=torch.bool, device=device)
        token_counts.append(int(token_count))
    return input_ids, labels, doc_ids, target_mask_t, token_counts


@torch.no_grad()
def _score_requests(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: list[CandidateRequest],
    *,
    batch_size: int,
    pad_token_id: int,
    seq_len: int,
    device: torch.device,
) -> list[dict[str, float]]:
    mask_format = attention_mask_format_for_model(model)
    scores: list[dict[str, float]] = []
    for offset in range(0, len(requests), int(batch_size)):
        batch = requests[offset : offset + int(batch_size)]
        input_ids, labels, doc_ids, target_mask, token_counts = _pack_candidate_batch(
            tokenizer, batch, pad_token_id, int(seq_len), device
        )
        attention_mask = document_causal_attention_mask(doc_ids, mask_format=mask_format).to(device)
        logits = unwrap_logits(model(input_ids, attention_mask=attention_mask, use_cache=False)).float()
        logp = F.log_softmax(logits, dim=-1)
        gold_logp = logp.gather(-1, labels[:, :, None]).squeeze(-1)
        for row_idx in range(len(batch)):
            mask = target_mask[row_idx]
            selected = gold_logp[row_idx, mask]
            if selected.numel() == 0:
                scores.append({"mean_logp": float("-inf"), "sum_logp": float("-inf"), "nll": float("inf"), "first_token_nll": float("inf"), "token_count": 0.0})
            else:
                scores.append(
                    {
                        "mean_logp": float(selected.mean().item()),
                        "sum_logp": float(selected.sum().item()),
                        "nll": float((-selected).mean().item()),
                        "first_token_nll": float((-selected[0]).item()),
                        "token_count": float(token_counts[row_idx]),
                    }
                )
    return scores


def _summarize_case_scores(
    cases: list[ProbeCase],
    requests: list[CandidateRequest],
    scores: list[dict[str, float]],
    off_scores: list[dict[str, float]] | None,
    preview_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: list[list[tuple[str, dict[str, float], dict[str, float] | None]]] = [[] for _ in cases]
    for req, score_idx in zip(requests, range(len(requests))):
        grouped[req.case_index].append(
            (
                req.candidate,
                scores[score_idx],
                off_scores[score_idx] if off_scores is not None else None,
            )
        )

    records: list[dict[str, Any]] = []
    for case_idx, case in enumerate(cases):
        values = grouped[case_idx]
        gold_norms = {_normalize(g) for g in case.gold_answers}
        gold_values = [(cand, score, off) for cand, score, off in values if _normalize(cand) in gold_norms]
        if not gold_values:
            continue
        gold_cand, gold_score, gold_off = max(gold_values, key=lambda item: item[1]["mean_logp"])
        non_gold = [(cand, score) for cand, score, _off in values if _normalize(cand) not in gold_norms]
        sorted_values = sorted(values, key=lambda item: item[1]["mean_logp"], reverse=True)
        rank = 1 + sum(1 for _cand, score, _off in values if score["mean_logp"] > gold_score["mean_logp"])
        best_non_gold = max((score["mean_logp"] for _cand, score in non_gold), default=float("nan"))
        margin = gold_score["mean_logp"] - best_non_gold if not math.isnan(best_non_gold) else float("nan")
        off_rank = float("nan")
        off_margin = float("nan")
        off_gold_nll = float("nan")
        active_minus_off_nll = float("nan")
        rank_delta = float("nan")
        if gold_off is not None:
            off_gold_nll = float(gold_off["nll"])
            active_minus_off_nll = float(gold_score["nll"]) - off_gold_nll
            off_rank = 1 + sum(
                1
                for cand, _score, off in values
                if off is not None and _normalize(cand) not in gold_norms and off["mean_logp"] > gold_off["mean_logp"]
            )
            off_non_gold = [
                off["mean_logp"]
                for cand, _score, off in values
                if off is not None and _normalize(cand) not in gold_norms
            ]
            if off_non_gold:
                off_margin = gold_off["mean_logp"] - max(off_non_gold)
            rank_delta = float(rank) - float(off_rank)

        records.append(
            {
                "case_id": case.case_id,
                "variant": case.variant,
                "context_length": case.context_length,
                "question_type": case.question_type,
                "depth_percent": case.depth_percent,
                "test_name": case.test_name,
                "gold": gold_cand,
                "top_candidate": sorted_values[0][0] if sorted_values else "",
                "candidate_count": len(values),
                "gold_nll": float(gold_score["nll"]),
                "gold_first_token_nll": float(gold_score["first_token_nll"]),
                "gold_rank": float(rank),
                "candidate_top1": float(rank == 1),
                "candidate_mrr": 1.0 / float(rank),
                "gold_margin": float(margin),
                "adapter_off_gold_nll": off_gold_nll,
                "adapter_active_minus_off_gold_nll": active_minus_off_nll,
                "adapter_off_gold_rank": float(off_rank),
                "adapter_rank_delta": rank_delta,
                "adapter_margin_delta": float(margin - off_margin) if not math.isnan(off_margin) and not math.isnan(margin) else float("nan"),
            }
        )

    def mean(key: str, rows: list[dict[str, Any]]) -> float:
        vals = [float(row[key]) for row in rows if key in row and not math.isnan(float(row[key]))]
        return sum(vals) / len(vals) if vals else float("nan")

    def one(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "cases": len(rows),
            "gold_nll": mean("gold_nll", rows),
            "gold_first_token_nll": mean("gold_first_token_nll", rows),
            "candidate_top1_acc": mean("candidate_top1", rows),
            "candidate_mrr": mean("candidate_mrr", rows),
            "gold_rank_mean": mean("gold_rank", rows),
            "gold_margin": mean("gold_margin", rows),
            "adapter_active_minus_off_gold_nll": mean("adapter_active_minus_off_gold_nll", rows),
            "adapter_rank_delta": mean("adapter_rank_delta", rows),
            "adapter_margin_delta": mean("adapter_margin_delta", rows),
        }

    by_qtype: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_qtype.setdefault(str(record["question_type"]), []).append(record)
    summary = {
        "overall": one(records),
        "by_question_type": {key: one(rows) for key, rows in sorted(by_qtype.items())},
        "previews": records[: int(preview_limit)],
    }
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct NoLiMA answer-likelihood behavior probe for CURE checkpoints.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix-scripts-dir", default="/data/user/xnie012/pythonprojects/prefix/scripts")
    parser.add_argument("--needle-set-path", required=True)
    parser.add_argument("--haystack-dir", default="/data/user/xnie012/cache/nolima/haystack/rand_shuffle")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-candidate-tokens", type=int, default=8)
    parser.add_argument("--max-books", type=int, default=5)
    parser.add_argument("--max-experiments", type=int, default=10)
    parser.add_argument("--document-depth-percent-min", type=float, default=0.0)
    parser.add_argument("--document-depth-percent-max", type=float, default=100.0)
    parser.add_argument("--document-depth-percent-intervals", type=int, default=26)
    parser.add_argument("--sample-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--preview-limit", type=int, default=24)
    parser.add_argument("--save-records", action="store_true")
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
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise SystemExit("tokenizer must have pad_token_id or eos_token_id")

    cases = _build_cases(args, tokenizer)
    requests: list[CandidateRequest] = []
    for case_idx, case in enumerate(cases):
        for candidate in case.candidates:
            requests.append(CandidateRequest(case_idx, candidate, case.prompt_ids))

    try:
        scores = _score_requests(
            model,
            tokenizer,
            requests,
            batch_size=int(args.batch_size),
            pad_token_id=int(pad_token_id),
            seq_len=int(args.seq_len),
            device=device,
        )
        off_scores = None
        if adapters is not None:
            for handle in handles:
                handle.remove()
            handles = []
            off_scores = _score_requests(
                model,
                tokenizer,
                requests,
                batch_size=int(args.batch_size),
                pad_token_id=int(pad_token_id),
                seq_len=int(args.seq_len),
                device=device,
            )
    finally:
        for handle in handles:
            handle.remove()

    summary, records = _summarize_case_scores(cases, requests, scores, off_scores, int(args.preview_limit))
    result = {
        "protocol": {
            "model_tag": args.model_tag,
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "variant": args.variant,
            "context_length": int(args.context_length),
            "seq_len": int(args.seq_len),
            "needle_set_path": str(Path(args.needle_set_path).resolve()),
            "haystack_dir": str(Path(args.haystack_dir).resolve()),
            "max_books": args.max_books,
            "max_experiments": args.max_experiments,
            "depth_intervals": args.document_depth_percent_intervals,
            "sample_fraction": args.sample_fraction,
            "sample_offset": args.sample_offset,
            "cases": len(cases),
            "candidate_requests": len(requests),
            "candidate_set": "MC options for mc; character_set plus gold answers otherwise.",
            "rank_metric": "Candidates ranked by length-normalized answer mean logprob.",
            "adapter_delta": "active_minus_off; negative NLL delta means adapters improved gold likelihood.",
        },
        "summary": summary,
    }
    if args.save_records:
        result["records"] = records
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**result["protocol"], **result["summary"]["overall"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
