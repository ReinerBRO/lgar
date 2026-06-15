from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .model_eval_utils import load_model_tokenizer_for_eval


LONGBENCH_ROOT = Path(os.environ.get("LONGBENCH_ROOT", "/data/user/xnie012/cache/data/LongBench"))
BABILONG_ROOT = Path(os.environ.get("BABILONG_ROOT", "/data/user/xnie012/cache/data/babilong"))
LONGEVAL_ROOT = Path(os.environ.get("LONGEVAL_ROOT", "/data/user/xnie012/cache/data/longeval/evaluation"))

LONGBENCH_TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
)
LONGBENCH_E_TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)
BABILONG_TASKS = ("qa1", "qa2", "qa3", "qa4", "qa5")
LONGEVAL_TASKS = ("topics", "lines")

LONGBENCH_PROMPTS = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}
LONGBENCH_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}
LONGBENCH_METRICS = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "multifieldqa_zh": "qa_f1_zh",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "triviaqa": "qa_f1",
    "dureader": "rouge_l_zh",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "vcsum": "rouge_l_zh",
    "trec": "classification",
    "samsum": "rouge_l",
    "lsht": "classification",
    "passage_count": "count",
    "passage_retrieval_en": "retrieval",
    "passage_retrieval_zh": "retrieval_zh",
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}
BABILONG_TASK_LABELS = {
    "qa1": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa2": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa3": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa4": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa5": ["Bill", "Fred", "Jeff", "Mary", "apple", "football", "milk"],
}
LONGEVAL_TOPIC_LENGTHS = (5, 10, 15, 20, 25)
LONGEVAL_LINE_LENGTHS = (200, 300, 400, 500, 600, 680)


@dataclass(frozen=True)
class EvalRow:
    benchmark: str
    task: str
    subtask: str
    index: int
    prompt: str
    targets: list[str]
    metric: str
    max_new_tokens: int
    metadata: dict[str, Any]


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        limit = None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _read_json_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path, limit)
    if limit is not None and limit <= 0:
        limit = None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("data", "test", "validation", "examples", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        else:
            raise ValueError(f"JSON file has no list payload: {path}")
    else:
        raise ValueError(f"unsupported JSON payload in {path}: {type(payload).__name__}")
    if limit is not None:
        rows = rows[: int(limit)]
    return [dict(row) for row in rows]


def _load_longbench_task(task: str, max_examples: int, *, use_e: bool = False) -> list[EvalRow]:
    base_task = task[:-2] if task.endswith("_e") else task
    file_task = f"{base_task}_e" if use_e and not task.endswith("_e") else task
    candidates = [
        LONGBENCH_ROOT / file_task / "test.jsonl",
        LONGBENCH_ROOT / f"{file_task}.jsonl",
        LONGBENCH_ROOT / base_task / "test.jsonl",
        LONGBENCH_ROOT / f"{base_task}.jsonl",
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"LongBench task {file_task!r} not found under {LONGBENCH_ROOT}")
    rows = _read_jsonl(path, max_examples)
    out: list[EvalRow] = []
    for idx, row in enumerate(rows):
        prompt = LONGBENCH_PROMPTS[base_task].format(context=row["context"], input=row["input"])
        out.append(
            EvalRow(
                benchmark="longbench",
                task=file_task,
                subtask=file_task,
                index=idx,
                prompt=prompt,
                targets=[str(x) for x in row.get("answers", [])],
                metric=LONGBENCH_METRICS[base_task],
                max_new_tokens=LONGBENCH_MAX_NEW_TOKENS[base_task],
                metadata={"source_length": row.get("length"), "id": row.get("_id"), "all_classes": row.get("all_classes", [])},
            )
        )
    return out


def _load_babilong_task(task: str, max_examples: int, *, length_key: str = "") -> list[EvalRow]:
    candidates: list[Path]
    if length_key:
        candidates = [
            BABILONG_ROOT / task / f"{length_key}.json",
            BABILONG_ROOT / task / f"{length_key}.jsonl",
            BABILONG_ROOT / f"{task}_{length_key}.json",
            BABILONG_ROOT / f"{task}_{length_key}.jsonl",
        ]
    else:
        candidates = [
            BABILONG_ROOT / task / "test.jsonl",
            BABILONG_ROOT / task / "1k.json",
            BABILONG_ROOT / task / "1k.jsonl",
        ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        expected = ", ".join(str(item) for item in candidates)
        raise FileNotFoundError(f"BABILong task {task!r} length={length_key or '<default>'} not found. Expected one of: {expected}")
    rows = _read_json_records(path, max_examples)
    out: list[EvalRow] = []
    for idx, row in enumerate(rows):
        context = row.get("context", row.get("background", row.get("input", "")))
        if isinstance(context, list):
            context = "\n".join(str(x) for x in context)
        question = str(row.get("question", row.get("query", ""))).strip()
        answer_value = row.get("answer", row.get("target", row.get("answers", "")))
        if isinstance(answer_value, list):
            answer = str(answer_value[0]) if answer_value else ""
        else:
            answer = str(answer_value)
        prompt = (
            f"{str(context).strip()}\n"
            f"Question: {question}\n"
            "Answer:"
        )
        out.append(
            EvalRow(
                benchmark="babilong",
                task=task,
                subtask=task,
                index=idx,
                prompt=prompt,
                targets=[answer],
                metric="babilong_exact",
                max_new_tokens=32,
                metadata={"question": question, "length": length_key or path.stem, "path": str(path)},
            )
        )
    return out


def _load_longeval_rows(task: str, max_examples: int) -> list[EvalRow]:
    if task == "topics":
        lengths = LONGEVAL_TOPIC_LENGTHS
        max_new_tokens = 50
    elif task == "lines":
        lengths = LONGEVAL_LINE_LENGTHS
        max_new_tokens = 100
    else:
        raise ValueError(f"unsupported LongEval task: {task}")

    out: list[EvalRow] = []
    for length in lengths:
        path = LONGEVAL_ROOT / task / "testcases" / f"{length}_{task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"LongEval split missing: {path}")
        for idx, row in enumerate(_read_jsonl(path, max_examples)):
            targets: list[str]
            if task == "topics":
                targets = [str(row["topics"][0])]
                metric = "contains_norm"
            else:
                targets = [str(row["expected_number"])]
                metric = "first_number"
            out.append(
                EvalRow(
                    benchmark="longeval",
                    task=task,
                    subtask=f"{task}_{length}",
                    index=idx,
                    prompt=str(row["prompt"]),
                    targets=targets,
                    metric=metric,
                    max_new_tokens=max_new_tokens,
                    metadata={"length": length},
                )
            )
    return out


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def _normalize_zh_answer(text: str) -> str:
    cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    punctuation = set(string.punctuation + cn_punctuation)
    return "".join(ch for ch in text.lower() if ch not in punctuation and not ch.isspace())


def _f1_score(prediction_tokens: list[str], target_tokens: list[str]) -> float:
    if not prediction_tokens or not target_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(target_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return (2.0 * precision * recall) / max(precision + recall, 1e-8)


def _qa_f1(prediction: str, targets: list[str]) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    return max((_f1_score(pred_tokens, _normalize_answer(target).split()) for target in targets), default=0.0)


def _qa_f1_zh(prediction: str, targets: list[str]) -> float:
    pred_tokens = list(_normalize_zh_answer(prediction))
    return max((_f1_score(pred_tokens, list(_normalize_zh_answer(target))) for target in targets), default=0.0)


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def _rouge_l(prediction: str, targets: list[str]) -> float:
    pred_tokens = prediction.split()
    best = 0.0
    for target in targets:
        target_tokens = str(target).split()
        lcs = _lcs_len(pred_tokens, target_tokens)
        if lcs == 0:
            continue
        precision = lcs / max(len(pred_tokens), 1)
        recall = lcs / max(len(target_tokens), 1)
        best = max(best, (2 * precision * recall) / max(precision + recall, 1e-8))
    return best


def _rouge_l_zh(prediction: str, targets: list[str]) -> float:
    pred_tokens = list(_normalize_zh_answer(prediction))
    best = 0.0
    for target in targets:
        target_tokens = list(_normalize_zh_answer(str(target)))
        lcs = _lcs_len(pred_tokens, target_tokens)
        if lcs == 0:
            continue
        precision = lcs / max(len(pred_tokens), 1)
        recall = lcs / max(len(target_tokens), 1)
        best = max(best, (2 * precision * recall) / max(precision + recall, 1e-8))
    return best


def _classification_score(prediction: str, targets: list[str], all_classes: list[str]) -> float:
    if not targets:
        return 0.0
    ground_truth = targets[0]
    matches = [class_name for class_name in all_classes if str(class_name) in prediction]
    matches = [item for item in matches if not (item in ground_truth and item != ground_truth)]
    if ground_truth in matches:
        return 1.0 / max(len(matches), 1)
    return 0.0


def _count_score(prediction: str, targets: list[str]) -> float:
    if not targets:
        return 0.0
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1 for number in numbers if str(number) == str(targets[0])) / len(numbers)


def _retrieval_score(prediction: str, targets: list[str], *, zh: bool = False) -> float:
    if not targets:
        return 0.0
    pattern = r"段落(\d+)" if zh else r"Paragraph (\d+)"
    matches = re.findall(pattern, targets[0])
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1 for number in numbers if str(number) == str(ground_truth_id)) / len(numbers)


def _code_sim_score(prediction: str, targets: list[str]) -> float:
    if not targets:
        return 0.0
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    import difflib

    return difflib.SequenceMatcher(None, candidate, targets[0]).ratio()


def _score_babilong_answer(task: str, prediction: str, target: str, question: str) -> float:
    text = prediction.strip().lower()
    text = text.split(".", 1)[0]
    task_key = task.split("_", 1)[0]
    labels = {label.lower() for label in BABILONG_TASK_LABELS.get(task_key, [])}
    if not labels:
        return 1.0 if target.lower() in text else 0.0
    labels_in_question = {label for label in labels if label in question.lower()}
    labels_in_output = {label for label in labels if label in text}
    labels_in_output -= labels_in_question
    return 1.0 if target.lower() in labels_in_output and len(labels_in_output) == 1 else 0.0


def _parse_first_number(text: str) -> str | None:
    match = re.search(r"<\s*(\d+)\s*>", text)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d+)\b", text)
    return match.group(1) if match else None


def _score_row(row: EvalRow, prediction: str) -> float:
    if row.task.rsplit("_", 1)[0] in {"trec", "triviaqa", "samsum", "lsht"} or row.task in {"trec", "triviaqa", "samsum", "lsht"}:
        prediction = prediction.lstrip("\n").split("\n", 1)[0]
    if row.metric == "qa_f1":
        return _qa_f1(prediction, row.targets)
    if row.metric == "qa_f1_zh":
        return _qa_f1_zh(prediction, row.targets)
    if row.metric == "rouge_l":
        return _rouge_l(prediction, row.targets)
    if row.metric == "rouge_l_zh":
        return _rouge_l_zh(prediction, row.targets)
    if row.metric == "classification":
        return _classification_score(prediction, row.targets, [str(item) for item in row.metadata.get("all_classes", [])])
    if row.metric == "count":
        return _count_score(prediction, row.targets)
    if row.metric == "retrieval":
        return _retrieval_score(prediction, row.targets, zh=False)
    if row.metric == "retrieval_zh":
        return _retrieval_score(prediction, row.targets, zh=True)
    if row.metric == "code_sim":
        return _code_sim_score(prediction, row.targets)
    if row.metric == "babilong_exact":
        return _score_babilong_answer(row.task, prediction, row.targets[0], str(row.metadata.get("question", "")))
    if row.metric == "contains_norm":
        return 1.0 if _normalize_answer(row.targets[0]) in _normalize_answer(prediction) else 0.0
    if row.metric == "first_number":
        return 1.0 if _parse_first_number(prediction) == row.targets[0] else 0.0
    raise ValueError(f"unsupported metric: {row.metric}")


def _truncate_prompt_ids(prompt_ids: list[int], max_prompt_tokens: int) -> tuple[list[int], bool]:
    if len(prompt_ids) <= max_prompt_tokens:
        return prompt_ids, False
    half = max_prompt_tokens // 2
    return prompt_ids[:half] + prompt_ids[-(max_prompt_tokens - half) :], True


@torch.no_grad()
def _generate_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    seq_len: int,
    device: torch.device,
    stop_on_newline: bool,
    use_cache: bool,
) -> list[tuple[str, int, bool]]:
    if not prompts:
        return []

    prompt_id_rows: list[list[int]] = []
    prompt_tokens: list[int] = []
    truncated_flags: list[bool] = []
    max_prompt_tokens = max(seq_len - max_new_tokens, 1)
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids, truncated = _truncate_prompt_ids(prompt_ids, max_prompt_tokens)
        prompt_id_rows.append(prompt_ids)
        prompt_tokens.append(len(prompt_ids))
        truncated_flags.append(truncated)

    max_width = max(len(row) for row in prompt_id_rows)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full(
        (len(prompt_id_rows), max_width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((len(prompt_id_rows), max_width), dtype=torch.long, device=device)
    for row_idx, prompt_ids in enumerate(prompt_id_rows):
        width = len(prompt_ids)
        input_ids[row_idx, -width:] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        attention_mask[row_idx, -width:] = 1

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        use_cache=bool(use_cache),
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    results: list[tuple[str, int, bool]] = []
    new_token_rows = generated[:, input_ids.size(1):]
    for row_idx in range(new_token_rows.size(0)):
        text = tokenizer.decode(new_token_rows[row_idx].tolist(), skip_special_tokens=True).strip()
        if stop_on_newline and "\n" in text:
            text = text.split("\n", 1)[0].strip()
        results.append((text, prompt_tokens[row_idx], truncated_flags[row_idx]))
    return results


def _dist_context() -> tuple[int, int, torch.device]:
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
    return rank, world_size, device


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _parse_checkpoints(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--checkpoint must be NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    if not out:
        raise SystemExit("at least one --checkpoint NAME=PATH is required")
    return out


def _load_rows_for_benchmark(
    benchmark: str,
    tasks: list[str],
    max_examples: int,
    *,
    longbench_e: bool = False,
    babilong_length: str = "",
) -> dict[str, list[EvalRow]]:
    rows: dict[str, list[EvalRow]] = {}
    if benchmark == "longbench":
        task_list = tasks or list(LONGBENCH_E_TASKS if longbench_e else LONGBENCH_TASKS)
        for task in task_list:
            rows[task] = _load_longbench_task(task, max_examples, use_e=longbench_e)
        return rows
    if benchmark == "babilong":
        task_list = tasks or list(BABILONG_TASKS)
        for task in task_list:
            rows[task] = _load_babilong_task(task, max_examples, length_key=babilong_length)
        return rows
    if benchmark == "longeval":
        task_list = tasks or list(LONGEVAL_TASKS)
        for task in task_list:
            rows[task] = _load_longeval_rows(task, max_examples)
        return rows
    raise ValueError(f"unsupported benchmark: {benchmark}")


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(str(record["task"]), []).append(record)

    def one(items: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [float(item["score"]) for item in items]
        return {
            "examples": len(items),
            "score": 100.0 * sum(scores) / max(len(scores), 1),
            "truncated_examples": sum(1 for item in items if item.get("truncated")),
            "avg_prompt_tokens": sum(int(item["prompt_tokens"]) for item in items) / max(len(items), 1),
        }

    return {
        "overall": one(records),
        "by_task": {task: one(items) for task, items in sorted(by_task.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CURE checkpoints on LongBench, LongEval, and BABILong.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--benchmark", choices=["longbench", "babilong", "longeval"], required=True)
    parser.add_argument("--tasks", default="", help="Comma-separated task list. Empty uses cached default tasks.")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--longbench-e", action="store_true", help="Evaluate LongBench-E files such as qasper_e.jsonl.")
    parser.add_argument("--babilong-length", default="", help="BABILong file key, e.g. 4k/8k/12k/16k/24k/32k.")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--preview-limit", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--use-cache", action="store_true", help="Enable KV cache during generation.")
    args = parser.parse_args()

    rank, world_size, device = _dist_context()
    checkpoints = _parse_checkpoints(args.checkpoint)
    requested_tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    rows_by_task = _load_rows_for_benchmark(
        args.benchmark,
        requested_tasks,
        int(args.max_examples),
        longbench_e=bool(args.longbench_e),
        babilong_length=str(args.babilong_length),
    )

    output = Path(args.output)
    rank_dir = output.parent / f"{output.stem}_rank_parts"
    rank_dir.mkdir(parents=True, exist_ok=True)
    expected_rank_records = sum(len(rows[rank::world_size]) for rows in rows_by_task.values())

    for model_name, checkpoint_path in checkpoints.items():
        part = rank_dir / f"{model_name}.rank{rank:02d}.json"
        if part.exists():
            try:
                existing = json.loads(part.read_text(encoding="utf-8"))
                existing_records = existing.get("records", [])
            except Exception:
                existing_records = []
            if isinstance(existing_records, list) and len(existing_records) == expected_rank_records:
                print(
                    json.dumps(
                        {
                            "event": "model_skip_existing_rank_part",
                            "model": model_name,
                            "rank": rank,
                            "records": len(existing_records),
                            "path": str(part),
                            "use_cache": bool(args.use_cache),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue

        print(
            json.dumps(
                {
                    "event": "model_start",
                    "model": model_name,
                    "rank": rank,
                    "benchmark": args.benchmark,
                    "seq_len": int(args.seq_len),
                    "output": str(output),
                    "batch_size": int(args.batch_size),
                    "use_cache": bool(args.use_cache),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        model, tokenizer, _adapters, handles, _checkpoint = load_model_tokenizer_for_eval(
            args.model_path,
            checkpoint_path,
            device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        model.config.use_cache = bool(args.use_cache)
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = bool(args.use_cache)
        records: list[dict[str, Any]] = []
        try:
            for task, rows in rows_by_task.items():
                shard = rows[rank::world_size]
                stop_on_newline = args.benchmark != "longbench" or task != "qmsum"
                for batch_start in range(0, len(shard), int(args.batch_size)):
                    batch_rows = shard[batch_start: batch_start + int(args.batch_size)]
                    batch_results = _generate_batch(
                        model,
                        tokenizer,
                        [row.prompt for row in batch_rows],
                        max_new_tokens=batch_rows[0].max_new_tokens,
                        seq_len=int(args.seq_len),
                        device=device,
                        stop_on_newline=stop_on_newline,
                        use_cache=bool(args.use_cache),
                    )
                    for batch_offset, row in enumerate(batch_rows):
                        local_idx = batch_start + batch_offset
                        prediction, prompt_tokens, truncated = batch_results[batch_offset]
                        score = _score_row(row, prediction)
                        record = {
                            "model": model_name,
                            "benchmark": args.benchmark,
                            "task": row.task,
                            "subtask": row.subtask,
                            "index": row.index,
                            "score": float(score),
                            "prediction": prediction,
                            "targets": row.targets,
                            "prompt_tokens": int(prompt_tokens),
                            "truncated": bool(truncated),
                            "metric": row.metric,
                            "rank": rank,
                        }
                        if len(records) < int(args.preview_limit):
                            record["prompt_preview"] = row.prompt[:400]
                        records.append(record)
                        if (local_idx + 1) % 10 == 0 or (local_idx + 1) == len(shard):
                            print(
                                json.dumps(
                                    {
                                        "event": "progress",
                                        "model": model_name,
                                        "benchmark": args.benchmark,
                                        "seq_len": int(args.seq_len),
                                        "task": task,
                                        "rank": rank,
                                        "done": local_idx + 1,
                                        "total": len(shard),
                                        "batch_size": int(args.batch_size),
                                        "use_cache": bool(args.use_cache),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                print(
                    json.dumps(
                        {
                            "event": "task_done",
                            "model": model_name,
                            "benchmark": args.benchmark,
                            "seq_len": int(args.seq_len),
                            "task": task,
                            "rank": rank,
                            "items": len(shard),
                            "use_cache": bool(args.use_cache),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        finally:
            for handle in handles:
                handle.remove()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        part.write_text(json.dumps({"model": model_name, "rank": rank, "records": records}, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "event": "model_done",
                    "model": model_name,
                    "rank": rank,
                    "benchmark": args.benchmark,
                    "seq_len": int(args.seq_len),
                    "records": len(records),
                    "use_cache": bool(args.use_cache),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    _barrier()
    if rank == 0:
        summaries: dict[str, Any] = {}
        for model_name in checkpoints:
            merged: list[dict[str, Any]] = []
            for part in sorted(rank_dir.glob(f"{model_name}.rank*.json")):
                merged.extend(json.loads(part.read_text(encoding="utf-8"))["records"])
            summaries[model_name] = _summarize(merged)
        result = {
            "benchmark": args.benchmark,
            "roots": {
                "longbench": str(LONGBENCH_ROOT),
                "babilong": str(BABILONG_ROOT),
                "longeval": str(LONGEVAL_ROOT),
            },
            "max_examples_per_task": int(args.max_examples),
            "seq_len": int(args.seq_len),
            "longbench_e": bool(args.longbench_e),
            "babilong_length": str(args.babilong_length),
            "use_cache": bool(args.use_cache),
            "checkpoints": checkpoints,
            "summaries": summaries,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    _barrier()


if __name__ == "__main__":
    main()
