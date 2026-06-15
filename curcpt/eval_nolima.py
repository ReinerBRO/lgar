from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model_eval_utils import load_model_tokenizer_for_eval
from .rh_bottleneck import greedy_generate as greedy_generate_eval
from .rh_bottleneck import resolve_retrieval_heads


def _load_prefix_nolima(prefix_scripts_dir: str) -> Any:
    path = Path(prefix_scripts_dir) / "eval_nolima.py"
    project_root = path.resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location("prefix_eval_nolima", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import prefix NoLiMa evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_stride(sample_fraction: float) -> int:
    if sample_fraction <= 0 or sample_fraction > 1:
        raise ValueError("--sample-fraction must be in (0, 1]")
    return 1 if sample_fraction >= 1 else max(1, round(1.0 / sample_fraction))


def _truncate_prompt_ids(prompt_ids: list[int], max_prompt_tokens: int) -> list[int]:
    if len(prompt_ids) <= max_prompt_tokens:
        return prompt_ids
    half = max_prompt_tokens // 2
    return prompt_ids[:half] + prompt_ids[-(max_prompt_tokens - half) :]


@torch.no_grad()
def _greedy_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    seq_len: int,
    eval_mode: str,
    retrieval_heads: set[tuple[int, int]],
    local_window: int,
) -> str:
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    return greedy_generate_eval(
        model,
        tokenizer,
        [prompt],
        seq_len,
        max_new_tokens,
        device,
        eval_mode=eval_mode,
        retrieval_heads=retrieval_heads,
        local_window=local_window,
        stop_on_newline=True,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CURE checkpoints on prefix-style NoLiMa.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix-scripts-dir", default="/data/user/xnie012/pythonprojects/prefix/scripts")
    parser.add_argument("--needle-set-path", default="/data/user/xnie012/cache/nolima/needlesets/needle_set_hard.json")
    parser.add_argument("--haystack-dir", default="/data/user/xnie012/cache/nolima/haystack/rand_shuffle")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--metric", default="contains")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--preview-limit", type=int, default=12)
    parser.add_argument("--max-books", type=int, default=5)
    parser.add_argument("--max-experiments", type=int, default=8)
    parser.add_argument("--document-depth-percent-min", type=float, default=0.0)
    parser.add_argument("--document-depth-percent-max", type=float, default=100.0)
    parser.add_argument("--document-depth-percent-intervals", type=int, default=5)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--eval-mode", choices=["full", "rh_bottleneck", "rh_layer_bottleneck"], default="full")
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--retrieval-heads", default=None, help="Comma-separated L:H heads for rh_bottleneck eval.")
    parser.add_argument("--retrieval-heads-json", default=None, help="Ablation JSON containing retrieval_heads.")
    parser.add_argument("--reference-checkpoint-path", default=None, help="Checkpoint to read retrieval heads from when this checkpoint has none.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    prefix_nolima = _load_prefix_nolima(args.prefix_scripts_dir)
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

    needle_set = json.loads(Path(args.needle_set_path).read_text(encoding="utf-8"))
    tests = prefix_nolima._build_tests(needle_set, max_experiments=args.max_experiments)
    haystack_paths = sorted(path for path in Path(args.haystack_dir).glob("*.txt") if not path.name.startswith("._"))
    if args.max_books is not None:
        haystack_paths = haystack_paths[: int(args.max_books)]
    if not haystack_paths:
        raise FileNotFoundError(f"no haystack books found in {args.haystack_dir}")

    depth_values = np.linspace(
        float(args.document_depth_percent_min),
        float(args.document_depth_percent_max),
        int(args.document_depth_percent_intervals),
    )
    sample_stride = _sample_stride(float(args.sample_fraction))
    sample_offset = int(args.sample_offset) % sample_stride
    total_examples_before_sampling = len(haystack_paths) * len(tests) * len(depth_values)

    num_examples = 0
    selected_examples = 0
    num_correct = 0
    num_correct_norm = 0
    prompt_tokens: list[int] = []
    overlength_examples = 0
    previews: list[dict[str, Any]] = []
    max_prompt_tokens = int(args.seq_len) - int(args.max_new_tokens)
    candidate_index = 0

    try:
        for book_index, haystack_path in enumerate(haystack_paths):
            haystack = prefix_nolima.BookHaystack(str(haystack_path))
            for test in tests:
                rng = np.random.default_rng(int(test["seed"]) + book_index)
                for depth_percent in depth_values:
                    current_index = candidate_index
                    candidate_index += 1
                    if current_index % sample_stride != sample_offset:
                        continue
                    selected_examples += 1
                    needle = str(test["needle"])
                    question = str(test["retrieval_question"])
                    gold_answers = list(test["gold_answers"])
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
                    prompt_tokens.append(len(prompt_ids))
                    if len(prompt_ids) + int(args.max_new_tokens) > int(args.seq_len):
                        overlength_examples += 1
                        prompt_ids = _truncate_prompt_ids(prompt_ids, max_prompt_tokens)

                    prediction = _greedy_generate(
                        model,
                        tokenizer,
                        prompt_ids,
                        int(args.max_new_tokens),
                        device,
                        int(args.seq_len),
                        args.eval_mode,
                        retrieval_heads,
                        int(args.local_window),
                    )
                    correct = prefix_nolima._official_metric(prediction, gold_answers, args.metric)
                    correct_norm = prefix_nolima._normalized_metric(prediction, gold_answers, args.metric)
                    num_examples += 1
                    num_correct += int(correct)
                    num_correct_norm += int(correct_norm)
                    if len(previews) < int(args.preview_limit):
                        previews.append(
                            {
                                "book": haystack_path.name,
                                "depth_percent": round(float(depth_percent), 3),
                                "gold_answers": gold_answers,
                                "prediction": prediction,
                                "prompt_tokens": len(prompt_ids),
                                "question": question,
                                "test_name": test["test_name"],
                                "correct_official": bool(correct),
                                "correct_normalized": bool(correct_norm),
                            }
                        )
    finally:
        for handle in handles:
            handle.remove()

    result = {
        "checkpoint": str(Path(args.checkpoint_path).resolve()),
        "adapters_active": adapters is not None,
        "num_retrieval_heads": adapters.num_heads() if adapters is not None else 0,
        "eval_mode": args.eval_mode,
        "rh_bottleneck_retrieval_heads": sorted([list(head) for head in retrieval_heads]),
        "context_length": int(args.context_length),
        "seq_len": int(args.seq_len),
        "needle_set_path": str(Path(args.needle_set_path).resolve()),
        "haystack_dir": str(Path(args.haystack_dir).resolve()),
        "num_books": len(haystack_paths),
        "num_tests": len(tests),
        "num_examples": num_examples,
        "selected_examples_before_overlength": selected_examples,
        "total_examples_before_sampling": total_examples_before_sampling,
        "sample_fraction": float(args.sample_fraction),
        "sample_stride": sample_stride,
        "sample_offset": sample_offset,
        "document_depth_percent_intervals": int(args.document_depth_percent_intervals),
        "max_books": args.max_books,
        "max_experiments": args.max_experiments,
        "metric": args.metric,
        "max_new_tokens": int(args.max_new_tokens),
        "preview_limit": int(args.preview_limit),
        "official_accuracy": round(100.0 * num_correct / max(num_examples, 1), 4),
        "normalized_accuracy": round(100.0 * num_correct_norm / max(num_examples, 1), 4),
        "overlength_examples": overlength_examples,
        "prompt_tokens": {
            "max": max(prompt_tokens) if prompt_tokens else 0,
            "mean": round(float(np.mean(prompt_tokens)), 3) if prompt_tokens else 0.0,
            "min": min(prompt_tokens) if prompt_tokens else 0,
        },
        "previews": previews,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
