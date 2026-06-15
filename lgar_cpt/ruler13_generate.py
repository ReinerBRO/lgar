from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import LGARParams, Paths
from .hard_lgar import qwen_lgar_forward
from .mining import attention_mask_format_for_model, document_causal_attention_mask
from .modeling import load_qwen_causal_lm, load_tokenizer, unwrap_logits
from .router import SharedQueryRouter
from .utils import ensure_dir, read_jsonl, write_json


OFFICIAL_RULER13_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]


def _env_rank() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def _load_py_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_official_task_configs(ruler_dir: Path) -> dict[str, dict[str, Any]]:
    yaml_path = ruler_dir / "scripts" / "synthetic.yaml"
    constants_path = ruler_dir / "_assets" / "RULER" / "scripts" / "data" / "synthetic" / "constants.py"
    if not constants_path.exists():
        constants_path = ruler_dir / "scripts" / "data" / "synthetic" / "constants.py"
    tasks_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    tasks_base = _load_py_module(constants_path, "ruler13_synthetic_constants").TASKS
    merged: dict[str, dict[str, Any]] = {}
    for task_name, task_cfg in tasks_yaml.items():
        out = dict(task_cfg)
        out.update(tasks_base[task_cfg["task"]])
        merged[task_name] = out
    return merged


def _merge_lgar_params(ckpt_params: dict[str, Any], args: argparse.Namespace) -> LGARParams:
    values = {field.name: getattr(LGARParams(), field.name) for field in fields(LGARParams)}
    values.update({k: v for k, v in ckpt_params.items() if k in values})
    for key in (
        "seq_len",
        "short_window",
        "local_window",
        "router_target_budget",
        "final_global_budget",
        "routed_layer_fraction",
        "router_hidden_dim",
    ):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    return LGARParams(**values)


def _load_model_bundle(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, LGARParams, SharedQueryRouter | None, set[tuple[int, int]]]:
    tokenizer = load_tokenizer(args.model_path)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    ckpt_params = checkpoint.get("extra", {}).get("lgar_params", {})
    lgar_params = _merge_lgar_params(ckpt_params, args)
    model = load_qwen_causal_lm(
        args.model_path,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    if checkpoint.get("adapters"):
        from curcpt.forward import _register_adapter_hooks
        from curcpt.model_eval_utils import load_adapters_from_checkpoint

        adapters = load_adapters_from_checkpoint(checkpoint, model, device)
        model._curcpt_adapters = adapters  # keep modules alive for generation hooks
        model._curcpt_adapter_handles = _register_adapter_hooks(model, adapters)

    retrieval_heads: set[tuple[int, int]] = set()
    if args.mode == "rh_bottleneck":
        from curcpt.rh_bottleneck import resolve_retrieval_heads

        retrieval_heads = resolve_retrieval_heads(
            checkpoint,
            explicit_heads=args.retrieval_heads,
            heads_json=args.retrieval_heads_json,
            reference_checkpoint_path=args.reference_checkpoint_path,
        )
        if not retrieval_heads:
            raise SystemExit("rh_bottleneck evaluation requires retrieval heads")

    router: SharedQueryRouter | None = None
    if args.mode in {"router_aux", "routed"}:
        router = SharedQueryRouter(int(model.config.hidden_size), lgar_params.router_hidden_dim)
        router_state = checkpoint.get("router")
        if not router_state:
            raise SystemExit("routed evaluation requires router weights in checkpoint.pt")
        router.load_state_dict(router_state, strict=True)
        router.to(device)
        router.eval()
    return model, tokenizer, lgar_params, router, retrieval_heads


def _slice_prompt_ids(tokenizer: Any, prompt: str, seq_len: int) -> list[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > int(seq_len):
        ids = ids[-int(seq_len) :]
    return ids


def _decode_new_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


@torch.no_grad()
def _generate_full(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    seq_len: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    prompt_ids = [_slice_prompt_ids(tokenizer, prompt, seq_len) for prompt in prompts]
    batch = tokenizer.pad(
        {"input_ids": prompt_ids},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    model.config.use_cache = True
    generated = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_width = int(batch["input_ids"].shape[1])
    texts = [_decode_new_tokens(tokenizer, row[prompt_width:].tolist()) for row in generated]
    tokenizer.padding_side = original_padding_side
    return texts


def _pack_routed_batch(
    prompt_token_ids: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths = [len(ids) for ids in prompt_token_ids]
    max_len = max(lengths)
    input_ids = torch.full((len(prompt_token_ids), max_len), int(pad_token_id), dtype=torch.long, device=device)
    doc_ids = torch.full((len(prompt_token_ids), max_len), -1, dtype=torch.long, device=device)
    for row_idx, ids in enumerate(prompt_token_ids):
        row = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[row_idx, : len(ids)] = row
        doc_ids[row_idx, : len(ids)] = 0
    return input_ids, doc_ids, lengths


@torch.no_grad()
def _generate_routed(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    seq_len: int,
    max_new_tokens: int,
    device: torch.device,
    params: LGARParams,
    router: SharedQueryRouter,
    mode: str,
    target_budget: float | None,
    force_last_query_global: bool,
) -> list[str]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise SystemExit("tokenizer must expose pad_token_id or eos_token_id")
    eos_token_id = tokenizer.eos_token_id
    sequences = [_slice_prompt_ids(tokenizer, prompt, seq_len) for prompt in prompts]
    generated: list[list[int]] = [[] for _ in sequences]
    finished = [False for _ in sequences]
    mask_format = attention_mask_format_for_model(model)

    for _step in range(int(max_new_tokens)):
        input_ids, doc_ids, lengths = _pack_routed_batch(sequences, pad_token_id, device)
        force_global_query_mask = None
        if force_last_query_global:
            force_global_query_mask = torch.zeros_like(doc_ids, dtype=torch.bool)
            for row_idx, length in enumerate(lengths):
                if not finished[row_idx] and length > 0:
                    force_global_query_mask[row_idx, int(length) - 1] = True
        forward = qwen_lgar_forward(
            model=model,
            input_ids=input_ids,
            full_attention_mask=document_causal_attention_mask(doc_ids, mask_format=mask_format),
            doc_ids=doc_ids,
            router=router,
            params=params,
            routing_valid_mask=doc_ids >= 0,
            force_global_query_mask=force_global_query_mask,
            mode=mode,
            target_budget=target_budget,
        )
        logits = forward.logits
        all_done = True
        for row_idx, length in enumerate(lengths):
            if finished[row_idx]:
                continue
            next_token = int(torch.argmax(logits[row_idx, length - 1]).item())
            generated[row_idx].append(next_token)
            if eos_token_id is not None and next_token == int(eos_token_id):
                finished[row_idx] = True
                continue
            sequences[row_idx].append(next_token)
            if len(sequences[row_idx]) > int(seq_len):
                sequences[row_idx] = sequences[row_idx][-int(seq_len) :]
            all_done = False
        if all_done:
            break
    return [_decode_new_tokens(tokenizer, token_ids) for token_ids in generated]


def _load_existing_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    rows = read_jsonl(path)
    return {int(row["index"]) for row in rows if "index" in row}


def _task_prompt(row: dict[str, Any], append_answer_prefix: bool) -> str:
    prompt = str(row["input"])
    if append_answer_prefix and row.get("answer_prefix"):
        prompt += str(row["answer_prefix"])
    return prompt


def _run_task(
    task_name: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    batch_size: int,
    generator_fn: Any,
    max_new_tokens: int,
    seq_len: int,
    append_answer_prefix: bool,
    rank: int,
) -> dict[str, Any]:
    ensure_dir(output_path.parent)
    existing = _load_existing_indices(output_path)
    pending = [row for row in rows if int(row["index"]) not in existing]
    start = time.time()
    if not pending:
        return {"task": task_name, "pending": 0, "seconds": 0.0, "rank": rank}
    with output_path.open("a", encoding="utf-8") as fout:
        for offset in range(0, len(pending), int(batch_size)):
            batch_rows = pending[offset : offset + int(batch_size)]
            prompts = [_task_prompt(row, append_answer_prefix=append_answer_prefix) for row in batch_rows]
            preds = generator_fn(prompts, seq_len=seq_len, max_new_tokens=max_new_tokens)
            for row, pred in zip(batch_rows, preds):
                item = dict(row)
                item["pred"] = pred
                item["others"] = item.get("others", {}) or {}
                fout.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "task": task_name,
        "pending": len(pending),
        "seconds": round(time.time() - start, 3),
        "rank": rank,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate official RULER-13 predictions from project checkpoints.")
    defaults = Paths()
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--ruler-dir", default=defaults.ruler_dir)
    parser.add_argument("--data-dir", required=True, help="Official RULER data root for one length, e.g. generated/official1k")
    parser.add_argument("--save-dir", required=True, help="Prediction output directory for official scorer")
    parser.add_argument("--tasks", default=",".join(OFFICIAL_RULER13_TASKS))
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mode", choices=["full", "router_aux", "routed", "rh_bottleneck", "rh_layer_bottleneck"], default="full")
    parser.add_argument("--target-budget", type=float, default=None)
    parser.add_argument("--force-last-query-global", action="store_true")
    parser.add_argument("--append-answer-prefix", action="store_true")
    parser.add_argument("--retrieval-heads", default=None, help="Comma-separated L:H heads for rh_bottleneck eval.")
    parser.add_argument("--retrieval-heads-json", default=None, help="Ablation JSON containing retrieval_heads.")
    parser.add_argument("--reference-checkpoint-path", default=None, help="Checkpoint to read retrieval heads from when this checkpoint has none.")
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
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    task_configs = _load_official_task_configs(ruler_dir)
    requested_tasks = [task.strip() for task in str(args.tasks).split(",") if task.strip()]
    for task in requested_tasks:
        if task not in task_configs:
            raise SystemExit(f"unknown RULER task: {task}")

    model, tokenizer, lgar_params, router, retrieval_heads = _load_model_bundle(args, device)
    seq_len = int(lgar_params.seq_len if args.seq_len is None else args.seq_len)

    if args.mode == "full":
        def generator(prompts: list[str], seq_len: int, max_new_tokens: int) -> list[str]:
            return _generate_full(model, tokenizer, prompts, seq_len, max_new_tokens, device)
    elif args.mode in {"rh_bottleneck", "rh_layer_bottleneck"}:
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
    else:
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

    summaries: list[dict[str, Any]] = []
    for task_name in requested_tasks:
        task_rows = read_jsonl(data_dir / task_name / f"{args.subset}.jsonl", limit=args.max_samples)
        shard_rows = task_rows[rank::world_size]
        output_path = save_dir / f"{task_name}-{rank}.jsonl"
        summary = _run_task(
            task_name=task_name,
            rows=shard_rows,
            output_path=output_path,
            batch_size=args.batch_size,
            generator_fn=generator,
            max_new_tokens=int(task_configs[task_name]["tokens_to_generate"]),
            seq_len=seq_len,
            append_answer_prefix=bool(args.append_answer_prefix),
            rank=rank,
        )
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    meta = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": str(device),
        "mode": args.mode,
        "rh_bottleneck_retrieval_heads": sorted([list(head) for head in retrieval_heads]),
        "checkpoint_path": args.checkpoint_path,
        "data_dir": str(data_dir),
        "save_dir": str(save_dir),
        "append_answer_prefix": bool(args.append_answer_prefix),
        "batch_size": int(args.batch_size),
        "lgar_params": asdict(lgar_params),
        "tasks": summaries,
    }
    write_json(save_dir / f"rank{rank:02d}.meta.json", meta)


if __name__ == "__main__":
    main()
