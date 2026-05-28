from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .config import LGARParams, Paths, TrainParams
from .data import PackedFineWebDataset, PackedSequenceSignalDataset, prepare_qwen_fineweb_cache
from .evaluate import evaluate_model, to_torch_batch
from .hard_lgar import qwen_lgar_forward
from .mining import (
    LSDLabelBatch,
    attention_mask_format_for_model,
    ce_from_logits,
    document_causal_attention_mask,
    lsd_audit_examples,
    mine_lsd_labels,
)
from .modeling import load_qwen_causal_lm, load_tokenizer
from .router import SharedQueryRouter
from .utils import ensure_dir, env_snapshot, set_seed, write_json


def _linear_lr(step: int, total_steps: int, lr: float, min_lr: float, warmup_fraction: float) -> float:
    warmup = max(1, int(total_steps * warmup_fraction))
    if step < warmup:
        return lr * float(step + 1) / warmup
    pct = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, pct)))


def _set_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _barrier() -> None:
    if _distributed_ready():
        dist.barrier()


def _all_reduce_float(value: float, device: torch.device, op: dist.ReduceOp) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if _distributed_ready():
        dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def _ddp_sum(value: float, device: torch.device) -> float:
    return _all_reduce_float(value, device, dist.ReduceOp.SUM)


def _ddp_mean(value: float, device: torch.device, world_size: int) -> float:
    return _ddp_sum(value, device) / max(1, int(world_size))


def _sync_module_grads(module: torch.nn.Module | None, world_size: int) -> None:
    if module is None or int(world_size) <= 1 or not _distributed_ready():
        return
    for param in module.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(int(world_size))


def _reduce_metric(key: str, value: float, device: torch.device, world_size: int) -> float:
    sum_keys = {
        "label/labels_per_batch",
        "router/valid_tokens",
    }
    if key in sum_keys:
        return _ddp_sum(value, device)
    return _ddp_mean(value, device, world_size)


def _mean_tail(rows: list[dict[str, Any]], key: str, part: str) -> float:
    values = [float(x[key]) for x in rows if key in x and math.isfinite(float(x[key]))]
    if not values:
        return float("nan")
    n = max(1, len(values) // 4)
    use = values[:n] if part == "first" else values[-n:]
    return float(sum(use) / len(use))


def _effective_routed_target_budget(
    method: str,
    params: LGARParams,
    routed_target_budget: float | None,
) -> float | None:
    if method != "lgar_routed":
        return None
    if routed_target_budget is not None:
        return float(routed_target_budget)
    return float(params.final_global_budget)


def _effective_routing_supervision_start(params: LGARParams) -> int:
    return min(int(params.local_window), int(params.short_window + params.min_remote_margin))


def _aligned_routing_params(method: str, params: LGARParams) -> LGARParams:
    if method not in {"router_aux", "lgar_routed"}:
        return params
    aligned_budget = float(params.final_global_budget)
    return replace(
        params,
        lsd_top_fraction=aligned_budget,
        router_target_budget=aligned_budget,
    )


def _offline_label_batch_from_batch(batch: dict[str, torch.Tensor], require_lsd: bool) -> LSDLabelBatch:
    if "offline_lsd_labels" not in batch or "offline_lsd_valid" not in batch:
        raise RuntimeError("offline signal batch is missing cached labels/valid masks")
    valid = batch["offline_lsd_valid"].bool()
    positive = batch["offline_lsd_labels"].bool()
    lsd = batch.get("offline_lsd")
    if lsd is None:
        if require_lsd:
            raise RuntimeError("offline signal cache is missing offline_lsd scores required by LongCE")
        lsd = torch.zeros_like(valid, dtype=torch.float32)
    else:
        lsd = lsd.float()
    pos = positive & valid
    stats = {
        "positive_label_fraction": float(pos.float().sum().item() / max(1.0, valid.float().sum().item())),
        "labels_per_batch": float(pos.float().sum().item()),
        "valid_tokens": float(valid.float().sum().item()),
        "lsd_mean": float(lsd[valid].mean().item()) if valid.any() else 0.0,
    }
    zeros = torch.zeros_like(lsd)
    return LSDLabelBatch(
        labels=positive.detach(),
        valid=valid.detach(),
        lsd=lsd.detach(),
        long_logp=zeros.detach(),
        short_logp=zeros.detach(),
        stats=stats,
    )


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _latest_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint_latest.pt"


def _latest_rank_state_path(run_dir: Path, rank: int) -> Path:
    return run_dir / f"checkpoint_latest.rank{int(rank)}.pt"


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _completed_eval_events_before_step(start_step: int, eval_interval: int) -> int:
    if start_step <= 0:
        return 0
    interval = max(1, int(eval_interval))
    return 1 + (int(start_step) - 1) // interval


def _advance_dataset_state_for_resume(
    dataset: PackedFineWebDataset,
    val_dataset: PackedFineWebDataset,
    batch_size: int,
    start_step: int,
    eval_interval: int,
    eval_batches: int,
) -> None:
    for _ in range(int(start_step)):
        dataset.sample(batch_size)
    eval_events = _completed_eval_events_before_step(start_step, eval_interval)
    for _ in range(int(eval_events) * int(eval_batches)):
        val_dataset.sample(batch_size)


def _save_rank_resume_state(
    run_dir: Path,
    rank: int,
    next_step: int,
    dataset: PackedFineWebDataset,
    val_dataset: PackedFineWebDataset,
    device: torch.device,
) -> None:
    payload: dict[str, Any] = {
        "next_step": int(next_step),
        "dataset_rng_state": dataset.rng.bit_generator.state,
        "val_dataset_rng_state": val_dataset.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }
    if device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    _atomic_torch_save(payload, _latest_rank_state_path(run_dir, rank))


def _try_restore_rank_resume_state(
    run_dir: Path,
    rank: int,
    expected_step: int,
    dataset: PackedFineWebDataset,
    val_dataset: PackedFineWebDataset,
    device: torch.device,
) -> bool:
    path = _latest_rank_state_path(run_dir, rank)
    if not path.exists():
        return False
    state = torch.load(path, map_location="cpu")
    if int(state.get("next_step", -1)) != int(expected_step):
        return False
    dataset.rng.bit_generator.state = state["dataset_rng_state"]
    val_dataset.rng.bit_generator.state = state["val_dataset_rng_state"]
    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda" and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"], device)
    return True


def _save_latest_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    router: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    metrics: list[dict[str, Any]],
    extra: dict[str, Any],
    eval_metrics: dict[str, float],
    next_step: int,
    tokens_seen: int,
    elapsed_wall_seconds: float,
) -> None:
    payload = {
        "model": model.state_dict(),
        "router": router.state_dict() if router is not None else None,
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "extra": extra,
        "eval": eval_metrics,
        "next_step": int(next_step),
        "tokens_seen": int(tokens_seen),
        "elapsed_wall_seconds": float(elapsed_wall_seconds),
    }
    _atomic_torch_save(payload, _latest_checkpoint_path(run_dir))


def collect_label_audit(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: PackedFineWebDataset,
    params: LGARParams,
    device: torch.device,
    sequences: int,
    batch_size: int,
    example_limit: int = 20,
) -> dict[str, Any]:
    model.eval()
    stats_accum: dict[str, float] = {}
    stat_count = 0
    examples: list[dict[str, Any]] = []
    batches = max(1, math.ceil(int(sequences) / max(1, int(batch_size))))
    with torch.no_grad():
        for batch_idx in range(batches):
            batch = to_torch_batch(dataset.sample(batch_size), device)
            labels = mine_lsd_labels(model, batch, tokenizer, params)
            for key, value in labels.stats.items():
                stats_accum[key] = stats_accum.get(key, 0.0) + float(value)
            stat_count += 1
            if len(examples) < example_limit:
                examples.extend(lsd_audit_examples(tokenizer, batch, labels, limit=example_limit - len(examples)))
            if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == batches:
                seen = min(sequences, (batch_idx + 1) * batch_size)
                mean_pos = stats_accum.get("positive_label_fraction", 0.0) / max(1, stat_count)
                print(
                    json.dumps(
                        {
                            "kind": "label_audit_progress",
                            "seen_sequences": int(seen),
                            "target_sequences": int(sequences),
                            "mean_positive_label_fraction": float(mean_pos),
                            "examples": len(examples),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return {
        "requested_sequences": int(sequences),
        "observed_batches": int(batches),
        "mean_stats": {k: v / max(1, stat_count) for k, v in sorted(stats_accum.items())},
        "examples": examples[:example_limit],
    }


def _train_one(
    run_name: str,
    method: str,
    paths: Paths,
    params: LGARParams,
    train: TrainParams,
    device: torch.device,
    stage_name: str = "stage0",
    routed_target_budget: float | None = None,
    rank: int = 0,
    world_size: int = 1,
    final_eval: bool = False,
    final_short_limit: int = 16,
    final_long_limit: int = 8,
    final_long_lengths: list[int] | tuple[int, ...] | None = None,
    save_interval: int = 0,
    resume_if_available: bool = False,
    offline_signal_dir: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(train.seed)
    params = _aligned_routing_params(method, params)
    mode = "full"
    if method == "router_aux":
        mode = "router_aux"
    elif method == "lgar_routed":
        mode = "routed"
    tokenizer = load_tokenizer(paths.model_path)
    use_offline_signals = offline_signal_dir is not None and method in {"longce", "router_aux", "lgar_routed"}
    if use_offline_signals:
        dataset = PackedSequenceSignalDataset(
            cache_dir=paths.cache_dir,
            signal_dir=offline_signal_dir,
            pad_token_id=int(tokenizer.pad_token_id),
            seed=train.seed + int(rank) * 1009,
        )
        if int(dataset.seq_len) != int(params.seq_len):
            raise RuntimeError(
                f"offline signal cache seq_len={dataset.seq_len} does not match training seq_len={params.seq_len}"
            )
    else:
        dataset = PackedFineWebDataset(
            paths.cache_dir,
            params.seq_len,
            int(tokenizer.pad_token_id),
            "train",
            seed=train.seed + int(rank) * 1009,
        )
    val_dataset = PackedFineWebDataset(paths.cache_dir, params.seq_len, int(tokenizer.pad_token_id), "val", seed=train.seed)
    model = load_qwen_causal_lm(
        paths.model_path,
        dtype_name=train.dtype,
        attn_implementation=train.attn_implementation,
        gradient_checkpointing=bool(getattr(train, "gradient_checkpointing", True)),
    ).to(device)
    mask_format = attention_mask_format_for_model(model)
    router = (
        SharedQueryRouter(int(model.config.hidden_size), params.router_hidden_dim).to(device)
        if method in {"router_aux", "lgar_routed"}
        else None
    )
    set_seed(train.seed + int(rank) * 1009)
    model.train()
    if router is not None:
        router.train()
    opt_params = list(model.parameters()) + (list(router.parameters()) if router is not None else [])
    optimizer = torch.optim.AdamW(opt_params, lr=train.lr, weight_decay=train.weight_decay)
    run_dir = ensure_dir(Path(paths.output_dir) / "runs" / stage_name / run_name)
    metrics: list[dict[str, Any]] = []
    tokens_seen = 0
    start_step = 0
    resumed_from: str | None = None
    elapsed_wall_before = 0.0
    effective_routed_budget = _effective_routed_target_budget(method, params, routed_target_budget)
    if resume_if_available:
        latest_checkpoint = _latest_checkpoint_path(run_dir)
        if latest_checkpoint.exists():
            checkpoint = torch.load(latest_checkpoint, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            router_state = checkpoint.get("router")
            if router is not None:
                if router_state is None:
                    raise RuntimeError(f"missing router state in {latest_checkpoint}")
                router.load_state_dict(router_state)
            optimizer.load_state_dict(checkpoint["optimizer"])
            _optimizer_to_device(optimizer, device)
            metrics = list(checkpoint.get("metrics", []))
            start_step = int(checkpoint.get("next_step", 0))
            tokens_seen = int(checkpoint.get("tokens_seen", 0))
            elapsed_wall_before = float(checkpoint.get("elapsed_wall_seconds", 0.0))
            resumed_from = str(latest_checkpoint)
            checkpoint_extra = checkpoint.get("extra", {}) if isinstance(checkpoint.get("extra", {}), dict) else {}
            resume_world_size = int(checkpoint_extra.get("world_size", world_size))
            if start_step > int(train.steps) or resume_world_size != int(world_size):
                expected_tokens_per_step = max(1, int(params.seq_len) * int(train.batch_size) * max(1, int(world_size)))
                adjusted_step = min(int(tokens_seen // expected_tokens_per_step), int(train.steps))
                if tokens_seen > 0 and adjusted_step == 0:
                    adjusted_step = 1
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "event": "resume_step_adjusted",
                                "run": run_name,
                                "checkpoint_world_size": int(resume_world_size),
                                "current_world_size": int(world_size),
                                "checkpoint_next_step": int(start_step),
                                "adjusted_next_step": int(adjusted_step),
                                "tokens_seen": int(tokens_seen),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                start_step = adjusted_step
                restored_rank_state = False
            else:
                restored_rank_state = _try_restore_rank_resume_state(run_dir, rank, start_step, dataset, val_dataset, device)
            if not restored_rank_state:
                _advance_dataset_state_for_resume(
                    dataset,
                    val_dataset,
                    train.batch_size,
                    start_step,
                    train.eval_interval,
                    train.eval_batches,
                )
    start_wall = time.time() - elapsed_wall_before
    if resumed_from is not None and rank == 0:
        print(
            json.dumps(
                {
                    "event": "run_resumed",
                    "run": run_name,
                    "resume": resumed_from,
                    "next_step": int(start_step),
                    "tokens_seen": int(tokens_seen),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for step in range(start_step, train.steps):
        lr = _linear_lr(step, train.steps, train.lr, train.min_lr, train.warmup_fraction)
        _set_lr(optimizer, lr)
        batch = to_torch_batch(dataset.sample(train.batch_size), device)
        input_doc = batch["doc_ids_full"][:, :-1]
        attention_mask = document_causal_attention_mask(input_doc, mask_format=mask_format)
        needs_labels = method in {"longce", "router_aux", "lgar_routed"}

        labels: LSDLabelBatch | None = None
        if needs_labels:
            if use_offline_signals:
                labels = _offline_label_batch_from_batch(batch, require_lsd=(method == "longce"))
            else:
                was_training = model.training
                model.eval()
                valid_offset_threshold = None
                positive_fraction = None
                if method in {"router_aux", "lgar_routed"}:
                    valid_offset_threshold = _effective_routing_supervision_start(params)
                    positive_fraction = float(params.final_global_budget)
                with torch.no_grad():
                    labels = mine_lsd_labels(
                        model,
                        batch,
                        tokenizer,
                        params,
                        valid_offset_threshold=valid_offset_threshold,
                        positive_fraction=positive_fraction,
                    )
                model.train(was_training)

        optimizer.zero_grad(set_to_none=True)
        forward = qwen_lgar_forward(
            model=model,
            input_ids=batch["input_ids"],
            full_attention_mask=attention_mask,
            doc_ids=input_doc,
            router=router,
            params=params,
            label_targets=labels.labels if labels is not None and method in {"router_aux", "lgar_routed"} else None,
            label_valid_mask=labels.valid if labels is not None and method in {"router_aux", "lgar_routed"} else None,
            routing_valid_mask=batch["loss_mask"],
            mode=mode,
            target_budget=effective_routed_budget if method == "lgar_routed" else None,
            router_budget_target=float(params.final_global_budget) if method in {"router_aux", "lgar_routed"} else None,
        )
        logits = forward.logits
        ce = ce_from_logits(logits, batch["labels"], batch["loss_mask"])
        loss = ce
        if method == "longce":
            if labels is None:
                raise RuntimeError("LongCE requires LSD labels")
            per_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                batch["labels"].reshape(-1),
                reduction="none",
            ).view_as(batch["labels"])
            weights = torch.exp(labels.lsd.float()).clamp(min=0.25, max=4.0)
            mask = batch["loss_mask"].bool()
            if mask.any():
                weights = (weights / weights[mask].mean().clamp_min(1e-8)).detach()
                loss = (weights[mask] * per_ce[mask]).mean()
            else:
                weights = (weights / weights.mean().clamp_min(1e-8)).detach()
                loss = (weights * per_ce).mean()
        router_metrics: dict[str, float] = {}
        if method in {"router_aux", "lgar_routed"}:
            loss = ce + forward.router_loss
            router_metrics.update(forward.metrics)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{run_name} step {step}: non-finite loss {float(loss.detach().item())}")
        loss.backward()
        _sync_module_grads(model, world_size)
        _sync_module_grads(router, world_size)
        grad_norm = torch.nn.utils.clip_grad_norm_(opt_params, train.grad_clip)
        optimizer.step()
        global_step_tokens = int(_ddp_sum(float(batch["loss_mask"].sum().item()), device))
        tokens_seen += global_step_tokens

        row: dict[str, Any] = {
            "step": step,
            "tokens_seen": tokens_seen,
            "lr": float(lr),
            "loss": _ddp_mean(float(loss.detach().float().item()), device, world_size),
            "ce_loss": _ddp_mean(float(ce.detach().float().item()), device, world_size),
            "grad_norm": _ddp_mean(
                float(grad_norm.detach().float().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                device,
                world_size,
            ),
            "wall_seconds": float(time.time() - start_wall),
            "true_wall_tok_s": float(tokens_seen / max(1e-6, time.time() - start_wall)),
        }
        if method == "longce":
            row["loss/longce"] = row["loss"]
            local_weight_mean = float(weights[batch["loss_mask"].bool()].mean().detach().float().item()) if batch["loss_mask"].any() else float(weights.mean().detach().float().item())
            row["longce/weight_mean"] = _ddp_mean(local_weight_mean, device, world_size)
        if labels is not None:
            row.update({f"label/{k}": _reduce_metric(f"label/{k}", float(v), device, world_size) for k, v in labels.stats.items()})
        row.update({k: _reduce_metric(k, float(v), device, world_size) for k, v in router_metrics.items()})
        if rank == 0:
            metrics.append(row)
            write_json(run_dir / "metrics_partial.json", {"metrics": metrics})
        should_save = int(save_interval) > 0 and (step + 1) < train.steps and (step + 1) % int(save_interval) == 0
        if should_save:
            _barrier()
            if rank == 0:
                extra_partial = {
                    "run_name": run_name,
                    "method": method,
                    "tokens_seen": tokens_seen,
                    "wall_seconds": float(time.time() - start_wall),
                    "true_wall_tok_s": float(tokens_seen / max(1e-6, time.time() - start_wall)),
                    "train_params": train.__dict__,
                    "lgar_params": params.__dict__,
                    "routed_target_budget": effective_routed_budget,
                    "offline_signal_dir": str(offline_signal_dir) if offline_signal_dir is not None else None,
                    "world_size": int(world_size),
                }
                _save_latest_checkpoint(
                    run_dir=run_dir,
                    model=model,
                    router=router,
                    optimizer=optimizer,
                    metrics=metrics,
                    extra=extra_partial,
                    eval_metrics={},
                    next_step=step + 1,
                    tokens_seen=tokens_seen,
                    elapsed_wall_seconds=float(time.time() - start_wall),
                )
            _barrier()
            _save_rank_resume_state(run_dir, rank, step + 1, dataset, val_dataset, device)
            _barrier()
        if step % max(1, train.eval_interval) == 0 or step == train.steps - 1:
            if rank == 0:
                model.eval()
                with torch.no_grad():
                    eval_losses = []
                    for _ in range(train.eval_batches):
                        val_batch = to_torch_batch(val_dataset.sample(train.batch_size), device)
                        val_mask = document_causal_attention_mask(
                            val_batch["doc_ids_full"][:, :-1],
                            mask_format=mask_format,
                        )
                        val_forward = qwen_lgar_forward(
                            model=model,
                            input_ids=val_batch["input_ids"],
                            full_attention_mask=val_mask,
                            doc_ids=val_batch["doc_ids_full"][:, :-1],
                            router=router,
                            params=params,
                            routing_valid_mask=val_batch["loss_mask"],
                            mode=mode,
                            target_budget=effective_routed_budget if method == "lgar_routed" else None,
                            router_budget_target=float(params.final_global_budget) if method in {"router_aux", "lgar_routed"} else None,
                        )
                        eval_losses.append(float(ce_from_logits(val_forward.logits, val_batch["labels"], val_batch["loss_mask"]).item()))
                    row["quick_val_ce"] = float(sum(eval_losses) / len(eval_losses))
                model.train()
                write_json(run_dir / "metrics_partial.json", {"metrics": metrics})
                print(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "run": run_name,
                            "step": int(step),
                            "tokens_seen": int(tokens_seen),
                            "ce_loss": row["ce_loss"],
                            "true_wall_tok_s": row["true_wall_tok_s"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            _barrier()

    eval_metrics: dict[str, float] = {}
    if final_eval:
        model.eval()
        if router is not None:
            router.eval()
        with torch.no_grad():
            eval_metrics = evaluate_model(
                model,
                tokenizer,
                paths,
                params,
                train,
                batches=train.eval_batches,
                short_limit=final_short_limit,
                long_limit=final_long_limit,
                long_lengths=final_long_lengths,
                device=device,
                router=router,
                mode=mode,
                target_budget=effective_routed_budget if method == "lgar_routed" else None,
            )
        model.train()
        if router is not None:
            router.train()
        _barrier()

    extra = {
        "run_name": run_name,
        "method": method,
        "tokens_seen": tokens_seen,
        "wall_seconds": float(time.time() - start_wall),
        "true_wall_tok_s": float(tokens_seen / max(1e-6, time.time() - start_wall)),
        "train_params": train.__dict__,
        "lgar_params": params.__dict__,
        "routed_target_budget": effective_routed_budget,
        "offline_signal_dir": str(offline_signal_dir) if offline_signal_dir is not None else None,
        "world_size": int(world_size),
    }
    if rank == 0:
        checkpoint_path = run_dir / "checkpoint.pt"
        extra["checkpoint_path"] = str(checkpoint_path)
        write_json(run_dir / "metrics.json", {"metrics": metrics, "extra": extra, "eval": eval_metrics})
        torch.save(
            {
                "model": model.state_dict(),
                "router": router.state_dict() if router is not None else None,
                "extra": extra,
                "eval": eval_metrics,
            },
            checkpoint_path,
        )
        if router is not None:
            torch.save({"router": router.state_dict(), "extra": extra}, run_dir / "router.pt")
    _barrier()
    return {"run_dir": str(run_dir), "metrics": metrics, "extra": extra, "eval": eval_metrics}


def _summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    rows = run["metrics"]
    summary = {
        "run_dir": run["run_dir"],
        "tokens_seen": run["extra"]["tokens_seen"],
        "true_wall_tok_s": run["extra"]["true_wall_tok_s"],
        "ce_first_quarter": _mean_tail(rows, "ce_loss", "first"),
        "ce_last_quarter": _mean_tail(rows, "ce_loss", "last"),
        "router_bce_first_quarter": _mean_tail(rows, "router/loss_bce", "first"),
        "router_bce_last_quarter": _mean_tail(rows, "router/loss_bce", "last"),
        "router_auc_last_quarter": _mean_tail(rows, "router/auc", "last"),
        "router_precision10_last_quarter": _mean_tail(rows, "router/precision_at_10pct", "last"),
        "router_random_precision_last_quarter": _mean_tail(rows, "router/random_precision", "last"),
        "actual_budget_last_quarter": _mean_tail(rows, "router/actual_budget", "last"),
        "positive_labels_total": float(sum(float(x.get("label/labels_per_batch", 0.0)) for x in rows)),
        "nonfinite_loss": any(not math.isfinite(float(x["loss"])) for x in rows),
    }
    if run.get("eval"):
        summary["eval"] = run["eval"]
    if run["extra"].get("checkpoint_path"):
        summary["checkpoint_path"] = run["extra"]["checkpoint_path"]
    return summary


def _decision(train_summary: dict[str, Any], label_audit: dict[str, Any]) -> tuple[dict[str, bool], str]:
    ce = train_summary["Qwen_CE_CPT_100step"]
    aux = train_summary["Qwen_RouterAux_CPT_100step"]
    routed = train_summary["Qwen_LGAR_HighBudget_CPT_100step"]
    auc = aux["router_auc_last_quarter"]
    p10 = aux["router_precision10_last_quarter"]
    rand = aux["router_random_precision_last_quarter"]
    checks = {
        "label_audit_has_20_examples": len(label_audit["examples"]) >= 20,
        "positive_labels_nonzero": aux["positive_labels_total"] > 0 and label_audit["mean_stats"].get("positive_label_fraction", 0.0) > 0.0,
        "no_nan": not ce["nonfinite_loss"] and not aux["nonfinite_loss"],
        "router_bce_stable_or_decreasing": (
            math.isfinite(aux["router_bce_first_quarter"])
            and math.isfinite(aux["router_bce_last_quarter"])
            and aux["router_bce_last_quarter"] <= aux["router_bce_first_quarter"] + 0.02
        ),
        "router_beats_random": (
            (math.isfinite(auc) and auc > 0.55)
            or (math.isfinite(p10) and math.isfinite(rand) and p10 > rand + 0.01)
        ),
        "ce_no_spike_over_baseline": aux["ce_last_quarter"] <= ce["ce_last_quarter"] + 0.05,
        "lgar_high_budget_no_nan": not routed["nonfinite_loss"],
        "lgar_high_budget_ce_no_explosion": routed["ce_last_quarter"] <= ce["ce_last_quarter"] + 0.25,
        "lgar_high_budget_matches_target": abs(routed["actual_budget_last_quarter"] - 0.75) <= 0.02,
        "hard_lgar_layer_specific_forward_available": True,
    }
    if all(checks.values()):
        decision = (
            "Stage0 passes label, RouterAux, and high-budget hard L-GAR smoke review. Stage1 may be launched after final report review."
        )
    else:
        decision = "Stage0 does not pass; do not launch Stage1. Fix failed checks first."
    return checks, decision


def generate_stage0_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# L-GAR-CPT-Qwen0.5B Stage0 Report",
        "",
        "## Scope",
        "",
        "This run validates LSD mining, RouterAux, and hard L-GAR upper-layer routing for Qwen2.5-0.5B CPT.",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(summary["env"], indent=2, sort_keys=True),
        "```",
        "",
        "## Data Cache",
        "",
        "```json",
        json.dumps(summary["cache"], indent=2, sort_keys=True),
        "```",
        "",
        "## Zero-Training Evaluation",
        "",
        "```json",
        json.dumps(summary["zero_eval"], indent=2, sort_keys=True),
        "```",
        "",
        "## LSD Label Audit",
        "",
        "```json",
        json.dumps(summary["label_audit"]["mean_stats"], indent=2, sort_keys=True),
        "```",
        "",
        f"Collected examples: {len(summary['label_audit']['examples'])}",
        "",
    ]
    for i, item in enumerate(summary["label_audit"]["examples"][:20], start=1):
        lines.extend(
            [
                f"### Example {i}",
                "",
                f"- label: `{item['label']}`",
                f"- target_text: `{item['target_text']}`",
                f"- long_nll: `{item['long_nll']:.6f}`",
                f"- short_nll: `{item['short_nll']:.6f}`",
                f"- LSD: `{item['lsd']:.6f}`",
                "",
                "Remote prefix excerpt:",
                "",
                "```text",
                item["remote_prefix_excerpt"][:1200],
                "```",
                "",
                "Short-window excerpt:",
                "",
                "```text",
                item["short_window_excerpt"][:1200],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 100-Step Stability",
            "",
            "```json",
            json.dumps(summary["train_summary"], indent=2, sort_keys=True),
            "```",
            "",
            "## Decision Checks",
            "",
            "```json",
            json.dumps(summary["decision_checks"], indent=2, sort_keys=True),
            "```",
            "",
            "## Stage0 Decision",
            "",
            summary["decision"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mandatory Stage0 for L-GAR-CPT-Qwen0.5B.")
    defaults = Paths()
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--raw-data-dir", default=defaults.raw_data_dir)
    parser.add_argument("--cache-dir", default=defaults.cache_dir)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--target-cache-tokens", type=int, default=64_000_000)
    parser.add_argument("--max-shards", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--short-window", type=int, default=1024)
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--label-audit-sequences", type=int, default=10_000)
    parser.add_argument("--label-audit-batch-size", type=int, default=None)
    parser.add_argument("--label-audit-json", default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--short-limit", type=int, default=16)
    parser.add_argument("--long-limit", type=int, default=8)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-downstream-eval", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    paths = Paths(
        model_path=args.model_path,
        raw_data_dir=args.raw_data_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    output_dir = ensure_dir(paths.output_dir)
    cache_info_path = Path(paths.cache_dir) / "cache_info.json"
    if not cache_info_path.exists():
        cache_info = prepare_qwen_fineweb_cache(
            raw_data_dir=paths.raw_data_dir,
            model_path=paths.model_path,
            cache_dir=paths.cache_dir,
            target_tokens=args.target_cache_tokens,
            max_shards=args.max_shards,
        )
    else:
        cache_info = json.loads(cache_info_path.read_text(encoding="utf-8"))

    params = LGARParams(seq_len=args.seq_len, short_window=args.short_window, local_window=args.local_window)
    train = TrainParams(
        batch_size=args.batch_size,
        steps=args.steps,
        eval_batches=args.eval_batches,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(paths.model_path)
    base_model = load_qwen_causal_lm(paths.model_path, dtype_name=args.dtype, attn_implementation=args.attn_implementation).to(device)
    base_model.eval()
    if args.skip_downstream_eval:
        zero_eval: dict[str, float] = {"skipped": 1.0}
    else:
        zero_eval = evaluate_model(
            base_model,
            tokenizer,
            paths,
            params,
            train,
            batches=args.eval_batches,
            short_limit=args.short_limit,
            long_limit=args.long_limit,
            device=device,
        )
    if args.label_audit_json:
        label_audit = json.loads(Path(args.label_audit_json).read_text(encoding="utf-8"))
    else:
        audit_dataset = PackedFineWebDataset(paths.cache_dir, params.seq_len, int(tokenizer.pad_token_id), "train", seed=args.seed)
        label_audit = collect_label_audit(
            base_model,
            tokenizer,
            audit_dataset,
            params,
            device,
            sequences=args.label_audit_sequences,
            batch_size=args.label_audit_batch_size or args.batch_size,
        )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ce_run = _train_one("Qwen_CE_CPT_100step", "ce", paths, params, train, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    aux_run = _train_one("Qwen_RouterAux_CPT_100step", "router_aux", paths, params, train, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    routed_run = _train_one(
        "Qwen_LGAR_HighBudget_CPT_100step",
        "lgar_routed",
        paths,
        params,
        train,
        device,
        routed_target_budget=0.75,
    )
    train_summary = {
        "Qwen_CE_CPT_100step": _summarize_run(ce_run),
        "Qwen_RouterAux_CPT_100step": _summarize_run(aux_run),
        "Qwen_LGAR_HighBudget_CPT_100step": _summarize_run(routed_run),
    }
    decision_checks, decision = _decision(train_summary, label_audit)
    summary = {
        "env": env_snapshot(),
        "cache": cache_info,
        "zero_eval": zero_eval,
        "label_audit": label_audit,
        "train_summary": train_summary,
        "hard_lgar_available": True,
        "decision_checks": decision_checks,
        "decision": decision,
    }
    report_dir = ensure_dir(output_dir / "reports")
    write_json(report_dir / "stage0_summary.json", summary)
    generate_stage0_report(report_dir / "stage0_report.md", summary)
    print(json.dumps({"decision": decision, "decision_checks": decision_checks, "report": str(report_dir / "stage0_report.md")}, indent=2))


if __name__ == "__main__":
    main()
