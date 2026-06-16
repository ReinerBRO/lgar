from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import BackwardPrefetch
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
except Exception:  # pragma: no cover - depends on the installed torch build.
    FSDP = None  # type: ignore[assignment]
    BackwardPrefetch = None  # type: ignore[assignment]
    FullStateDictConfig = None  # type: ignore[assignment]
    StateDictType = None  # type: ignore[assignment]
    transformer_auto_wrap_policy = None  # type: ignore[assignment]

from lgar_cpt.config import Paths, TrainParams
from lgar_cpt.data import PackedFineWebDataset, PackedSequenceSignalDataset
from lgar_cpt.mining import (
    attention_mask_format_for_model,
    ce_from_logits,
    document_causal_attention_mask,
    mine_lsd_labels,
)
from lgar_cpt.modeling import load_qwen_causal_lm, load_tokenizer
from lgar_cpt.utils import ensure_dir, set_seed, write_json

from .adapters import HeadAdapterSet
from .config import CUREParams
from .forward import cure_forward
from .head_ablation import load_ablation_results
from .losses import (
    apply_isolated_adapter_grads,
    cure_ce_loss,
    cure_full_hu_loss,
    cure_nonhu_logp_consistency_loss,
    cure_rh_loss,
    select_cure_main_loss_mask,
)
from .sae import (
    EvidenceFeatureSet,
    SparseAutoencoder,
    load_evidence_feature_set,
    load_sparse_autoencoder,
    lsor_loss,
    sae_feature_losses,
    short_kl_loss,
)


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


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _validate_sae_layer_covers_retrieval_heads(
    sae_layer: int,
    retrieval_heads: set[tuple[int, int]],
) -> None:
    if not retrieval_heads:
        raise ValueError("SAE-CURE requires non-empty selected retrieval heads")
    selected_layers = {int(layer) for layer, _head in retrieval_heads}
    max_selected_layer = max(selected_layers)
    if int(sae_layer) < max_selected_layer:
        raise ValueError(
            "SAE layer must be at or after every selected retrieval-head layer "
            f"so SAE loss has a gradient path to all selected heads; "
            f"sae_layer={int(sae_layer)} max_selected_head_layer={max_selected_layer}. "
            "Use features from a later layer or reduce the selected-head set."
        )


def _sync_module_grads(module: torch.nn.Module, world_size: int) -> None:
    if world_size <= 1:
        return
    for param in module.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)


def _unwrap_fsdp(model: torch.nn.Module) -> torch.nn.Module:
    if FSDP is not None and isinstance(model, FSDP):
        return model.module
    return model


def _enable_input_require_grads(model: torch.nn.Module) -> None:
    base_model = _unwrap_fsdp(model)
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
        return
    if hasattr(base_model, "get_input_embeddings"):
        embeddings = base_model.get_input_embeddings()
    else:
        backbone = getattr(base_model, "model", None) or getattr(base_model, "base_model", None)
        embeddings = getattr(backbone, "embed_tokens", None)
    if embeddings is None:
        raise AttributeError("cannot locate input embeddings for gradient checkpointing")

    def _make_outputs_require_grad(_module, _inputs, output):
        if torch.is_tensor(output):
            output.requires_grad_(True)

    embeddings.register_forward_hook(_make_outputs_require_grad)


def _decoder_layer_classes(model: torch.nn.Module) -> set[type[torch.nn.Module]]:
    base_model = _unwrap_fsdp(model)
    backbone = getattr(base_model, "model", None) or getattr(base_model, "base_model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None or len(layers) == 0:
        return set()
    return {layers[0].__class__}


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _fsdp_backward_prefetch_from_env() -> Any | None:
    if BackwardPrefetch is None:
        return None
    value = os.environ.get("FSDP_BACKWARD_PREFETCH", "pre").strip().lower()
    if value in {"0", "false", "none", "off", "no"}:
        return None
    if value in {"post", "backward_post"}:
        return BackwardPrefetch.BACKWARD_POST
    if value in {"pre", "backward_pre"}:
        return BackwardPrefetch.BACKWARD_PRE
    raise ValueError(f"unknown FSDP_BACKWARD_PREFETCH={value!r}; use pre, post, or none")


def _wrap_model_fsdp(model: torch.nn.Module, device: torch.device, rank: int) -> torch.nn.Module:
    if FSDP is None or transformer_auto_wrap_policy is None:
        raise RuntimeError("FSDP is unavailable in this PyTorch build")
    layer_classes = _decoder_layer_classes(model)
    auto_wrap_policy = (
        partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_classes)
        if layer_classes
        else None
    )
    limit_all_gathers = _env_flag("FSDP_LIMIT_ALL_GATHERS", True)
    forward_prefetch = _env_flag("FSDP_FORWARD_PREFETCH", False)
    backward_prefetch = _fsdp_backward_prefetch_from_env()
    if _is_rank0(rank):
        class_names = ",".join(sorted(cls.__name__ for cls in layer_classes)) or "none"
        print(
            "[train] enabling FSDP "
            f"auto_wrap_layers={class_names} "
            f"limit_all_gathers={int(limit_all_gathers)} "
            f"forward_prefetch={int(forward_prefetch)} "
            f"backward_prefetch={backward_prefetch}",
            flush=True,
        )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device if device.type == "cuda" else None,
        use_orig_params=True,
        limit_all_gathers=limit_all_gathers,
        forward_prefetch=forward_prefetch,
        backward_prefetch=backward_prefetch,
    )


def _model_state_dict_for_checkpoint(model: torch.nn.Module, fsdp_active: bool) -> dict[str, Any] | None:
    if not fsdp_active:
        return None
    if FSDP is None or FullStateDictConfig is None or StateDictType is None:
        raise RuntimeError("FSDP full-state checkpointing is unavailable in this PyTorch build")
    state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config):
        return dict(model.state_dict())


def _make_tensorboard_writer(enabled: bool, log_dir: Path, rank: int) -> Any | None:
    if not enabled or not _is_rank0(rank):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"[train] tensorboard disabled: failed to import SummaryWriter: {exc}", flush=True)
        return None
    ensure_dir(log_dir)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"tensorboard_log_dir={log_dir}", flush=True)
    return writer


def _as_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _write_tensorboard_entry(writer: Any | None, entry: dict[str, Any], world_size: int) -> None:
    if writer is None:
        return
    step_value = entry.get("step")
    try:
        step = int(step_value)
    except (TypeError, ValueError):
        return

    tokens = _as_finite_float(entry.get("tokens"))
    if tokens is not None:
        writer.add_scalar("progress/tokens_per_rank", tokens, step)
        writer.add_scalar("progress/global_tokens", tokens * int(world_size), step)
        writer.add_scalar("progress/global_tokens_b", tokens * int(world_size) / 1_000_000_000.0, step)

    wall_elapsed = _as_finite_float(entry.get("wall_elapsed"))
    if wall_elapsed is not None:
        writer.add_scalar("time/wall_elapsed_sec", wall_elapsed, step)

    for key, value in entry.items():
        if key in {"step", "tokens", "wall_elapsed"}:
            continue
        scalar = _as_finite_float(value)
        if scalar is None:
            continue
        tag = "eval/val_ce" if key == "val_ce" else str(key)
        writer.add_scalar(tag, scalar, step)
    writer.flush()


def _lr_schedule(step: int, total_steps: int, warmup_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _latest_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint_latest.pt"


def _interval_checkpoint_path(run_dir: Path, next_step: int, global_tokens_seen: int) -> Path:
    token_millions = int(round(int(global_tokens_seen) / 1_000_000))
    return run_dir / f"checkpoint_step{int(next_step):06d}_gtokens{token_millions:04d}m.pt"


def _latest_rank_state_path(run_dir: Path, rank: int) -> Path:
    return run_dir / f"checkpoint_latest.rank{int(rank)}.pt"


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _rng_state(obj: Any) -> dict[str, Any] | None:
    rng = getattr(obj, "rng", None)
    if rng is None:
        return None
    return rng.bit_generator.state


def _restore_rng_state(obj: Any, state: dict[str, Any] | None) -> None:
    if state is None:
        return
    rng = getattr(obj, "rng", None)
    if rng is not None:
        rng.bit_generator.state = state


def _advance_dataset_state_for_resume(
    dataset: PackedFineWebDataset | PackedSequenceSignalDataset,
    val_dataset: PackedFineWebDataset,
    batch_size: int,
    start_step: int,
    eval_interval: int,
) -> None:
    for _ in range(int(start_step)):
        dataset.sample(batch_size)
    val_events = int(start_step) // max(1, int(eval_interval) * 5)
    for _ in range(val_events):
        val_dataset.sample(batch_size)


def _save_rank_resume_state(
    run_dir: Path,
    rank: int,
    next_step: int,
    dataset: PackedFineWebDataset | PackedSequenceSignalDataset,
    val_dataset: PackedFineWebDataset,
    device: torch.device,
) -> None:
    payload: dict[str, Any] = {
        "next_step": int(next_step),
        "dataset_rng_state": _rng_state(dataset),
        "dataset_base_rng_state": _rng_state(getattr(dataset, "base", None)),
        "val_dataset_rng_state": _rng_state(val_dataset),
        "torch_rng_state": torch.get_rng_state(),
    }
    if device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    _atomic_torch_save(payload, _latest_rank_state_path(run_dir, rank))


def _try_restore_rank_resume_state(
    run_dir: Path,
    rank: int,
    expected_step: int,
    dataset: PackedFineWebDataset | PackedSequenceSignalDataset,
    val_dataset: PackedFineWebDataset,
    device: torch.device,
) -> bool:
    path = _latest_rank_state_path(run_dir, rank)
    if not path.exists():
        return False
    state = torch.load(path, map_location="cpu")
    if int(state.get("next_step", -1)) != int(expected_step):
        return False
    _restore_rng_state(dataset, state.get("dataset_rng_state"))
    _restore_rng_state(getattr(dataset, "base", None), state.get("dataset_base_rng_state"))
    _restore_rng_state(val_dataset, state.get("val_dataset_rng_state"))
    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda" and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"], device)
    return True


def _save_latest_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    adapters: HeadAdapterSet | None,
    optimizer: torch.optim.Optimizer,
    metrics_log: list[dict[str, Any]],
    next_step: int,
    tokens_seen: int,
    elapsed_wall_seconds: float,
    extra: dict[str, Any],
    model_state_dict: dict[str, Any] | None = None,
    include_optimizer: bool = True,
) -> None:
    payload: dict[str, Any] = {
        "model": model_state_dict if model_state_dict is not None else model.state_dict(),
        "metrics_log": metrics_log,
        "next_step": int(next_step),
        "tokens_seen": int(tokens_seen),
        "elapsed_wall_seconds": float(elapsed_wall_seconds),
        "extra": extra,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    if adapters is not None:
        payload["adapters"] = adapters.state_dict()
    _atomic_torch_save(payload, _latest_checkpoint_path(run_dir))


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train_cure(
    run_name: str,
    method: str,
    paths: Paths,
    cure_params: CUREParams,
    train: TrainParams,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    ablation_results_path: str | Path | None = None,
    offline_signal_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    save_interval: int = 0,
    resume_if_available: bool = False,
    tensorboard: bool = True,
    tensorboard_dir: str | Path | None = None,
    archive_interval_checkpoints: bool = True,
    attention_mask_mode: str = "document",
    use_fsdp: bool = False,
    gradient_checkpointing: bool | None = None,
) -> dict[str, Any]:
    """Run a CURE-CPT training run.

    method: "ce_cpt", "adapter_ce", "longce_cpt", "cure_cpt"
    """
    if attention_mask_mode not in {"document", "causal"}:
        raise ValueError(f"unknown attention_mask_mode={attention_mask_mode!r}")
    fsdp_active = bool(use_fsdp and world_size > 1)
    if use_fsdp and method not in {"ce_cpt", "longce_cpt", "adapter_ce", "cure_cpt"}:
        raise ValueError("FSDP is currently supported for ce_cpt, longce_cpt, adapter_ce, and cure_cpt")
    set_seed(train.seed + rank * 1009)

    tokenizer = load_tokenizer(paths.model_path)
    adapter_methods = {"adapter_ce", "cure_cpt"}
    use_adapters = method in adapter_methods
    require_freeze_base = _env_flag("CURE_REQUIRE_FREEZE_BASE", False)
    if method == "cure_cpt" and require_freeze_base and not cure_params.freeze_base_model:
        raise ValueError(
            "CURE_REQUIRE_FREEZE_BASE=1 but --freeze-base-model was not set. "
            "Use FREEZE_BASE_MODEL=1 in the launcher for frozen-base CURE runs."
        )
    if rank == 0 and method == "cure_cpt":
        print(
            "[train] cure_flags "
            f"freeze_base_model={int(cure_params.freeze_base_model)} "
            f"require_freeze_base={int(require_freeze_base)} "
            f"fsdp={int(fsdp_active)} "
            f"checkpoint_path={checkpoint_path or 'none'}",
            flush=True,
        )
    if cure_params.sae_enabled:
        if method != "cure_cpt":
            raise ValueError("SAE steering is only supported for method=cure_cpt")
        if not cure_params.freeze_base_model:
            raise ValueError("SAE steering requires --freeze-base-model so only retrieval-head LoRA trains")
        if not cure_params.sae_features_path:
            raise ValueError("SAE steering requires --sae-features")
        if offline_signal_dir is None:
            raise ValueError("SAE steering requires --offline-signal-dir with precomputed feature targets")
    if cure_params.lsor_enabled and method != "cure_cpt":
        raise ValueError("LSOR is only supported for method=cure_cpt")
    gradient_checkpointing_enabled = (
        cure_params.gradient_checkpointing
        if gradient_checkpointing is None
        else bool(gradient_checkpointing)
    )
    # Adapter methods inject LoRA via forward hooks. With a frozen base model,
    # checkpointed blocks may have no grad-tracked tensor inputs, so the in-hook
    # LoRA params can be invisible at the checkpoint boundary. For full
    # CURE-from-base training the base remains trainable, so keep checkpointing
    # enabled to make 16k runs fit in memory.
    disable_grad_ckpt_for_adapter = (
        use_adapters and cure_params.freeze_base_model and not fsdp_active
    )
    use_grad_ckpt = gradient_checkpointing_enabled and not disable_grad_ckpt_for_adapter
    if gradient_checkpointing_enabled and not use_grad_ckpt and rank == 0:
        print(f"[train] disabling gradient_checkpointing for adapter method '{method}'")
    if rank == 0:
        print(f"[train] gradient_checkpointing={int(use_grad_ckpt)}", flush=True)
    model = load_qwen_causal_lm(
        paths.model_path,
        dtype_name=cure_params.dtype,
        attn_implementation=cure_params.attn_implementation,
        gradient_checkpointing=use_grad_ckpt,
    ).to(device)

    if checkpoint_path is not None:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    if cure_params.freeze_base_model:
        for param in model.parameters():
            param.requires_grad_(False)
        if rank == 0:
            print("[train] freezing base model; only adapters are trainable")
    if use_adapters and cure_params.freeze_base_model and use_grad_ckpt:
        _enable_input_require_grads(model)
        if rank == 0:
            print("[train] enabled input require_grads for frozen adapter checkpointing", flush=True)

    # Determine retrieval heads
    retrieval_heads: set[tuple[int, int]] = set()
    adapters: HeadAdapterSet | None = None

    if ablation_results_path is not None and Path(ablation_results_path).exists():
        retrieval_heads, _ = load_ablation_results(ablation_results_path)
    elif use_adapters:
        # Default: top 1/3 layers, all heads (fallback)
        num_layers = int(model.config.num_hidden_layers)
        n_heads = int(model.config.num_attention_heads)
        from .config import ablation_candidate_layers
        for l in ablation_candidate_layers(num_layers, cure_params.ablation_layer_fraction):
            for h in range(n_heads):
                retrieval_heads.add((l, h))

    if use_adapters and retrieval_heads:
        head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
        adapters = HeadAdapterSet(
            retrieval_heads,
            head_dim,
            cure_params.lora_rank,
            cure_params.lora_alpha,
            hidden_size=int(model.config.hidden_size),
        ).to(device)
    if cure_params.sae_enabled and adapters is None:
        raise ValueError("SAE steering requires selected retrieval-head adapters")
    if cure_params.lsor_enabled and adapters is None:
        raise ValueError("LSOR requires selected retrieval-head adapters")

    sae_model: SparseAutoencoder | None = None
    sae_features: EvidenceFeatureSet | None = None
    if cure_params.sae_enabled:
        sae_features = load_evidence_feature_set(
            str(cure_params.sae_features_path),
            require_validated=cure_params.sae_require_validated_features,
        )
        _validate_sae_layer_covers_retrieval_heads(int(sae_features.layer), retrieval_heads)
        sae_ckpt_path = cure_params.sae_checkpoint_path or sae_features.sae_checkpoint
        if not sae_ckpt_path:
            raise ValueError("SAE steering requires --sae-checkpoint or sae_checkpoint in feature JSON")
        sae_model = load_sparse_autoencoder(sae_ckpt_path, device=device)
        hidden_size = int(model.config.hidden_size)
        if int(sae_model.encoder.in_features) != hidden_size:
            raise ValueError(
                f"SAE input_dim={sae_model.encoder.in_features} does not match model hidden_size={hidden_size}"
            )
        if _is_rank0(rank):
            print(
                json.dumps(
                    {
                        "event": "sae_steering_loaded",
                        "layer": int(sae_features.layer),
                        "hook_point": sae_features.hook_point,
                        "num_features": len(sae_features.feature_ids),
                        "validation_passed": bool(sae_features.validation_passed),
                        "sae_checkpoint": str(sae_ckpt_path),
                        "features": str(cure_params.sae_features_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    mask_format = attention_mask_format_for_model(model)
    if fsdp_active:
        model = _wrap_model_fsdp(model, device, rank)
    elif use_fsdp and _is_rank0(rank):
        print("[train] FSDP requested but world_size=1; running without FSDP", flush=True)

    # Data
    is_cure = method == "cure_cpt"
    use_offline_signals = offline_signal_dir is not None and method in {"longce_cpt", "cure_cpt"}

    if use_offline_signals:
        dataset = PackedSequenceSignalDataset(
            cache_dir=paths.cache_dir,
            signal_dir=offline_signal_dir,
            pad_token_id=int(tokenizer.pad_token_id),
            seed=train.seed + rank * 1009,
        )
    else:
        dataset = PackedFineWebDataset(
            paths.cache_dir, cure_params.seq_len, int(tokenizer.pad_token_id), "train",
            seed=train.seed + rank * 1009,
        )

    val_dataset = PackedFineWebDataset(
        paths.cache_dir, cure_params.seq_len, int(tokenizer.pad_token_id), "val", seed=train.seed
    )

    # Optimizer
    model_params = [param for param in model.parameters() if param.requires_grad]
    adapter_params = adapters.adapter_parameters() if adapters is not None else []
    all_params = model_params + adapter_params

    adapter_lr = cure_params.adapter_lr if cure_params.adapter_lr is not None else train.lr
    param_groups: list[dict[str, Any]] = []
    if model_params:
        param_groups.append({
            "name": "model",
            "params": model_params,
            "lr": train.lr,
            "base_lr": train.lr,
            "weight_decay": train.weight_decay,
        })
    if adapter_params:
        param_groups.append(
            {
                "name": "adapter",
                "params": adapter_params,
                "lr": adapter_lr,
                "base_lr": adapter_lr,
                "weight_decay": cure_params.adapter_weight_decay,
            }
        )
    if not param_groups:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(param_groups)

    # Training state
    model.train()
    if adapters is not None:
        adapters.train()

    run_dir = ensure_dir(Path(paths.output_dir) / "runs" / "cure" / run_name)
    metrics_log: list[dict[str, Any]] = []
    tokens_seen = 0
    total_steps = max(1, cure_params.tokens_per_run // max(1, cure_params.seq_len * train.batch_size))
    warmup_steps = max(1, int(total_steps * train.warmup_fraction))
    start_step = 0
    elapsed_wall_before = 0.0
    resumed_from: str | None = None
    tb_log_dir = Path(tensorboard_dir) if tensorboard_dir is not None else run_dir / "tensorboard"
    tb_writer = _make_tensorboard_writer(tensorboard, tb_log_dir, rank)

    fsdp_adapter_only_resume = bool(
        fsdp_active and resume_if_available and use_adapters and cure_params.freeze_base_model
    )
    if fsdp_active and resume_if_available and not fsdp_adapter_only_resume and _is_rank0(rank):
        print("[train] FSDP resume is disabled; saving full model checkpoints for eval only", flush=True)
    if fsdp_active and not fsdp_adapter_only_resume:
        resume_if_available = False
    if fsdp_adapter_only_resume and _is_rank0(rank):
        print("[train] FSDP adapter-only resume is enabled", flush=True)

    if resume_if_available:
        latest_checkpoint = _latest_checkpoint_path(run_dir)
        if latest_checkpoint.exists():
            checkpoint = torch.load(latest_checkpoint, map_location="cpu")
            if fsdp_adapter_only_resume:
                if _is_rank0(rank):
                    print("[train] skipping FSDP base-model load for frozen adapter-only resume", flush=True)
            else:
                model.load_state_dict(checkpoint["model"], strict=True)
            if adapters is not None and checkpoint.get("adapters") is not None:
                adapters.load_state_dict(checkpoint["adapters"], strict=True)
            if checkpoint.get("optimizer") is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
                _optimizer_to_device(optimizer, device)
            elif _is_rank0(rank):
                print("[train] checkpoint has no optimizer state; continuing with fresh optimizer", flush=True)
            metrics_log = list(checkpoint.get("metrics_log", []))
            start_step = int(checkpoint.get("next_step", 0))
            tokens_seen = int(checkpoint.get("tokens_seen", 0))
            elapsed_wall_before = float(checkpoint.get("elapsed_wall_seconds", 0.0))
            resumed_from = str(latest_checkpoint)
            if start_step > total_steps:
                start_step = total_steps
            if not _try_restore_rank_resume_state(run_dir, rank, start_step, dataset, val_dataset, device):
                _advance_dataset_state_for_resume(
                    dataset,
                    val_dataset,
                    train.batch_size,
                    start_step,
                    train.eval_interval,
                )

    wall_start = time.monotonic() - elapsed_wall_before
    if resumed_from is not None and _is_rank0(rank):
        for historical_entry in metrics_log:
            _write_tensorboard_entry(tb_writer, historical_entry, world_size)
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

    for step in range(start_step, total_steps):
        # LR schedule
        lr_now = _lr_schedule(step, total_steps, warmup_steps, train.lr, cure_params.min_lr)
        adapter_lr_now = _lr_schedule(
            step,
            total_steps,
            warmup_steps,
            adapter_lr,
            min(cure_params.min_lr, adapter_lr),
        )
        for pg in optimizer.param_groups:
            pg["lr"] = adapter_lr_now if pg.get("name") == "adapter" else lr_now

        # Sample batch
        batch = dataset.sample(train.batch_size)
        input_ids = torch.as_tensor(batch["input_ids"], device=device)
        labels = torch.as_tensor(batch["labels"], device=device)
        loss_mask = torch.as_tensor(batch["loss_mask"], device=device)
        doc_ids_full = torch.as_tensor(batch["doc_ids_full"], device=device)
        input_doc_ids = doc_ids_full[:, :-1]

        full_mask = (
            None
            if attention_mask_mode == "causal"
            else document_causal_attention_mask(input_doc_ids, mask_format=mask_format).to(device)
        )

        # High-utility mask / utility scores
        utility_scores: torch.Tensor | None = None
        if use_offline_signals and "offline_lsd_labels" in batch:
            high_utility = torch.as_tensor(batch["offline_lsd_labels"], device=device).bool()
            if "offline_lsd_valid" in batch:
                valid = torch.as_tensor(batch["offline_lsd_valid"], device=device).bool()
            else:
                valid = loss_mask
            if "offline_lsd" in batch:
                utility_scores = torch.as_tensor(batch["offline_lsd"], device=device).float()
        elif method in {"longce_cpt", "cure_cpt"}:
            # Online mining — convert batch to tensors for model forward
            t_batch = {k: torch.as_tensor(v, device=device) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
            label_batch = mine_lsd_labels(
                model, t_batch, tokenizer, cure_params,
                positive_fraction=cure_params.utility_top_fraction_training,
            )
            high_utility = label_batch.labels
            valid = label_batch.valid
            utility_scores = label_batch.lsd.float()
        else:
            high_utility = torch.zeros_like(loss_mask, dtype=torch.bool)
            valid = loss_mask

        # Forward
        use_adapter_forward = adapters is not None and retrieval_heads
        sae_capture_layers = {int(sae_features.layer)} if sae_features is not None else set()
        lsor_capture_layers = (
            {int(layer) for layer, _head in retrieval_heads}
            if cure_params.lsor_enabled and cure_params.lambda_lsor > 0.0
            else set()
        )
        resid_capture_layers = sae_capture_layers | lsor_capture_layers
        need_rh_bottleneck = (
            is_cure
            and use_adapter_forward
            and (
                cure_params.lambda_rh_ce > 0.0
                or cure_params.lambda_rh_kd > 0.0
                or cure_params.lambda_cov > 0.0
            )
        )
        out = cure_forward(
            model, input_ids, input_doc_ids, full_mask,
            retrieval_heads if use_adapter_forward else set(),
            adapters, cure_params.local_window,
            compute_rh_bottleneck=need_rh_bottleneck,
            rh_bottleneck_scope=cure_params.rh_bottleneck_scope,
            capture_resid_mid_layers=resid_capture_layers,
            capture_adapter_write_layers=lsor_capture_layers,
        )

        # Loss & backward with gradient isolation
        if is_cure:
            # CURE main CE can be restricted so frozen-base adapters do not
            # spend most of their capacity learning ordinary local tokens.
            main_loss_mask = select_cure_main_loss_mask(
                loss_mask=loss_mask,
                valid_mask=valid,
                high_utility_mask=high_utility,
                mode=cure_params.cure_main_loss_mask,
            )
            l_ce, ce_metrics = cure_ce_loss(out.logits_full, labels, main_loss_mask)
            ce_metrics.update({
                "cure/main_loss_mask": cure_params.cure_main_loss_mask,
                "utility/main_loss_count": float(main_loss_mask.sum().item()),
                "utility/main_loss_fraction": float(main_loss_mask.sum().item()) / max(1.0, float(loss_mask.sum().item())),
            })
            aux_losses: dict[str, torch.Tensor] = {}

            l_full_hu, full_hu_metrics = cure_full_hu_loss(
                out.logits_full,
                labels,
                high_utility,
                valid & loss_mask,
                cure_params.lambda_full_hu_ce,
            )
            if cure_params.lambda_full_hu_ce > 0.0:
                aux_losses["full_hu"] = l_full_hu

            teacher_logits = None
            if cure_params.lambda_nonhu_logp > 0.0 or cure_params.lambda_short_kl > 0.0:
                with torch.no_grad():
                    teacher_out = model(input_ids, attention_mask=full_mask, use_cache=False)
                    teacher_logits = (
                        teacher_out.logits
                        if hasattr(teacher_out, "logits")
                        else teacher_out["logits"]
                    )
            if cure_params.lambda_nonhu_logp > 0.0:
                assert teacher_logits is not None
                l_nonhu, nonhu_metrics = cure_nonhu_logp_consistency_loss(
                    out.logits_full,
                    teacher_logits,
                    labels,
                    high_utility,
                    valid & loss_mask,
                    cure_params.lambda_nonhu_logp,
                )
                aux_losses["nonhu_logp"] = l_nonhu
            else:
                nonhu_metrics = {
                    "loss/nonhu_logp_mse": 0.0,
                    "loss/nonhu_logp_total": 0.0,
                    "utility/nonhu_count": 0.0,
                    "utility/nonhu_fraction": 0.0,
                }

            if cure_params.lambda_short_kl > 0.0:
                assert teacher_logits is not None
                l_short_kl, short_kl_metrics = short_kl_loss(
                    out.logits_full,
                    teacher_logits,
                    high_utility,
                    valid & loss_mask,
                    cure_params.lambda_short_kl,
                )
                aux_losses["short_kl"] = l_short_kl
            else:
                short_kl_metrics = {
                    "loss/short_kl": 0.0,
                    "loss/short_kl_total": 0.0,
                    "utility/short_kl_count": 0.0,
                    "utility/short_kl_fraction": 0.0,
                }

            if sae_model is not None and sae_features is not None:
                required = [
                    "offline_sae_feature_full",
                    "offline_sae_feature_neg",
                    "offline_sae_feature_mask",
                ]
                missing = [key for key in required if key not in batch]
                if missing:
                    raise ValueError(
                        "SAE steering requires offline feature targets in signal_dir: "
                        + ", ".join(missing)
                    )
                sae_target_full = torch.as_tensor(batch["offline_sae_feature_full"], device=device)
                sae_target_neg = torch.as_tensor(batch["offline_sae_feature_neg"], device=device)
                sae_target_mask = torch.as_tensor(batch["offline_sae_feature_mask"], device=device)
                sae_losses, sae_metrics = sae_feature_losses(
                    out.resid_mid_by_layer or {},
                    sae_model,
                    sae_features,
                    sae_target_full,
                    sae_target_neg,
                    sae_target_mask,
                    valid & loss_mask,
                    high_utility,
                    beta_margin=cure_params.lambda_sae_margin,
                    beta_match=cure_params.lambda_sae_match,
                    gamma=cure_params.sae_margin_gamma,
                    match_clip=cure_params.sae_match_clip,
                )
                aux_losses.update(sae_losses)
            else:
                sae_metrics = {
                    "sae/enabled": 0.0,
                    "loss/sae_margin_total": 0.0,
                    "loss/sae_match_total": 0.0,
                }

            if cure_params.lsor_enabled and cure_params.lambda_lsor > 0.0:
                if tokens_seen >= int(cure_params.lsor_warmup_tokens):
                    l_lsor, lsor_metrics = lsor_loss(
                        out.adapter_writes_by_layer or {},
                        out.resid_mid_by_layer or {},
                        input_doc_ids,
                        high_utility & valid & loss_mask,
                        top_k=cure_params.lsor_top_k,
                        window=cure_params.lsor_window,
                        max_context_tokens=cure_params.lsor_max_context_tokens,
                        lambda_lsor=cure_params.lambda_lsor,
                    )
                    aux_losses["lsor"] = l_lsor
                else:
                    lsor_metrics = {
                        "loss/lsor": 0.0,
                        "loss/lsor_total": 0.0,
                        "lsor/projection_ratio": 0.0,
                        "lsor/active_layers": 0.0,
                        "lsor/warmup_active": 1.0,
                    }
            else:
                lsor_metrics = {
                    "loss/lsor": 0.0,
                    "loss/lsor_total": 0.0,
                    "lsor/projection_ratio": 0.0,
                    "lsor/active_layers": 0.0,
                }

            if out.logits_rh_only is not None:
                l_rh, rh_metrics = cure_rh_loss(
                    out.logits_full, out.logits_rh_only, labels,
                    high_utility, valid & loss_mask,
                    cure_params.lambda_rh_ce, cure_params.lambda_rh_kd, cure_params.lambda_cov,
                    tokens_seen, cure_params.cov_warmup_tokens,
                )
                aux_losses["rh"] = l_rh
            else:
                hu_valid = high_utility & valid & loss_mask
                rh_metrics = {
                    "loss/rh_ce": 0.0,
                    "loss/rh_kd": 0.0,
                    "loss/cov": 0.0,
                    "loss/rh_total": 0.0,
                    "utility/high_utility_count": float(hu_valid.sum().item()),
                    "utility/high_utility_fraction": float(hu_valid.sum().item()) / max(1.0, float((valid & loss_mask).sum().item())),
                }

            optimizer.zero_grad()

            # Gradient isolation: selected main CE follows normal trainable
            # params; auxiliaries are explicitly restricted to adapters.
            grad_metrics = apply_isolated_adapter_grads(
                l_ce,
                aux_losses,
                adapter_params,
                base_frozen=cure_params.freeze_base_model,
            )
            aux_total = sum(aux_losses.values(), start=l_ce.new_tensor(0.0))

            loss_metrics = {
                **ce_metrics,
                **full_hu_metrics,
                **nonhu_metrics,
                **short_kl_metrics,
                **sae_metrics,
                **lsor_metrics,
                **rh_metrics,
                **grad_metrics,
                "loss/total": float((l_ce + aux_total).detach().item()),
            }
        elif method == "longce_cpt":
            # LongCE: reweight CE by utility
            utility = (
                utility_scores
                if utility_scores is not None
                else torch.zeros_like(labels, dtype=torch.float32, device=device)
            )
            weights = utility.exp().clamp(0.25, 4.0)
            if loss_mask.any():
                weights = weights / weights[loss_mask].mean().clamp(min=1e-6)
            else:
                weights = weights / weights.mean().clamp(min=1e-6)
            ce_per_pos = F.cross_entropy(
                out.logits_full.float().reshape(-1, out.logits_full.size(-1)),
                labels.reshape(-1), reduction="none",
            ).view_as(labels)
            if loss_mask.any():
                total_loss = (ce_per_pos[loss_mask] * weights[loss_mask]).mean()
            else:
                total_loss = ce_per_pos.mean()
            loss_metrics = {
                "loss/ce": float(total_loss.detach().item()),
                "longce/weight_mean": float(weights[loss_mask].detach().float().mean().item()) if loss_mask.any() else float(weights.detach().float().mean().item()),
            }
            optimizer.zero_grad()
            total_loss.backward()
        else:
            # CE-CPT or Adapter-CE
            total_loss = ce_from_logits(out.logits_full, labels, loss_mask)
            loss_metrics = {"loss/ce": float(total_loss.detach().item())}
            optimizer.zero_grad()
            total_loss.backward()

        # FSDP owns model-gradient reduction. Head adapters live outside FSDP,
        # so they still need explicit data-parallel averaging.
        if world_size > 1:
            if not fsdp_active:
                _sync_module_grads(model, world_size)
            if adapters is not None:
                _sync_module_grads(adapters, world_size)

        # Clip and step
        if fsdp_active and hasattr(model, "clip_grad_norm_"):
            model.clip_grad_norm_(train.grad_clip)
            if adapter_params:
                torch.nn.utils.clip_grad_norm_(adapter_params, train.grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(all_params, train.grad_clip)
        optimizer.step()

        tokens_seen += input_ids.numel()

        # Logging
        if (step + 1) % train.eval_interval == 0 or step == 0:
            entry = {
                "step": step + 1,
                "tokens": tokens_seen,
                "lr": lr_now,
                "lr/adapter": adapter_lr_now if adapter_params else 0.0,
                "wall_elapsed": time.monotonic() - wall_start,
                **loss_metrics,
                **out.metrics,
            }
            metrics_log.append(entry)
            if _is_rank0(rank):
                _write_tensorboard_entry(tb_writer, entry, world_size)
                print(json.dumps(entry, sort_keys=True), flush=True)

        # Val CE
        if (step + 1) % (train.eval_interval * 5) == 0:
            val_ce = _eval_val_ce(
                model,
                val_dataset,
                train.batch_size,
                device,
                mask_format,
                attention_mask_mode=attention_mask_mode,
            )
            if _is_rank0(rank):
                val_entry = {
                    "step": step + 1,
                    "tokens": tokens_seen,
                    "val_ce": val_ce,
                    "wall_elapsed": time.monotonic() - wall_start,
                }
                metrics_log.append(val_entry)
                _write_tensorboard_entry(tb_writer, val_entry, world_size)
                print(json.dumps(val_entry, sort_keys=True), flush=True)

        should_save = int(save_interval) > 0 and (step + 1) < total_steps and (step + 1) % int(save_interval) == 0
        if should_save:
            _barrier()
            checkpoint_model_state = _model_state_dict_for_checkpoint(model, fsdp_active)
            include_optimizer_state = bool((not fsdp_active) or (fsdp_active and use_adapters and cure_params.freeze_base_model))
            if _is_rank0(rank):
                checkpoint_extra = {
                    "run_name": run_name,
                    "method": method,
                    "total_steps": int(total_steps),
                    "save_interval": int(save_interval),
                    "offline_signal_dir": str(offline_signal_dir) if offline_signal_dir is not None else None,
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                    "world_size": int(world_size),
                    "fsdp": bool(fsdp_active),
                    "optimizer_saved": include_optimizer_state,
                }
                _save_latest_checkpoint(
                    run_dir=run_dir,
                    model=model,
                    adapters=adapters,
                    optimizer=optimizer,
                    metrics_log=metrics_log,
                    next_step=step + 1,
                    tokens_seen=tokens_seen,
                    elapsed_wall_seconds=time.monotonic() - wall_start,
                    extra=checkpoint_extra,
                    model_state_dict=checkpoint_model_state,
                    include_optimizer=include_optimizer_state,
                )
                archive_path = None
                if archive_interval_checkpoints:
                    archive_path = _interval_checkpoint_path(run_dir, step + 1, tokens_seen * world_size)
                    shutil.copy2(_latest_checkpoint_path(run_dir), archive_path)
                write_json(run_dir / "metrics_partial.json", {"metrics": metrics_log})
                if tb_writer is not None:
                    tb_writer.flush()
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_saved",
                            "run": run_name,
                            "next_step": int(step + 1),
                            "path": str(_latest_checkpoint_path(run_dir)),
                            "archive_path": str(archive_path) if archive_path is not None else None,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            _barrier()
            _save_rank_resume_state(run_dir, rank, step + 1, dataset, val_dataset, device)
            _barrier()
            del checkpoint_model_state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    wall_total = time.monotonic() - wall_start
    _barrier()
    include_optimizer_state = bool((not fsdp_active) or (fsdp_active and use_adapters and cure_params.freeze_base_model))

    summary = {
        "run_name": run_name,
        "method": method,
        "tokens_seen": tokens_seen,
        "total_steps": total_steps,
        "wall_seconds": wall_total,
        "retrieval_heads": sorted([list(h) for h in retrieval_heads]),
        "num_retrieval_heads": len(retrieval_heads),
        "adapter_parameters": adapters.num_parameters() if adapters is not None else 0,
        "lora_rank": cure_params.lora_rank,
        "lora_alpha": cure_params.lora_alpha,
        "lambda_rh_ce": cure_params.lambda_rh_ce,
        "lambda_rh_kd": cure_params.lambda_rh_kd,
        "lambda_cov": cure_params.lambda_cov,
        "lambda_full_hu_ce": cure_params.lambda_full_hu_ce,
        "lambda_nonhu_logp": cure_params.lambda_nonhu_logp,
        "lambda_short_kl": cure_params.lambda_short_kl,
        "cure_main_loss_mask": cure_params.cure_main_loss_mask,
        "rh_bottleneck_scope": cure_params.rh_bottleneck_scope,
        "sae_enabled": cure_params.sae_enabled,
        "sae_checkpoint_path": cure_params.sae_checkpoint_path,
        "sae_features_path": cure_params.sae_features_path,
        "sae_require_validated_features": cure_params.sae_require_validated_features,
        "lambda_sae_margin": cure_params.lambda_sae_margin,
        "lambda_sae_match": cure_params.lambda_sae_match,
        "sae_margin_gamma": cure_params.sae_margin_gamma,
        "lsor_enabled": cure_params.lsor_enabled,
        "lambda_lsor": cure_params.lambda_lsor,
        "lsor_top_k": cure_params.lsor_top_k,
        "lsor_window": cure_params.lsor_window,
        "lsor_warmup_tokens": cure_params.lsor_warmup_tokens,
        "lsor_max_context_tokens": cure_params.lsor_max_context_tokens,
        "adapter_lr": adapter_lr if adapter_params else None,
        "adapter_weight_decay": cure_params.adapter_weight_decay if adapter_params else None,
        "freeze_base_model": cure_params.freeze_base_model,
        "save_interval": int(save_interval),
        "resume_if_available": bool(resume_if_available),
        "resumed_from": resumed_from,
        "tensorboard_dir": str(tb_log_dir) if tensorboard else None,
        "archive_interval_checkpoints": bool(archive_interval_checkpoints),
        "attention_mask_mode": attention_mask_mode,
        "gradient_checkpointing": bool(use_grad_ckpt),
        "fsdp": bool(fsdp_active),
        "optimizer_saved": include_optimizer_state,
        "final_metrics": metrics_log[-1] if metrics_log else {},
    }

    final_model_state = _model_state_dict_for_checkpoint(model, fsdp_active)
    if _is_rank0(rank):
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "metrics.jsonl", metrics_log)
        # Save checkpoint
        ckpt = {
            "model": final_model_state if final_model_state is not None else model.state_dict(),
            "summary": summary,
        }
        if include_optimizer_state:
            ckpt["optimizer"] = optimizer.state_dict()
        if adapters is not None:
            ckpt["adapters"] = adapters.state_dict()
        torch.save(ckpt, run_dir / "checkpoint.pt")
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    return summary


@torch.no_grad()
def _eval_val_ce(
    model: torch.nn.Module,
    val_dataset: PackedFineWebDataset,
    batch_size: int,
    device: torch.device,
    mask_format: str,
    attention_mask_mode: str = "document",
) -> float:
    model.eval()
    batch = val_dataset.sample(batch_size)
    input_ids = torch.as_tensor(batch["input_ids"], device=device)
    labels = torch.as_tensor(batch["labels"], device=device)
    loss_mask = torch.as_tensor(batch["loss_mask"], device=device)
    doc_ids = torch.as_tensor(batch["doc_ids_full"], device=device)[:, :-1]
    full_mask = (
        None
        if attention_mask_mode == "causal"
        else document_causal_attention_mask(doc_ids, mask_format=mask_format).to(device)
    )
    out = model(input_ids, attention_mask=full_mask, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out["logits"]
    ce = ce_from_logits(logits, labels, loss_mask)
    model.train()
    return float(ce.detach().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="CURE-CPT training")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--method", required=True, choices=["ce_cpt", "adapter_ce", "longce_cpt", "cure_cpt"])
    parser.add_argument("--model-path", default="/gemini/space/private/zjc/models/Qwen2.5-0.5B")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ablation-results", default=None)
    parser.add_argument("--offline-signal-dir", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--tokens-per-run", type=int, default=100_000_000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--adapter-lr", type=float, default=None)
    parser.add_argument("--adapter-weight-decay", type=float, default=0.0)
    parser.add_argument("--freeze-base-model", action="store_true")
    parser.add_argument("--lambda-rh-ce", type=float, default=0.1)
    parser.add_argument("--lambda-rh-kd", type=float, default=0.05)
    parser.add_argument("--lambda-cov", type=float, default=0.005)
    parser.add_argument("--lambda-full-hu-ce", type=float, default=0.0)
    parser.add_argument("--lambda-nonhu-logp", type=float, default=0.0)
    parser.add_argument("--lambda-short-kl", type=float, default=0.0)
    parser.add_argument(
        "--cure-main-loss-mask",
        choices=["all", "valid_remote", "high_utility", "none"],
        default="all",
    )
    parser.add_argument("--sae-enable", action="store_true")
    parser.add_argument("--sae-checkpoint", default=None)
    parser.add_argument("--sae-features", default=None)
    parser.add_argument("--allow-unvalidated-sae-features", action="store_true")
    parser.add_argument("--lambda-sae-margin", type=float, default=0.02)
    parser.add_argument("--lambda-sae-match", type=float, default=0.10)
    parser.add_argument("--sae-margin-gamma", type=float, default=0.5)
    parser.add_argument("--sae-match-clip", type=float, default=None)
    parser.add_argument("--lsor-enable", action="store_true")
    parser.add_argument("--lambda-lsor", type=float, default=0.0)
    parser.add_argument("--lsor-top-k", type=int, default=16)
    parser.add_argument("--lsor-window", type=int, default=128)
    parser.add_argument("--lsor-warmup-tokens", type=int, default=10_000_000)
    parser.add_argument("--lsor-max-context-tokens", type=int, default=4096)
    parser.add_argument("--utility-top-fraction-training", type=float, default=0.10)
    parser.add_argument("--rh-bottleneck-scope", choices=["all_layers", "routed_layers"], default="all_layers")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--resume-if-available", action="store_true")
    parser.add_argument("--no-archive-interval-checkpoints", action="store_true")
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--attention-mask-mode", choices=["document", "causal"], default="document")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    rank, world_size, local_rank, device = _distributed_context()

    paths = Paths(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    cure_params = CUREParams(
        seq_len=args.seq_len,
        local_window=args.local_window,
        tokens_per_run=args.tokens_per_run,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        adapter_lr=args.adapter_lr,
        adapter_weight_decay=args.adapter_weight_decay,
        freeze_base_model=args.freeze_base_model,
        lambda_rh_ce=args.lambda_rh_ce,
        lambda_rh_kd=args.lambda_rh_kd,
        lambda_cov=args.lambda_cov,
        lambda_full_hu_ce=args.lambda_full_hu_ce,
        lambda_nonhu_logp=args.lambda_nonhu_logp,
        lambda_short_kl=args.lambda_short_kl,
        cure_main_loss_mask=args.cure_main_loss_mask,
        sae_enabled=args.sae_enable,
        sae_checkpoint_path=args.sae_checkpoint,
        sae_features_path=args.sae_features,
        sae_require_validated_features=not args.allow_unvalidated_sae_features,
        lambda_sae_margin=args.lambda_sae_margin,
        lambda_sae_match=args.lambda_sae_match,
        sae_margin_gamma=args.sae_margin_gamma,
        sae_match_clip=args.sae_match_clip,
        lsor_enabled=args.lsor_enable,
        lambda_lsor=args.lambda_lsor,
        lsor_top_k=args.lsor_top_k,
        lsor_window=args.lsor_window,
        lsor_warmup_tokens=args.lsor_warmup_tokens,
        lsor_max_context_tokens=args.lsor_max_context_tokens,
        utility_top_fraction_training=args.utility_top_fraction_training,
        rh_bottleneck_scope=args.rh_bottleneck_scope,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
    )
    train_params = TrainParams(
        batch_size=args.batch_size,
        steps=args.steps or max(1, args.tokens_per_run // max(1, args.seq_len * args.batch_size)),
        lr=args.lr,
        seed=args.seed,
        eval_interval=args.eval_interval,
    )

    train_cure(
        run_name=args.run_name,
        method=args.method,
        paths=paths,
        cure_params=cure_params,
        train=train_params,
        device=device,
        rank=rank,
        world_size=world_size,
        ablation_results_path=args.ablation_results,
        offline_signal_dir=args.offline_signal_dir,
        checkpoint_path=args.checkpoint_path,
        save_interval=args.save_interval,
        resume_if_available=args.resume_if_available,
        tensorboard=not args.no_tensorboard,
        tensorboard_dir=args.tensorboard_dir,
        archive_interval_checkpoints=not args.no_archive_interval_checkpoints,
        attention_mask_mode=args.attention_mask_mode,
        use_fsdp=args.fsdp,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )


if __name__ == "__main__":
    main()
