from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Paths:
    model_path: str = "/gemini/space/private/zjc/models/Qwen2.5-0.5B"
    raw_data_dir: str = "/gemini/space/private/zjc/data/fineweb_edu_100BT-shuffled"
    cache_dir: str = "/gemini/space/private/zjc/goals/lgar/data/qwen_fineweb_stage0"
    output_dir: str = "/gemini/space/private/zjc/goals/lgar"
    ruler_dir: str = "/gemini/space/private/zjc/data/RULER"
    nolima_dir: str = "/gemini/space/private/zjc/data/NoLiMa"
    short_mc_dir: str = "/gemini/space/private/zjc/data"


@dataclass(frozen=True)
class LGARParams:
    seq_len: int = 8192
    short_window: int = 1024
    min_remote_margin: int = 256
    local_window: int = 1024
    lsd_top_fraction: float = 0.10
    long_nll_max_quantile: float = 0.70
    router_target_budget: float = 0.10
    final_global_budget: float = 0.25
    routed_layer_fraction: float = 1.0 / 3.0
    router_hidden_dim: int = 512
    lambda_router_final: float = 0.02
    lambda_budget: float = 0.005
    entropy_weight: float = 0.0


@dataclass(frozen=True)
class TrainParams:
    batch_size: int = 1
    steps: int = 100
    lr: float = 1.0e-5
    min_lr: float = 1.0e-6
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_fraction: float = 0.03
    eval_interval: int = 20
    eval_batches: int = 4
    seed: int = 1337
    dtype: str = "bf16"
    attn_implementation: str = "eager"
    gradient_checkpointing: bool = True


def routed_layer_indices(num_layers: int, routed_layer_fraction: float) -> list[int]:
    count = max(1, int(round(num_layers * float(routed_layer_fraction))))
    start = max(0, int(num_layers) - count)
    return list(range(start, int(num_layers)))


# Compatibility for the copied evaluation/data helpers.
C3TParams = LGARParams
