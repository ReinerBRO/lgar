from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CUREParams:
    # Stage A: utility mining
    seq_len: int = 8192
    short_window: int = 1024       # used by LGAR mining (local_window alias)
    local_window: int = 1024       # used by CURE forward for RH-bottleneck
    min_remote_margin: int = 256
    utility_top_fraction_ablation: float = 0.05
    utility_top_fraction_training: float = 0.10
    full_nll_max_quantile: float = 0.80
    long_nll_max_quantile: float = 0.80   # alias for LGAR mining compatibility
    lsd_top_fraction: float = 0.10        # alias for LGAR mining compatibility

    # Stage B: head ablation
    ablation_layer_fraction: float = 1.0 / 3.0
    ablation_top_k_fraction: float = 0.05
    ablation_min_delta: float = 0.01
    ablation_batch_size: int = 4
    ablation_calibration_sequences: int = 512
    ablation_split_half: bool = True

    # Stage C: adapters
    lora_rank: int = 8
    lora_alpha: float = 16.0
    adapter_lr: float | None = None
    adapter_weight_decay: float = 0.0

    # Training
    lambda_rh_ce: float = 0.1
    lambda_rh_kd: float = 0.05
    lambda_cov: float = 0.005
    lambda_full_hu_ce: float = 0.0
    lambda_nonhu_logp: float = 0.0
    lambda_short_kl: float = 0.0
    cure_main_loss_mask: str = "all"
    cov_warmup_tokens: int = 10_000_000
    rh_bottleneck_scope: str = "all_layers"

    # CURE-v2 SAE evidence-feature steering. Defaults are off so CURE-v0 is
    # behaviorally unchanged unless the caller provides validated features.
    sae_enabled: bool = False
    sae_checkpoint_path: str | None = None
    sae_features_path: str | None = None
    sae_require_validated_features: bool = True
    lambda_sae_margin: float = 0.02
    lambda_sae_match: float = 0.10
    sae_margin_gamma: float = 0.5
    sae_match_clip: float | None = None

    # Optional Local Subspace Orthogonal Residual-write regularizer.
    lsor_enabled: bool = False
    lambda_lsor: float = 0.0
    lsor_top_k: int = 16
    lsor_window: int = 128
    lsor_warmup_tokens: int = 10_000_000
    lsor_max_context_tokens: int = 4096

    context_len: int = 4096
    tokens_per_run: int = 100_000_000
    lr: float = 1.0e-5
    min_lr: float = 1.0e-6
    warmup_fraction: float = 0.03
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    freeze_base_model: bool = False
    seed: int = 1337
    dtype: str = "bf16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True


def ablation_candidate_layers(num_layers: int, fraction: float) -> list[int]:
    count = max(1, int(round(num_layers * float(fraction))))
    start = max(0, int(num_layers) - count)
    return list(range(start, int(num_layers)))
