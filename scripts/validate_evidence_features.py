#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curcpt.head_ablation import load_ablation_results
from curcpt.sae import (
    load_evidence_feature_set,
    load_sparse_autoencoder,
    sae_decoder_feature_contribution,
)
from scripts.precompute_sae_feature_targets import (
    _capture_selected_features,
    _device_from_arg,
    _materialize_indices,
    _register_masked_head_hooks,
)


def _normalise_feature_payload(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("validation currently expects one feature layer")
        payload = payload[0]
    if "layers" in payload:
        layers = payload["layers"]
        if not isinstance(layers, list) or len(layers) != 1:
            raise ValueError("validation currently expects one feature layer")
        payload = layers[0]
    return dict(payload)


def _register_feature_intervention_hook(
    model: torch.nn.Module,
    layer: int,
    sae: torch.nn.Module,
    feature_ids: tuple[int, ...],
    token_mask: torch.Tensor,
    mode: str,
    target_acts: torch.Tensor | None = None,
) -> torch.utils.hooks.RemovableHandle:
    from lgar_cpt.modeling import qwen_layers

    hook_point = getattr(qwen_layers(model)[int(layer)], "post_attention_layernorm", None)
    if hook_point is None:
        raise AttributeError(
            f"Layer {layer} has no post_attention_layernorm; cannot validate resid_mid features"
        )
    token_mask = token_mask.bool()

    def hook_fn(module, args, kwargs):
        residual = args[0]
        acts = sae.encode(residual)
        index = torch.as_tensor(feature_ids, device=acts.device, dtype=torch.long)
        selected = acts.index_select(dim=-1, index=index)
        if mode == "ablate":
            values = selected
            contribution = sae_decoder_feature_contribution(sae, values, feature_ids)
            updated = residual - contribution * token_mask[..., None].to(contribution.dtype)
        elif mode == "patch":
            if target_acts is None:
                raise ValueError("patch mode requires target_acts")
            delta = target_acts.to(device=selected.device, dtype=selected.dtype) - selected
            contribution = sae_decoder_feature_contribution(sae, delta, feature_ids)
            updated = residual + contribution * token_mask[..., None].to(contribution.dtype)
        else:
            raise ValueError(f"unknown feature intervention mode={mode!r}")
        updated = updated.to(dtype=residual.dtype)
        return (updated,) + args[1:], kwargs

    return hook_point.register_forward_pre_hook(hook_fn, with_kwargs=True)


@torch.no_grad()
def _logp_for_mask(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    token_mask: torch.Tensor,
    extra_handles: list[torch.utils.hooks.RemovableHandle] | None = None,
) -> torch.Tensor:
    from lgar_cpt.mining import gather_logprob
    from lgar_cpt.modeling import unwrap_logits

    try:
        logits = unwrap_logits(model(input_ids, attention_mask=attention_mask, use_cache=False))
    finally:
        for handle in extra_handles or []:
            handle.remove()
    logp = gather_logprob(logits, labels)
    return logp[token_mask.bool()].detach().float()


def _mean_or_zero(values: list[torch.Tensor]) -> float:
    if not values:
        return 0.0
    joined = torch.cat([v.reshape(-1).float().cpu() for v in values if v.numel() > 0], dim=0)
    if joined.numel() == 0:
        return 0.0
    return float(joined.mean().item())


def _random_feature_ids(feature_dim: int, selected: tuple[int, ...], seed: int) -> tuple[int, ...]:
    selected_set = {int(x) for x in selected}
    candidates = [idx for idx in range(int(feature_dim)) if idx not in selected_set]
    if len(candidates) < len(selected):
        candidates = [idx for idx in range(int(feature_dim))]
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(np.asarray(candidates, dtype=np.int64), size=len(selected), replace=False)
    return tuple(int(x) for x in picked.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causal ablation/patching validation for mined SAE evidence features."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--offline-signal-dir", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--sae-checkpoint", default=None)
    parser.add_argument("--ablation-results", default=None)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-features", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequences", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--local-window", type=int, default=1024)
    parser.add_argument("--min-ablation-margin", type=float, default=0.01)
    parser.add_argument("--min-feature-ablation-logp-drop", type=float, default=0.01)
    parser.add_argument("--min-patch-gain", type=float, default=0.01)
    parser.add_argument("--min-masked-head-logp-drop", type=float, default=0.01)
    parser.add_argument("--min-masked-head-drop", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if not args.ablation_results:
        raise ValueError(
            "Feature validation requires --ablation-results so masked retrieval heads "
            "can be checked against answer logprob"
        )

    from lgar_cpt.data import PackedSequenceLayoutDataset
    from lgar_cpt.mining import (
        attention_mask_format_for_model,
        document_causal_attention_mask,
        local_document_attention_mask,
    )
    from lgar_cpt.modeling import load_qwen_causal_lm, load_tokenizer

    device = _device_from_arg(args.device)
    tokenizer = load_tokenizer(args.model_path)
    model = load_qwen_causal_lm(
        args.model_path,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=False,
    ).to(device)
    model.eval()

    features = load_evidence_feature_set(args.features, require_validated=False)
    sae_path = args.sae_checkpoint or features.sae_checkpoint
    if not sae_path:
        raise ValueError("--sae-checkpoint is required unless features JSON has sae_checkpoint")
    sae = load_sparse_autoencoder(sae_path, device=device)
    if int(sae.encoder.in_features) != int(model.config.hidden_size):
        raise ValueError(
            f"SAE input_dim={sae.encoder.in_features} does not match model hidden_size={model.config.hidden_size}"
        )
    random_ids = _random_feature_ids(int(sae.encoder.out_features), features.feature_ids, args.seed)

    retrieval_heads: set[tuple[int, int]] = set()
    if args.ablation_results:
        retrieval_heads, _ = load_ablation_results(args.ablation_results)

    signal_dir = Path(args.offline_signal_dir)
    offline_labels = np.load(signal_dir / "offline_lsd_labels.npy", mmap_mode="r")
    offline_valid = np.load(signal_dir / "offline_lsd_valid.npy", mmap_mode="r")
    dataset = PackedSequenceLayoutDataset(
        cache_dir=args.cache_dir,
        signal_dir=signal_dir,
        pad_token_id=int(tokenizer.pad_token_id),
        seed=int(args.seed),
    )
    num_sequences = min(dataset.num_sequences, int(args.max_sequences))
    mask_format = attention_mask_format_for_model(model)

    selected_ablation_drops: list[torch.Tensor] = []
    random_ablation_drops: list[torch.Tensor] = []
    short_patch_gains: list[torch.Tensor] = []
    masked_head_logp_drops: list[torch.Tensor] = []
    masked_head_drops: list[torch.Tensor] = []
    token_total = 0

    for start in range(0, num_sequences, int(args.batch_size)):
        if token_total >= int(args.max_tokens):
            break
        end = min(start + int(args.batch_size), num_sequences)
        indices = np.arange(start, end, dtype=np.int64)
        batch = _materialize_indices(dataset, indices)
        input_ids = torch.as_tensor(batch["input_ids"], device=device)
        labels = torch.as_tensor(batch["labels"], device=device)
        doc_ids = torch.as_tensor(batch["doc_ids_full"][:, :-1], device=device)
        loss_mask = torch.as_tensor(batch["loss_mask"], device=device).bool()
        token_mask = (
            torch.as_tensor(offline_labels[indices], device=device).bool()
            & torch.as_tensor(offline_valid[indices], device=device).bool()
            & loss_mask
        )
        if not token_mask.any():
            continue
        remaining = int(args.max_tokens) - token_total
        coords = token_mask.nonzero(as_tuple=False)
        if coords.shape[0] > remaining:
            keep = coords[:remaining]
            clipped = torch.zeros_like(token_mask)
            clipped[keep[:, 0], keep[:, 1]] = True
            token_mask = clipped
        token_total += int(token_mask.sum().item())

        full_mask = document_causal_attention_mask(doc_ids, mask_format=mask_format).to(device)
        short_mask = local_document_attention_mask(
            doc_ids,
            int(args.local_window),
            mask_format=mask_format,
        ).to(device)

        full_logp = _logp_for_mask(model, input_ids, labels, full_mask, token_mask)

        ablate_handle = _register_feature_intervention_hook(
            model,
            int(features.layer),
            sae,
            features.feature_ids,
            token_mask,
            mode="ablate",
        )
        full_ablated_logp = _logp_for_mask(
            model,
            input_ids,
            labels,
            full_mask,
            token_mask,
            extra_handles=[ablate_handle],
        )
        selected_ablation_drops.append(full_logp - full_ablated_logp)

        random_handle = _register_feature_intervention_hook(
            model,
            int(features.layer),
            sae,
            random_ids,
            token_mask,
            mode="ablate",
        )
        random_ablated_logp = _logp_for_mask(
            model,
            input_ids,
            labels,
            full_mask,
            token_mask,
            extra_handles=[random_handle],
        )
        random_ablation_drops.append(full_logp - random_ablated_logp)

        full_acts = _capture_selected_features(
            model,
            sae,
            int(features.layer),
            features.feature_ids,
            input_ids,
            full_mask,
        )
        short_logp = _logp_for_mask(model, input_ids, labels, short_mask, token_mask)
        patch_handle = _register_feature_intervention_hook(
            model,
            int(features.layer),
            sae,
            features.feature_ids,
            token_mask,
            mode="patch",
            target_acts=full_acts,
        )
        patched_short_logp = _logp_for_mask(
            model,
            input_ids,
            labels,
            short_mask,
            token_mask,
            extra_handles=[patch_handle],
        )
        short_patch_gains.append(patched_short_logp - short_logp)

        masked_logp_handles = _register_masked_head_hooks(
            model,
            retrieval_heads,
            doc_ids,
            int(args.local_window),
            mask_format,
        )
        masked_head_logp = _logp_for_mask(
            model,
            input_ids,
            labels,
            full_mask,
            token_mask,
            extra_handles=masked_logp_handles,
        )
        masked_head_logp_drops.append(full_logp - masked_head_logp)

        masked_feature_handles = _register_masked_head_hooks(
            model,
            retrieval_heads,
            doc_ids,
            int(args.local_window),
            mask_format,
        )
        masked_acts = _capture_selected_features(
            model,
            sae,
            int(features.layer),
            features.feature_ids,
            input_ids,
            full_mask,
            extra_handles=masked_feature_handles,
        )
        masked_head_drops.append((full_acts - masked_acts)[token_mask].mean(dim=-1))

        print(
            json.dumps(
                {
                    "event": "validate_evidence_features_progress",
                    "start": int(start),
                    "end": int(end),
                    "tokens": int(token_total),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    selected_drop = _mean_or_zero(selected_ablation_drops)
    random_drop = _mean_or_zero(random_ablation_drops)
    patch_gain = _mean_or_zero(short_patch_gains)
    masked_head_answer_drop = _mean_or_zero(masked_head_logp_drops)
    masked_drop = _mean_or_zero(masked_head_drops)
    passed = (
        token_total > 0
        and selected_drop >= float(args.min_feature_ablation_logp_drop)
        and selected_drop >= random_drop + float(args.min_ablation_margin)
        and masked_head_answer_drop >= float(args.min_masked_head_logp_drop)
        and patch_gain >= float(args.min_patch_gain)
        and masked_drop >= float(args.min_masked_head_drop)
    )
    report = {
        "passed": bool(passed),
        "layer": int(features.layer),
        "hook_point": features.hook_point,
        "feature_ids": list(features.feature_ids),
        "random_feature_ids": list(random_ids),
        "num_tokens": int(token_total),
        "selected_ablation_delta_logp": selected_drop,
        "random_ablation_delta_logp": random_drop,
        "masked_head_answer_delta_logp": masked_head_answer_drop,
        "short_patch_delta_logp": patch_gain,
        "masked_head_feature_drop": masked_drop,
        "thresholds": {
            "min_ablation_margin": float(args.min_ablation_margin),
            "min_feature_ablation_logp_drop": float(args.min_feature_ablation_logp_drop),
            "min_patch_gain": float(args.min_patch_gain),
            "min_masked_head_logp_drop": float(args.min_masked_head_logp_drop),
            "min_masked_head_drop": float(args.min_masked_head_drop),
        },
    }
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    payload = _normalise_feature_payload(args.features)
    payload["validation"] = report
    payload["validation_passed"] = bool(passed)
    payload.setdefault("sae_checkpoint", str(sae_path))
    output_features = (
        Path(args.output_features)
        if args.output_features
        else output_report.with_name(output_report.stem + "_features.json")
    )
    output_features.parent.mkdir(parents=True, exist_ok=True)
    output_features.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "validate_evidence_features_done",
                "passed": bool(passed),
                "report": str(output_report),
                "features": str(output_features),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
