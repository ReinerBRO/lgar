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

from curcpt.sae import load_evidence_feature_set, load_sparse_autoencoder
from scripts.precompute_sae_feature_targets import _device_from_arg, _materialize_indices
from scripts.validate_evidence_features import _logp_for_mask


def _register_grouped_single_feature_ablation_hook(
    model: torch.nn.Module,
    layer: int,
    sae: torch.nn.Module,
    feature_ids_per_row: torch.Tensor,
    token_mask: torch.Tensor,
) -> list[torch.utils.hooks.RemovableHandle]:
    from lgar_cpt.modeling import qwen_layers

    decoder_layer = qwen_layers(model)[int(layer)]
    hook_point = getattr(decoder_layer, "post_attention_layernorm", None)
    if hook_point is None:
        raise AttributeError(
            f"Layer {layer} has no post_attention_layernorm; cannot ablate resid_mid features"
        )
    token_mask = token_mask.bool()
    state: dict[str, torch.Tensor] = {}

    def hook_fn(module, args, kwargs):
        residual = args[0]
        row_feature_ids = feature_ids_per_row.to(device=residual.device, dtype=torch.long)
        if int(row_feature_ids.shape[0]) != int(residual.shape[0]):
            raise ValueError(
                "feature_ids_per_row must have one feature id per repeated batch row; "
                f"got {tuple(row_feature_ids.shape)} for residual batch={int(residual.shape[0])}"
            )
        acts = sae.encode(residual)
        gather_index = row_feature_ids.view(-1, 1, 1).expand(-1, int(acts.shape[1]), 1)
        values = acts.gather(dim=-1, index=gather_index).squeeze(-1)
        vectors = sae.decoder.weight[:, row_feature_ids].T.to(
            device=residual.device,
            dtype=values.dtype,
        )
        vectors = vectors * sae.input_scale.to(device=residual.device, dtype=values.dtype)[None, :]
        contribution = values[..., None] * vectors[:, None, :]
        updated = residual - contribution * token_mask[..., None].to(contribution.dtype)
        updated = updated.to(dtype=residual.dtype)
        state["delta"] = (residual - updated).detach()
        return (updated,) + args[1:], kwargs

    def layer_output_hook(module, args, kwargs, output):
        delta = state.pop("delta", None)
        if delta is None:
            return output
        if torch.is_tensor(output):
            return output - delta.to(device=output.device, dtype=output.dtype)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            first = output[0] - delta.to(device=output[0].device, dtype=output[0].dtype)
            return (first, *output[1:])
        raise TypeError(f"unsupported decoder layer output type for SAE intervention: {type(output)!r}")

    return [
        hook_point.register_forward_pre_hook(hook_fn, with_kwargs=True),
        decoder_layer.register_forward_hook(layer_output_hook, with_kwargs=True),
    ]


def _mean_or_zero(values: list[torch.Tensor]) -> float:
    if not values:
        return 0.0
    tensors = [value.reshape(-1).float().cpu() for value in values if value.numel() > 0]
    if not tensors:
        return 0.0
    return float(torch.cat(tensors, dim=0).mean().item())


def _normalise_feature_payload(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("expected one feature layer")
        payload = payload[0]
    if "layers" in payload:
        layers = payload["layers"]
        if not isinstance(layers, list) or len(layers) != 1:
            raise ValueError("expected one feature layer")
        payload = layers[0]
    return dict(payload)


def _random_feature_ids(feature_dim: int, exclude: set[int], count: int, seed: int) -> tuple[int, ...]:
    candidates = [idx for idx in range(int(feature_dim)) if idx not in exclude]
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(np.asarray(candidates, dtype=np.int64), size=int(count), replace=False)
    return tuple(int(x) for x in picked.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank SAE features by direct answer-logprob causal ablation.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--offline-signal-dir", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--sae-checkpoint", default=None)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-features", required=True)
    parser.add_argument("--top-k-features", type=int, default=8)
    parser.add_argument("--candidate-feature-ids", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequences", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--feature-group-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from lgar_cpt.data import PackedSequenceLayoutDataset
    from lgar_cpt.mining import attention_mask_format_for_model, document_causal_attention_mask
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

    if args.candidate_feature_ids:
        candidate_ids = tuple(int(x) for x in str(args.candidate_feature_ids).split(",") if x.strip())
    else:
        candidate_ids = tuple(int(x) for x in features.feature_ids)
    if not candidate_ids:
        raise ValueError("no candidate feature ids")
    random_ids = _random_feature_ids(
        int(sae.encoder.out_features),
        exclude={int(x) for x in candidate_ids},
        count=len(candidate_ids),
        seed=int(args.seed),
    )
    all_eval_ids = list(candidate_ids) + list(random_ids)

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

    drops_by_feature: dict[int, list[torch.Tensor]] = {int(fid): [] for fid in all_eval_ids}
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

        attention_mask = document_causal_attention_mask(doc_ids, mask_format=mask_format).to(device)
        full_logp = _logp_for_mask(model, input_ids, labels, attention_mask, token_mask)
        per_sample_token_counts = token_mask.sum(dim=1).detach().cpu().tolist()
        full_segments = list(torch.split(full_logp, [int(x) for x in per_sample_token_counts]))
        group_size = max(1, int(args.feature_group_size))
        for group_start in range(0, len(all_eval_ids), group_size):
            group_ids = [int(x) for x in all_eval_ids[group_start : group_start + group_size]]
            repeated_input_ids = input_ids.repeat_interleave(len(group_ids), dim=0)
            repeated_labels = labels.repeat_interleave(len(group_ids), dim=0)
            repeated_token_mask = token_mask.repeat_interleave(len(group_ids), dim=0)
            repeated_attention_mask = attention_mask.repeat_interleave(len(group_ids), dim=0)
            feature_ids_per_row = torch.as_tensor(
                group_ids * int(input_ids.shape[0]),
                device=device,
                dtype=torch.long,
            )
            handles = _register_grouped_single_feature_ablation_hook(
                model,
                int(features.layer),
                sae,
                feature_ids_per_row,
                repeated_token_mask,
            )
            ablated = _logp_for_mask(
                model,
                repeated_input_ids,
                repeated_labels,
                repeated_attention_mask,
                repeated_token_mask,
                extra_handles=handles,
            )
            offset = 0
            for sample_idx, count in enumerate(per_sample_token_counts):
                count = int(count)
                full_segment = full_segments[sample_idx]
                for feature_id in group_ids:
                    ablated_segment = ablated[offset : offset + count]
                    offset += count
                    drops_by_feature[int(feature_id)].append(full_segment - ablated_segment)
        print(
            json.dumps(
                {
                    "event": "rank_sae_features_progress",
                    "start": int(start),
                    "end": int(end),
                    "tokens": int(token_total),
                    "features": int(len(all_eval_ids)),
                    "feature_group_size": int(group_size),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    candidate_records = []
    random_records = []
    for feature_id in candidate_ids:
        candidate_records.append(
            {
                "feature_id": int(feature_id),
                "answer_logp_drop": _mean_or_zero(drops_by_feature[int(feature_id)]),
            }
        )
    for feature_id in random_ids:
        random_records.append(
            {
                "feature_id": int(feature_id),
                "answer_logp_drop": _mean_or_zero(drops_by_feature[int(feature_id)]),
            }
        )
    candidate_records.sort(key=lambda item: item["answer_logp_drop"], reverse=True)
    random_records.sort(key=lambda item: item["answer_logp_drop"], reverse=True)
    selected = candidate_records[: int(args.top_k_features)]

    payload = _normalise_feature_payload(args.features)
    old_scores = dict(payload.get("scores", {}))
    payload["feature_ids"] = [int(item["feature_id"]) for item in selected]
    payload["scores"] = {
        str(item["feature_id"]): {
            **dict(old_scores.get(str(item["feature_id"]), {})),
            "feature_id": int(item["feature_id"]),
            "answer_logp_drop": float(item["answer_logp_drop"]),
            "causal_rank_source": "single_feature_answer_logp_ablation",
        }
        for item in selected
    }
    random_mean = float(np.mean([item["answer_logp_drop"] for item in random_records])) if random_records else 0.0
    selected_mean = float(np.mean([item["answer_logp_drop"] for item in selected])) if selected else 0.0
    payload["validation_passed"] = bool(token_total > 0 and selected_mean > random_mean)
    payload["validation"] = {
        "passed": bool(payload["validation_passed"]),
        "policy": "single_feature_answer_logp_drop_selected_mean_gt_random_mean",
        "num_tokens": int(token_total),
        "selected_mean_answer_logp_drop": selected_mean,
        "random_mean_answer_logp_drop": random_mean,
        "top_random_answer_logp_drop": float(random_records[0]["answer_logp_drop"]) if random_records else 0.0,
    }
    payload["sae_checkpoint"] = str(sae_path)

    report = {
        "layer": int(features.layer),
        "hook_point": features.hook_point,
        "num_tokens": int(token_total),
        "candidate_feature_ids": [int(x) for x in candidate_ids],
        "random_feature_ids": [int(x) for x in random_ids],
        "selected_feature_ids": payload["feature_ids"],
        "candidate_records": candidate_records,
        "random_records": random_records,
        "selected_mean_answer_logp_drop": selected_mean,
        "random_mean_answer_logp_drop": random_mean,
        "top_random_answer_logp_drop": float(random_records[0]["answer_logp_drop"]) if random_records else 0.0,
        "output_features": str(args.output_features),
    }
    output_report = Path(args.output_report)
    output_features = Path(args.output_features)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_features.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output_features.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "rank_sae_features_done",
                "report": str(output_report),
                "features": str(output_features),
                "selected_mean_answer_logp_drop": selected_mean,
                "random_mean_answer_logp_drop": random_mean,
                "num_tokens": int(token_total),
                "selected_feature_ids": payload["feature_ids"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
