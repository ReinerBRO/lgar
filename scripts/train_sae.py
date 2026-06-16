#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curcpt.sae import SparseAutoencoder
from scripts.precompute_sae_feature_targets import _device_from_arg


def _load_activation_matrix(path: Path, views: list[str]) -> np.ndarray:
    payload = np.load(path)
    arrays = []
    for view in views:
        if view not in payload:
            raise ValueError(f"{path} has no view {view!r}")
        arrays.append(np.asarray(payload[view], dtype=np.float32))
    return np.concatenate(arrays, axis=0)


def _explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    residual_var = (x - x_hat).float().pow(2).mean()
    total_var = (x.float() - x.float().mean(dim=0, keepdim=True)).pow(2).mean().clamp(min=1.0e-12)
    return 1.0 - residual_var / total_var


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-layer ReLU SAE on collected resid_mid activations.")
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated layer ids, or 'auto' from activation_meta.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", default="full,short,masked_head")
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--l1", type=float, default=1.0e-3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max-activations", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    activation_dir = Path(args.activation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = [view.strip() for view in str(args.views).split(",") if view.strip()]
    if args.layers == "auto":
        meta = json.loads((activation_dir / "activation_meta.json").read_text(encoding="utf-8"))
        layers = [int(layer) for layer in meta["layers"]]
    else:
        layers = [int(x) for x in str(args.layers).split(",") if x.strip()]
    device = _device_from_arg(args.device)

    for layer in layers:
        path = activation_dir / f"layer{int(layer)}_activations.npz"
        x_np = _load_activation_matrix(path, views)
        if x_np.shape[0] == 0:
            raise ValueError(f"{path} contains no activations")
        if x_np.shape[0] > int(args.max_activations):
            keep = rng.choice(x_np.shape[0], size=int(args.max_activations), replace=False)
            x_np = x_np[keep]
        x_cpu = torch.from_numpy(np.asarray(x_np, dtype=np.float32))
        mean = x_cpu.mean(dim=0)
        scale = x_cpu.std(dim=0).clamp(min=1.0e-6)
        input_dim = int(x_cpu.shape[1])
        feature_dim = input_dim * int(args.expansion)
        sae = SparseAutoencoder(input_dim, feature_dim, mean, scale).to(device)
        opt = torch.optim.AdamW(sae.parameters(), lr=float(args.lr))

        for step in range(int(args.steps)):
            idx = torch.randint(0, x_cpu.shape[0], (int(args.batch_size),))
            x = x_cpu[idx].to(device)
            x_hat, acts = sae(x)
            mse = F.mse_loss(x_hat.float(), x.float())
            l1 = acts.float().abs().mean()
            loss = mse + float(args.l1) * l1
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == int(args.steps) - 1:
                print(
                    json.dumps(
                        {
                            "event": "train_sae_progress",
                            "layer": int(layer),
                            "step": int(step),
                            "loss": float(loss.detach().item()),
                            "mse": float(mse.detach().item()),
                            "l1": float(l1.detach().item()),
                            "l0": float((acts.detach() > 0).float().sum(dim=-1).mean().item()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        with torch.no_grad():
            eval_idx = torch.arange(min(x_cpu.shape[0], 8192))
            x_eval = x_cpu[eval_idx].to(device)
            x_hat, acts = sae(x_eval)
            mse = F.mse_loss(x_hat.float(), x_eval.float())
            ev = _explained_variance(x_eval, x_hat)
            l0 = (acts > 0).float().sum(dim=-1).mean()
            dead = (acts > 0).float().mean(dim=0).eq(0).float().mean()
        metrics = {
            "layer": int(layer),
            "hook_point": "resid_mid",
            "num_activations": int(x_cpu.shape[0]),
            "input_dim": int(input_dim),
            "feature_dim": int(feature_dim),
            "reconstruction_mse": float(mse.detach().item()),
            "explained_variance": float(ev.detach().item()),
            "average_l0": float(l0.detach().item()),
            "dead_feature_ratio": float(dead.detach().item()),
            "views": views,
        }
        ckpt_path = output_dir / f"sae_layer{int(layer)}.pt"
        torch.save({"model": sae.state_dict(), "metrics": metrics}, ckpt_path)
        (output_dir / f"sae_layer{int(layer)}_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"event": "train_sae_done", "checkpoint": str(ckpt_path), **metrics},
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
