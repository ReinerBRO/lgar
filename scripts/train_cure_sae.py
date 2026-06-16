#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thin wrapper for SAE-Steered CURE training; unknown args pass through to curcpt.train."
    )
    parser.add_argument("--sae-ckpt", "--sae-checkpoint", dest="sae_checkpoint", required=True)
    parser.add_argument("--features", "--sae-features", dest="sae_features", required=True)
    parser.add_argument("--beta-margin", type=float, default=0.02)
    parser.add_argument("--beta-match", type=float, default=0.10)
    parser.add_argument("--short-kl", type=float, default=0.0)
    parser.add_argument("--lsor-enable", action="store_true")
    parser.add_argument("--lsor-alpha", type=float, default=0.0)
    parser.add_argument("--lsor-topk", type=int, default=16)
    parser.add_argument("--lsor-window", type=int, default=128)
    known, rest = parser.parse_known_args()

    forwarded = [
        "curcpt.train",
        "--method",
        "cure_cpt",
        "--freeze-base-model",
        "--sae-enable",
        "--sae-checkpoint",
        known.sae_checkpoint,
        "--sae-features",
        known.sae_features,
        "--lambda-sae-margin",
        str(float(known.beta_margin)),
        "--lambda-sae-match",
        str(float(known.beta_match)),
        "--lambda-short-kl",
        str(float(known.short_kl)),
    ]
    if known.lsor_enable or float(known.lsor_alpha) > 0.0:
        forwarded.extend(
            [
                "--lsor-enable",
                "--lambda-lsor",
                str(float(known.lsor_alpha)),
                "--lsor-top-k",
                str(int(known.lsor_topk)),
                "--lsor-window",
                str(int(known.lsor_window)),
            ]
        )
    forwarded.extend(rest)
    sys.argv = forwarded
    from curcpt.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
