#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _model_dims(model_path: str | Path | None, num_layers: int | None, num_heads: int | None) -> tuple[int, int]:
    if num_layers is not None and num_heads is not None:
        return int(num_layers), int(num_heads)
    if model_path is None:
        raise SystemExit("pass --model-path or both --num-layers and --num-heads")
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"missing model config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    layers = int(num_layers if num_layers is not None else cfg["num_hidden_layers"])
    heads = int(num_heads if num_heads is not None else cfg["num_attention_heads"])
    return layers, heads


def _candidate_layers(num_layers: int, layer_fraction: float) -> list[int]:
    count = max(1, int(round(num_layers * float(layer_fraction))))
    start = max(0, int(num_layers) - count)
    return list(range(start, int(num_layers)))


def _heads_from_json(path: str | Path | None) -> list[list[int]]:
    data = _load_json(path)
    return [[int(layer), int(head)] for layer, head in data.get("retrieval_heads", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible random retrieval-head JSON.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-selected", type=int, default=None)
    parser.add_argument("--reference-heads-json", default=None, help="Use its retrieval_heads count when --num-selected is omitted.")
    parser.add_argument("--exclude-heads-json", default=None, help="Optional heads to exclude from the random pool.")
    parser.add_argument("--layer-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--selection-name", default="random_heads")
    args = parser.parse_args()

    num_layers, num_heads = _model_dims(args.model_path, args.num_layers, args.num_heads)
    ref_heads = _heads_from_json(args.reference_heads_json)
    num_selected = int(args.num_selected if args.num_selected is not None else len(ref_heads))
    if num_selected <= 0:
        raise SystemExit("num_selected must be positive; pass --num-selected or a reference JSON with retrieval_heads")

    candidate_layers = _candidate_layers(num_layers, args.layer_fraction)
    candidates = [(layer, head) for layer in candidate_layers for head in range(num_heads)]
    excluded = {tuple(head) for head in _heads_from_json(args.exclude_heads_json)}
    pool = [head for head in candidates if head not in excluded]
    if num_selected > len(pool):
        raise SystemExit(f"cannot select {num_selected} heads from pool size {len(pool)}")

    rng = random.Random(int(args.seed))
    selected = sorted(rng.sample(pool, num_selected))
    payload = {
        "retrieval_heads": [[int(layer), int(head)] for layer, head in selected],
        "num_selected": len(selected),
        "selection": args.selection_name,
        "seed": int(args.seed),
        "model_path": str(args.model_path) if args.model_path else None,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "candidate_layers": candidate_layers,
        "candidate_pool_size": len(pool),
        "layer_fraction": float(args.layer_fraction),
        "reference_heads_json": str(args.reference_heads_json) if args.reference_heads_json else None,
        "reference_num_heads": len(ref_heads),
        "exclude_heads_json": str(args.exclude_heads_json) if args.exclude_heads_json else None,
        "excluded_num_heads": len(excluded),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"num_selected={len(selected)} pool={len(pool)} seed={args.seed}")
    print("heads=" + json.dumps(payload["retrieval_heads"]))


if __name__ == "__main__":
    main()
