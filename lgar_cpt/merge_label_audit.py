from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge L-GAR label-audit shards.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-shards", type=int, default=8)
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).glob("shard_*.json"))
    if len(paths) != int(args.expected_shards):
        raise SystemExit(f"expected {args.expected_shards} shards, found {len(paths)} under {args.input_dir}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    total_batches = sum(int(s["observed_batches"]) for s in shards)
    total_sequences = sum(int(s["requested_sequences"]) for s in shards)
    merged_stats: dict[str, float] = {}
    for shard in shards:
        weight = int(shard["observed_batches"]) / max(1, total_batches)
        for key, value in shard["mean_stats"].items():
            merged_stats[key] = merged_stats.get(key, 0.0) + float(value) * weight
    examples = []
    for shard in shards:
        for item in shard["examples"]:
            item = dict(item)
            item["audit_shard_id"] = shard["shard_id"]
            examples.append(item)
            if len(examples) >= 20:
                break
        if len(examples) >= 20:
            break
    merged = {
        "requested_sequences": total_sequences,
        "observed_batches": total_batches,
        "mean_stats": dict(sorted(merged_stats.items())),
        "examples": examples,
        "shards": [{"path": str(path), "shard_id": shard["shard_id"], "seed": shard["seed"]} for path, shard in zip(paths, shards)],
    }
    write_json(args.output, merged)
    print(json.dumps({"output": args.output, "requested_sequences": total_sequences, "examples": len(examples)}, sort_keys=True))


if __name__ == "__main__":
    main()
