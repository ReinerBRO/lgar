#!/usr/bin/env python3
"""Convert Stage2 JSON metrics/logs to TensorBoard event files."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty label in {value!r}")
    return label, Path(path)


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def json_rows_from_text(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            match = re.search(r"\{.*\}", line)
            if match is None:
                continue
            try:
                item = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
        return rows

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("metrics", "metrics_log"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        value = data.get("final_metrics")
        if isinstance(value, dict):
            return [value]
        return [data]
    return []


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for name in ("metrics.jsonl", "metrics_partial.json", "summary.json"):
            candidate = path / name
            if candidate.exists():
                rows.extend(json_rows_from_text(candidate.read_text(encoding="utf-8")))
        return dedupe(rows)
    return dedupe(json_rows_from_text(path.read_text(encoding="utf-8")))


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("step"),
            row.get("tokens"),
            row.get("loss/ce"),
            row.get("loss/total"),
            row.get("val_ce"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_rows(label: str, rows: list[dict[str, Any]], out_dir: Path, world_size: int) -> int:
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(str(out_dir / label))
    flat_writer = EventFileWriter(str(out_dir))
    written = 0

    def add_scalar(tag: str, value: float, step: int) -> None:
        nonlocal written
        per_run_event = Event(
            wall_time=time.time(),
            step=int(step),
            summary=Summary(value=[Summary.Value(tag=str(tag), simple_value=float(value))]),
        )
        flat_event = Event(
            wall_time=time.time(),
            step=int(step),
            summary=Summary(value=[Summary.Value(tag=f"{label}/{tag}", simple_value=float(value))]),
        )
        writer.add_event(per_run_event)
        flat_writer.add_event(flat_event)
        written += 1

    for row in rows:
        try:
            step = int(row["step"])
        except (KeyError, TypeError, ValueError):
            continue

        tokens = as_float(row.get("tokens"))
        if tokens is not None:
            add_scalar("progress/tokens_per_rank", tokens, step)
            add_scalar("progress/global_tokens", tokens * int(world_size), step)
            add_scalar("progress/global_tokens_b", tokens * int(world_size) / 1_000_000_000.0, step)

        wall_elapsed = as_float(row.get("wall_elapsed"))
        if wall_elapsed is not None:
            add_scalar("time/wall_elapsed_sec", wall_elapsed, step)

        for key, value in row.items():
            if key in {"step", "tokens", "wall_elapsed"}:
                continue
            scalar = as_float(value)
            if scalar is None:
                continue
            tag = "eval/val_ce" if key == "val_ce" else str(key)
            add_scalar(tag, scalar, step)
    writer.flush()
    writer.close()
    flat_writer.flush()
    flat_writer.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Stage2 metrics to TensorBoard event files")
    parser.add_argument("--input", action="append", required=True, help="LABEL=RUN_DIR_OR_METRICS_OR_LOG")
    parser.add_argument("--out", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in args.input:
        label, path = parse_label_path(item)
        rows = load_rows(path)
        written = write_rows(label, rows, out_dir, args.world_size)
        print(f"{label}: rows={len(rows)} scalars={written} logdir={out_dir / label}")
    print(f"tensorboard_logdir={out_dir}")


if __name__ == "__main__":
    main()
