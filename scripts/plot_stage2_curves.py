#!/usr/bin/env python3
"""Render Stage2 training metrics as a self-contained HTML/SVG report."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "loss/ce",
    "loss/total",
    "val_ce",
    "lr",
    "lr/adapter",
    "longce/weight_mean",
    "loss/full_hu_ce",
    "loss/full_hu_total",
    "loss/nonhu_logp",
    "utility/full_hu_fraction",
    "utility/high_utility_fraction",
    "grad/adapter_aux_to_ce_ratio",
]

COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
]


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty label in {value!r}")
    return label, Path(path)


def _json_rows_from_text(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            start = line.find("{")
            if start < 0:
                continue
            try:
                item = json.loads(line[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
        return rows

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        metrics = data.get("metrics")
        if isinstance(metrics, list):
            return [x for x in metrics if isinstance(x, dict)]
        metrics_log = data.get("metrics_log")
        if isinstance(metrics_log, list):
            return [x for x in metrics_log if isinstance(x, dict)]
        final_metrics = data.get("final_metrics")
        if isinstance(final_metrics, dict):
            return [final_metrics]
        return [data]
    return []


def load_run_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for name in ("metrics.jsonl", "metrics_partial.json", "summary.json"):
            candidate = path / name
            if candidate.exists():
                rows.extend(_json_rows_from_text(candidate.read_text(encoding="utf-8")))
        return dedupe_rows(rows)
    return dedupe_rows(_json_rows_from_text(path.read_text(encoding="utf-8")))


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def x_value(row: dict[str, Any], world_size: int, seq_len: int, batch_size: int) -> float | None:
    tokens = as_float(row.get("tokens"))
    if tokens is not None:
        return tokens * world_size / 1_000_000_000.0
    step = as_float(row.get("step"))
    if step is not None:
        return step * seq_len * batch_size * world_size / 1_000_000_000.0
    return None


def format_num(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
        return f"{value:.3e}"
    return f"{value:.6g}"


def polyline(points: list[tuple[float, float]], x_min: float, x_max: float, y_min: float, y_max: float) -> str:
    width, height = 940.0, 260.0
    left, top = 56.0, 18.0
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    coords = []
    for x, y in points:
        px = left + (x - x_min) / (x_max - x_min) * width
        py = top + (1.0 - (y - y_min) / (y_max - y_min)) * height
        coords.append(f"{px:.1f},{py:.1f}")
    return " ".join(coords)


def render_chart(metric: str, series: dict[str, list[tuple[float, float]]]) -> str:
    nonempty = {label: pts for label, pts in series.items() if pts}
    if not nonempty:
        return ""
    xs = [x for pts in nonempty.values() for x, _ in pts]
    ys = [y for pts in nonempty.values() for _, y in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_max > y_min:
        pad = (y_max - y_min) * 0.06
        y_min -= pad
        y_max += pad

    lines = []
    legend = []
    for idx, (label, pts) in enumerate(nonempty.items()):
        color = COLORS[idx % len(COLORS)]
        points = polyline(pts, x_min, x_max, y_min, y_max)
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" '
            f'stroke-linejoin="round" stroke-linecap="round" points="{points}" />'
        )
        legend.append(
            f'<span><i style="background:{color}"></i>{html.escape(label)}</span>'
        )

    return f"""
<section class="chart">
  <h2>{html.escape(metric)}</h2>
  <svg viewBox="0 0 1040 330" role="img" aria-label="{html.escape(metric)} curve">
    <line x1="56" y1="278" x2="996" y2="278" class="axis" />
    <line x1="56" y1="18" x2="56" y2="278" class="axis" />
    <text x="56" y="308" class="tick">{format_num(x_min)}B</text>
    <text x="940" y="308" class="tick">{format_num(x_max)}B global tokens</text>
    <text x="8" y="282" class="tick">{format_num(y_min)}</text>
    <text x="8" y="24" class="tick">{format_num(y_max)}</text>
    {''.join(lines)}
  </svg>
  <div class="legend">{''.join(legend)}</div>
</section>
"""


def latest_table(runs: dict[str, list[dict[str, Any]]], metrics: list[str], world_size: int, seq_len: int, batch_size: int) -> str:
    rows = []
    for label, data in runs.items():
        cells = [f"<td>{html.escape(label)}</td>"]
        xs = [
            x
            for row in data
            for x in [x_value(row, world_size, seq_len, batch_size)]
            if x is not None
        ]
        latest_x = max(xs) if xs else None
        cells.append(f"<td>{format_num(latest_x)}B</td>")
        for metric in metrics:
            value = None
            for row in reversed(data):
                if metric in row:
                    value = as_float(row.get(metric))
                    break
            cells.append(f"<td>{format_num(value)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    headers = "".join(f"<th>{html.escape(m)}</th>" for m in metrics)
    return f"""
<table>
  <thead><tr><th>run</th><th>latest x</th>{headers}</tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
"""


def render_report(runs: dict[str, list[dict[str, Any]]], metrics: list[str], args: argparse.Namespace) -> str:
    series_by_metric: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for metric in metrics:
        metric_series: dict[str, list[tuple[float, float]]] = {}
        for label, rows in runs.items():
            pts: list[tuple[float, float]] = []
            for row in rows:
                y = as_float(row.get(metric))
                x = x_value(row, args.world_size, args.seq_len, args.batch_size)
                if x is None or y is None:
                    continue
                pts.append((x, y))
            pts.sort(key=lambda item: item[0])
            metric_series[label] = pts
        series_by_metric[metric] = metric_series

    charts = "".join(render_chart(metric, series_by_metric[metric]) for metric in metrics)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage2 Training Curves</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    h1 {{ font-size: 24px; margin: 0 0 16px; }}
    h2 {{ font-size: 16px; margin: 0 0 8px; }}
    .chart {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin: 16px 0; }}
    svg {{ width: 100%; height: auto; background: #ffffff; }}
    .axis {{ stroke: #94a3b8; stroke-width: 1; }}
    .tick {{ fill: #64748b; font-size: 12px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 12px; font-size: 13px; color: #334155; }}
    .legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }}
    table {{ border-collapse: collapse; background: white; border: 1px solid #e5e7eb; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 7px 9px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f1f5f9; }}
    .note {{ color: #64748b; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Stage2 Training Curves</h1>
  <p class="note">X axis uses global tokens: logged per-rank tokens * world_size / 1e9.</p>
  {latest_table(runs, metrics, args.world_size, args.seq_len, args.batch_size)}
  {charts}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="LABEL=RUN_DIR_OR_METRICS_FILE")
    parser.add_argument("--log", action="append", default=[], help="LABEL=LOG_FILE; useful for val_ce from old runs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--metric", action="append", default=[])
    args = parser.parse_args()

    runs: dict[str, list[dict[str, Any]]] = {}
    for item in args.run:
        label, path = parse_label_path(item)
        runs.setdefault(label, []).extend(load_run_rows(path))
    for item in args.log:
        label, path = parse_label_path(item)
        runs.setdefault(label, []).extend(load_run_rows(path))

    if not runs:
        raise SystemExit("pass at least one --run LABEL=PATH or --log LABEL=PATH")
    metrics = args.metric or DEFAULT_METRICS
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(runs, metrics, args), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
