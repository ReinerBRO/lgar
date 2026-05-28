from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .utils import read_jsonl, write_json


def _load_py_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_official_task_configs(ruler_dir: Path) -> dict[str, dict[str, Any]]:
    yaml_path = ruler_dir / "scripts" / "synthetic.yaml"
    constants_path = ruler_dir / "scripts" / "eval" / "synthetic" / "constants.py"
    tasks_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    tasks_base = _load_py_module(constants_path, "ruler13_eval_constants").TASKS
    merged: dict[str, dict[str, Any]] = {}
    for task_name, task_cfg in tasks_yaml.items():
        out = dict(task_cfg)
        out.update(tasks_base[task_cfg["task"]])
        merged[task_name] = out
    return merged


def _postprocess_pred(text: str) -> str:
    text = str(text).strip()
    return re.compile(r"[\x00-\x1f]").sub("\n", text).strip()


def _aggregate_chunk_files(folder: Path) -> None:
    chunk_groups: dict[str, list[Path]] = defaultdict(list)
    for path in folder.glob("*.jsonl"):
        match = re.match(r"(.+)-(\d+)\.jsonl$", path.name)
        if match:
            chunk_groups[match.group(1)].append(path)
    for task_name, files in chunk_groups.items():
        lines: list[dict[str, Any]] = []
        for file_path in sorted(files):
            lines.extend(read_jsonl(file_path))
            file_path.unlink()
        merged_path = folder / f"{task_name}.jsonl"
        with merged_path.open("w", encoding="utf-8") as fout:
            for row in lines:
                fout.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _run_metric(metric_fn: Any, rows: list[dict[str, Any]]) -> tuple[float, str, list[str], list[Any]]:
    preds = [_postprocess_pred(row.get("pred", "")) for row in rows]
    refs = [row.get("outputs", [row.get("output", "")]) for row in rows]
    nulls = f"{sum(1 for x in preds if not x)}/{len(preds)}"
    score = float(metric_fn(preds, refs)) if refs and refs[0] and refs[0][0] is not None else 0.0
    indices = [(row.get("others") or {}).get("id", row.get("index")) for row in rows]
    return score, nulls, preds, indices


def _write_summary_csv(path: Path, tasks: list[str], scores: list[float], nulls: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tasks", *tasks])
        writer.writerow(["Score", *scores])
        writer.writerow(["Nulls", *nulls])


def main() -> None:
    parser = argparse.ArgumentParser(description="Score official RULER-13 predictions with official task metrics.")
    parser.add_argument("--ruler-dir", required=True)
    parser.add_argument("--data-dir", required=True, help="prediction directory containing task jsonl or task-rank jsonl files")
    args = parser.parse_args()

    ruler_dir = Path(args.ruler_dir)
    data_dir = Path(args.data_dir)
    task_configs = _load_official_task_configs(ruler_dir)
    _aggregate_chunk_files(data_dir)

    task_names: list[str] = []
    scores: list[float] = []
    nulls: list[str] = []
    submission_rows: list[dict[str, Any]] = []
    task_scores: dict[str, float] = {}

    for task_name, task_cfg in task_configs.items():
        task_path = data_dir / f"{task_name}.jsonl"
        if not task_path.exists():
            continue
        rows = read_jsonl(task_path)
        score, task_nulls, preds, indices = _run_metric(task_cfg["metric_fn"], rows)
        task_names.append(task_name)
        scores.append(score)
        nulls.append(task_nulls)
        task_scores[task_name] = score
        submission_rows.extend(
            {"Task": task_name, "ID": idx, "Prediction": pred}
            for idx, pred in zip(indices, preds)
        )

    _write_summary_csv(data_dir / "summary.csv", task_names, scores, nulls)
    with (data_dir / "submission.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Task", "ID", "Prediction"])
        writer.writeheader()
        writer.writerows(submission_rows)

    write_json(
        data_dir / "summary.json",
        {
            "tasks": task_names,
            "task_scores": task_scores,
            "mean_score": sum(scores) / len(scores) if scores else float("nan"),
            "task_nulls": dict(zip(task_names, nulls)),
        },
    )
    print(json.dumps({"tasks": task_scores, "mean_score": sum(scores) / len(scores) if scores else float("nan")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
