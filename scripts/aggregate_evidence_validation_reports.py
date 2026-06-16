#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "selected_ablation_delta_logp",
    "random_ablation_delta_logp",
    "masked_head_answer_delta_logp",
    "short_patch_delta_logp",
    "masked_head_feature_drop",
)


def _normalise_feature_payload(path: str | Path) -> dict[str, Any]:
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


def _weighted_mean(reports: list[dict[str, Any]], key: str) -> float:
    total = sum(int(report.get("num_tokens", 0)) for report in reports)
    if total <= 0:
        return 0.0
    return float(
        sum(float(report.get(key, 0.0)) * int(report.get("num_tokens", 0)) for report in reports)
        / total
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sharded SAE evidence validation reports.")
    parser.add_argument("--reports-glob", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-features", required=True)
    args = parser.parse_args()

    report_paths = sorted(Path(path) for path in glob.glob(args.reports_glob))
    if not report_paths:
        raise ValueError(f"no reports matched {args.reports_glob!r}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    total_tokens = sum(int(report.get("num_tokens", 0)) for report in reports)
    thresholds = dict(reports[0].get("thresholds", {}))
    aggregate = {
        "passed": False,
        "layer": int(reports[0]["layer"]),
        "hook_point": reports[0]["hook_point"],
        "feature_ids": list(reports[0]["feature_ids"]),
        "random_feature_ids_by_shard": {
            str(report.get("sequence_rank", idx)): list(report.get("random_feature_ids", []))
            for idx, report in enumerate(reports)
        },
        "num_tokens": int(total_tokens),
        "num_shards": len(reports),
        "shard_reports": [str(path) for path in report_paths],
        "thresholds": thresholds,
    }
    for key in METRIC_KEYS:
        aggregate[key] = _weighted_mean(reports, key)
    aggregate["passed"] = bool(
        total_tokens > 0
        and aggregate["selected_ablation_delta_logp"]
        >= float(thresholds.get("min_feature_ablation_logp_drop", 0.0))
        and aggregate["selected_ablation_delta_logp"]
        >= aggregate["random_ablation_delta_logp"] + float(thresholds.get("min_ablation_margin", 0.0))
        and aggregate["masked_head_answer_delta_logp"]
        >= float(thresholds.get("min_masked_head_logp_drop", 0.0))
        and aggregate["short_patch_delta_logp"] >= float(thresholds.get("min_patch_gain", 0.0))
        and aggregate["masked_head_feature_drop"] >= float(thresholds.get("min_masked_head_drop", 0.0))
    )

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")

    payload = _normalise_feature_payload(args.features)
    payload["validation"] = aggregate
    payload["validation_passed"] = bool(aggregate["passed"])
    output_features = Path(args.output_features)
    output_features.parent.mkdir(parents=True, exist_ok=True)
    output_features.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "aggregate_evidence_validation_done",
                "passed": bool(aggregate["passed"]),
                "num_tokens": int(total_tokens),
                "selected_ablation_delta_logp": aggregate["selected_ablation_delta_logp"],
                "random_ablation_delta_logp": aggregate["random_ablation_delta_logp"],
                "masked_head_answer_delta_logp": aggregate["masked_head_answer_delta_logp"],
                "short_patch_delta_logp": aggregate["short_patch_delta_logp"],
                "masked_head_feature_drop": aggregate["masked_head_feature_drop"],
                "report": str(output_report),
                "features": str(output_features),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
