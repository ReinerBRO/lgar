from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .utils import ensure_dir, write_json


REQUIRED_RUNS = ["Qwen_CE_CPT", "Qwen_LongCE_CPT", "Qwen_RouterAux_CPT", "Qwen_LGAR_CPT"]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _get_eval(run: dict[str, Any], key: str) -> float:
    return float(run.get("eval", {}).get(key, float("nan")))


def _long_avg(run: dict[str, Any]) -> float:
    keys = [
        "ruler_mini_4096_accuracy",
        "ruler_mini_8192_accuracy",
        "nolima_style_4096_accuracy",
        "nolima_style_8192_accuracy",
    ]
    values = [_get_eval(run, key) for key in keys]
    values = [x for x in values if math.isfinite(x)]
    return float(sum(values) / len(values)) if values else float("nan")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def load_stage1_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Stage1 summary must be a JSON object, got {type(data).__name__}")
    return data


def decide(summary: dict[str, Any], min_tokens: int, budget: float, budget_tolerance: float) -> dict[str, Any]:
    missing = [name for name in REQUIRED_RUNS if name not in summary]
    checks: dict[str, bool] = {
        "all_required_runs_present": not missing,
    }
    if missing:
        return {
            "status": "incomplete",
            "missing_runs": missing,
            "checks": checks,
            "decision": "Stage1 summary is incomplete; keep training or rerun missing jobs.",
        }

    ce = summary["Qwen_CE_CPT"]
    router = summary["Qwen_RouterAux_CPT"]
    lgar = summary["Qwen_LGAR_CPT"]
    runs = [summary[name] for name in REQUIRED_RUNS]

    eval_present = all(bool(summary[name].get("eval")) for name in REQUIRED_RUNS)
    normal_eval_safety = True
    short_mc_safety = True
    long_context_improvements = 0
    if eval_present:
        normal_eval_safety = _get_eval(lgar, "normal_val_ce") <= _get_eval(ce, "normal_val_ce") + 0.02
        short_mc_safety = _get_eval(lgar, "short_mc_avg_accuracy") >= _get_eval(ce, "short_mc_avg_accuracy") - 0.01
        long_context_improvements = sum(
            [
                _get_eval(lgar, "high_lsd_ce") < _get_eval(ce, "high_lsd_ce"),
                _get_eval(lgar, "long_4096_target_ce") < _get_eval(ce, "long_4096_target_ce"),
                _long_avg(lgar) > _long_avg(ce),
                _long_avg(lgar) > _long_avg(router),
            ]
        )

    lgar_auc = float(lgar.get("router_auc_last_quarter", float("nan")))
    lgar_p10 = float(lgar.get("router_precision10_last_quarter", float("nan")))
    lgar_rand = float(lgar.get("router_random_precision_last_quarter", float("nan")))
    router_auc = float(router.get("router_auc_last_quarter", float("nan")))
    router_p10 = float(router.get("router_precision10_last_quarter", float("nan")))
    router_rand = float(router.get("router_random_precision_last_quarter", float("nan")))

    lgar_signal = (_finite(lgar_auc) and lgar_auc > 0.55) or (
        _finite(lgar_p10) and _finite(lgar_rand) and lgar_p10 > lgar_rand + 0.01
    )
    router_signal = (_finite(router_auc) and router_auc > 0.55) or (
        _finite(router_p10) and _finite(router_rand) and router_p10 > router_rand + 0.01
    )

    checks.update(
        {
            "all_runs_reached_token_budget": all(float(run.get("tokens_seen", 0.0)) >= float(min_tokens) for run in runs),
            "no_nonfinite_loss": all(not bool(run.get("nonfinite_loss", True)) for run in runs),
            "true_wall_clock_reported": all(float(run.get("true_wall_tok_s", 0.0)) > 0 for run in runs),
            "checkpoints_written": all(
                (
                    bool(run.get("checkpoint_path"))
                    and Path(run["checkpoint_path"]).exists()
                )
                or (
                    bool(run.get("run_dir"))
                    and (Path(run["run_dir"]) / "checkpoint.pt").exists()
                )
                for run in runs
            ),
            "lgar_budget_matches_target": abs(float(lgar.get("actual_budget_last_quarter", float("nan"))) - float(budget)) <= float(budget_tolerance),
            "lgar_router_signal": lgar_signal,
            "routeraux_signal": router_signal,
            "train_ce_no_large_explosion": float(lgar.get("ce_last_quarter", float("inf"))) <= float(ce.get("ce_last_quarter", -float("inf"))) + 0.25,
            "final_eval_present": eval_present,
            "normal_eval_ce_safety": normal_eval_safety,
            "short_mc_safety": short_mc_safety,
            "long_context_criteria_count_met": long_context_improvements >= 2 if eval_present else False,
        }
    )

    blocking = [name for name, ok in checks.items() if not ok]
    needs_fix = not checks["lgar_router_signal"] or not checks["lgar_budget_matches_target"] or not checks["train_ce_no_large_explosion"]
    if blocking:
        status = "needs_fix" if needs_fix else "blocked_for_review"
        decision = (
            "Stage1 does not pass current review; run fix rounds before claiming L-GAR."
            if needs_fix
            else "Stage1 has missing or weak evidence; review artifacts before claiming completion."
        )
    else:
        status = "pass"
        decision = "Stage1 passes safety, checkpoint, evaluation, budget, and router-mechanism review."

    return {
        "status": status,
        "missing_runs": missing,
        "checks": checks,
        "blocking_checks": blocking,
        "mechanism": {
            "lgar_auc_last_quarter": lgar_auc,
            "lgar_precision10_last_quarter": lgar_p10,
            "lgar_random_precision_last_quarter": lgar_rand,
            "routeraux_auc_last_quarter": router_auc,
            "routeraux_precision10_last_quarter": router_p10,
            "routeraux_random_precision_last_quarter": router_rand,
            "long_context_improvements": long_context_improvements,
        },
        "decision": decision,
    }


def write_report(path: Path, summary: dict[str, Any], review: dict[str, Any]) -> None:
    lines = [
        "# L-GAR Stage1 Review",
        "",
        "## Scope",
        "",
        "This report reviews the four-run Stage1 CPT screen for Qwen2.5-0.5B L-GAR.",
        "",
        "## Training Summary",
        "",
        "| run | tokens | CE first | CE last | tok/s | checkpoint |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in REQUIRED_RUNS:
        run = summary.get(name, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(run.get("tokens_seen", "missing")),
                    _fmt(run.get("ce_first_quarter", float("nan"))),
                    _fmt(run.get("ce_last_quarter", float("nan"))),
                    _fmt(run.get("true_wall_tok_s", float("nan"))),
                    "yes"
                    if (
                        run.get("checkpoint_path")
                        or (run.get("run_dir") and (Path(run["run_dir"]) / "checkpoint.pt").exists())
                    )
                    else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Router Quality",
            "",
            "| run | AUC | precision@10pct | random precision | actual budget |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ["Qwen_RouterAux_CPT", "Qwen_LGAR_CPT"]:
        run = summary.get(name, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(run.get("router_auc_last_quarter", float("nan"))),
                    _fmt(run.get("router_precision10_last_quarter", float("nan"))),
                    _fmt(run.get("router_random_precision_last_quarter", float("nan"))),
                    _fmt(run.get("actual_budget_last_quarter", float("nan"))),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Evaluation Summary", ""])
    if all(bool(summary.get(name, {}).get("eval")) for name in REQUIRED_RUNS):
        lines.extend(
            [
                "| run | normal CE | high-LSD CE | long4096 CE | RULER/NoLiMa avg | short MC avg |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name in REQUIRED_RUNS:
            run = summary[name]
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        _fmt(_get_eval(run, "normal_val_ce")),
                        _fmt(_get_eval(run, "high_lsd_ce")),
                        _fmt(_get_eval(run, "long_4096_target_ce")),
                        _fmt(_long_avg(run)),
                        _fmt(_get_eval(run, "short_mc_avg_accuracy")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("Final evaluation metrics are missing from one or more runs.")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "```json",
            json.dumps(review, indent=2, sort_keys=True),
            "```",
            "",
            review["decision"],
            "",
        ]
    )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review completed L-GAR Stage1 results.")
    parser.add_argument("--summary", default="/gemini/space/private/zjc/goals/lgar/reports/stage1_summary.json")
    parser.add_argument("--output-json", default="/gemini/space/private/zjc/goals/lgar/reports/stage1_review.json")
    parser.add_argument("--output-md", default="/gemini/space/private/zjc/goals/lgar/reports/stage1_review.md")
    parser.add_argument("--min-tokens", type=int, default=145_000_000)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--budget-tolerance", type=float, default=0.03)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"Stage1 summary not found: {summary_path}")
    summary = load_stage1_summary(summary_path)
    review = decide(summary, args.min_tokens, args.budget, args.budget_tolerance)
    write_json(args.output_json, review)
    write_report(Path(args.output_md), summary, review)
    print(json.dumps(review, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
