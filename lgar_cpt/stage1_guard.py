from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate L-GAR Stage1 launch on Stage0 review results.")
    parser.add_argument("--summary", default="reports/stage0_summary.json")
    parser.add_argument("--allow-routeraux-only", action="store_true")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    checks = summary.get("decision_checks", {})
    failed = [k for k, v in checks.items() if k != "hard_lgar_layer_specific_forward_available" and not v]
    if failed:
        raise SystemExit(f"Stage1 blocked: failed Stage0 checks: {failed}")
    if not summary.get("hard_lgar_available", False) and not args.allow_routeraux_only:
        raise SystemExit(
            "Stage1 hard L-GAR blocked: Stage0 did not mark hard L-GAR available. "
            "Use --allow-routeraux-only only for a RouterAux diagnostic screen, not for claiming L-GAR."
        )
    print("stage1_guard_ok")


if __name__ == "__main__":
    main()
