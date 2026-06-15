#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SKILL_SCRIPT="${TIANYI_ZJC_SYNC_SCRIPT:-$HOME/.codex/skills/tianyi-cloud-sync/scripts/ty-zjc-sync.mjs}"
ZJC_ROOT="/gemini/space/private/zjc/goals"

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_tianyi_dev.sh upload                 # Full Mac project -> Tianyi D:\<project>
  scripts/sync_tianyi_dev.sh upload --changed-only  # Upload only files changed since local manifest
  scripts/sync_tianyi_dev.sh upload --only RELPATH  # Upload one project-relative file or directory
  scripts/sync_tianyi_dev.sh upload --only RELPATH --include-cmd
  scripts/sync_tianyi_dev.sh mark-synced            # Seed manifest after a verified full sync
  scripts/sync_tianyi_dev.sh make-cmd               # Generate Windows rsync launchers locally only
  scripts/sync_tianyi_dev.sh help

Notes:
  - Tianyi target is D:\<project>\.
  - Server target is dev:/gemini/space/private/zjc/goals/<project>/.
  - Ignore rules live in .tianyi-syncignore. Add data/ or other directories there.
  - Server transfer is rsync incremental and does not delete remote-only files.
EOF
}

mode="${1:-upload}"
case "$mode" in
  upload|sync)
    shift || true
    exec node "$SKILL_SCRIPT" upload --local "$PROJECT_ROOT" --zjc-root "$ZJC_ROOT" "$@"
    ;;
  make-cmd|cmd)
    shift || true
    exec node "$SKILL_SCRIPT" make-cmd --local "$PROJECT_ROOT" --zjc-root "$ZJC_ROOT" "$@"
    ;;
  mark-synced|mark)
    shift || true
    exec node "$SKILL_SCRIPT" mark-synced --local "$PROJECT_ROOT" --zjc-root "$ZJC_ROOT" "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    usage >&2
    exit 2
    ;;
esac
