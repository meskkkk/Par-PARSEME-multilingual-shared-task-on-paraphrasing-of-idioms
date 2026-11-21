#!/usr/bin/env bash
# The team's single evaluation command (wraps the official scorer).
#   scripts/evaluate.sh <predictions-dir> [dev|trial]
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/evaluate.py "$1" --split "${2:-dev}"
