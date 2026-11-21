#!/usr/bin/env bash
# Download the organisers' Par-PARSEME data (train, trial, official tools).
# Source: https://gitlab.com/parseme/sharedtask-data  (folder par-parseme/)
# When the test data is released (10 Jan 2027) just run this script again.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
if [ -d data/raw/sharedtask-data/.git ]; then
  git -C data/raw/sharedtask-data pull --quiet
else
  git clone --quiet --depth 1 --filter=blob:none --sparse \
      https://gitlab.com/parseme/sharedtask-data.git data/raw/sharedtask-data
  git -C data/raw/sharedtask-data sparse-checkout set par-parseme
fi
# Short, stable path used by configs/shared.yaml
ln -sfn sharedtask-data/par-parseme data/raw/par-parseme
echo "Data commit: $(git -C data/raw/sharedtask-data rev-parse --short HEAD)"
ls data/raw/par-parseme
