#!/usr/bin/env bash
# Run every experiment in sequence (E2 first because figures reuse its score dump).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTHONHASHSEED=0
mkdir -p logs results
for e in "$@"; do
  echo "=== $(date -u +%H:%M:%S) $e"
  python3 -m "anomalyguard.experiments.$e" > "logs/$e.log" 2>&1
done
echo "=== $(date -u +%H:%M:%S) done"
