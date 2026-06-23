#!/usr/bin/env bash
set -euo pipefail

# ===================================================
#  Launch a grid of jobs: CONFIG_IDXS x REPEATS
#  The paper averages over 10 repeats; this fires one job per (config, repeat)
#
#  Usage:
#     ./submit_sweep.sh "1 2" "0 1 2"     # configs 1,2 x repeats 0,1,2 = 6 jobs
#     ./submit_sweep.sh "1"               # config 1, repeats 0-9 (paper default)
# ===================================================
CONFIG_IDXS="${1:-1}"
REPEATS="${2:-0 1 2 3 4 5 6 7 8 9}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

n=0
for cfg in $CONFIG_IDXS; do
  for rep in $REPEATS; do
    echo ">>> launching config_idx=$cfg repeat=$rep"
    WAIT=0 "$SCRIPT_DIR/submit_experiment.sh" "$cfg" "$rep"
    n=$((n + 1))
    sleep 2
  done
done
echo "Submitted $n jobs."
echo "After they finish, aggregate with: python scripts/aggregate_results.py <results_root>"