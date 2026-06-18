#!/usr/bin/env bash
set -euo pipefail

# ===================================================
#  Reproduce the Fig. 7 trend: main-task accuracy vs number of free-riders.
#  Launches one job per free-rider count for a given Stage-2 config.
#
#  Usage:
#     ./submit_fig7.sh [CONFIG_IDX] [REPEAT] [COUNTS...]
#  Examples:
#     ./submit_fig7.sh 7 0                 # fast MNIST smoke, counts 0 2 4 6 8
#     ./submit_fig7.sh 8 0 0 2 4 6 8       # ResNet-18/CIFAR-10, previous_models
#     ./submit_fig7.sh 9 0 0 2 4 6 8       # ResNet-18/CIFAR-10, gaussian
#  The config's own `attack` is used unless you export ATTACK=... yourself.
# ===================================================
CONFIG_IDX="${1:-7}"
REPEAT="${2:-0}"
shift 2 || true
COUNTS=("$@")
if [ ${#COUNTS[@]} -eq 0 ]; then
  COUNTS=(0 2 4 6 8)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for k in "${COUNTS[@]}"; do
  echo ">>> free-riders=$k (config_idx=$CONFIG_IDX repeat=$REPEAT)"
  NUM_FREE_RIDERS="$k" "$SCRIPT_DIR/submit_experiment.sh" "$CONFIG_IDX" "$REPEAT"
  sleep 2
done

echo ""
echo "All Fig.7 jobs submitted. When done, summarise the trend with:"
echo "  python scripts/aggregate_results.py /mnt/nfs/home/zu/results"
