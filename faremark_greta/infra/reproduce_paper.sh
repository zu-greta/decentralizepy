#!/usr/bin/env bash
# TODO
# ============================================================================
# reproduce_paper.sh — submit the full FareMark experiment grid.
#
# Drives every table/figure off a small set of base configs plus per-cell
# overrides (MODEL/DATASET/ATTACK/NUM_FREE_RIDERS/WM_* are passed through
# submit_experiment.sh as env vars). All jobs are submitted non-blocking
# (WAIT=0), so they queue and the cluster runs them as GPUs free up.
#
# Usage:
#   ./reproduce_paper.sh fidelity     # Table I + II   (watermark, 0 FR, repeats)
#   ./reproduce_paper.sh fig7         # Fig. 7         (acc vs #free-riders)
#   ./reproduce_paper.sh detection    # Table III+Fig8 (watermark + FR sweep)
#   ./reproduce_paper.sh robustness   # Figs. 9-10 + Table VI
#   ./reproduce_paper.sh all          # everything
#
# Edit the arrays below to widen/narrow the grid. Start small (one model,
# REPEATS="0") to validate, then scale up to the full grid + 10 repeats.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$HERE/submit_experiment.sh"
FIG7="$HERE/submit_fig7.sh"

# ---- the grid (paper §V-A). Add shufflenet/googlenet + food100 once implemented.
MODELS=(resnet18 alexnet)                 # paper: + shufflenet, googlenet
DATASETS=(cifar10 mnist)                  # paper: + cifar100, food100
REPEATS="0 1 2 3 4 5 6 7 8 9"             # paper averages 10; use "0" to smoke-test
FR_COUNTS="0 2 4 6 8"                     # free-rider counts for Fig.7 / detection
ATTACKS=(previous_models gaussian)

# Base config indices (see config.py): 11 = watermarked ResNet/CIFAR (fidelity),
# 12 = watermarked + free-riders (detection). We override model/dataset per cell.
WM_BASE=11
WM_FR_BASE=12

phase="${1:-all}"

submit_cell() {  # $1=base_cfg  $2..=extra "KEY=VAL" env overrides
  local base="$1"; shift
  env WAIT=0 WATERMARK=1 "$@" "$SUBMIT" "$base" "$REP"
}

# ---------------------------------------------------------------- FIDELITY (I+II)
if [[ "$phase" == "fidelity" || "$phase" == "all" ]]; then
  echo "### Fidelity (Table I) + watermark accuracy (Table II): watermark on, 0 FR"
  for m in "${MODELS[@]}"; do for d in "${DATASETS[@]}"; do for REP in $REPEATS; do
    submit_cell "$WM_BASE" MODEL="$m" DATASET="$d" NUM_FREE_RIDERS=0 WM_NUM_TRIGGERS=100
    sleep 2
  done; done; done
fi

# ---------------------------------------------------------------- FIG. 7 (acc vs FR)
if [[ "$phase" == "fig7" || "$phase" == "all" ]]; then
  echo "### Fig. 7: accuracy vs #free-riders (no watermark needed; 4 panels)"
  # (a,c) ResNet/CIFAR-10  (b,d) AlexNet/MNIST  x {previous_models, gaussian}
  for spec in "resnet18 cifar10" "alexnet mnist"; do
    set -- $spec; m="$1"; d="$2"
    for atk in "${ATTACKS[@]}"; do
      for REP in $REPEATS; do
        env MODEL="$m" DATASET="$d" ATTACK="$atk" "$FIG7" 1 "$REP" $FR_COUNTS
        sleep 2
      done
    done
  done
fi

# ---------------------------------------------------------------- DETECTION (III + Fig.8)
if [[ "$phase" == "detection" || "$phase" == "all" ]]; then
  echo "### Table III + Fig. 8: watermark ON, sweep free-rider rate, both attacks"
  for m in "${MODELS[@]}"; do for d in "${DATASETS[@]}"; do
    for atk in "${ATTACKS[@]}"; do for fr in $FR_COUNTS; do for REP in $REPEATS; do
      submit_cell "$WM_FR_BASE" MODEL="$m" DATASET="$d" ATTACK="$atk" \
                  NUM_FREE_RIDERS="$fr" WM_NUM_TRIGGERS=50
      sleep 2
    done; done; done
  done; done
fi

# ---------------------------------------------------------------- ROBUSTNESS (Figs.9-10, VI)
if [[ "$phase" == "robustness" || "$phase" == "all" ]]; then
  echo "### Figs. 9-10 + Table VI: run_robustness.py (fine-tune / prune / quantize)"
  echo "    Run on the cluster after a watermarked model exists:"
  for m in "${MODELS[@]}"; do for d in "${DATASETS[@]}"; do
    echo "    python scripts/run_robustness.py --config_idx $WM_BASE --repeat 0 \\"
    echo "        --output_dir \$RESULTS/robust_${m}_${d} --data_root \$DATA   # MODEL=$m DATASET=$d"
  done; done
fi

echo ""
echo "Submitted phase='$phase'. Watch:  runai list jobs"
echo "Aggregate when done:  python scripts/aggregate_results.py \$RESULTS_ROOT [--fig7]"
