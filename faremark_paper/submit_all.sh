#!/usr/bin/env bash
# =============================================================================
# FareMark — RunAI Job Submission Script
# =============================================================================
# Usage:
#   bash submit_all.sh                  # submit ALL experiments
#   bash submit_all.sh table1           # submit only Table I jobs
#   bash submit_all.sh table2
#   bash submit_all.sh fig7
#   bash submit_all.sh fig8
#   bash submit_all.sh table3
#   bash submit_all.sh table4_5
#   bash submit_all.sh table6
#   bash submit_all.sh ablations        # tables 7,8,9 + figs 9,10
#
# Prerequisites:
#   - runai CLI installed and logged in (runai login)
#   - PROJECT set to your RunAI project name
#   - IMAGE set to your Docker image (pushed to a registry)
#   - PVC set to your persistent volume claim name (for /results)
#
# =============================================================================

set -euo pipefail

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────
PROJECT="${RUNAI_PROJECT:-sacs-zu}"    # runai project name
IMAGE="${DOCKER_IMAGE:-sacs-zu/faremark:latest}"
# PVC="${RUNAI_PVC:-faremark-results}"            # PVC mounted at /results
# DATA_PVC="${RUNAI_DATA_PVC:-faremark-data}"     # PVC mounted at /data (datasets)
GPUS=1                                          # GPUs per job (1 A100 or V100)
REPEATS=10                                      # paper averages 10 repetitions
# ─────────────────────────────────────────────────────────────────────────────

EXP_FILTER="${1:-all}"

# Helper: submit one RunAI training job
#   submit_job <job-name> <python-command>
submit_job() {
  local name="$1"
  local cmd="$2"
  echo "Submitting: $name"
  runai submit "$name" \
    --project "$PROJECT" \
    --image   "$IMAGE" \
    -g "$GPUS" \
    --node-pool default \
    --pvc sacs-scratch:/mnt/nfs \
    -- bash -c "cd /workspace && $cmd"
}

    # --pvc "${PVC}:/results" \
    # --pvc "${DATA_PVC}:/data" \

# ── TABLE I ──────────────────────────────────────────────────────────────────
submit_table1() {
  echo "=== TABLE I — Fidelity ==="
  # 6 configs × 10 repeats = 60 jobs
  for idx in 0 1 2 3 4 5; do
    for rep in $(seq 0 $((REPEATS-1))); do
      submit_job "fm-t1-c${idx}-r${rep}" \
        "python scripts/exp_table1.py --config_idx $idx --repeat $rep \
         --output_dir /mnt/nfs/results --device cuda --data_root /mnt/nfs/data"
    done
  done
}

# ── TABLE II ─────────────────────────────────────────────────────────────────
submit_table2() {
  echo "=== TABLE II — Watermark Detection ==="
  for idx in 0 1 2 3 4 5; do
    for rep in $(seq 0 $((REPEATS-1))); do
      submit_job "fm-t2-c${idx}-r${rep}" \
        "python scripts/exp_table2.py --config_idx $idx --repeat $rep \
         --output_dir /mnt/nfs/results --device cuda --data_root /mnt/nfs/data"
    done
  done
}

# ── FIGURE 7 ─────────────────────────────────────────────────────────────────
submit_fig7() {
  echo "=== FIGURE 7 — Accuracy vs FR Count ==="
  # 4 subfigs × 9 FR counts × 10 repeats = 360 jobs
  # Tip: reduce REPEATS to 3 if cluster is congested
  for sf in a b c d; do
    for nfr in 0 1 2 3 4 5 6 7 8; do
      for rep in $(seq 0 $((REPEATS-1))); do
        submit_job "fm-f7-${sf}-fr${nfr}-r${rep}" \
          "python scripts/exp_fig7.py --subfig $sf --num_fr $nfr --repeat $rep \
           --output_dir /mnt/nfs/results --device cuda --data_root /mnt/nfs/data"
      done
    done
  done
}

# ── FIGURE 8 ─────────────────────────────────────────────────────────────────
submit_fig8() {
  echo "=== FIGURE 8 — Detection Rate Over Rounds ==="
  for sf in a b; do
    for rep in $(seq 0 $((REPEATS-1))); do
      submit_job "fm-f8-${sf}-r${rep}" \
        "python scripts/exp_fig8.py --subfig $sf --repeat $rep \
         --output_dir /mnt/nfs/results --device cuda --data_root /mnt/nfs/data"
    done
  done
}

# ── TABLE III ────────────────────────────────────────────────────────────────
submit_table3() {
  echo "=== TABLE III — FR Detection at Varying Ratios ==="
  for fr_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    for fr_type in previous_models gaussian_noise; do
      for rep in $(seq 0 $((REPEATS-1))); do
        fr_tag=$(echo $fr_ratio | tr '.' 'p')
        ft_tag=$(echo $fr_type | cut -c1-4)
        submit_job "fm-t3-fr${fr_tag}-${ft_tag}-r${rep}" \
          "python scripts/exp_table3.py --fr_ratio $fr_ratio --fr_type $fr_type \
           --repeat $rep --output_dir /mnt/nfs/results --device cuda --data_root /mnt/nfs/data"
      done
    done
  done
}

# ── TABLES IV & V ────────────────────────────────────────────────────────────
submit_table4_5() {
  echo "=== TABLE IV & V — Advanced FR Scenarios ==="
  for tbl in 4 5; do
    for rep in $(seq 0 $((REPEATS-1))); do
      submit_job "fm-t${tbl}-r${rep}" \
        "python scripts/exp_table4_5.py --table $tbl --repeat $rep \
         --output_dir /mnt/nfs/results --device cuda --data_root /mnt/nfs/data"
    done
  done
}

# ── TABLE VI ─────────────────────────────────────────────────────────────────
submit_table6() {
  echo "=== TABLE VI — Differential Privacy ==="
  for nm in 0.0 0.5 1.0 1.5 2.0; do
    for ep in 0 5; do
      for rep in $(seq 0 $((REPEATS-1))); do
        nm_tag=$(echo $nm | tr '.' 'p')
        submit_job "fm-t6-nm${nm_tag}-ep${ep}-r${rep}" \
          "python scripts/exp_table6.py --noise_mult $nm --extra_epochs $ep \
           --repeat $rep --output_dir /mnt/nfs/results --device cuda --data_root /mnt/nfs/data"
      done
    done
  done
}

# ── ABLATIONS (Tables VII, VIII, IX + Figs 9, 10) ───────────────────────────
submit_ablations() {
  echo "=== ABLATION STUDIES ==="
  for exp in table7 table8 table9 fig9 fig10; do
    for rep in $(seq 0 $((REPEATS-1))); do
      submit_job "fm-${exp}-r${rep}" \
        "python scripts/exp_ablations.py --exp $exp --repeat $rep \
         --output_dir /mnt/nfs/results --device cuda --data_root /mnt/nfs/data"
    done
  done
}

# ── DISPATCH ─────────────────────────────────────────────────────────────────
case "$EXP_FILTER" in
  table1)   submit_table1  ;;
  table2)   submit_table2  ;;
  fig7)     submit_fig7    ;;
  fig8)     submit_fig8    ;;
  table3)   submit_table3  ;;
  table4_5) submit_table4_5 ;;
  table6)   submit_table6  ;;
  ablations) submit_ablations ;;
  all)
    submit_table1
    submit_table2
    submit_fig7
    submit_fig8
    submit_table3
    submit_table4_5
    submit_table6
    submit_ablations
    ;;
  *)
    echo "Unknown filter: $EXP_FILTER"
    echo "Use: all | table1 | table2 | fig7 | fig8 | table3 | table4_5 | table6 | ablations"
    exit 1
    ;;
esac

echo ""
echo "Jobs submitted. Monitor with:"
echo "  runai list jobs --project $PROJECT"
echo "  watch -n 10 'runai list jobs --project $PROJECT | grep fm-'"