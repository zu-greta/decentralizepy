#!/usr/bin/env bash
# run_tests_noniid.sh — non-IID counterpart of run_tests.sh.
# Mirrors Tests 1-3 under a Dirichlet(alpha) partition, swept over alpha.
# The free-rider ESTIMATES eta (no oracle) so the calibration-TIMING effect is real.
#
#   ./run_tests_noniid.sh            # submit (ALPHAS="0.1 0.5 1.0", SEEDS="0 1 2")
#   RES=/path ./run_tests_noniid.sh PLOT
#
# COST: 3 alpha x 25 configs x 3 seeds = 225 runs. To start small, set
#   ALPHAS=0.5  CPC_HOPS="0 5 -1"  SEEDS=0
set -uo pipefail
SEEDS="${SEEDS:-0 1 2}"
ALPHAS="${ALPHAS:-0.1 0.5 1.0}"
RES="${RES:-/mnt/nfs/home/zu/results}"
CFG="${CFG:-14}"
CPC_HOPS="${CPC_HOPS:-0 5 10 20 50 -1}"
POS_A="${POS_A:-3,6}"; POS_B="${POS_B:-1,7}"
PT="python scripts/plot_tests.py"; PA="python scripts/plot_analysis.py"

aid(){ echo "$1" | tr -d '.'; }   # 0.5 -> 05  (for family tags)

# ---------------------------------------------------------------- PLOT ----
if [ "${1:-}" = "PLOT" ]; then
  OUT="${OUT:-figs_noniid}"; mkdir -p "$OUT"; ALL="$RES/*/result.json"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }
  for A in $ALPHAS; do AT=$(aid $A)
    run $PT test1_fpr --in "'$ALL'" --family t1_noniid_a$AT --out "$OUT/t1_noniid_a$AT"
    for P in A B; do PS=$([ $P = A ] && echo full || echo full)
      run $PT test_data --in "'$ALL'" --family t2_noniid_a${AT}_pos$P --scope full \
          --title "'Test2 non-IID a=$A pos$P'" --out "$OUT/t2_noniid_a${AT}_pos$P"
      run $PT test_data --in "'$ALL'" --family t3_noniid_a${AT}_pos$P --scope block2 \
          --title "'Test3 non-IID a=$A pos$P'" --out "$OUT/t3_noniid_a${AT}_pos$P"
    done
    # at-a-glance: effort frontier (full vs block2, posA) + scorecard + thresholds
    run $PA frontier --in "'$ALL'" \
        --families t2_noniid_a${AT}_posA t3_noniid_a${AT}_posA \
        --title "'Effort frontier non-IID a=$A (posA: full vs block2)'" --out "$OUT/frontier_noniid_a$AT"
    run $PA scorecard --in "'$ALL'" \
        --families t2_noniid_a${AT}_posA t3_noniid_a${AT}_posA t2_noniid_a${AT}_posB t3_noniid_a${AT}_posB \
        --title "'Scorecard non-IID a=$A'" --out "$OUT/scorecard_noniid_a$AT"
    run $PA thresholds --in "'$ALL'" --family t1_noniid_a$AT \
        --title "'Threshold FPR non-IID a=$A'" --out "$OUT/thresholds_noniid_a$AT"
  done
  echo "READ: per-alpha FPR + BER/effort + frontier/scorecard. Fair eta is auto-recomputed per family."
  exit 0
fi

# ------------------------------------------------------------- SUBMIT ----
# NOTE: no oracle -> the FR estimates & freezes eta from its own honest phase.
common(){ echo "ATTACK=autopilot AUTOP_HONEST_UNTIL=12 AUTOP_HONEST_EXTRA=3 \
  AUTOP_MARGIN0=0.06 ROUNDS=50 CALIB_ON_ALL=0 PARTITION=dirichlet"; }

for A in $ALPHAS; do AT=$(aid $A)
 for R in $SEEDS; do
  # T1 non-IID: all honest
  env ATTACK=none ROUNDS=50 CALIB_ON_ALL=0 PARTITION=dirichlet DIRICHLET_ALPHA=$A \
      FAMILY=t1_noniid_a$AT NOTE="t1 noniid a=$A" WAIT=0 ./submit_experiment.sh $CFG $R
  for NC in $CPC_HOPS; do
    # T2 full scope
    env $(common) DIRICHLET_ALPHA=$A AUTOP_SCOPE=full AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$POS_A \
        FAMILY=t2_noniid_a${AT}_posA SWEEP_LEVEL=$NC NOTE="t2 noniid a=$A posA cpc=$NC" WAIT=0 ./submit_experiment.sh $CFG $R
    env $(common) DIRICHLET_ALPHA=$A AUTOP_SCOPE=full AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$POS_B \
        FAMILY=t2_noniid_a${AT}_posB SWEEP_LEVEL=$NC NOTE="t2 noniid a=$A posB cpc=$NC" WAIT=0 ./submit_experiment.sh $CFG $R
    # T3 block2 scope
    env $(common) DIRICHLET_ALPHA=$A AUTOP_SCOPE=block2 AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$POS_A \
        FAMILY=t3_noniid_a${AT}_posA SWEEP_LEVEL=$NC NOTE="t3 noniid a=$A posA cpc=$NC" WAIT=0 ./submit_experiment.sh $CFG $R
    env $(common) DIRICHLET_ALPHA=$A AUTOP_SCOPE=block2 AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$POS_B \
        FAMILY=t3_noniid_a${AT}_posB SWEEP_LEVEL=$NC NOTE="t3 noniid a=$A posB cpc=$NC" WAIT=0 ./submit_experiment.sh $CFG $R
  done
 done
done
echo "submitted non-IID sweep. When done: RES=$RES ./run_tests_noniid.sh PLOT"
