#!/usr/bin/env bash
# run_tests.sh — the only sweep. Autopilot free-rider, config idx 14. 3 seeds each.
#
#   TEST 1  all-honest FPR check (no free-rider): does every honest client sit
#           under the fair threshold, or are hard-position clients flagged?
#   TEST 2  full-scope data sweep (2 free-riders), TWO pinned position sets:
#           triggers-only -> +N/class -> full shard, trained every round like honest.
#   TEST 3  same as TEST 2 but scope=block2 (backbone frozen; cheaper GPU).
#
#   ./run_tests.sh              # submit (SEEDS="0 1 2")
#   RES=/path ./run_tests.sh PLOT
set -uo pipefail
SEEDS="${SEEDS:-0 1 2}"
RES="${RES:-/mnt/nfs/home/zu/results}"
CFG="${CFG:-14}"                          # autopilot config index (config.AUTOPILOT_IDX)
ORACLE="${ORACLE:-0.09}"                  # oracle eta for testing (set "" to make the FR estimate)
CPC_HOPS="${CPC_HOPS:-0 5 10 20 50 -1}"   # triggers-only -> +N/class -> full shard(-1)
POS_A="${POS_A:-3,6}"                      # position set A (mixed/hard floors)
POS_B="${POS_B:-1,7}"                      # position set B (easy floors)
PT="python scripts/plot_tests.py"

# ---------------------------------------------------------------- PLOT ----
if [ "${1:-}" = "PLOT" ]; then
  OUT="${OUT:-figs}"; mkdir -p "$OUT"; ALL="$RES/*/result.json"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }
  run $PT test1_fpr  --in "'$ALL'" --family t1_all_honest --out "$OUT/test1_fpr"
  run $PT test_data  --in "'$ALL'" --family t2_full_posA --scope full \
      --title "'TEST 2 (full scope, positions $POS_A)'"   --out "$OUT/test2_full_posA"
  run $PT test_data  --in "'$ALL'" --family t2_full_posB --scope full \
      --title "'TEST 2 (full scope, positions $POS_B)'"   --out "$OUT/test2_full_posB"
  run $PT test_data  --in "'$ALL'" --family t3_block2_posA --scope block2 \
      --title "'TEST 3 (block2 scope, positions $POS_A)'" --out "$OUT/test3_block2_posA"
  run $PT test_data  --in "'$ALL'" --family t3_block2_posB --scope block2 \
      --title "'TEST 3 (block2 scope, positions $POS_B)'" --out "$OUT/test3_block2_posB"
  echo "READ: test1_fpr (honest FPR under two eta defs); test2/3 (per-FR & per-honest BER + effort)."
  exit 0
fi

# ------------------------------------------------------------- SUBMIT ----
# Common autopilot knobs: honest until eta converges (~round 12), then tap every
# round at a FIXED honest-style budget; cost = scope + data (CPC).
common(){ echo "ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_HONEST_UNTIL=12 \
  AUTOP_HONEST_EXTRA=3 AUTOP_MARGIN0=0.06 ROUNDS=50 CALIB_ON_ALL=0"; }

for R in $SEEDS; do
  # ---- TEST 1: all-honest control (no free-rider) ----
  env ATTACK=none ROUNDS=50 CALIB_ON_ALL=0 \
      FAMILY=t1_all_honest manifest.sweep_var="none" NOTE="test1 all honest" \
      WAIT=0 ./submit_experiment.sh $CFG $R

  # ---- TEST 2: full scope, data sweep, two position sets ----
  for NC in $CPC_HOPS; do
    env $(common) AUTOP_SCOPE=full AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$POS_A \
        FAMILY=t2_full_posA SWEEP_VAR=autop_common_per_class SWEEP_LEVEL=$NC \
        NOTE="test2 full posA cpc=$NC" WAIT=0 ./submit_experiment.sh $CFG $R
    env $(common) AUTOP_SCOPE=full AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$POS_B \
        FAMILY=t2_full_posB SWEEP_VAR=autop_common_per_class SWEEP_LEVEL=$NC \
        NOTE="test2 full posB cpc=$NC" WAIT=0 ./submit_experiment.sh $CFG $R
  done

  # ---- TEST 3: block2 scope, data sweep, two position sets ----
  for NC in $CPC_HOPS; do
    env $(common) AUTOP_SCOPE=block2 AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$POS_A \
        FAMILY=t3_block2_posA SWEEP_VAR=autop_common_per_class SWEEP_LEVEL=$NC \
        NOTE="test3 block2 posA cpc=$NC" WAIT=0 ./submit_experiment.sh $CFG $R
    env $(common) AUTOP_SCOPE=block2 AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$POS_B \
        FAMILY=t3_block2_posB SWEEP_VAR=autop_common_per_class SWEEP_LEVEL=$NC \
        NOTE="test3 block2 posB cpc=$NC" WAIT=0 ./submit_experiment.sh $CFG $R
  done
done
echo "submitted. When done: RES=$RES ./run_tests.sh PLOT"
