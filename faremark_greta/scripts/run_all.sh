#!/usr/bin/env bash
# run_all.sh — one entry point for every experiment. Autopilot free-rider, config 14.
#
#   ./run_all.sh iid          # Tests 1-3 (IID): FPR + data sweep x2 positions x {full,block2}
#   ./run_all.sh submarine    # Test 4: coast-when-safe (stay_min) at easy vs hard positions
#   ./run_all.sh noniid       # Tests 1-3 under Dirichlet(alpha), swept alpha
#   ./run_all.sh all          # everything above
#   RES=/path ./run_all.sh PLOT [iid|submarine|noniid|all]
#
# Knobs (env):  SEEDS="0 1 2"  ALPHAS="0.1 0.5 1.0"  CPC_HOPS="0 5 10 20 50 -1"
#               POS_A=3,6  POS_B=1,7  ORACLE=0.09  CFG=14  RES=/mnt/nfs/home/zu/results
set -uo pipefail
SEEDS="${SEEDS:-0 1 2}"
ALPHAS="${ALPHAS:-0.1 0.5 1.0}"
CPC_HOPS="${CPC_HOPS:-0 5 10 20 50 -1}"
POS_A="${POS_A:-3,6}"; POS_B="${POS_B:-1,7}"
ORACLE="${ORACLE:-0.09}"; CFG="${CFG:-14}"
RES="${RES:-/mnt/nfs/home/zu/results}"
PT="python scripts/plot_tests.py"; PA="python scripts/plot_analysis.py"
aid(){ echo "$1" | tr -d '.'; }

sub(){ WAIT=0 ./submit_experiment.sh "$CFG" "$1"; }   # $1 = seed; env vars carry the rest

# ------- IID Tests 1-3 (tap every round; cost = data x scope) -------
run_iid(){
 for R in $SEEDS; do
  env ATTACK=none ROUNDS=50 FAMILY=t1_all_honest NOTE="t1 all honest" sub $R
  for NC in $CPC_HOPS; do for PP in "posA:$POS_A" "posB:$POS_B"; do P=${PP%%:*}; IDS=${PP##*:}
   env ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_HONEST_UNTIL=12 AUTOP_MARGIN0=0.06 ROUNDS=50 \
       AUTOP_SCOPE=full  AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$IDS \
       FAMILY=t2_full_$P SWEEP_LEVEL=$NC NOTE="t2 full $P cpc=$NC" sub $R
   env ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_HONEST_UNTIL=12 AUTOP_MARGIN0=0.06 ROUNDS=50 \
       AUTOP_SCOPE=block2 AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$IDS \
       FAMILY=t3_block2_$P SWEEP_LEVEL=$NC NOTE="t3 block2 $P cpc=$NC" sub $R
  done; done
 done
}

# ------- Test 4: SUBMARINE / coast (stay_min=1) -------
# Coast (no training) while safely under target; tap only when the mark drifts up.
# Point: at EASY positions the FR already passes -> submarine makes it near-free.
#        at HARD positions it's still caught (coast BER >= tap BER >= eta).
run_submarine(){
 for R in $SEEDS; do for NC in 5 -1; do for PP in "posA:$POS_A" "posB:$POS_B"; do
   P=${PP%%:*}; IDS=${PP##*:}
   env ATTACK=autopilot AUTOP_ORACLE_ETA=$ORACLE AUTOP_HONEST_UNTIL=12 AUTOP_MARGIN0=0.06 ROUNDS=50 \
       AUTOP_STAY_MIN=1 AUTOP_SCOPE=full AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$IDS \
       FAMILY=t4_sub_${P}_cpc$NC SWEEP_LEVEL=$NC NOTE="t4 submarine $P cpc=$NC" sub $R
 done; done; done
}

# ------- Non-IID Tests 1-3 (FR ESTIMATES eta; no oracle) -------
run_noniid(){
 for A in $ALPHAS; do AT=$(aid $A); for R in $SEEDS; do
   env ATTACK=none ROUNDS=50 PARTITION=dirichlet DIRICHLET_ALPHA=$A \
       FAMILY=t1_noniid_a$AT NOTE="t1 noniid a=$A" sub $R
   for NC in $CPC_HOPS; do for PP in "posA:$POS_A" "posB:$POS_B"; do P=${PP%%:*}; IDS=${PP##*:}
    env ATTACK=autopilot AUTOP_HONEST_UNTIL=12 AUTOP_MARGIN0=0.06 ROUNDS=50 PARTITION=dirichlet DIRICHLET_ALPHA=$A \
        AUTOP_SCOPE=full  AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$IDS \
        FAMILY=t2_noniid_a${AT}_$P SWEEP_LEVEL=$NC NOTE="t2 noniid a=$A $P cpc=$NC" sub $R
    env ATTACK=autopilot AUTOP_HONEST_UNTIL=12 AUTOP_MARGIN0=0.06 ROUNDS=50 PARTITION=dirichlet DIRICHLET_ALPHA=$A \
        AUTOP_SCOPE=block2 AUTOP_COMMON_PER_CLASS=$NC FREE_RIDER_IDS=$IDS \
        FAMILY=t3_noniid_a${AT}_$P SWEEP_LEVEL=$NC NOTE="t3 noniid a=$A $P cpc=$NC" sub $R
   done; done
 done; done
}

# =================== PLOT ===================
if [ "${1:-}" = "PLOT" ]; then
  WHAT="${2:-all}"; OUT="${OUT:-figs}"; mkdir -p "$OUT"; ALL="$RES/*/result.json"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }
  if [ "$WHAT" = "iid" ] || [ "$WHAT" = "all" ]; then
    run $PT test1_fpr --in "'$ALL'" --family t1_all_honest --out "$OUT/test1_fpr"
    for P in posA posB; do
      run $PT test_data --in "'$ALL'" --family t2_full_$P  --scope full   --title "'Test2 full $P'"   --out "$OUT/test2_full_$P"
      run $PT test_data --in "'$ALL'" --family t3_block2_$P --scope block2 --title "'Test3 block2 $P'" --out "$OUT/test3_block2_$P"
    done
    run $PA frontier   --in "'$ALL'" --families t2_full_posA t3_block2_posA --title "'Effort frontier (posA)'" --out "$OUT/frontier_posA"
    run $PA scorecard  --in "'$ALL'" --families t2_full_posA t3_block2_posA t2_full_posB t3_block2_posB --out "$OUT/scorecard_iid"
    run $PA thresholds --in "'$ALL'" --family t1_all_honest --out "$OUT/thresholds_iid"
  fi
  if [ "$WHAT" = "submarine" ] || [ "$WHAT" = "all" ]; then
    for f in $(cd "$RES" 2>/dev/null && ls -d */ 2>/dev/null | sed 's#/##' | grep t4_sub | sed 's/_[0-9]*$//' | sort -u); do :; done
    run $PA timeline  --in "'$ALL'" --family t4_sub_posB_cpc-1 --title "'Submarine — easy positions (coast)'" --out "$OUT/sub_timeline_posB"
    run $PA timeline  --in "'$ALL'" --family t4_sub_posA_cpc-1 --title "'Submarine — hard positions (coast)'" --out "$OUT/sub_timeline_posA"
  fi
  if [ "$WHAT" = "noniid" ] || [ "$WHAT" = "all" ]; then
    for A in $ALPHAS; do AT=$(aid $A)
      run $PT test1_fpr --in "'$ALL'" --family t1_noniid_a$AT --out "$OUT/t1_noniid_a$AT"
      for P in posA posB; do
        run $PT test_data --in "'$ALL'" --family t2_noniid_a${AT}_$P --scope full   --title "'Test2 noniid a=$A $P'" --out "$OUT/t2_noniid_a${AT}_$P"
        run $PT test_data --in "'$ALL'" --family t3_noniid_a${AT}_$P --scope block2 --title "'Test3 noniid a=$A $P'" --out "$OUT/t3_noniid_a${AT}_$P"
      done
      run $PA frontier   --in "'$ALL'" --families t2_noniid_a${AT}_posA t3_noniid_a${AT}_posA --title "'Frontier noniid a=$A'" --out "$OUT/frontier_noniid_a$AT"
      run $PA scorecard  --in "'$ALL'" --families t2_noniid_a${AT}_posA t3_noniid_a${AT}_posA t2_noniid_a${AT}_posB t3_noniid_a${AT}_posB --out "$OUT/scorecard_noniid_a$AT"
      run $PA thresholds --in "'$ALL'" --family t1_noniid_a$AT --out "$OUT/thresholds_noniid_a$AT"
    done
  fi
  echo "plotted -> $OUT"; exit 0
fi

# =================== SUBMIT ===================
case "${1:-all}" in
  iid)       run_iid ;;
  submarine) run_submarine ;;
  noniid)    run_noniid ;;
  all)       run_iid; run_submarine; run_noniid ;;
  *) echo "usage: ./run_all.sh [iid|submarine|noniid|all]  |  RES=... ./run_all.sh PLOT [iid|submarine|noniid|all]"; exit 1 ;;
esac
echo "submitted '${1:-all}'. When done: RES=$RES ./run_all.sh PLOT ${1:-all}"