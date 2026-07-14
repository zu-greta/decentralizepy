#!/usr/bin/env bash
# run_all.sh 
#
# DIMENSIONS (env-overridable):
#   PARTS   partitions        : "iid dir0.1 dir0.5 dir1.0"
#   ETAS    threshold source  : "oracle estimate"        (oracle=0.09 for testing; estimate=realistic)
#   SCOPES  params per tap     : "full block2"
#   STAYS   attack mode        : "tap coast"             (coast = submarine/stay_min)
#   CPC_HOPS data per tap      : "0 5 10 20 50 -1"       (trig-only -> +N/class -> full shard)
#   POSES   positions          : "posA:3,6 posB:1,7"     (hard / easy)
#   SEEDS                      : "0 1 2"
#
# USAGE
#   ./run_all.sh matrix                 # submit EVERYTHING (large!)
#   PARTS=iid STAYS=tap ./run_all.sh matrix        # a slice
#   ./run_all.sh quick                  # tiny sanity slice (iid, estimate, full, tap, posA/B, 1 seed)
#   RES=/path ./run_all.sh PLOT [iid|dir0.5|...]   # plot a partition's results
#
# SIZE: full matrix = 4 parts x 2 eta x 2 scope x 2 stay x 6 cpc x 2 pos x 3 seeds
#       = 1152 runs (+ 4 all-honest x 3). Run slices; start with `quick`.
set -uo pipefail
CFG="${CFG:-14}"; RES="${RES:-/mnt/nfs/home/zu/results}"
PARTS="${PARTS:-iid dir0.1 dir0.5 dir1.0}"
ETAS="${ETAS:-oracle estimate}"
SCOPES="${SCOPES:-full block2}"
STAYS="${STAYS:-tap coast}"
CPC_HOPS="${CPC_HOPS:-0 5 10 20 50 -1}"
POSES="${POSES:-posA:3,6 posB:1,7}"
SEEDS="${SEEDS:-0 1 2}"
PT="python scripts/plot_tests.py"; PA="python scripts/plot_analysis.py"

part_env(){ case "$1" in
  iid)     echo "" ;;
  dir*)    echo "PARTITION=dirichlet DIRICHLET_ALPHA=${1#dir}" ;;
esac; }
part_tag(){ echo "$1" | tr -d '.'; }         # dir0.5 -> dir05

submit_one(){  # args: part eta scope stay cpc posname posids seed
  local part=$1 eta=$2 scope=$3 stay=$4 cpc=$5 pn=$6 ids=$7 seed=$8
  local E="ROUNDS=50 CALIB_ON_ALL=0 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 AUTOP_MARGIN0=0.06"
  E="$E $(part_env $part) AUTOP_SCOPE=$scope AUTOP_COMMON_PER_CLASS=$cpc FREE_RIDER_IDS=$ids ATTACK=autopilot"
  [ "$eta" = oracle ] && E="$E AUTOP_ORACLE_ETA=0.09"
  [ "$stay" = coast ] && E="$E AUTOP_STAY_MIN=1"
  local fam="$(part_tag $part)_${eta}_${scope}_${stay}_${pn}"
  env $E FAMILY="$fam" SWEEP_LEVEL=$cpc NOTE="$fam cpc=$cpc" WAIT=0 ./submit_experiment.sh "$CFG" "$seed"
}

run_matrix(){
  for part in $PARTS; do
    # all-honest reference per partition (for the thresholds/FPR)
    for seed in $SEEDS; do
      env ROUNDS=50 ATTACK=none $(part_env $part) FAMILY="t1_$(part_tag $part)" \
          NOTE="all honest $part" WAIT=0 ./submit_experiment.sh "$CFG" "$seed"
    done
    for eta in $ETAS; do for scope in $SCOPES; do for stay in $STAYS; do
      for cpc in $CPC_HOPS; do for pp in $POSES; do
        pn=${pp%%:*}; ids=${pp##*:}
        for seed in $SEEDS; do submit_one $part $eta $scope $stay $cpc $pn $ids $seed; done
      done; done
    done; done; done
  done
}

# =============================== PLOT ===============================
if [ "${1:-}" = "PLOT" ]; then
  P="${2:-iid}"; PT_="$(part_tag $P)"; OUT="${OUT:-figs/$PT_}"; mkdir -p "$OUT"; ALL="$RES/*/result.json"
  ETA="${ETA_MODE:-estimate}"; STAY="${STAY_MODE:-tap}"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }
  # per-scope x position BER+effort
  for scope in full block2; do for pn in posA posB; do
    F="${PT_}_${ETA}_${scope}_${STAY}_${pn}"
    run $PT test_data --in "'$ALL'" --family $F --scope $scope --title "'$F'" --out "$OUT/${F}"
  done; done
  # timeline (representative: full, posA), effort frontier, scorecard, and ALL thresholds
  run $PA timeline   --in "'$ALL'" --family ${PT_}_${ETA}_full_${STAY}_posA --level -1 --out "$OUT/timeline_${PT_}"
  run $PA frontier   --in "'$ALL'" --families ${PT_}_${ETA}_full_${STAY}_posA ${PT_}_${ETA}_block2_${STAY}_posA --title "'Effort frontier $P (posA)'" --out "$OUT/frontier_${PT_}"
  run $PA scorecard  --in "'$ALL'" --families ${PT_}_${ETA}_full_${STAY}_posA ${PT_}_${ETA}_block2_${STAY}_posA ${PT_}_${ETA}_full_${STAY}_posB ${PT_}_${ETA}_block2_${STAY}_posB --out "$OUT/scorecard_${PT_}"
  run $PA all_thresholds --in "'$ALL'" --family ${PT_}_${ETA}_full_${STAY}_posA --honest_family t1_${PT_} --title "'All thresholds $P'" --out "$OUT/all_thresholds_${PT_}"
  run $PT test1_fpr  --in "'$ALL'" --family t1_${PT_} --out "$OUT/test1_fpr_${PT_}"
  echo "plotted -> $OUT   (ETA_MODE=$ETA STAY_MODE=$STAY)"; exit 0
fi

# =============================== SUBMIT ===============================
case "${1:-matrix}" in
  matrix) run_matrix ;;
  quick)  PARTS=iid ETAS=estimate SCOPES=full STAYS=tap SEEDS=0 CPC_HOPS="0 5 -1" run_matrix ;;
  *) echo "usage: ./run_all.sh [matrix|quick]  |  RES=... ./run_all.sh PLOT <partition>"; exit 1 ;;
esac
echo "submitted '${1:-matrix}'. When done:  RES=$RES ./run_all.sh PLOT <partition>"