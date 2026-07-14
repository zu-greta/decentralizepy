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
  # warmup schedule: fixed (default, position-independent) or dynamic (convergence-detected)
  [ -n "${AUTOP_WARMUP_MODE:-}" ] && E="$E AUTOP_WARMUP_MODE=${AUTOP_WARMUP_MODE}"
  # FAM_SUFFIX keeps variant runs (e.g. dynamic warmup) in their own family
  local fam="$(part_tag $part)_${eta}_${scope}_${stay}_${pn}${FAM_SUFFIX:-}"
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
  P="${2:-iid}"; PT_="$(part_tag $P)"; ALL="$RES/*/result.json"
  ETA="${ETA_MODE:-estimate}"; STAY="${STAY_MODE:-tap}"; SFX="${FAM_SUFFIX:-}"
  # OUT defaults to a slice-specific dir so different slices don't overwrite each other
  OUT="${OUT:-figs/${PT_}${SFX}$([ "$ETA" = estimate ] || echo "_$ETA")$([ "$STAY" = tap ] || echo "_$STAY")}"
  mkdir -p "$OUT"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }
  # per-scope x position BER+effort
  for scope in full block2; do for pn in posA posB; do
    F="${PT_}_${ETA}_${scope}_${STAY}_${pn}${SFX}"
    run $PT test_data --in "'$ALL'" --family $F --scope $scope --title "'$F'" --out "$OUT/${F}"
  done; done
  # timeline (representative: full, posA), effort frontier, scorecard, and ALL thresholds
  run $PA timeline   --in "'$ALL'" --family ${PT_}_${ETA}_full_${STAY}_posA${SFX} --level -1 --out "$OUT/timeline_${PT_}"
  run $PA frontier   --in "'$ALL'" --families ${PT_}_${ETA}_full_${STAY}_posA${SFX} ${PT_}_${ETA}_block2_${STAY}_posA${SFX} --title "'Effort frontier $P (posA)'" --out "$OUT/frontier_${PT_}"
  run $PA scorecard  --in "'$ALL'" --families ${PT_}_${ETA}_full_${STAY}_posA${SFX} ${PT_}_${ETA}_block2_${STAY}_posA${SFX} ${PT_}_${ETA}_full_${STAY}_posB${SFX} ${PT_}_${ETA}_block2_${STAY}_posB${SFX} --out "$OUT/scorecard_${PT_}"
  run $PA all_thresholds --in "'$ALL'" --family ${PT_}_${ETA}_full_${STAY}_posA${SFX} --honest_family t1_${PT_} --title "'All thresholds $P'" --out "$OUT/all_thresholds_${PT_}"
  run $PT test1_fpr  --in "'$ALL'" --family t1_${PT_} --out "$OUT/test1_fpr_${PT_}"
  echo "plotted -> $OUT   (ETA_MODE=$ETA STAY_MODE=$STAY FAM_SUFFIX='$SFX')"; exit 0
fi

# ---- PLOTALL: every slice that `slim` produces, into its own figs/ subdir ----
if [ "${1:-}" = "PLOTALL" ]; then
  echo "### iid  estimate/tap  (HEADLINE: tier0+tier1+tier2-block2)"
  bash "$0" PLOT iid
  echo "### iid  estimate/COAST (tier2)"
  STAY_MODE=coast bash "$0" PLOT iid
  echo "### iid  ORACLE/tap    (tier4)"
  ETA_MODE=oracle bash "$0" PLOT iid
  echo "### iid  DYNAMIC warmup (tier5 robustness)"
  FAM_SUFFIX=_dyn bash "$0" PLOT iid
  for p in dir0.5 dir0.1 dir1.0; do
    echo "### $p  estimate/tap  (tier3)"
    bash "$0" PLOT "$p"
  done
  echo "ALL PLOTS DONE -> figs/"
  exit 0
fi

# =============================== SLIM (tiered) =======================
# The full matrix is ~1164 runs = ~1164 separate runai jobs. Most cells are
# redundant: block2 has the SAME BER as full (only cheaper GPU), oracle only
# changes the FR's target (not the floor), coast only matters at easy positions,
# and the cpc knee is at +5/class. `slim` keeps every scientific comparison but
# only spends seeds/levels where they change a conclusion -> ~134 runs (~8.7x fewer).
#
# Run ONE TIER AT A TIME so you never queue more than ~50 jobs:
#     ./run_all.sh tier0 ... ./run_all.sh tier5     (or ./run_all.sh slim = all of them)
#
# WARMUP MODE: default is FIXED (config.py autop_warmup_mode="fixed") -> warmup
# length is the same at every position, so the position comparison is not
# confounded by warmup cost. tier5 re-runs the headline with WARMUP_MODE=dynamic
# as a robustness cell.

tier0(){  # all-honest reference (the FPR / threshold plots come from these)
  echo "== TIER 0: all-honest reference (iid + dir0.5), 3 seeds  [6 jobs]"
  for part in iid dir0.5; do for seed in 0 1 2; do
    env ROUNDS=50 ATTACK=none $(part_env $part) FAMILY="t1_$(part_tag $part)" \
        NOTE="all honest $part" WAIT=0 ./submit_experiment.sh "$CFG" "$seed"
  done; done
}

tier1(){  # THE HEADLINE: data sweep x position
  echo "== TIER 1: HEADLINE iid/estimate/full/tap, posA+posB, all 6 cpc, 3 seeds  [36 jobs]"
  PARTS=iid ETAS=estimate SCOPES=full STAYS=tap CPC_HOPS="0 5 10 20 50 -1" \
    POSES="posA:3,6 posB:1,7" SEEDS="0 1 2" run_matrix_noref
}

tier2(){  # scope (params) + mode (coast) story
  echo "== TIER 2: SCOPE+MODE iid/estimate {block2 tap, full coast, block2 coast}, cpc 0/5/-1, 3 seeds  [54 jobs]"
  PARTS=iid ETAS=estimate SCOPES=block2 STAYS=tap   CPC_HOPS="0 5 -1" SEEDS="0 1 2" run_matrix_noref
  PARTS=iid ETAS=estimate SCOPES=full   STAYS=coast CPC_HOPS="0 5 -1" SEEDS="0 1 2" run_matrix_noref
  PARTS=iid ETAS=estimate SCOPES=block2 STAYS=coast CPC_HOPS="0 5 -1" SEEDS="0 1 2" run_matrix_noref
}

tier3(){  # non-IID generalisation
  echo "== TIER 3: NON-IID estimate/full/tap, cpc 0/5/-1; dir0.5 x3 seeds, dir0.1 & dir1.0 x1  [32 jobs]"
  PARTS=dir0.5 ETAS=estimate SCOPES=full STAYS=tap CPC_HOPS="0 5 -1" SEEDS="0 1 2" run_matrix_noref
  PARTS="dir0.1 dir1.0" ETAS=estimate SCOPES=full STAYS=tap CPC_HOPS="0 5 -1" SEEDS="0" run_matrix_noref
  for part in dir0.1 dir1.0; do
    env ROUNDS=50 ATTACK=none $(part_env $part) FAMILY="t1_$(part_tag $part)" \
        NOTE="all honest $part" WAIT=0 ./submit_experiment.sh "$CFG" 0
  done
}

tier4(){  # oracle control: FR GIVEN eta (isolates estimation error)
  echo "== TIER 4: ORACLE control iid/oracle/full/tap, cpc 0/5/-1, 1 seed  [6 jobs]"
  PARTS=iid ETAS=oracle SCOPES=full STAYS=tap CPC_HOPS="0 5 -1" SEEDS="0" run_matrix_noref
}

tier5(){  # ROBUSTNESS: dynamic (convergence-detected) warmup instead of fixed
  echo "== TIER 5: DYNAMIC-WARMUP robustness iid/estimate/full/tap, posA+posB, cpc 0/5/-1, 1 seed  [6 jobs]"
  AUTOP_WARMUP_MODE=dynamic FAM_SUFFIX=_dyn \
    PARTS=iid ETAS=estimate SCOPES=full STAYS=tap CPC_HOPS="0 5 -1" \
    POSES="posA:3,6 posB:1,7" SEEDS="0" run_matrix_noref
}

run_slim(){ tier0; tier1; tier2; tier3; tier4; tier5; }

# run_matrix without the per-partition all-honest ref (tiers add refs explicitly)
run_matrix_noref(){
  for part in $PARTS; do
    for eta in $ETAS; do for scope in $SCOPES; do for stay in $STAYS; do
      for cpc in $CPC_HOPS; do for pp in $POSES; do
        pn=${pp%%:*}; ids=${pp##*:}
        for seed in $SEEDS; do submit_one $part $eta $scope $stay $cpc $pn $ids $seed; done
      done; done
    done; done; done
  done
}

# =============================== SUBMIT ===============================
case "${1:-matrix}" in
  matrix) run_matrix ;;
  slim)   run_slim ;;
  tier0)  tier0 ;;
  tier1)  tier1 ;;
  tier2)  tier2 ;;
  tier3)  tier3 ;;
  tier4)  tier4 ;;
  tier5)  tier5 ;;
  quick)  PARTS=iid ETAS=estimate SCOPES=full STAYS=tap SEEDS=0 CPC_HOPS="0 5 -1" run_matrix ;;
  *) echo "usage: ./run_all.sh [quick|tier0..tier5|slim|matrix]\n       RES=... ./run_all.sh PLOT <partition>   (ETA_MODE/STAY_MODE/FAM_SUFFIX to pick a slice)\n       RES=... ./run_all.sh PLOTALL            (every slim slice -> figs/*)"; exit 1 ;;
esac
echo "submitted '${1:-matrix}'. When done:  RES=$RES ./run_all.sh PLOT <partition>"