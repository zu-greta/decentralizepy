#!/usr/bin/env bash
# run_full_sweep.sh — push scope=block from a BOUNDARY result to (hopefully) a
# clean break, at 3 seeds. Four ablations, each isolating one lever:
#   E0 baseline        : block, current settings (reproduce the boundary result)
#   E1 eta-fix         : real mu+3sigma estimate (Fix 1) — aims lower, taps harder
#   E2 honest-schedule : embeds on the SAME schedule as honest clients (Fix 2)
#   E3 deeper slice    : block2 = last TWO stages (Fix 3) — better generalisation
#   E4 bigger margin   : larger safety gap below eta
# Requires the code fixes 1-4 pushed first. CALIB_ON_ALL=0 = fair (honest-only) eta.
#
#   ./run_full_sweep.sh            # submit all (SEEDS="0 1 2")
#   RES=/path ./run_full_sweep.sh PLOT   # when done, build every figure
set -uo pipefail
SEEDS="${SEEDS:-0 1 2}"
RES="${RES:-/mnt/nfs/home/zu/results}"
PT="python scripts/plot_thresholds.py"; SB="python scripts/seedband.py"; PA="python scripts/plot_adaptive.py"

if [ "${1:-}" = "PLOT" ]; then
  OUT="${OUT:-figs}"; mkdir -p "$OUT"; ALL="$RES/*/result.json"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }
  # go/no-go with error bars across all ablations
  run $PT evade_bars --in "'$ALL'" --family blk_base blk_etafix blk_honest blk2 blk_margin confirm_head --out "$OUT/evade_ablations"
  # seed-band (mean +/- std shaded, honest BER, actual + estimated eta) per ablation
  for N in "block baseline 3seed" "block etafix 3seed" "block honest-sched 3seed" "block2 3seed" "block bigmargin 3seed"; do
    tag=$(echo "$N" | tr ' =' '__')
    run $SB --in "'$ALL'" --note "'$N'" --title "'$N'" --out "$OUT/seedband_$tag"
  done
  # why it does / doesn't clear the line: estimated vs actual eta
  run $PT estimate  --in "'$ALL'" --family blk_etafix --out "$OUT/estimate_etafix"
  # the submarine + timeline on the best ablation (edit family after reading evade)
  run $PT submarine --in "'$ALL'" --family blk_etafix --out "$OUT/submarine_etafix"
  run $PT timeline  --in "'$ALL'" --family blk_etafix --out "$OUT/timeline_etafix"
  echo; echo "Read evade_ablations.png FIRST: is any 'frozen'/'converged' bar HIGH with a SMALL error bar? that's a clean break."
  exit 0
fi

sub(){ local r="$1"; shift; env "$@" ROUNDS=50 CALIB_ON_ALL=0 WAIT=0 ./submit_experiment.sh 17 "$r"; }
for R in $SEEDS; do
  # E0 baseline (reproduce boundary)
  sub $R ATTACK=autopilot AUTOP_SCOPE=block  AUTOP_MAX_BATCHES=250 AUTOP_PROTECT_UNTIL=8 \
      FAMILY=blk_base    SWEEP_VAR=none NOTE="block baseline 3seed"
  # E1 eta-fix (needs Fix 1)
  sub $R ATTACK=autopilot AUTOP_SCOPE=block  AUTOP_MAX_BATCHES=250 AUTOP_MARGIN0=0.10 AUTOP_PROTECT_UNTIL=8 \
      FAMILY=blk_etafix  SWEEP_VAR=none NOTE="block etafix 3seed"
  # E2 honest-schedule (needs Fix 2); honest clients converge ~round 6-8
  sub $R ATTACK=autopilot AUTOP_SCOPE=block  AUTOP_MAX_BATCHES=250 AUTOP_HONEST_UNTIL=8 AUTOP_PROTECT_UNTIL=8 \
      FAMILY=blk_honest  SWEEP_VAR=none NOTE="block honest-sched 3seed"
  # E3 deeper slice (needs Fix 3)
  sub $R ATTACK=autopilot AUTOP_SCOPE=block2 AUTOP_MAX_BATCHES=250 AUTOP_PROTECT_UNTIL=8 \
      FAMILY=blk2        SWEEP_VAR=none NOTE="block2 3seed"
  # E4 bigger margin
  sub $R ATTACK=autopilot AUTOP_SCOPE=block  AUTOP_MAX_BATCHES=250 AUTOP_MARGIN0=0.15 AUTOP_PROTECT_UNTIL=8 \
      FAMILY=blk_margin  SWEEP_VAR=none NOTE="block bigmargin 3seed"
  # head baseline for the scope curve (no fix needed)
  sub $R ATTACK=autopilot AUTOP_SCOPE=head   AUTOP_MAX_BATCHES=250 AUTOP_PROTECT_UNTIL=8 \
      FAMILY=confirm_head SWEEP_VAR=none NOTE="autopilot cifar100 scope=head 3seed"
done
echo "submitted $(echo $SEEDS | wc -w) seeds x 6 configs. When done: RES=$RES ./run_confirm_sweep.sh PLOT"