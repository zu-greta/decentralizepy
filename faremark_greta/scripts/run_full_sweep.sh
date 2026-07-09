#!/usr/bin/env bash
# run_honestcal_sweep.sh — test the HONEST-ROUND eta calibration fix.
# The free-rider trains fully-honest for the first AUTOP_HONEST_UNTIL rounds, and
# calibrates its eta estimate on THOSE rounds (the same converged-honest distribution
# the server's fair eta uses) -> aims below the REAL line instead of the pessimistic
# probe estimate (~0.25). Then it coasts/taps. Four arms, 3 seeds each:
#   hc_block   : honest-cal + block  scope  (cheap re-embed)
#   hc_block2  : honest-cal + block2 scope  (deeper, best generalization)
#   hc_full    : honest-cal + full   scope  (control: does deep re-embed clear it?)
#   hc_block_nocal (AUTOP_HONEST_UNTIL=0) : block WITHOUT honest-cal = the old estimate,
#                                           as an A/B baseline to isolate the fix's effect.
# AUTOP_HONEST_UNTIL=10 so there are >=3 CONVERGED honest rounds to calibrate on.
# AUTOP_MARGIN0=0.06 = a big safety margin below the (now-accurate) eta, per the plan.
#
#   ./run_honestcal_sweep.sh              # submit all (SEEDS="0 1 2")
#   RES=/path ./run_honestcal_sweep.sh PLOT
set -uo pipefail
SEEDS="${SEEDS:-0 1 2}"; RES="${RES:-/mnt/nfs/home/zu/results}"
PT="python scripts/plot_thresholds.py"; SB="python scripts/seedband.py"

if [ "${1:-}" = "PLOT" ]; then
  OUT="${OUT:-figs}"; mkdir -p "$OUT"; ALL="$RES/*/result.json"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }
  # go/no-go with error bars across all arms
  run $PT evade_bars --in "'$ALL'" --family hc_block hc_block2 hc_full hc_block_nocal --out "$OUT/honestcal_evade"
  # seed-band (std shaded + honest BER + ACTUAL vs ESTIMATED eta) per arm — the key read:
  # does the grey ESTIMATED-eta line now sit DOWN at ~0.09 (fix working) and the red FR
  # line dip UNDER the green ACTUAL-eta line?
  for N in "honestcal block 3seed" "honestcal block2 3seed" "honestcal full 3seed" "block nocal 3seed"; do
    tag=$(echo "$N" | tr ' ' '_')
    run $SB --in "'$ALL'" --note "'$N'" --title "'$N'" --out "$OUT/seedband_$tag"
  done
  # why it clears (or not): estimated vs actual eta on the block arm
  run $PT estimate  --in "'$ALL'" --family hc_block  --out "$OUT/estimate_hc_block"
  run $PT estimate  --in "'$ALL'" --family hc_block2 --out "$OUT/estimate_hc_block2"
  # the submarine + timeline on block2 (edit family to the winner after reading evade)
  run $PT submarine --in "'$ALL'" --family hc_block2 --out "$OUT/submarine_hc_block2"
  run $PT timeline  --in "'$ALL'" --family hc_block2 --out "$OUT/timeline_hc_block2"
  echo; echo "READ FIRST: honestcal_evade.png — is a FAIR (frozen/converged) bar HIGH with a SMALL error bar?"
  echo "Then seedband_*: has the grey ESTIMATED-eta dropped to ~0.09, and does red dip under green?"
  exit 0
fi

sub(){ local r="$1"; shift; env "$@" ROUNDS=50 CALIB_ON_ALL=0 WAIT=0 ./submit_experiment.sh 17 "$r"; }
for R in $SEEDS; do
  sub $R ATTACK=autopilot AUTOP_SCOPE=block  AUTOP_HONEST_UNTIL=10 AUTOP_MARGIN0=0.06 AUTOP_MAX_BATCHES=250 \
      FAMILY=hc_block       SWEEP_VAR=none NOTE="honestcal block 3seed"
  sub $R ATTACK=autopilot AUTOP_SCOPE=block2 AUTOP_HONEST_UNTIL=10 AUTOP_MARGIN0=0.06 AUTOP_MAX_BATCHES=250 \
      FAMILY=hc_block2      SWEEP_VAR=none NOTE="honestcal block2 3seed"
  sub $R ATTACK=autopilot AUTOP_SCOPE=full   AUTOP_HONEST_UNTIL=10 AUTOP_MARGIN0=0.06 AUTOP_MAX_BATCHES=250 \
      FAMILY=hc_full        SWEEP_VAR=none NOTE="honestcal full 3seed"
  # A/B control: block WITHOUT honest-cal (honest_until=0) -> old pessimistic estimate
  sub $R ATTACK=autopilot AUTOP_SCOPE=block  AUTOP_HONEST_UNTIL=0  AUTOP_MARGIN0=0.06 AUTOP_MAX_BATCHES=250 \
      FAMILY=hc_block_nocal SWEEP_VAR=none NOTE="block nocal 3seed"
done
echo "submitted $(echo $SEEDS|wc -w) seeds x 4 arms. When done: RES=$RES ./run_honestcal_sweep.sh PLOT"