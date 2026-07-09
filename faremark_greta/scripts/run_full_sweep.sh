#!/usr/bin/env bash
# run_full_sweeep.sh — the three experiments from the meeting notes.
#   (A) ORACLE eta: the free-rider is GIVEN the true fair eta (~0.09) — a diagnostic
#       upper bound: "if it knew the line, can it stay under with < honest effort?"
#       Run for the two scopes with the most potential: full and block.
#   (B) DATA-SHARD ablation: how little data does the re-embed need? Sweep
#       autop_common_per_class = 0 (triggers only), 10, 20, 40 (+N random imgs per
#       common class), on full and block. Plot BER vs data-samples.
# All use the honest-schedule + honest-round eta calibration already in the code.
# CALIB_ON_ALL=0 = fair (honest-only) eta.  ORACLE eta value = 0.09 (the known fair eta).
#
#   ./run_full_sweeep.sh          # submit all (SEEDS="0 1 2")
#   RES=/path ./run_full_sweeep.sh PLOT
set -uo pipefail
SEEDS="${SEEDS:-0 1 2}"; RES="${RES:-/mnt/nfs/home/zu/results}"
PT="python scripts/plot_thresholds.py"; SB="python scripts/seedband.py"

if [ "${1:-}" = "PLOT" ]; then
  OUT="${OUT:-figs}"; mkdir -p "$OUT"; ALL="$RES/*/result.json"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }
  # (A) oracle: does knowing eta let it stay under at < honest effort?
  run $PT evade_bars --in "'$ALL'" --family oracle_full oracle_block --out "$OUT/oracle_evade"
  run $SB --in "'$ALL'" --note "'oracle full 3seed'"  --title "'ORACLE eta, full scope'"  --out "$OUT/seedband_oracle_full"
  run $SB --in "'$ALL'" --note "'oracle block 3seed'" --title "'ORACLE eta, block scope'" --out "$OUT/seedband_oracle_block"
  # (B) data-shard ablation: BER (and effort) vs #common images per class
  run $PT knob --in "'$ALL'" --family data_full  --sweep_var autop_common_per_class --out "$OUT/data_full"
  run $PT knob --in "'$ALL'" --family data_block --sweep_var autop_common_per_class --out "$OUT/data_block"
  echo; echo "READ: oracle_evade.png (can it stay under when it KNOWS eta, cheaply?),"
  echo "      data_{full,block}.png (BER vs data used — does it still work with less data?)."
  exit 0
fi

sub(){ local r="$1"; shift; env "$@" ROUNDS=50 CALIB_ON_ALL=0 WAIT=0 ./submit_experiment.sh 17 "$r"; }
ORACLE=0.09   # the known fair (frozen) eta on CIFAR-100/ResNet-18; adjust if your runs differ
for R in $SEEDS; do
  # (A) ORACLE eta — full and block
  sub $R ATTACK=autopilot AUTOP_SCOPE=full  AUTOP_ORACLE_ETA=$ORACLE AUTOP_HONEST_UNTIL=10 AUTOP_MARGIN0=0.06 AUTOP_MAX_BATCHES=250 \
      FAMILY=oracle_full  SWEEP_VAR=none NOTE="oracle full 3seed"
  sub $R ATTACK=autopilot AUTOP_SCOPE=block AUTOP_ORACLE_ETA=$ORACLE AUTOP_HONEST_UNTIL=10 AUTOP_MARGIN0=0.06 AUTOP_MAX_BATCHES=250 \
      FAMILY=oracle_block SWEEP_VAR=none NOTE="oracle block 3seed"
  # (B) DATA-SHARD ablation — sweep common-images-per-class, full and block
  for NC in 0 10 20 40; do
    sub $R ATTACK=autopilot AUTOP_SCOPE=full  AUTOP_COMMON_PER_CLASS=$NC AUTOP_HONEST_UNTIL=10 AUTOP_MAX_BATCHES=250 \
        FAMILY=data_full  SWEEP_VAR=autop_common_per_class NOTE="data full cpc=$NC"
    sub $R ATTACK=autopilot AUTOP_SCOPE=block AUTOP_COMMON_PER_CLASS=$NC AUTOP_HONEST_UNTIL=10 AUTOP_MAX_BATCHES=250 \
        FAMILY=data_block SWEEP_VAR=autop_common_per_class NOTE="data block cpc=$NC"
  done
done
echo "submitted. When done: RES=$RES ./run_meeting_sweep.sh PLOT"
