#!/usr/bin/env bash
# run_full_sweep.sh — the clean experiment design from the meeting spec.
#
# Every run: the free-rider acts EXACTLY like an honest client (full model, full epoch)
# until its honest BER FLATTENS (auto-detected) + AUTOP_HONEST_EXTRA rounds, so both the
# server AND the free-rider get a clean, un-inflated eta estimate. The FR then FREEZES
# that eta and coasts/taps to stay under it.
#
# TWO SCOPES (the only two kept — see attacks_adaptive.py):
#   full   = taps re-train the whole model (like honest, but only until safe). Most
#            effort of the three, but strongest/most-generalizing watermark.
#   block2 = taps re-train only the LAST TWO blocks (less backprop -> less effort than
#            full, still deep enough to embed).
#
# DATA ABLATION per scope (AUTOP_COMMON_PER_CLASS):
#   0   = trigger samples ONLY
#   20  = trigger + 20 random imgs per common class
#   -1  = full shard (like an honest client)
#
# Plus an ORACLE arm (AUTOP_ORACLE_ETA=0.09): the FR is GIVEN the true eta — the
# diagnostic upper bound ("is staying under even POSSIBLE cheaply?").
#
#   ./run_full_sweep.sh            # submit (SEEDS="0 1 2")
#   RES=/path ./run_full_sweep.sh PLOT
set -uo pipefail
SEEDS="${SEEDS:-0 1 2}"; RES="${RES:-/mnt/nfs/home/zu/results}"
PT="python scripts/plot_thresholds.py"; SB="python scripts/seedband.py"; ORACLE=0.09

if [ "${1:-}" = "PLOT" ]; then
  OUT="${OUT:-figs}"; mkdir -p "$OUT"; ALL="$RES/*/result.json"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }

  # ============ OVERVIEW / COMPARISON (all arms together) ============
  # go/no-go: do full / block2 / oracle clear the FAIR eta?  (mean +/- std)
  run $PT evade_bars --in "'$ALL'" --family full_full block2_full oracle_full oracle_block2 --out "$OUT/final_evade"
  # worth: effort + BER-vs-eta + accuracy, stacked, all arms
  run $PT worth      --in "'$ALL'" --family full_full block2_full oracle_full oracle_block2 --out "$OUT/final_worth"
  # compute-meter comparison: which cost metric best captures the scope attack
  run $PT meters     --in "'$ALL'" --family full_full block2_full oracle_full oracle_block2 --out "$OUT/meters"

  # ============ PER-SCOPE: BER vs rounds + taps + effort + threshold refs ============
  # timeline = FR BER + honest BER + fair eta + attacker's estimated/ORACLE eta,
  #            warmup/tap markers, and cumulative effort (fraction of honest) at the bottom.
  for FAM in full_full block2_full oracle_full oracle_block2; do
    run $PT timeline  --in "'$ALL'" --family $FAM --out "$OUT/timeline_$FAM"
    run $PT submarine --in "'$ALL'" --family $FAM --out "$OUT/submarine_$FAM"   # coast/tap detail, cost in samples
  done
  # seed-bands (std shaded + honest BER + estimated eta) — headline scopes
  run $SB --in "'$ALL'" --note "'full fullshard 3seed'"   --title "'full scope (3 seeds)'"   --out "$OUT/seedband_full"
  run $SB --in "'$ALL'" --note "'block2 fullshard 3seed'" --title "'block2 scope (3 seeds)'" --out "$OUT/seedband_block2"
  run $SB --in "'$ALL'" --note "'oracle full 3seed'"      --title "'oracle full (3 seeds)'"  --out "$OUT/seedband_oracle_full"
  run $SB --in "'$ALL'" --note "'oracle block2 3seed'"    --title "'oracle block2 (3 seeds)'" --out "$OUT/seedband_oracle_block2"

  # ============ PER-SCOPE: DATA ABLATION (samples used vs BER & effort) ============
  # knob = BER (top) and effort (bottom) vs autop_common_per_class
  #        (0 = triggers only, 20 = +20/common class, -1 = full shard).
  run $PT knob --in "'$ALL'" --family data_full   --sweep_var autop_common_per_class --out "$OUT/data_full"
  run $PT knob --in "'$ALL'" --family data_block2 --sweep_var autop_common_per_class --out "$OUT/data_block2"

  echo
  echo "READ ORDER:"
  echo "  1) final_evade.png  — does ANY arm (incl. ORACLE) clear the FAIR eta? decides attack-vs-defense."
  echo "  2) timeline_*.png   — per scope: BER vs rounds, fair eta + estimated/oracle eta, taps, effort at bottom."
  echo "  3) data_*.png       — does it still work with less data (triggers only -> full shard)?"
  echo "  4) meters.png / final_worth.png — cross-arm cost & health comparison."
  exit 0
fi

sub(){ local r="$1"; shift; env "$@" ROUNDS=50 CALIB_ON_ALL=0 AUTOP_HONEST_UNTIL=12 AUTOP_HONEST_EXTRA=3 \
        AUTOP_MARGIN0=0.06 AUTOP_MAX_BATCHES=250 WAIT=0 ./submit_experiment.sh 17 "$r"; }
for R in $SEEDS; do
  # headline: full shard, both scopes
  sub $R ATTACK=autopilot AUTOP_SCOPE=full   FAMILY=full_full    SWEEP_VAR=none NOTE="full fullshard 3seed"
  sub $R ATTACK=autopilot AUTOP_SCOPE=block2 FAMILY=block2_full  SWEEP_VAR=none NOTE="block2 fullshard 3seed"
  # ORACLE (given true eta) — is it even possible cheaply?
  sub $R ATTACK=autopilot AUTOP_SCOPE=full   AUTOP_ORACLE_ETA=$ORACLE FAMILY=oracle_full   SWEEP_VAR=none NOTE="oracle full 3seed"
  sub $R ATTACK=autopilot AUTOP_SCOPE=block2 AUTOP_ORACLE_ETA=$ORACLE FAMILY=oracle_block2 SWEEP_VAR=none NOTE="oracle block2 3seed"
  # DATA ABLATION: trigger-only(0), trigger+common(20), full-shard(-1), each scope
  for NC in 0 20 -1; do
    sub $R ATTACK=autopilot AUTOP_SCOPE=full   AUTOP_COMMON_PER_CLASS=$NC FAMILY=data_full   SWEEP_VAR=autop_common_per_class NOTE="data full cpc=$NC"
    sub $R ATTACK=autopilot AUTOP_SCOPE=block2 AUTOP_COMMON_PER_CLASS=$NC FAMILY=data_block2 SWEEP_VAR=autop_common_per_class NOTE="data block2 cpc=$NC"
  done
done
echo "submitted. When done: RES=$RES ./run_final_sweep.sh PLOT"