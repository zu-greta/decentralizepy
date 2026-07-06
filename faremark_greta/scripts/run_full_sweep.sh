#!/usr/bin/env bash
# =====================================================================
# run_full_sweep.sh — one night, every attack, find each weak point.
# All WAIT=0 (queue & walk away). 1 seed by default (SEEDS="0 1" for more).
# ~22 runs @ ~3h each -> ~15h on 4 GPUs. Ordered by priority.
#
#   push code first, then from repo root:  ./scripts/run_full_sweep.sh
#   DRY=1 ./scripts/run_full_sweep.sh   # preview, submit nothing
#
# Answers, per attack:
#   SUBMARINE  S_warmup (Q1 warmup to fall under η) · S_samples (Q2 how much to
#              train) · S_coast (Q3 coast type: replay/blend/transplant/noise/
#              do-nothing) · effort comes out of every run (compute.summary)
#   MEMORY     M_warmup (Q1 warmup to a good point + Q2 effort for evasion)
#   REEMBED    R_frontier (scope × steps = effort-vs-evasion frontier)
# Primary metrics to read: wm_fr_ber (below η = evades) and final_acc
#   (≈72 = healthy, low = poisoned). NOT recall (η-regime dependent).
# =====================================================================
set -uo pipefail
SEEDS="${SEEDS:-0}"
DRY="${DRY:-0}"
sub(){ local cfg="$1" rep="$2"; shift 2
  if [ "$DRY" = 1 ]; then echo "[DRY] $* ROUNDS=50 CALIB_ON_ALL=0 WAIT=0 ./submit_experiment.sh $cfg $rep"
  else env "$@" ROUNDS=50 CALIB_ON_ALL=0 WAIT=0 ./submit_experiment.sh "$cfg" "$rep"; fi; }

echo "###### PRIORITY 1 — REEMBED frontier (the theoretically-motivated attack) ######"
# scope × steps: where does fr_ber fall under η at low effort AND acc stays healthy?
for SC in head block full; do for ST in 10 40 100; do for R in $SEEDS; do
  sub 16 $R ATTACK=reembed REEMBED_SCOPE=$SC REEMBED_STEPS=$ST \
      FAMILY=R_frontier SWEEP_VAR=reembed_effort NOTE="reembed $SC×$ST"
done; done; done

echo "###### PRIORITY 2 — MEMORY warmup (Q1 good point, Q2 effort) ######"
for W in 2 5 8 12; do for R in $SEEDS; do
  sub 15 $R ATTACK=memory_exploit WARMUP_ROUNDS=$W \
      FAMILY=M_warmup SWEEP_VAR=warmup_rounds NOTE="memory warmup=$W"
done; done

echo "###### PRIORITY 3 — SUBMARINE warmup (Q1) ######"
for W in 3 8 12; do for R in $SEEDS; do
  sub 14 $R ATTACK=submarine SUB_WARMUP=$W SUB_COAST_MODE=blend \
      FAMILY=S_warmup SWEEP_VAR=sub_warmup NOTE="submarine warmup=$W"
done; done

echo "###### PRIORITY 4 — SUBMARINE coast type (Q3) at fixed warmup=8 ######"
for CM in replay blend transplant noise global; do for R in $SEEDS; do
  sub 14 $R ATTACK=submarine SUB_WARMUP=8 SUB_COAST_MODE=$CM \
      FAMILY=S_coast SWEEP_VAR=sub_coast_mode NOTE="coast=$CM"
done; done

echo "###### PRIORITY 5 — SUBMARINE train-per-tap (Q2 how many samples) ######"
for BB in 20 150; do for R in $SEEDS; do
  sub 14 $R ATTACK=submarine SUB_WARMUP=8 SUB_COAST_MODE=blend SUB_MAX_BURST_BATCHES=$BB \
      FAMILY=S_samples SWEEP_VAR=sub_max_burst_batches NOTE="burst=$BB"
done; done

echo; echo "submitted. runai list jobs. When done: ./scripts/make_sweep_figs.sh"
