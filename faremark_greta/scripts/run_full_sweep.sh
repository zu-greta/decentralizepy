#!/usr/bin/env bash
# =====================================================================
# run_full_sweep.sh — ONE command: run every attack's weak-point sweep,
# then (in PLOT mode) generate all figures. Fire-and-forget (WAIT=0).
#
#   push code first, then from the repo root:
#     ./scripts/run_full_sweep.sh          # submit everything
#     SEEDS="0 1" ./scripts/run_full_sweep.sh
#     DRY=1 ./scripts/run_full_sweep.sh    # preview, submit nothing
#
#   when the jobs finish:
#     RES=/mnt/nfs/home/zu/results ./scripts/run_full_sweep.sh PLOT
#
# Attacks & questions:
#   REEMBED    (idx16) R_frontier   — scope x steps effort-vs-evasion frontier
#   AUTOPILOT  (idx17) autopilot    — fully self-tuning attack
#   SUBMARINE  (idx14) S_samples/S_warmup/S_coast — tap size / warmup / coast type
#   MEMORY     (idx15) M_warmup     — warmup vs fr_ber vs poisoning
# Read wm_fr_ber (below eta = evades) and final_acc (~72 healthy, low = poisoned).
# Priority order: reembed + autopilot finish first.
# =====================================================================
set -uo pipefail
SEEDS="${SEEDS:-0}"
DRY="${DRY:-0}"
RES="${RES:-/mnt/nfs/home/zu/results}"

if [ "${1:-}" = "PLOT" ]; then
  OUT="${OUT:-figs}"; mkdir -p "$OUT"
  ALL="$RES/*/result.json"; PA="python scripts/plot_adaptive.py"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }
  run python scripts/plot_frontier.py --in "'$ALL'" \
      --family R_frontier autopilot autopilot_scope S_samples S_warmup M_warmup S_coast --out "$OUT/weakpoint_all"
  run $PA sweep --in "'$ALL'" --family R_frontier --sweep_var reembed_effort --metric wm_fr_ber --out "$OUT/reembed_frber"
  run $PA sweep --in "'$ALL'" --family R_frontier --sweep_var reembed_effort --metric final_acc --out "$OUT/reembed_acc"
  run $PA sweep --in "'$ALL'" --family R_frontier --sweep_var reembed_effort --metric effort_ratio_samples --out "$OUT/reembed_effort"
  run $PA duty  --in "'$RES/*autopilot*rep0*/result.json'" --out "$OUT/autopilot_duty"
  run $PA sweep --in "'$ALL'" --family autopilot_scope --sweep_var autop_scope --metric wm_fr_ber --out "$OUT/autopilot_scope_frber"
  run $PA sweep --in "'$ALL'" --family autopilot_scope --sweep_var autop_scope --metric effort_ratio_samples --out "$OUT/autopilot_scope_effort"
  run $PA sweep --in "'$ALL'" --family M_warmup --sweep_var warmup_rounds --metric wm_fr_ber --out "$OUT/memory_frber"
  run $PA sweep --in "'$ALL'" --family M_warmup --sweep_var warmup_rounds --metric final_acc --out "$OUT/memory_acc"
  run $PA sweep --in "'$ALL'" --family S_samples --sweep_var sub_max_burst_batches --metric wm_fr_ber --out "$OUT/sub_taps_frber"
  run $PA sweep --in "'$ALL'" --family S_warmup  --sweep_var sub_warmup --metric wm_fr_ber --out "$OUT/sub_warmup_frber"
  run $PA sweep --in "'$ALL'" --family S_coast   --sweep_var sub_coast_mode --metric wm_fr_ber --out "$OUT/sub_coast_frber"
  run $PA sweep --in "'$ALL'" --family S_coast   --sweep_var sub_coast_mode --metric final_acc --out "$OUT/sub_coast_acc"
  echo; echo "Figures in $OUT/. Upload them (and any result.json) for the deck."
  exit 0
fi

sub(){ local cfg="$1" rep="$2"; shift 2
  if [ "$DRY" = 1 ]; then echo "[DRY] $* ROUNDS=50 CALIB_ON_ALL=0 WAIT=0 ./submit_experiment.sh $cfg $rep"
  else env "$@" ROUNDS=50 CALIB_ON_ALL=0 WAIT=0 ./submit_experiment.sh "$cfg" "$rep"; fi; }

echo "###### P1 — REEMBED frontier (idx16): scope x steps ######"
for SC in head block full; do for ST in 10 40 100; do for R in $SEEDS; do
  sub 16 $R ATTACK=reembed REEMBED_SCOPE=$SC REEMBED_STEPS=$ST \
      FAMILY=R_frontier SWEEP_VAR=reembed_effort NOTE="reembed $SC $ST"
done; done; done

echo "###### P2 — AUTOPILOT (idx17): the self-tuning attack ######"
for R in $SEEDS; do
  sub 17 $R ATTACK=autopilot FAMILY=autopilot SWEEP_VAR=none NOTE="autopilot self-tuning"
done
for MB in 120 250; do for R in $SEEDS; do
  sub 17 $R ATTACK=autopilot AUTOP_MAX_BATCHES=$MB \
      FAMILY=autopilot SWEEP_VAR=autop_max_batches NOTE="autopilot maxtap=$MB"
done; done

echo "###### P2b — AUTOPILOT TRAINING SCOPE (idx17): is full-model/full-shard worth it? ######"
# same self-tuning controller, but re-train only head / last block / whole model.
# head/block freeze the backbone (skip its backward) -> far cheaper per batch.
# The graph: effort vs fr_ber by scope -> does a cheap head-only re-embed still evade?
for SC in head block full; do for R in $SEEDS; do
  sub 17 $R ATTACK=autopilot AUTOP_SCOPE=$SC \
      FAMILY=autopilot_scope SWEEP_VAR=autop_scope NOTE="autopilot scope=$SC"
done; done

echo "###### P3 — MEMORY warmup (idx15) ######"
for W in 2 5 8 12; do for R in $SEEDS; do
  sub 15 $R ATTACK=memory_exploit WARMUP_ROUNDS=$W \
      FAMILY=M_warmup SWEEP_VAR=warmup_rounds NOTE="memory warmup=$W"
done; done

echo "###### P4 — SUBMARINE tap-size (idx14): confirm bb=150 ######"
for BB in 20 60 150; do for R in $SEEDS; do
  sub 14 $R ATTACK=submarine SUB_WARMUP=8 SUB_COAST_MODE=blend SUB_MAX_BURST_BATCHES=$BB \
      FAMILY=S_samples SWEEP_VAR=sub_max_burst_batches NOTE="tap=$BB"
done; done

echo "###### P5 — SUBMARINE warmup (idx14) ######"
for W in 3 8 12; do for R in $SEEDS; do
  sub 14 $R ATTACK=submarine SUB_WARMUP=$W SUB_COAST_MODE=blend \
      FAMILY=S_warmup SWEEP_VAR=sub_warmup NOTE="submarine warmup=$W"
done; done

echo "###### P6 — SUBMARINE coast type (idx14) ######"
for CM in replay blend transplant noise global; do for R in $SEEDS; do
  sub 14 $R ATTACK=submarine SUB_WARMUP=8 SUB_COAST_MODE=$CM \
      FAMILY=S_coast SWEEP_VAR=sub_coast_mode NOTE="coast=$CM"
done; done

echo; echo "submitted. Check: runai list jobs"
echo "When done:  RES=$RES ./scripts/run_full_sweep.sh PLOT"