#!/usr/bin/env bash
# =====================================================================
# run_full_sweep.sh — autopilot free-rider vs FareMark, the focused sweep.
# ONE command to submit every autopilot experiment (CIFAR-100 AND CIFAR-10),
# then (PLOT mode) build every figure. Fire-and-forget (WAIT=0).
#
#   push code first, then from the repo root:
#     ./scripts/run_full_sweep.sh              # submit everything (scout: 1 seed)
#     SEEDS="0 1 2" ./scripts/run_full_sweep.sh   # 3-seed confirmation (error bars)
#     DATASETS="cifar100" ./scripts/run_full_sweep.sh   # one dataset only
#     DRY=1 ./scripts/run_full_sweep.sh        # preview, submit nothing
#   when the jobs finish:
#     RES=/mnt/nfs/home/zu/results ./scripts/run_full_sweep.sh PLOT
#
# FOCUS: autopilot submarine (idx 17). memory_exploit dropped (can't be healthy
# AND below the fair eta). All results judged vs the FAIR threshold `frozen`
# (post-convergence, fixed) — see faremark/thresholds.py — not the swingy
# cumulative one. Read wm_fr_ber (below eta_frozen = evades) and final_acc
# (~72 c100 / ~90 c10 = healthy). Attacker forces honest warmup through the
# calibration window (AUTOP_PROTECT_UNTIL), then coasts.
#
# ---------------------------------------------------------------------
# !!! WIRING CHECK (submit_experiment.sh must forward these ENV -> CLI flags):
#     AUTOP_PROTECT_UNTIL -> --autop_protect_until   AUTOP_MARGIN0 -> --autop_margin0
#     DATASET -> --dataset   MODEL -> --model
#   All of the above are now wired in submit_experiment.sh. (idx 17 defaults to
#   10 clients, so cifar10 needs only DATASET+MODEL; num_clients is left default.)
#   Run DRY=1 first and eyeball one CIFAR-10 command; confirm the first run's log
#   (run_experiment.py dumps cfg at startup) shows the right dataset + that a tap
#   can exceed 200 batches when AUTOP_MAX_BATCHES>200.
# =====================================================================
set -uo pipefail
SEEDS="${SEEDS:-0}"                       # scout at 1 seed; re-run winners at "0 1 2"
DATASETS="${DATASETS:-cifar100 cifar10}"
DRY="${DRY:-0}"
RES="${RES:-/mnt/nfs/home/zu/results}"

# ------------------------------- PLOT MODE -------------------------------
if [ "${1:-}" = "PLOT" ]; then
  OUT="${OUT:-figs}"; mkdir -p "$OUT"
  ALL="$RES/*/result.json"
  PA="python scripts/plot_adaptive.py"; PT="python scripts/plot_thresholds.py"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }
  for DS in $DATASETS; do
    AP="ap_$DS"; SCOPE="apscope_$DS"
    # 1) GO/NO-GO: does it evade the FAIR threshold, or only the swingy one?
    run $PT evade_bars --in "'$ALL'" --family $AP $SCOPE --out "$OUT/evade_bars_$DS"
    # 2) MECHANISM: watermark decay (coast) + re-embed cost (tap) vs round
    run $PT decay   --in "'$ALL'" --family $AP --out "$OUT/decay_$DS"
    # 3) SAWTOOTH: fr_ber vs benign vs all eta lines, warmup/tap marked, %evade
    run $PT overlay --in "'$ALL'" --family $AP --out "$OUT/sawtooth_$DS"
    # 4) WORTH: effort + BER-vs-eta + accuracy, stacked, mean+/-std over seeds
    run $PT worth   --in "'$ALL'" --family $AP $SCOPE --out "$OUT/worth_$DS"
    # 5) WEAK-POINT MAP under each threshold (frozen = fair headline first)
    for V in frozen converged cumulative; do
      run python scripts/plot_frontier.py --in "'$ALL'" \
          --family $AP $SCOPE --eta $V --out "$OUT/weakpoint_${DS}_$V"
    done
    # 6) SCOPE question (is head-only / "output layer cheap to forge" enough?)
    run $PA sweep --in "'$ALL'" --family $SCOPE --sweep_var autop_scope \
        --metric wm_fr_ber --out "$OUT/scope_frber_$DS"
    run $PA sweep --in "'$ALL'" --family $SCOPE --sweep_var autop_scope \
        --metric effort_ratio_samples --out "$OUT/scope_effort_$DS"
    # 7) EFFORT-vs-STRENGTH and the two secondary knobs
    run $PA sweep --in "'$ALL'" --family $AP --sweep_var autop_max_batches \
        --metric wm_fr_ber --out "$OUT/maxtap_frber_$DS"
    run $PA sweep --in "'$ALL'" --family $AP --sweep_var autop_max_batches \
        --metric effort_ratio_samples --out "$OUT/maxtap_effort_$DS"
    run $PA sweep --in "'$ALL'" --family $AP --sweep_var autop_protect_until \
        --metric effort_ratio_samples --out "$OUT/protect_effort_$DS"
    run $PA sweep --in "'$ALL'" --family $AP --sweep_var autop_margin0 \
        --metric wm_fr_ber --out "$OUT/margin_frber_$DS"
    run $PA duty --in "'$RES/*${AP}*rep0*/result.json'" --out "$OUT/duty_$DS"
  done
  # coast-mode comparison (validates transplant as the coast) — cifar100 anchor
  run $PA sweep --in "'$ALL'" --family S_coast --sweep_var sub_coast_mode --metric wm_fr_ber  --out "$OUT/coast_frber"
  run $PA sweep --in "'$ALL'" --family S_coast --sweep_var sub_coast_mode --metric final_acc --out "$OUT/coast_acc"
  echo; echo "Figures in $OUT/. Read evade_bars_*.png FIRST (go/no-go), then decay_*.png."
  exit 0
fi

# ------------------------------- RUN MODE --------------------------------
# sub(): forward env->flags to submit. All autopilot runs FORCE honest warmup
# through the calibration window (protect), then coast; CALIB_ON_ALL=0 keeps eta
# on the known-honest pool.
sub(){ local cfg="$1" rep="$2"; shift 2
  if [ "$DRY" = 1 ]; then echo "[DRY] $* ROUNDS=50 CALIB_ON_ALL=0 WAIT=0 ./submit_experiment.sh $cfg $rep"
  else env "$@" ROUNDS=50 CALIB_ON_ALL=0 WAIT=0 ./submit_experiment.sh "$cfg" "$rep"; fi; }

for DS in $DATASETS; do
  # idx 17 is cifar100/resnet18/10-clients by default; override for cifar10.
  if [ "$DS" = "cifar10" ]; then DSENV="DATASET=cifar10 MODEL=resnet18"
  else                           DSENV="DATASET=cifar100 MODEL=resnet18"; fi
  AP="ap_$DS"; SCOPE="apscope_$DS"
  echo "############################## DATASET=$DS ##############################"

  echo "###### A1 — TAP STRENGTH (the effort frontier): full-model, forced warmup ######"
  for MB in 60 120 250 400; do for R in $SEEDS; do
    sub 17 $R $DSENV ATTACK=autopilot AUTOP_SCOPE=full AUTOP_PROTECT_UNTIL=8 \
        AUTOP_MAX_BATCHES=$MB FAMILY=$AP SWEEP_VAR=autop_max_batches \
        NOTE="autopilot $DS full maxtap=$MB"
  done; done

  echo "###### A2 — PROTECT WINDOW (forced-honest calibration rounds vs cheapness) ######"
  for PU in 4 8 12; do for R in $SEEDS; do
    sub 17 $R $DSENV ATTACK=autopilot AUTOP_SCOPE=full AUTOP_MAX_BATCHES=250 \
        AUTOP_PROTECT_UNTIL=$PU FAMILY=$AP SWEEP_VAR=autop_protect_until \
        NOTE="autopilot $DS protect=$PU"
  done; done

  echo "###### A3 — SAFETY DELTA (margin below estimated eta: safe vs cheap) ######"
  for MG in 0.04 0.08 0.12; do for R in $SEEDS; do
    sub 17 $R $DSENV ATTACK=autopilot AUTOP_SCOPE=full AUTOP_MAX_BATCHES=250 \
        AUTOP_PROTECT_UNTIL=8 AUTOP_MARGIN0=$MG FAMILY=$AP SWEEP_VAR=autop_margin0 \
        NOTE="autopilot $DS margin=$MG"
  done; done

  echo "###### A4 — TRAINING SCOPE (is head-only / 'output layer cheap to forge' enough?) ######"
  for SC in head block full; do for R in $SEEDS; do
    sub 17 $R $DSENV ATTACK=autopilot AUTOP_SCOPE=$SC AUTOP_MAX_BATCHES=250 \
        AUTOP_PROTECT_UNTIL=8 FAMILY=$SCOPE SWEEP_VAR=autop_scope \
        NOTE="autopilot $DS scope=$SC"
  done; done
done

echo "###### C — SUBMARINE coast type (cifar100 anchor: validates transplant coast) ######"
for CM in replay blend transplant noise global; do for R in $SEEDS; do
  sub 14 $R ATTACK=submarine SUB_WARMUP=8 SUB_COAST_MODE=$CM \
      FAMILY=S_coast SWEEP_VAR=sub_coast_mode NOTE="coast=$CM"
done; done

echo; echo "submitted. Check: runai list jobs"
echo "When done:  RES=$RES ./scripts/run_full_sweep.sh PLOT"
echo "NOTE: this is a SCOUT at SEEDS='$SEEDS'. Re-run winners with SEEDS='0 1 2' for error bars."