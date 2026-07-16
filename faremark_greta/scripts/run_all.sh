#!/usr/bin/env bash
# run_all.sh -- focused experiments (no more big sweeps).
#
# THREE experiments, all IID, CIFAR-100, paper-faithful:
#   honest    all-honest, multiple seeds. Used to CALIBRATE the one threshold,
#             and reported as its own (free-rider-free) baseline.
#   tap_every free-rider taps EVERY round on +5/common-class at FULL scope.
#   tap_stay  free-rider taps ONLY when needed to stay under the frozen eta
#             (coast otherwise). Uses the fixed coast/estimate logic.
#
# THRESHOLD: one canonical, pre-calibrated constant eta (calibrate_eta.py):
#   mean-over-clients per round -> mu+3*sigma over rounds, honest-only, multi-seed.
#   Frozen to eta_calibrated.json, then passed to every attack run as WM_ETA_FIXED.
#
# FLOW:
#   ./run_all.sh honest        # submit all-honest seeds
#   # ...wait for them to finish...
#   ./run_all.sh calibrate     # -> $RES/eta_calibrated.json
#   ./run_all.sh attacks       # submit tap_every + tap_stay (reads the frozen eta)
#   # ...wait...
#   ./run_all.sh PLOTALL       # timelines + class dynamics + thresholds + fidelity
set -uo pipefail
CFG="${CFG:-14}"; RES="${RES:-/mnt/nfs/home/zu/results}"
SEEDS="${SEEDS:-0 1 2}"
POS="${POS:-3,6}"                 # free-rider trigger positions (hard cls 3 & 6)
ETA_FILE="${ETA_FILE:-$RES/eta_calibrated.json}"
PL="python scripts/plots.py"   # all plotting consolidated
TH="python scripts/threshold.py"  # all threshold code consolidated

# common env for one autopilot run
COMMON_E="ROUNDS=50 AUTOP_WARMUP_MODE=fixed AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
AUTOP_ETA_MODE=tight AUTOP_NUM_CLIENTS_EST=10 AUTOP_MARGIN0=0.06 AUTOP_SAFETY=0.02 AUTOP_MAX_COAST=4"

read_eta(){ python - "$ETA_FILE" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1]))["eta"])
except Exception: print("")
PY
}

# =============================== EXPERIMENTS ===============================
honest(){
  echo "== HONEST (all-honest, seeds: $SEEDS) -> calibration + baseline"
  for s in $SEEDS; do
    env ROUNDS=50 ATTACK=none FAMILY="honest_iid" \
        NOTE="all honest iid" WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

tap_every(){
  local eta="$(read_eta)"; [ -z "$eta" ] && { echo "!! no $ETA_FILE -- run calibrate first"; return 1; }
  echo "== TAP_EVERY (+5/common, full scope, taps every round), eta=$eta, seeds: $SEEDS"
  for s in $SEEDS; do
    env $COMMON_E ATTACK=autopilot FREE_RIDER_IDS=$POS \
        AUTOP_SCOPE=full AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
        FAMILY="tap_every_iid" SWEEP_LEVEL=5 NOTE="tap every +5/cls full eta=$eta" \
        WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

tap_stay(){
  local eta="$(read_eta)"; [ -z "$eta" ] && { echo "!! no $ETA_FILE -- run calibrate first"; return 1; }
  echo "== TAP_STAY (coast + tap only to stay under eta), eta=$eta, seeds: $SEEDS"
  for s in $SEEDS; do
    env $COMMON_E ATTACK=autopilot FREE_RIDER_IDS=$POS AUTOP_STAY_MIN=1 \
        AUTOP_SCOPE=full AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
        FAMILY="tap_stay_iid" SWEEP_LEVEL=5 NOTE="tap-to-stay full eta=$eta" \
        WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

calibrate(){
  echo "== CALIBRATE canonical eta from honest-only runs -> $ETA_FILE"
  $TH calibrate --in "$RES/*/result.json" \
      --honest-family honest_iid --tail 20 --out "$ETA_FILE"
}

# =============================== PLOTALL ===============================
# 1) timelines (per family)   2) class dynamics + positions (hard classes)
# 3) canonical threshold derivation (+ fidelity)
plotall(){
  local ALL="$RES/*/result.json"
  local OUT="${OUT:-$RES/figs}"; mkdir -p "$OUT"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }

  for fam in honest_iid tap_every_iid tap_stay_iid; do
    # 1. timeline (prefix out: the fn appends .png)
    run $PL timeline       --in "'$ALL'" --family $fam --out "$OUT/timeline_${fam}"
    # 2. hard-class evidence (dir out)
    run $PL class_dynamics --in "'$ALL'" --family $fam --out "$OUT"
    run $PL positions      --in "'$ALL'" --family $fam --out "$OUT"
    # 3. canonical threshold derivation + fidelity (dir out)
    run $PL thresholds     --in "'$ALL'" --family $fam --out "$OUT"
    run $PL fidelity       --in "'$ALL'" --family $fam --out "$OUT"
  done
  # honest FPR against the frozen eta (prefix out)
  run $PL honest_fpr --in "'$ALL'" --family honest_iid --out "$OUT/honest_fpr"
  echo "PLOTALL done -> $OUT"
}

# =============================== DISPATCH ===============================
case "${1:-}" in
  honest)     honest ;;
  calibrate)  calibrate ;;
  tap_every)  tap_every ;;
  tap_stay)   tap_stay ;;
  attacks)    tap_every; tap_stay ;;
  all)        honest; echo "-> wait for honest jobs, then: ./run_all.sh calibrate && ./run_all.sh attacks" ;;
  PLOTALL)    plotall ;;
  *) echo "usage: ./run_all.sh [honest|calibrate|tap_every|tap_stay|attacks|PLOTALL]
  typical order:  honest  ->(wait)->  calibrate  ->  attacks  ->(wait)->  PLOTALL
  vars: CFG=$CFG RES=$RES SEEDS='$SEEDS' POS=$POS ETA_FILE=$ETA_FILE"; exit 1 ;;
esac