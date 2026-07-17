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
POS="${POS:-3,6}"                 # free-rider trigger CLASS IDs (hard cls 3 & 6)
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
  echo "== TAP_EVERY (+5/common, full scope), class ids=$POS -> family tap_every_iid_c${POS//,/}, eta=$eta, seeds: $SEEDS"
  for s in $SEEDS; do
    env $COMMON_E ATTACK=submarine FREE_RIDER_IDS=$POS \
        AUTOP_SCOPE=full AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
        FAMILY="tap_every_iid_c${POS//,/}" SWEEP_LEVEL=5 NOTE="tap every +5/cls full pos=$POS eta=$eta" \
        WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

tap_stay(){
  local eta="$(read_eta)"; [ -z "$eta" ] && { echo "!! no $ETA_FILE -- run calibrate first"; return 1; }
  echo "== TAP_STAY (coast), class ids=$POS -> family tap_stay_iid_c${POS//,/}, eta=$eta, seeds: $SEEDS"
  for s in $SEEDS; do
    env $COMMON_E ATTACK=submarine FREE_RIDER_IDS=$POS AUTOP_STAY_MIN=1 \
        AUTOP_SCOPE=full AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
        FAMILY="tap_stay_iid_c${POS//,/}" SWEEP_LEVEL=5 NOTE="tap-to-stay full pos=$POS eta=$eta" \
        WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

calibrate(){
  echo "== CALIBRATE canonical eta from honest-only runs -> $ETA_FILE"
  $TH calibrate --in "$RES/*/result.json" \
      --honest-family honest_iid --tail 20 --out "$ETA_FILE"
}

# =============================== PLOTALL ===============================
# 1) timelines (per family)   2) class difficulty + dynamics (harder class ids)
# 3) canonical threshold derivation (+ fidelity)
plotall(){
  # MINIMAL plot set -- only what proves the two claims + catches suspicious runs.
  local ALL="$RES/*/result.json"
  local OUT="${OUT:-$RES/figs}"; mkdir -p "$OUT"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }

  # --- 0. sanity FIRST (text): flag flat/zero BER, non-frozen eta, missing loss ---
  run $PL sanity --in "'$ALL'" --out "$OUT"

  # --- 1. CLAIM A: some class ids are harder to embed (all-honest) ---
  run $PL class_difficulty --in "'$ALL'" --family honest_iid --out "$OUT"   # per-class acc/loss vs BER
  run $PL thresholds       --in "'$ALL'" --family honest_iid --out "$OUT"   # the frozen eta + honest FPR

  # --- 2. CLAIM B: free-riding is possible in IID (+5/common and coast) ---
  # auto-discover every tap_* family present (each = one class-id assignment)
  FAMS="${FAMS:-$(python -c "import json,glob;fs=set(json.load(open(f)).get('manifest',{}).get('family') for f in glob.glob('$RES/*/result.json'));print(' '.join(sorted(x for x in fs if x and x.startswith('tap_'))))" 2>/dev/null)}"
  for fam in $FAMS; do
    run $PL timeline      --in "'$ALL'" --family $fam --out "$OUT/timeline_${fam}"  # BER vs eta, taps/coasts
    run $PL fidelity      --in "'$ALL'" --family $fam --out "$OUT"                  # FR vs honest BER + effort
    run $PL class_dynamics --in "'$ALL'" --family $fam --out "$OUT"                 # loss curves (diagnostic)
  done
  echo "PLOTALL (minimal) done -> $OUT"
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