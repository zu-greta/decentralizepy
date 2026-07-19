#!/usr/bin/env bash
# run_all.sh -- focused & cleaned.  CIFAR-100, IID, paper-faithful (trigger class INCLUDED).
#
# Difficulty is already CONFIRMED (softmax peakiness, not class accuracy) from the
# honest runs via class_difficulty_probe.py -- no new "difficulty" runs needed.
#
# Targets:
#   honest            all-honest seeds -> (a) eta calibration set, (b) baseline.
#   calibrate         freeze eta = avg over seeds of (mu_s + 3*sigma_s), honest-only.
#   reduced           the "+N" attacker: honest until round W, then trains ONLY on
#                     (all trigger imgs + N per common class) every round. No tapping.
#   reduced_majority  9 reduced free-riders + 1 honest anchor at an EASY class (cid 8).
#   PLOTALL           sanity + class_difficulty/thresholds (honest) + timeline/fidelity (attacks).
#
# tap_oracle (honest-path tap/coast with the TRUE eta as an oracle) is written below
# but COMMENTED OUT of the dispatch -- enable it after the reduced results look right.
#
# THRESHOLD: your eta_calibrated.json fetch is flaky, so pass the frozen constant on
# the CLI. Run every attack with USE_FIXED_ETA=1 to use FIXED_ETA below (=0.06397,
# the eta calibrated on the ORIGINAL 10 honest seeds -- the one you want to test against).
#
# FLOW:
#   ./run_all.sh honest                          # (only if you need fresh honest runs)
#   ./run_all.sh calibrate                       # (optional; we pass eta on CLI anyway)
#   SEEDS='0 1 2' POS=1,7 USE_FIXED_ETA=1 ./run_all.sh reduced           # easy classes
#   SEEDS='0 1 2' POS=3,6 USE_FIXED_ETA=1 ./run_all.sh reduced           # hard classes
#   SEEDS='0 1 2'         USE_FIXED_ETA=1 ./run_all.sh reduced_majority  # 9 FR + 1 honest
#   ./run_all.sh PLOTALL
set -uo pipefail
CFG="${CFG:-14}"; RES="${RES:-/mnt/nfs/home/zu/results}"
SEEDS="${SEEDS:-0 1 2}"
POS="${POS:-3,6}"                        # trigger CLASS IDs that free-ride
ETA_FILE="${ETA_FILE:-$RES/eta_calibrated.json}"
PL="python scripts/plots.py"
TH="python threshold.py"

# ---- frozen eta (calibrated on the original 10 honest seeds) ----
FIXED_ETA=0.06397
USE_FIXED_ETA="${USE_FIXED_ETA:-}"       # set =1 to use FIXED_ETA (CLI workaround for file-fetch)

# ---- only the warmup schedule matters for the reduced / tap attackers ----
#   W = AUTOP_HONEST_UNTIL (defect at round W), K = AUTOP_CALIB_ROUNDS (tags calib window [W-K, W-1]).
COMMON_E="ROUNDS=50 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4"

read_eta(){ python - "$ETA_FILE" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1]))["eta"])
except Exception: print("")
PY
}
get_eta(){ if [ -n "$USE_FIXED_ETA" ]; then echo "$FIXED_ETA"; else read_eta; fi; }

# ============================ EXPERIMENTS ============================
honest(){
  echo "== HONEST (all-honest, seeds: $SEEDS)"
  for s in $SEEDS; do
    env ROUNDS=50 ATTACK=none FAMILY="honest_iid" \
        NOTE="all honest iid" WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

calibrate(){
  echo "== CALIBRATE eta from honest-only runs -> $ETA_FILE"
  $TH calibrate --in "$RES/*/result.json" --honest-family honest_iid --tail 20 --out "$ETA_FILE"
}

# --- the "+N" attacker: honest warmup, then honest training on LESS data ---
reduced(){
  local eta; eta="$(get_eta)"; [ -z "$eta" ] && { echo "!! no eta (set USE_FIXED_ETA=1)"; return 1; }
  echo "== REDUCED (+5/common, honest path), pos=$POS -> reduced_iid_c${POS//,/}, eta=$eta, seeds: $SEEDS"
  for s in $SEEDS; do
    env $COMMON_E ATTACK=reduced FREE_RIDER_IDS=$POS \
        AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
        FAMILY="reduced_iid_c${POS//,/}" SWEEP_LEVEL=5 \
        NOTE="+5/cls honest-path pos=$POS eta=$eta" \
        WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

# --- majority: 9 reduced free-riders + 1 honest anchor at an EASY class (cid 8 -> class 8) ---
reduced_majority(){
  local eta; eta="$(get_eta)"; [ -z "$eta" ] && { echo "!! no eta (set USE_FIXED_ETA=1)"; return 1; }
  local FR="0,1,2,3,4,5,6,7,9"          # everyone EXCEPT cid 8 (honest anchor, easy class)
  echo "== REDUCED MAJORITY (9 FR + honest cid8), eta=$eta, seeds: $SEEDS"
  for s in $SEEDS; do
    env $COMMON_E ATTACK=reduced FREE_RIDER_IDS=$FR \
        AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
        FAMILY="reduced_iid_majority" SWEEP_LEVEL=5 \
        NOTE="9 reduced +5/cls, honest anchor cid8, eta=$eta" \
        WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

# --- (LATER) honest-path tap/coast with the TRUE eta as an ORACLE. Enable after reduced. ---
# tap_oracle(){
#   local eta; eta="$(get_eta)"; [ -z "$eta" ] && { echo "!! no eta (set USE_FIXED_ETA=1)"; return 1; }
#   echo "== TAP_ORACLE (tap/coast, oracle eta), pos=$POS -> taporacle_iid_c${POS//,/}, eta=$eta"
#   for s in $SEEDS; do
#     env $COMMON_E ATTACK=tap_oracle FREE_RIDER_IDS=$POS \
#         AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
#         FAMILY="taporacle_iid_c${POS//,/}" SWEEP_LEVEL=5 \
#         NOTE="oracle tap/coast pos=$POS eta=$eta" \
#         WAIT=0 ./submit_experiment.sh "$CFG" "$s"
#   done
# }

# ============================== PLOTS ==============================
plotall(){
  local ALL="$RES/*/result.json"
  local OUT="${OUT:-$RES/figs}"; mkdir -p "$OUT"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }

  # honest: difficulty figure + the frozen-eta / honest-FPR figure
  run $PL sanity           --in "'$ALL'" --out "$OUT"
  run $PL class_difficulty --in "'$ALL'" --family honest_iid --out "$OUT"
  run $PL thresholds       --in "'$ALL'" --family honest_iid --out "$OUT"

  # every non-honest family present -> timeline (BER lines vs eta) + fidelity (effort/BER)
  FAMS="${FAMS:-$(python -c "import json,glob;fs=set(json.load(open(f)).get('manifest',{}).get('family') for f in glob.glob('$RES/*/result.json'));print(' '.join(sorted(x for x in fs if x and not x.startswith('honest'))))" 2>/dev/null)}"
  for fam in $FAMS; do
    run $PL timeline  --in "'$ALL'" --family $fam --out "$OUT/timeline_${fam}"   # <- the BER lines you want
    run $PL fidelity  --in "'$ALL'" --family $fam --out "$OUT"
  done
  echo "PLOTALL done -> $OUT"
}

# ============================= DISPATCH =============================
case "${1:-}" in
  honest)           honest ;;
  calibrate)        calibrate ;;
  reduced)          reduced ;;
  reduced_majority) reduced_majority ;;
  # tap_oracle)     tap_oracle ;;       # enable after the reduced results look right
  PLOTALL)          plotall ;;
  *) echo "usage: ./run_all.sh [honest|calibrate|reduced|reduced_majority|PLOTALL]
  vars: CFG=$CFG RES=$RES SEEDS='$SEEDS' POS=$POS
  eta:  USE_FIXED_ETA=1 uses FIXED_ETA=$FIXED_ETA (CLI workaround for file-fetch)
  examples:
    SEEDS='0 1 2' POS=1,7 USE_FIXED_ETA=1 ./run_all.sh reduced          # easy classes
    SEEDS='0 1 2' POS=3,6 USE_FIXED_ETA=1 ./run_all.sh reduced          # hard classes
    SEEDS='0 1 2'         USE_FIXED_ETA=1 ./run_all.sh reduced_majority # 9 FR + 1 honest
    ./run_all.sh PLOTALL"; exit 1 ;;
esac