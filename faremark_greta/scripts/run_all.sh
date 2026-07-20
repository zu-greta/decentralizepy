#!/usr/bin/env bash
# run_all.sh -- CIFAR-10 reproduction + threshold recalibration.
#
# GOAL of this file:
#   1. honest-only CIFAR-10 (watermarked, no free-riders) -> should reproduce the
#      paper's clean picture: tight honest BER, low FPR. Run it with 10 SEEDS to
#      calibrate eta and check whether more seeds move the threshold.
#   2. the "+5" (reduced-data) attacker on CIFAR-10 -> does the same evasion hold
#      on the easy dataset the paper actually used?
#
# CONFIG:
#   CFG=11 = wm_resnet18_cifar10 (ResNet-18, CIFAR-10, watermark=True, all-honest
#            base). The +5 attacker is the SAME config with ATTACK=reduced and
#            FREE_RIDER_IDS set, so honest vs attack differ only in the attack.
#
# NOTE ON CIFAR-10 BER GRANULARITY: with n=10 classes the code uses m=2 bits
#   (m = n//10 -> 2), so a single client's BER is quantized to {0, 0.5, 1.0}; the
#   per-round mean-over-clients smooths it. Also l=5, so the random-key same-sign
#   artifact is ~2^(1-5)=6% (small but nonzero) -- unlike CIFAR-100 (~0.2%). Read
#   the honest floor with that in mind.
#
# FLOW:
#   SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_all.sh honest    # 10-seed honest
#   ./run_all.sh calibrate                              # -> $ETA_FILE (+ prints eta +/- std)
#   SEEDS="0 1 2" POS=1,7 ./run_all.sh reduced          # +5 attacker (reads $ETA_FILE)
#   ./run_all.sh PLOTALL
set -uo pipefail
CFG="${CFG:-11}"                                   # 11 = wm_resnet18_cifar10
RES="${RES:-/mnt/nfs/home/zu/results}"
SEEDS="${SEEDS:-0 1 2}"
POS="${POS:-1,7}"                                  # CIFAR-10 trigger CLASS IDs that free-ride
ETA_FILE="${ETA_FILE:-$RES/eta_calibrated_cifar10.json}"
PL="python scripts/plots.py"
TH="python threshold.py"

# ---- eta source ----
# Default: read the freshly calibrated CIFAR-10 eta from $ETA_FILE (written by
# `calibrate`). If your file-fetch is flaky, pass USE_FIXED_ETA=1 FIXED_ETA=<val>
# using the number `calibrate` printed. (Do NOT reuse the CIFAR-100 0.06397 here.)
FIXED_ETA="${FIXED_ETA:-}"
USE_FIXED_ETA="${USE_FIXED_ETA:-}"

# ---- warmup schedule for the reduced attacker ----
COMMON_E="ROUNDS=50 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4"

read_eta(){ python - "$ETA_FILE" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1]))["eta"])
except Exception: print("")
PY
}
get_eta(){
  if [ -n "$USE_FIXED_ETA" ]; then
    [ -z "$FIXED_ETA" ] && { echo ""; return; }
    echo "$FIXED_ETA"
  else
    read_eta
  fi
}

# ============================ CIFAR-10 EXPERIMENTS ============================
honest(){
  echo "== CIFAR-10 HONEST (all-honest, watermarked, seeds: $SEEDS)"
  for s in $SEEDS; do
    env ROUNDS=50 ATTACK=none FAMILY="honest_c10_iid" \
        NOTE="cifar10 all honest" WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

calibrate(){
  echo "== CALIBRATE CIFAR-10 eta from honest-only runs -> $ETA_FILE"
  $TH calibrate --in "$RES/*/result.json" \
      --honest-family honest_c10_iid --tail 20 --out "$ETA_FILE"
  echo "   (compare the printed eta +/- std at 10 seeds vs 3 seeds, and vs the"
  echo "    old CIFAR-100 0.06397 -- they should differ; CIFAR-10 is tighter.)"
}

reduced(){
  local eta; eta="$(get_eta)"
  [ -z "$eta" ] && { echo "!! no eta. Run 'calibrate' first, or pass USE_FIXED_ETA=1 FIXED_ETA=<val>"; return 1; }
  echo "== CIFAR-10 REDUCED (+5/common, honest path), pos=$POS -> reduced_c10_iid_c${POS//,/}, eta=$eta, seeds: $SEEDS"
  for s in $SEEDS; do
    env $COMMON_E ATTACK=reduced FREE_RIDER_IDS=$POS \
        AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
        FAMILY="reduced_c10_iid_c${POS//,/}" SWEEP_LEVEL=5 \
        NOTE="cifar10 +5/cls honest-path pos=$POS eta=$eta" \
        WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

# ======================= CIFAR-100 EXPERIMENTS (commented) ====================
# Re-enable by uncommenting the body AND the matching dispatch line below.
#
# reduced_c100(){            # +5 attacker on CIFAR-100 (CFG 14)
#   local eta; eta="$(get_eta)"; [ -z "$eta" ] && { echo "!! no eta"; return 1; }
#   for s in $SEEDS; do
#     env $COMMON_E ATTACK=reduced FREE_RIDER_IDS=$POS \
#         AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
#         FAMILY="reduced_iid_c${POS//,/}" SWEEP_LEVEL=5 \
#         NOTE="cifar100 +5/cls pos=$POS eta=$eta" \
#         WAIT=0 ./submit_experiment.sh 14 "$s"
#   done; }
#
# reduced_majority(){        # 9 reduced FR + 1 honest anchor (cid 8), CIFAR-100
#   local eta; eta="$(get_eta)"; [ -z "$eta" ] && { echo "!! no eta"; return 1; }
#   local FR="0,1,2,3,4,5,6,7,9"
#   for s in $SEEDS; do
#     env $COMMON_E ATTACK=reduced FREE_RIDER_IDS=$FR \
#         AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
#         FAMILY="reduced_iid_majority" SWEEP_LEVEL=5 \
#         NOTE="cifar100 9 reduced +5/cls, honest anchor cid8, eta=$eta" \
#         WAIT=0 ./submit_experiment.sh 14 "$s"
#   done; }
#
# tap_oracle(){              # honest-path tap/coast with the TRUE eta as oracle, CIFAR-100
#   local eta; eta="$(get_eta)"; [ -z "$eta" ] && { echo "!! no eta"; return 1; }
#   for s in $SEEDS; do
#     env $COMMON_E ATTACK=tap_oracle FREE_RIDER_IDS=$POS \
#         AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
#         FAMILY="taporacle_iid_c${POS//,/}" SWEEP_LEVEL=5 \
#         NOTE="cifar100 oracle tap/coast pos=$POS eta=$eta" \
#         WAIT=0 ./submit_experiment.sh 14 "$s"
#   done; }

# ============================== PLOTS ==============================
plotall(){
  local ALL="$RES/*/result.json"
  local OUT="${OUT:-$RES/figs}"; mkdir -p "$OUT"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }

  run $PL sanity           --in "'$ALL'" --out "$OUT"
  run $PL class_difficulty --in "'$ALL'" --family honest_c10_iid --out "$OUT"
  run $PL thresholds       --in "'$ALL'" --family honest_c10_iid --out "$OUT"

  # every non-honest family -> timeline (BER lines vs eta, with the per-class
  # honest-floor overlay pulled from the honest_c10_iid runs) + fidelity
  FAMS="${FAMS:-$(python -c "import json,glob;fs=set(json.load(open(f)).get('manifest',{}).get('family') for f in glob.glob('$RES/*/result.json'));print(' '.join(sorted(x for x in fs if x and not x.startswith('honest'))))" 2>/dev/null)}"
  for fam in $FAMS; do
    run $PL timeline --in "'$ALL'" --family $fam \
        --honest_in "'$ALL'" --honest_family honest_c10_iid \
        --out "$OUT/timeline_${fam}"
    run $PL fidelity --in "'$ALL'" --family $fam --out "$OUT"
  done
  echo "PLOTALL done -> $OUT"
}

# ============================= DISPATCH =============================
case "${1:-}" in
  honest)     honest ;;
  calibrate)  calibrate ;;
  reduced)    reduced ;;
  # reduced_c100)     reduced_c100 ;;       # CIFAR-100, uncomment to enable
  # reduced_majority) reduced_majority ;;   # CIFAR-100
  # tap_oracle)       tap_oracle ;;         # CIFAR-100
  PLOTALL)    plotall ;;
  *) echo "usage: ./run_all.sh [honest|calibrate|reduced|PLOTALL]
  vars: CFG=$CFG RES=$RES SEEDS='$SEEDS' POS=$POS ETA_FILE=$ETA_FILE
  order (CIFAR-10):
    SEEDS='0 1 2 3 4 5 6 7 8 9' ./run_all.sh honest      # 10-seed honest
    ./run_all.sh calibrate                                # -> eta (prints eta +/- std)
    SEEDS='0 1 2' POS=1,7 ./run_all.sh reduced            # +5 attacker
    ./run_all.sh PLOTALL
  eta override if file-fetch fails:  USE_FIXED_ETA=1 FIXED_ETA=<calibrated value>"; exit 1 ;;
esac