#!/usr/bin/env bash
# run_all.sh -- balanced keys + bit-count study.  CIFAR-100 (default) or CIFAR-10.
#
# PREREQ (one-line code edit, permanent):
#   wm_client.py build_watermarked_clients:
#     wm.make_key(..., balanced=False)  ->  balanced=True
#   Removes unembeddable (all-same-sign) key rows. Use a FRESH $RES dir so these
#   runs don't mix with old unbalanced results.
#
# Everything is TAGGED by dataset + bit count so runs never collide:
#   honest  family = honest_<DS>_b<BITS>_iid      (BITS omitted -> "bdef" = code default m)
#   reduced family = reduced_<DS>_b<BITS>_iid_c<POS>
#   eta file       = $RES/eta_<DS>_b<BITS>.json
# so a default-m run and a WM_BITS=20 run live side by side and plot separately.
#
# VARS:  DS=c100|c10   BITS=<int or empty>   SEEDS=...   POS=...
set -uo pipefail
DS="${DS:-c100}"                                   # c100 (CFG 14) or c10 (CFG 11)
case "$DS" in
  c100) CFG=14 ;;                                  # submarine_resnet18_cifar100
  c10)  CFG=11 ;;                                  # wm_resnet18_cifar10
  *) echo "DS must be c100 or c10"; exit 1 ;;
esac
CFG="${CFG_OVERRIDE:-$CFG}"
RES="${RES:-/mnt/nfs/home/zu/results}"
SEEDS="${SEEDS:-0 1 2}"
POS="${POS:-1,7}"
BITS="${BITS:-}"                                   # empty = code default m; else e.g. 20 (c100) / 5 (c10)
BT="b${BITS:-def}"                                 # bdef | b20 | b5
TAG="${DS}_${BT}"                                  # c100_bdef | c100_b20 | c10_b5
ETA_FILE="${ETA_FILE:-$RES/eta_${TAG}.json}"
PL="python scripts/plots.py"; TH="python threshold.py"
HL="python scripts/honest_class_lines.py"
FIXED_ETA="${FIXED_ETA:-}"; USE_FIXED_ETA="${USE_FIXED_ETA:-}"
COMMON_E="ROUNDS=50 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4"
BITSENV=""; [ -n "$BITS" ] && BITSENV="WM_BITS=$BITS"

HFAM="honest_${TAG}_iid"

read_eta(){ python - "$ETA_FILE" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1]))["eta"])
except Exception: print("")
PY
}
get_eta(){ if [ -n "$USE_FIXED_ETA" ]; then [ -z "$FIXED_ETA" ] && { echo ""; return; }; echo "$FIXED_ETA"; else read_eta; fi; }

# ============================ EXPERIMENTS ============================
honest(){
  echo "== HONEST $TAG (all-honest, balanced keys, seeds: $SEEDS)"
  for s in $SEEDS; do
    env ROUNDS=50 ATTACK=none $BITSENV FAMILY="$HFAM" \
        NOTE="$DS balanced-keys all honest bits=${BITS:-default}" \
        WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

calibrate(){
  echo "== CALIBRATE $TAG eta from honest-only ($HFAM) -> $ETA_FILE"
  $TH calibrate --in "$RES/*/result.json" --honest-family "$HFAM" --tail 20 --out "$ETA_FILE"
  echo "   check: eta_std_across_seeds should be SMALLER than the 0.027 unbalanced run,"
  echo "          and unembeddable_frac ~0 in the result.json."
}

reduced(){
  local eta; eta="$(get_eta)"
  [ -z "$eta" ] && { echo "!! no eta for $TAG. Run 'calibrate' first, or USE_FIXED_ETA=1 FIXED_ETA=<v>"; return 1; }
  local FAM="reduced_${TAG}_iid_c${POS//,/}"
  echo "== REDUCED $TAG (+5/common), pos=$POS -> $FAM, eta=$eta, seeds: $SEEDS"
  for s in $SEEDS; do
    env $COMMON_E $BITSENV ATTACK=reduced FREE_RIDER_IDS=$POS \
        AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
        FAMILY="$FAM" SWEEP_LEVEL=5 NOTE="$DS balanced +5/cls pos=$POS bits=${BITS:-default}" \
        WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

# ============================== PLOTS ==============================
# Plots ONLY the current TAG (so default-bits and more-bits don't get mixed).
plotall(){
  local ALL="$RES/*/result.json"
  local OUT="${OUT:-$RES/figs/$TAG}"; mkdir -p "$OUT"
  local eta; eta="$(get_eta)"; [ -z "$eta" ] && eta="0"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }

  run $PL sanity           --in "'$ALL'" --out "$OUT"
  run $PL class_difficulty --in "'$ALL'" --family "$HFAM" --out "$OUT"
  run $PL thresholds       --in "'$ALL'" --family "$HFAM" --out "$OUT"
  # honest per-class BER lines (auto), with this tag's eta drawn
  run $HL --in "'$ALL'" --family "$HFAM" --tail 20 --eta "$eta" \
          --out "$OUT/honest_class_lines_${TAG}.png"

  # attack families for THIS tag only
  FAMS="${FAMS:-$(python -c "import json,glob;fs=set(json.load(open(f)).get('manifest',{}).get('family') for f in glob.glob('$RES/*/result.json'));print(' '.join(sorted(x for x in fs if x and x.startswith('reduced_$TAG'))))" 2>/dev/null)}"
  for fam in $FAMS; do
    run $PL timeline --in "'$ALL'" --family $fam \
        --honest_in "'$ALL'" --honest_family "$HFAM" --out "$OUT/timeline_${fam}"
    run $PL fidelity --in "'$ALL'" --family $fam --out "$OUT"
  done
  echo "PLOTALL ($TAG) done -> $OUT"
}

# ============================= DISPATCH =============================
case "${1:-}" in
  honest)     honest ;;
  calibrate)  calibrate ;;
  reduced)    reduced ;;
  PLOTALL)    plotall ;;
  *) echo "usage: DS=c100|c10 [BITS=n] ./run_all.sh [honest|calibrate|reduced|PLOTALL]
  current: DS=$DS CFG=$CFG TAG=$TAG RES=$RES SEEDS='$SEEDS' POS=$POS
  default-bits CIFAR-100:
    DS=c100 SEEDS='0 1 2 3 4 5 6 7 8 9' ./run_all.sh honest
    DS=c100 ./run_all.sh calibrate
    DS=c100 SEEDS='0 1 2' POS=1,7 ./run_all.sh reduced
    DS=c100 SEEDS='0 1 2' POS=3,6 ./run_all.sh reduced
    DS=c100 ./run_all.sh PLOTALL
  more-bits (tagged separately, same RES):
    DS=c100 BITS=20 SEEDS='0 1 2 3 4 5 6 7 8 9' ./run_all.sh honest
    DS=c100 BITS=20 ./run_all.sh calibrate
    DS=c100 BITS=20 SEEDS='0 1 2' POS=1,7 ./run_all.sh reduced
    DS=c100 BITS=20 SEEDS='0 1 2' POS=3,6 ./run_all.sh reduced
    DS=c100 BITS=20 ./run_all.sh PLOTALL
  eta override if file-fetch fails:  USE_FIXED_ETA=1 FIXED_ETA=<calibrated value>"; exit 1 ;;
esac