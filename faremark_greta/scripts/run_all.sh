#!/usr/bin/env bash
# run_all.sh --  CIFAR-100 (default) or CIFAR-10.
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
WMF="${WMF:-}"                                     # empty = power (default); or 'sin' (paper Eq.9)
export BALANCED="${BALANCED:-0}"                   # 0 = paper-faithful UNBALANCED keys (default;
                                                   # spreads the honest floor -> clearer story).
                                                   # BALANCED=1 = sign-balanced keys (removes the
                                                   # F6 unembeddable-bit artifact; the airtight control)
FTAG=""; [ -n "$WMF" ] && FTAG="_${WMF}"           # tag so sin never mixes with power at calibration
FENV=""; [ -n "$WMF" ] && FENV="WM_F=$WMF"
TAG="${DS}_${BT}${FTAG}"                           # c100_bdef | c100_b20 | c100_bdef_sin
VTAG="${VTAG:-}"                                    # optional variant marker (bal|nc200|spread...)
TRIGMODE="${TRIGMODE:-}"                            # ""|class|client|client_train (verifier images)
TMENV=""; [ -n "$TRIGMODE" ] && TMENV="WM_TRIGGER_MODE=$TRIGMODE"
[ -n "$TRIGMODE" ] && [ "$TRIGMODE" != "class" ] && VTAG="${VTAG:+${VTAG}_}tm${TRIGMODE#client_}"
[ -n "$VTAG" ] && TAG="${TAG}_${VTAG}"             # -> unique family+eta when DS/BITS/PART/WMF don't differ
PART="${PART:-iid}"                                # iid | niid  (data partition)
ALPHA="${DIRICHLET_ALPHA:-0.5}"                    # Dirichlet concentration (non-IID severity)
PARTENV=""; [ "$PART" = "niid" ] && PARTENV="PARTITION=dirichlet DIRICHLET_ALPHA=${ALPHA}"
# PTAG carries alpha so an alpha SWEEP gets separate families + separate eta files.
# alpha=0.5 (the benchmark default) keeps the plain "niid" string -> back-compatible.
PTAG="$PART"
[ "$PART" = "niid" ] && [ "$ALPHA" != "0.5" ] && PTAG="niid_a${ALPHA//./}"
TCMAP="${TCMAP:-}"                                 # optional "cid:class,..." trigger-class override
TCENV=""; [ -n "$TCMAP" ] && TCENV="TRIGGER_CLASS_MAP=$TCMAP"  # for honest/reduced (NOT sameclass)
ETA_SUFFIX=""; [ "$PART" = "niid" ] && ETA_SUFFIX="_${PTAG}"
ETA_FILE="${ETA_FILE:-$RES/eta_${TAG}${ETA_SUFFIX}.json}"
PL="python scripts/plots.py"; TH="python threshold.py"
SEP="python scripts/separability.py"
HL="$PL honest_lines"                              # merged into plots.py (was honest_class_lines.py)
FIXED_ETA="${FIXED_ETA:-}"; USE_FIXED_ETA="${USE_FIXED_ETA:-}"
COMMON_E="ROUNDS=50 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4"
BITSENV=""; [ -n "$BITS" ] && BITSENV="WM_BITS=$BITS"

HFAM="honest_${TAG}_${PTAG}"                        # honest_c100_bdef_iid | ..._niid | ..._niid_a01

read_eta(){ python - "$ETA_FILE" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1]))["eta"])
except Exception: print("")
PY
}
get_eta(){ if [ -n "$USE_FIXED_ETA" ]; then [ -z "$FIXED_ETA" ] && { echo ""; return; }; echo "$FIXED_ETA"; else read_eta; fi; }

# ============================ EXPERIMENTS ============================
honest(){
  echo "== HONEST $TAG/$PART (all-honest, balanced keys, seeds: $SEEDS)"
  for s in $SEEDS; do
    env ROUNDS=50 ATTACK=none $BITSENV $PARTENV $FENV $TCENV $TMENV FAMILY="$HFAM" \
        NOTE="$DS $PART balanced-keys all honest bits=${BITS:-default}" \
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
  [ -z "$eta" ] && { echo "!! no eta for $TAG/$PART. Run 'calibrate' first, or USE_FIXED_ETA=1 FIXED_ETA=<v>"; return 1; }
  local FAM="reduced_${TAG}_${PTAG}_c${POS//,/}"
  echo "== REDUCED $TAG/$PART (+5/common), pos=$POS -> $FAM, eta=$eta, seeds: $SEEDS"
  for s in $SEEDS; do
    env $COMMON_E $BITSENV $PARTENV $FENV $TCENV $TMENV ATTACK=reduced FREE_RIDER_IDS=$POS \
        AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
        FAMILY="$FAM" SWEEP_LEVEL=5 NOTE="$DS $PART +5/cls pos=$POS bits=${BITS:-default}" \
        WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

# same-trigger-class control (the airtight non-separability slice): pin ONE free-rider
# onto a HARD honest class so a FR and an honest client share a trigger class. Their BER
# is then drawn from ONE class floor (keys/bits still differ per cid), so no threshold
# can tell them apart. SC_FR = which cid free-rides, SC_CLASS = the class it is pinned to.
sameclass(){
  local eta; eta="$(get_eta)"
  [ -z "$eta" ] && { echo "!! no eta for $TAG/$PART. Run 'calibrate' first."; return 1; }
  local FRC="${SC_FR:-0}"; local CLS="${SC_CLASS:-6}"
  local FAM="sameclass_${TAG}_${PTAG}_c${CLS}"
  echo "== SAMECLASS $TAG/$PART: FR cid$FRC pinned to class $CLS (honest cid$CLS shares it) -> $FAM, eta=$eta"
  for s in $SEEDS; do
    env $COMMON_E $BITSENV $PARTENV $FENV $TMENV ATTACK=reduced FREE_RIDER_IDS=$FRC \
        TRIGGER_CLASS_MAP="${FRC}:${CLS}" AUTOP_COMMON_PER_CLASS=5 WM_ETA_FIXED=$eta \
        FAMILY="$FAM" NOTE="$DS $PART same-class FR cid$FRC->cls$CLS bits=${BITS:-default}" \
        WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

# convenience: the full non-IID stress leg. Submits honest-niid (for its own eta) AND the
# reduced attack under Dirichlet, so the non-IID result is self-consistent (never mixes an
# iid-calibrated eta with non-iid data). After honest-niid finishes: `PART=niid ... calibrate`
# then `PART=niid ... reduced`. This target just fires the honest leg and prints the rest.
noniid(){
  PART=niid PARTENV="PARTITION=dirichlet DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-0.5}" \
  HFAM="honest_${TAG}_niid" honest
  echo "next: DS=$DS PART=niid ./run_all.sh calibrate  &&  DS=$DS PART=niid POS=$POS ./run_all.sh reduced"
}

# ============================== PLOTS ==============================
# Plots ONLY the current TAG/PART (so variants never get mixed).
plotall(){
  local ALL="$RES/*/result.json"
  local OUT="${OUT:-$RES/figs/${TAG}_${PTAG}}"; mkdir -p "$OUT"
  local eta; eta="$(get_eta)"; [ -z "$eta" ] && eta="0"
  run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }

  run $PL sanity           --in "'$ALL'" --out "$OUT"
  run $PL class_difficulty --in "'$ALL'" --family "$HFAM" --out "$OUT"
  run $PL thresholds       --in "'$ALL'" --family "$HFAM" --out "$OUT"
  # honest per-class BER lines (merged into plots.py), with this tag's eta drawn
  run $HL --in "'$ALL'" --family "$HFAM" --tail 20 --eta "$eta" \
          --out "$OUT/honest_lines_${TAG}_${PTAG}.png"

  # attack families for THIS tag+partition only (reduced + sameclass)
  FAMS="${FAMS:-$(python -c "import json,glob;fs=set(json.load(open(f)).get('manifest',{}).get('family') for f in glob.glob('$RES/*/result.json'));print(' '.join(sorted(x for x in fs if x and (x.startswith('reduced_${TAG}_${PTAG}') or x.startswith('sameclass_${TAG}_${PTAG}')))))" 2>/dev/null)}"
  for fam in $FAMS; do
    run $PL timeline --in "'$ALL'" --family $fam \
        --honest_in "'$ALL'" --honest_family "$HFAM" --out "$OUT/timeline_${fam}"
    run $PL fidelity --in "'$ALL'" --family $fam --out "$OUT"
    run $PL separability --in "'$ALL'" --family "$HFAM" --attack_family $fam \
        --out "$OUT/separability_${fam}"
  done
  echo "PLOTALL ($TAG/$PTAG) done -> $OUT"
}

# rule-independent non-separability tables (text + json) for every attack family vs the
# honest floor. This is the headline "no threshold works" evidence.
separability(){
  local ALL="$RES/*/result.json"
  local OUT="${OUT:-$RES/figs/${TAG}_${PTAG}}"; mkdir -p "$OUT"
  local FAMS; FAMS="$(python -c "import json,glob;fs=set(json.load(open(f)).get('manifest',{}).get('family') for f in glob.glob('$RES/*/result.json'));print(' '.join(sorted(x for x in fs if x and (x.startswith('reduced_${TAG}_${PTAG}') or x.startswith('sameclass_${TAG}_${PTAG}')))))" 2>/dev/null)"
  for fam in $FAMS; do
    echo "== SEPARABILITY  honest=$HFAM  attack=$fam"
    $SEP --honest-in "$ALL" --honest-family "$HFAM" \
         --attack-in "$ALL" --attack-family "$fam" --tail 20 --per-class \
         --emit "$OUT/separability_${fam}.json"
  done
}

# ============================= DISPATCH =============================
case "${1:-}" in
  honest)       honest ;;
  calibrate)    calibrate ;;
  reduced)      reduced ;;
  sameclass)    sameclass ;;
  noniid)       noniid ;;
  separability) separability ;;
  PLOTALL)      plotall ;;
  *) echo "usage: DS=c100|c10 [BITS=n] [PART=iid|niid] ./run_all.sh [honest|calibrate|reduced|sameclass|noniid|separability|PLOTALL]
  current: DS=$DS CFG=$CFG TAG=$TAG PART=$PART alpha=$ALPHA RES=$RES SEEDS='$SEEDS' POS=$POS
  honest family: $HFAM
  eta file:      $ETA_FILE
  default-bits CIFAR-100 (IID):
    DS=c100 SEEDS='0 1 2 3 4 5 6 7 8 9' ./run_all.sh honest
    DS=c100 ./run_all.sh calibrate
    DS=c100 SEEDS='0 1 2' POS=1,7 ./run_all.sh reduced        # easy positions
    DS=c100 SEEDS='0 1 2' POS=3,6 ./run_all.sh reduced        # hard positions
    DS=c100 SEEDS='0 1 2' SC_FR=0 SC_CLASS=6 ./run_all.sh sameclass
    DS=c100 ./run_all.sh separability                         # text/json tables
    DS=c100 ./run_all.sh PLOTALL                              # figures
  non-IID stress (own eta, never mixed with iid):
    DS=c100 PART=niid SEEDS='0 1 2 3 4 5 6 7 8 9' ./run_all.sh honest
    DS=c100 PART=niid ./run_all.sh calibrate
    DS=c100 PART=niid SEEDS='0 1 2' POS=3,6 ./run_all.sh reduced
    DS=c100 PART=niid ./run_all.sh PLOTALL
  more-bits (tagged separately, same RES):
    DS=c100 BITS=20 SEEDS='0 1 2 3 4 5 6 7 8 9' ./run_all.sh honest
    DS=c100 BITS=20 ./run_all.sh calibrate
    DS=c100 BITS=20 SEEDS='0 1 2' POS=1,7 ./run_all.sh reduced
    DS=c100 BITS=20 ./run_all.sh PLOTALL
  eta override if file-fetch fails:  USE_FIXED_ETA=1 FIXED_ETA=<calibrated value>"; exit 1 ;;
esac