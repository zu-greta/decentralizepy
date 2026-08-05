#!/usr/bin/env bash
# =============================================================================
# plot_groups.sh -- make the figures for each experiment group AS IT FINISHES.
#
#   RES=~/local/results ./plot_groups.sh A        # just group A
#   RES=~/local/results ./plot_groups.sh A C D     # several
#   RES=~/local/results ./plot_groups.sh all
#
# Every group gets the TWO figures you always want:
#   1. thresholds        -- the paper's eta, calibrated on PURE HONEST runs
#   2. timeline          -- BER vs round: honest BER, FR BER, and eta, over time
# plus group-specific figures. Threshold tables (.md) come from plot_all_thresholds.
#
# eta is calibrated ONCE from the honest family of each dataset and reused on the
# attack timelines, so the dashed line is the REAL threshold, not the provisional.
# =============================================================================
set -uo pipefail
RES="${RES:?set RES to your local results dir}"
OUT="${OUT:-figs}"; mkdir -p "$OUT"
ALL="$RES/*/result.json"
PL="python ../scripts/plots.py"
DET="python ../scripts/detection.py"
ATH="python ../scripts/plot_all_thresholds.py"
run(){ echo "== $*"; eval "$*" || echo "   (skipped -- family may not exist yet)"; }
WANT="${*:-all}"; has(){ [[ " $WANT " == *" $1 "* ]] || [[ "$WANT" == all ]]; }

# --- calibrate eta per dataset from the HONEST family, reuse on attacks --------
# c100 honest = A1; if you renamed, edit here.
calib_eta(){
  local hf="$1" fb="$2" tmp
  tmp="$(mktemp)"
  $DET calibrate --in $ALL --honest-family "$hf" --tail 20 --out "$tmp" >/dev/null 2>&1 || true
  local v
  v="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))[\"eta\"])" "$tmp" 2>/dev/null)"
  [ -n "$v" ] && echo "$v" || echo "$fb"
}
ETA_C100="${ETA_C100:-$(calib_eta A1_honest_c100 0.064)}"
echo ">> using calibrated eta (c100 honest) = $ETA_C100"

# helper: the two standard figures for one honest+attack family pair
pair(){  # $1 attack family  $2 honest family  $3 eta  $4 label
  local af="$1" hf="$2" eta="$3" lab="$4"
  run $ATH --in "'$ALL'" --family "$hf" --tail 20 --out "$OUT/${lab}_thresholds"     # (1) calibrated-threshold + .md table
  run $PL timeline --in "'$ALL'" --family "$af" --honest_in "'$ALL'" \
        --honest_family "$hf" --eta "$eta" --out "$OUT/${lab}_timeline"              # (2) BER vs round + eta
  run $DET separability --honest-in "'$ALL'" --honest-family "$hf" \
        --attack-in "'$ALL'" --attack-family "$af" --tail 20 --per-class \
        --emit "$OUT/${lab}_sep.json"                                               # the OVL / best-error numbers
}

# ============================ GROUP A =========================================
# proven baseline: class difficulty (honest) + reduced non-separability
if has A; then
  echo "### GROUP A"
  # honest baseline: the two always-on figures + class difficulty
  run $ATH --in "'$ALL'" --family A1_honest_c100 --tail 20 --out "$OUT/A1_thresholds"
  run $PL thresholds   --in "'$ALL'" --family A1_honest_c100 --out "$OUT/A1_thresholds_canonical"
  run $PL honest_lines --in "'$ALL'" --family A1_honest_c100 --tail 20 --out "$OUT/A1_class_floors.png"
  run $PL class_probe  --in "'$ALL'" --family A1_honest_c100 --out "$OUT/A1_class_probe"
  run $PL eta_stability --in "'$ALL'" --family A1_honest_c100 --out "$OUT/A1_eta_stability"   # the seed-variance of eta (F2)
  # attacks: easy (c17), hard (c36), sameclass, sameclass+samekey
  pair A2_reduced_c100_c17   A1_honest_c100 "$ETA_C100" A2_easy
  pair A3_reduced_c100_c36   A1_honest_c100 "$ETA_C100" A3_hard
  pair A4_sameclass_c100_c6  A1_honest_c100 "$ETA_C100" A4_sameclass
fi

# ============================ AK ==============================================
# same class + same key + same message: only effort differs
if has AK || has A; then
  echo "### AK"
  pair AK_sameclass_samekey_c6 A1_honest_c100 "$ETA_C100" AK_samekey
fi

# ============================ GROUP C =========================================
# smoothing function (does sin move the floors?)
if has C; then
  echo "### GROUP C"
  run $ATH --in "'$ALL'" --family C1_honest_sin_c100 --tail 20 --out "$OUT/C1_thresholds"
  run $PL honest_lines --in "'$ALL'" --family C1_honest_sin_c100 --tail 20 --out "$OUT/C1_class_floors.png"
  run $PL class_probe  --in "'$ALL'" --family C1_honest_sin_c100 --out "$OUT/C1_class_probe"
fi

# ============================ GROUP D =========================================
# +N free-riding spectrum: the price-of-invisibility curve
if has D; then
  echo "### GROUP D"
  FAMS=$(python - <<PY
print(' '.join(f"D1_reduced_c100_c36_n{t}" for t in ['-1','0','1','2','5','10']))
PY
)
  run $PL sweep --in "'$ALL'" --families $FAMS --honest_in "'$ALL'" \
        --honest_family A1_honest_c100 --eta "$ETA_C100" --out "$OUT/D1_spectrum"
fi

# ============================ GROUP E =========================================
# non-iid
if has E; then
  echo "### GROUP E"
  ETA_NIID="$(calib_eta E1_honest_niid_c100 0.161)"
  run $ATH --in "'$ALL'" --family E1_honest_niid_c100 --tail 20 --out "$OUT/E1_thresholds"
  run $PL honest_lines --in "'$ALL'" --family E1_honest_niid_c100 --tail 20 --out "$OUT/E1_class_floors.png"
  pair E2_reduced_niid_c36 E1_honest_niid_c100 "$ETA_NIID" E2_niid
fi

# ============================ GROUP F =========================================
# more clients than classes
if has F; then
  echo "### GROUP F"
  ETA_F="$(calib_eta F1_honest_nc200 0.384)"
  run $ATH --in "'$ALL'" --family F1_honest_nc200 --tail 20 --out "$OUT/F1_thresholds"
  pair F2_reduced_nc200_c67 F1_honest_nc200 "$ETA_F" F2_capacity
fi

echo
echo "done -> $OUT"
echo "For every *_sep.json, the two headline numbers are:"
echo "  overlap_coefficient          1.0 = honest & FR BER identical"
echo "  best_threshold_balanced_error 0.5 = no threshold beats a coin"