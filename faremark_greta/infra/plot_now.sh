#!/usr/bin/env bash
# =============================================================================
# plot_now.sh -- LOCAL. Makes exactly the four figure groups (a)-(d) and nothing
# else. Run after `scp -r <cluster>:$MOUNT/home/zu/results ~/local/results`.
#
#   RES=~/local/results ./plot_now.sh
#
# Deliberately does NOT call run_everything.sh plot / PLOTALL -- those emit
# dozens of figures across every family. Four groups, each with a stated claim.
# =============================================================================
set -uo pipefail
RES="${RES:?set RES to your local results dir}"
ALL="$RES/*/result.json"
OUT="${OUT:-figs}"; mkdir -p "$OUT"
PL="python scripts/plots.py"; DET="python scripts/detection.py"
ATH="python scripts/plot_all_thresholds.py"
run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }

# =============================================================================
# (a) EVERY THRESHOLD ON A TIMELINE + the table explaining each calculation
#     Claim: no horizontal line separates anything, and the paper's own rule
#     delivers a fraction of the 3-sigma headroom it advertises.
# =============================================================================
run $ATH --in "'$ALL'" --family R1_paper_c100_nc100 --tail 20 --out "$OUT/a_thresholds_c100"
run $ATH --in "'$ALL'" --family R2_paper_c10_m1     --tail 20 --out "$OUT/a_thresholds_c10"
run $ATH --in "'$ALL'" --family honest_c100_bdef_iid --tail 20 --out "$OUT/a_thresholds_iid10"
# the canonical single-eta derivation, for the appendix
run $PL thresholds --in "'$ALL'" --family R1_paper_c100_nc100 --out "$OUT"

# =============================================================================
# (b) IMPLEMENTATION MATCHES THE PAPER
#     Claim: our code reproduces Table I+II and cleanly catches the paper's OWN
#     free-riders. Without this, every negative result below is unfalsifiable.
# =============================================================================
run "RES=$RES ROW=c100 FAM=R1_paper_c100_nc100 ./paper_check.sh check"
run "RES=$RES ROW=c10  FAM=R2_paper_c10_m1     ./paper_check.sh check"
for fam in R3_crude_prevmodels_c100 R4_crude_gaussian_c100; do
  run $DET separability --honest-in "'$ALL'" --honest-family honest_c100_bdef_iid \
        --attack-in "'$ALL'" --attack-family $fam --tail 20 --per-class \
        --emit "$OUT/b_sep_${fam}.json"
  run $PL timeline --in "'$ALL'" --family $fam --honest_in "'$ALL'" \
        --honest_family honest_c100_bdef_iid --eta 0.147 --out "$OUT/b_timeline_${fam}"
done

# =============================================================================
# (c) CLASS DIFFICULTY, all 100 CIFAR-100 trigger classes
#     R1 has 100 clients so cid%100 covers every class in one run.
#     class_probe emits the correlation table (entropy / dominance / accuracy).
# =============================================================================
run $PL class_difficulty --in "'$ALL'" --family R1_paper_c100_nc100 --out "$OUT"
run $PL class_probe      --in "'$ALL'" --family R1_paper_c100_nc100 --out "$OUT"
run $PL honest_lines     --in "'$ALL'" --family R1_paper_c100_nc100 --tail 20 \
       --out "$OUT/c_honest_lines_all100.png"

# =============================================================================
# (d) ATTACKS -- timeline with the TIGHTEST honest-calibrated threshold, plus
#     the same-trigger-class runs.
#     Tightest = the smallest eta any honest-only rule produces. Using the
#     tightest is the strongest possible concession to the paper: if the
#     free-rider survives even the most aggressive threshold, no rule works.
#     Read it off column `eta` in $OUT/a_thresholds_*.md, then set ETA_* below.
# =============================================================================
ETA_IID="${ETA_IID:-0.063}"     # from a_thresholds_iid10.md
ETA_N200="${ETA_N200:-0.384}"   # from R5 honest calibration
ETA_BAL="${ETA_BAL:-0.001}"     # balanced keys: honest BER is 0, so any eta>0

# d1  same trigger class, unbalanced  (already-run family)
run $PL timeline --in "'$ALL'" --family sameclass_c100_bdef_iid_c6 \
      --honest_in "'$ALL'" --honest_family honest_c100_bdef_iid \
      --eta "$ETA_IID" --out "$OUT/d1_sameclass_iid_c6"
run $DET separability --honest-in "'$ALL'" --honest-family honest_c100_bdef_iid \
      --attack-in "'$ALL'" --attack-family sameclass_c100_bdef_iid_c6 \
      --tail 20 --per-class --emit "$OUT/d1_sep_sameclass_iid.json"

# d2  same trigger class, BALANCED keys -- the paper's own operating point
run $PL timeline --in "'$ALL'" --family R8_sameclass_bal_c6 \
      --honest_in "'$ALL'" --honest_family honest_c100_bdef_bal_iid \
      --eta "$ETA_BAL" --out "$OUT/d2_sameclass_balanced_c6"
run $DET separability --honest-in "'$ALL'" --honest-family honest_c100_bdef_bal_iid \
      --attack-in "'$ALL'" --attack-family R8_sameclass_bal_c6 \
      --tail 20 --per-class --emit "$OUT/d2_sep_sameclass_bal.json"

# d3  more clients than classes -- forced sharing of classes 6,7
run $PL timeline --in "'$ALL'" --family R6_reduced_nc200_c67 \
      --honest_in "'$ALL'" --honest_family R5_honest_nc200 \
      --eta "$ETA_N200" --out "$OUT/d3_reduced_nc200_c67"
run $DET separability --honest-in "'$ALL'" --honest-family R5_honest_nc200 \
      --attack-in "'$ALL'" --attack-family R6_reduced_nc200_c67 \
      --tail 20 --per-class --emit "$OUT/d3_sep_nc200.json"

# d4  adaptive free-rider (coast/tap)
run $PL timeline --in "'$ALL'" --family R7_tap_c100_c36 \
      --honest_in "'$ALL'" --honest_family honest_c100_bdef_iid \
      --eta "$ETA_IID" --out "$OUT/d4_tap_c36"
run $DET separability --honest-in "'$ALL'" --honest-family honest_c100_bdef_iid \
      --attack-in "'$ALL'" --attack-family R7_tap_c100_c36 \
      --tail 20 --per-class --emit "$OUT/d4_sep_tap.json"

# d5  non-IID same trigger class (already run)
run $PL timeline --in "'$ALL'" --family sameclass_c100_bdef_niid_c6 \
      --honest_in "'$ALL'" --honest_family honest_c100_bdef_niid \
      --eta 0.161 --out "$OUT/d5_sameclass_niid_c6"

echo; echo "done -> $OUT"
echo "In every separability json the two numbers that matter are:"
echo "  overlap_coefficient        1.0 = the two BER distributions are IDENTICAL"
echo "  best_threshold_balanced_error  0.5 = NO threshold beats a coin flip"
