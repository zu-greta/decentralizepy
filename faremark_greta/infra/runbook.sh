#!/usr/bin/env bash
# =============================================================================
# runbook.sh -- one entry point for the whole pipeline.
#
#   ./runbook.sh help          print the phase order
#   ./runbook.sh probe         0. embedding sanity check (gates paper rows)
#   ./runbook.sh manifest      1. build jobs.tsv for the next batch (run_now.sh)
#   ./runbook.sh submit        2. run the pool (PODS x WORKERS, submit_pool.sh)
#   ./runbook.sh monitor       3. watch progress
#   ./runbook.sh plot          4. ALL figures (plots.py, organised by group)
#   ./runbook.sh grade         5. paper_check tables (optional, paper-repro rows)
#
# Knobs (env): BATCH, PODS, WORKERS, RES, OUT, FAST_DATA(1), DETERMINISM(0), K4B(1)
#
# THIS TASK'S NEW WORK is groups Z and Y (see run_now.sh), which is the DEFAULT batch:
#   ./runbook.sh manifest && ./runbook.sh submit      # runs Z (no-wm) + Y (J4 c36+c17)
#   RES=~/local/results ./runbook.sh plot             # draws every storyline figure
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"

# ---- batch selection --------------------------------------------------------
# Default = the two NEW single-seed groups this task adds:
#   Z = no-watermark control (lambda=0; proves trig_acc~0 is caused by the watermark)
#   Y = J4 oracle-eta reproduction at classes 3,6 AND 1,7
# Other groups (already run / regenerable): A D E EA H K.  Combine letters, e.g.
#   BATCH=EEAK ./runbook.sh manifest   # E + EA + K together
BATCH="${BATCH:-ZY}"
PAPER_OK="${PAPER_OK:-0}"          # 1 = also build probe-gated paper rows (grade phase)
FAST_DATA="${FAST_DATA:-1}"        # 1 = GPU-resident loaders (kills DataLoader fork storms)
DETERMINISM="${DETERMINISM:-0}"    # 0 = cuDNN autotuner on (~1.3-2x; stat. identical over seeds)
PODS="${PODS:-2}"; WORKERS="${WORKERS:-6}"
MPS="${MPS:-1}"

RES="${RES:-/mnt/nfs/home/zu/results}"   # cluster results (submit) OR local dir (plot)
OUT="${OUT:-$RES/figs}"
ALL="$RES/*/result.json"

# ---- frozen references (STORYLINE 5.1 / E1). Change here, applied to every plot ----
HON=A1_honest_c100                 # honest calibration family (IID, c100, 10 clients)
HONCLASS="${HONCLASS:-A1_honest_c100}"   # all-honest family for the class-acc bar chart
ETA_T="0.064"; ETA_L="0.264"       # IID  eta tight / loose
ETA_T_NIID="0.161"; ETA_L_NIID="0.576"   # non-IID eta tight / loose

PL="python ../scripts/plots.py"    # THE single merged plotting script
PC="python ../scripts/paper_check.py"
run(){ echo "== $*"; eval "$*" || echo "   (skipped -- family may not exist yet)"; }

# ---------------------------------------------------------------------------
phase_probe(){
  echo ">>> PROBE: embedding sanity (watch run.log: ber_h should drop, pmax not nan)"
  NUM_WORKERS=0 FAMILY=probe_fix WM_BITS=2 WM_NUM_TRIGGERS=50 ROUNDS=25 \
      ./submit_experiment.sh 11 0
}

phase_manifest(){
  echo ">>> MANIFEST: groups=[$BATCH] PAPER_OK=$PAPER_OK FAST_DATA=$FAST_DATA DETERMINISM=$DETERMINISM"
  rm -f jobs.tsv
  FAST_DATA="$FAST_DATA" DETERMINISM="$DETERMINISM" PAPER_OK="$PAPER_OK" ./run_now.sh "$BATCH"
  echo "   -> jobs.tsv built. Review the per-family counts above."
}

phase_submit(){
  echo ">>> SUBMIT: $PODS pod(s) x $WORKERS workers, shared queue over jobs.tsv"
  [ -s jobs.tsv ] || { echo "!! no jobs.tsv -- run ./runbook.sh manifest first"; return 1; }
  [ "$MPS" = "1" ] && echo "   (MPS=1: start nvidia-cuda-mps-control -d in each pod before the workers)"
  unset DRYRUN
  WORKERS="$WORKERS" PODS="$PODS" ./submit_pool.sh
  echo "   monitor with:  ./runbook.sh monitor"
}

phase_monitor(){
  echo ">>> MONITOR"
  run "runai list jobs"
  echo "--- quick digest of whatever has landed ---"
  run "python ../scripts/resultio.py digest --in '$ALL'"
  echo "--- speed: seconds/round (last col) ---"
  run "for d in $RES/*/run.log; do echo \"\$d:\"; awk '\$2==\"R\" && \$3 ~ /^[0-9]/ {print \$3, \$NF}' \"\$d\" | tail -3; done"
}

# ---------------------------------------------------------------------------
# 4. ALL FIGURES. eta is FROZEN (references above), so no calibrate phase.
#    Organised strictly by experiment group; every call maps to a merged plots.py
#    subcommand. Families that are not present yet are skipped by run{} with a note.
# ---------------------------------------------------------------------------
phase_plot(){
  mkdir -p "$OUT"; echo ">>> PLOT -> $OUT"

  # ===================== GROUP A -- honest baseline =========================
  run "$PL honest_lines     --in '$ALL' --family $HON --tail 20 --out $OUT/A1_class_floors"
  run "$PL honest_per_round --in '$ALL' --family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/A1_honest_per_round"
  run "$PL class_acc        --in '$ALL' --family $HONCLASS --out $OUT/A0_class_acc"
  run "$PL timeline --in '$ALL' --family A2_reduced_c100_c17 --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/A2_easy_timeline"
  run "$PL timeline --in '$ALL' --family A3_reduced_c100_c36 --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/A3_hard_timeline"
  run "$PL iso_pair --honest_in '$ALL' --fr_in '$ALL' --family A3_reduced_c100_c36 --class 3 --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/iso_A3_c3"
  run "$PL iso_pair --honest_in '$ALL' --fr_in '$ALL' --family A3_reduced_c100_c36 --class 6 --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/iso_A3_c6"
  # isolated ACCURACY: hard class 6 (FR from K4), easy-medium class 7 (FR from A2)
  run "$PL iso_acc --honest_in '$ALL' --fr_in '$ALL' --family K4_alldyn_block2_c36 --class 6 --out $OUT/iso_acc_c6"
  run "$PL iso_acc --honest_in '$ALL' --fr_in '$ALL' --family A2_reduced_c100_c17 --class 7 --out $OUT/iso_acc_c7"

  # ===================== GROUP D -- reduced data-budget spectrum ============
  run "$PL sweep       --in '$ALL' --family D1_reduced_c100_c36 --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/D1_spectrum"
  run "$PL gpu_savings --in '$ALL' --family D1_reduced_c100_c36_n5 --out $OUT/gpu_savings_D1_reduced_c100_c36_n5"

  # ===================== GROUP E -- starved non-IID =========================
  run "$PL honest_lines     --in '$ALL' --family E1_honest_niid_c100 --tail 20 --out $OUT/E1_class_floors"
  run "$PL honest_per_round --in '$ALL' --family E1_honest_niid_c100 --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/E1_honest_per_round"
  run "$PL timeline --in '$ALL' --family E2_reduced_niid_c36 --honest_in '$RES/E1_honest_niid_c100_rep*/result.json' --honest_family E1_honest_niid_c100 --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/E2_niid_timeline"
  for at in a01 a10; do
    run "$PL timeline --in '$ALL' --family E3_reduced_niid_c36_${at} --honest_in '$ALL' --honest_family E3_honest_niid_c100_${at} --eta_tight $ETA_T_NIID --out $OUT/E3_${at}_timeline"
  done
  run "$PL iso_pair    --honest_in '$RES/E1_honest_niid_c100_rep*/result.json' --fr_in '$ALL' --family E2_reduced_niid_c36 --class 6 --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/iso_E2_c6"
  run "$PL gpu_savings --in '$ALL' --family E2_reduced_niid_c36 --out $OUT/gpu_savings_E2_reduced_niid_c36"
  run "$PL dirichlet_dist --out $OUT/dirichlet_dist"

  # ===================== GROUP EA -- fair non-IID (distribution assignment) ==
  run "$PL honest_per_round --in '$ALL' --family EA1_honest_niid_distrib_c100 --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/EA1_honest_per_round"
  run "$PL iso_pair    --honest_in '$RES/EA1_honest_niid_distrib_c100_rep*/result.json' --fr_in '$ALL' --family EA2_reduced_niid_distrib_c36 --class 6 --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/iso_EA2_c6"
  run "$PL gpu_savings --in '$ALL' --family EA2_reduced_niid_distrib_c36 --out $OUT/gpu_savings_EA2_reduced_niid_distrib_c36"

  # ===================== GROUP K -- the submarine ===========================
  # THREE views per free-rider (cid3 easy, cid6 hard) from the SAME data:
  #   tap_perfr   = seed-band single graph (mean over seeds; majority tap/coast markers)
  #   tap_perseed = one panel PER SEED (fixes the marker collisions across seeds)
  #   tap_effort  = BER + cumulative SAMPLES side by side (stealth vs effort in one look)
  for fam in K4_alldyn_block2_c36 K5_alldyn_full_c36 ${K4B:+K4b_alldyn_block2_fulldata_c36}; do
    run "$PL tap_perfr   --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_perfr_${fam}"
    run "$PL tap_perseed --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_perfr_${fam}"
    run "$PL tap_effort  --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_perfr_${fam}"
    run "$PL accuracy    --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --out $OUT/accuracy_${fam}"
    run "$PL gpu_savings --in '$ALL' --family $fam --out $OUT/gpu_savings_${fam}"
  done
  # isolated same-class twin (from A1) for cid3 (easy) and cid6 (hard), K4 + K5
  for cls in 3 6; do
    run "$PL iso_pair --honest_in '$RES/${HON}_rep*/result.json' --fr_in '$RES/K4_alldyn_block2_c36_rep*/result.json' --class $cls --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/iso_K4_c${cls}"
    run "$PL iso_pair --honest_in '$RES/${HON}_rep*/result.json' --fr_in '$RES/K5_alldyn_full_c36_rep*/result.json' --class $cls --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/iso_K5_c${cls}"
  done

  # ===================== GROUP Z -- NO-WATERMARK control ====================
  # A0_nowm = all-honest, lambda=0 (embedding OFF) but the verifier still logs
  # trig_acc. The bottom panel of honest_per_round should stay HIGH here, unlike
  # A1 where the watermark drives it to ~0 -- confirming the watermark embedding
  # (not the class) causes the zero trigger-class accuracy (STORYLINE 6c).
  run "$PL honest_per_round --in '$ALL' --family A0_nowm_honest_c100 --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/A0_nowm_per_round"
  run "$PL class_acc        --in '$ALL' --family A0_nowm_honest_c100 --out $OUT/A0_nowm_class_acc"

  # ===================== GROUP Y -- J4 oracle-eta reproduction ==============
  # J4 = the oracle-eta version of K4 (block2 + graft, given eta=0.264). Reproduced
  # at classes 3,6 (c36) and 1,7 (c17), single seed. Same 3 submarine views.
  for fam in J4_scope_graft_block2_c36 J4_scope_graft_block2_c17; do
    run "$PL tap_perfr  --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_${fam}"
    run "$PL tap_effort --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_${fam}"
  done

  echo "   done -> $OUT  (A honest / D reduced / E starved-niid / EA fair-niid / K+Y submarine / Z no-wm control)"
}

phase_grade(){
  # OPTIONAL paper-reproduction tables (needs the probe-gated paper rows, PAPER_OK=1).
  echo ">>> GRADE vs the FareMark paper (optional; needs paper-repro families)"
  run "$PC --row t9 --in '$ALL' --family F3_tableIX_c10_nc50 --heldout-family F3_tableIX_c10_nc50_heldout"
  run "$PC --row c10 --in '$ALL' --family H1_honest_c10"
}

case "${1:-help}" in
  probe)     phase_probe ;;
  manifest)  phase_manifest ;;
  submit)    phase_submit ;;
  monitor)   phase_monitor ;;
  plot)      phase_plot ;;
  grade)     phase_grade ;;
  all-submit) phase_probe; phase_manifest; phase_submit ;;   # 0 -> 1 -> 2
  *)
    cat <<USAGE
runbook.sh -- run phases in this order (wait for the cluster between submit and plot):

  ON THE CLUSTER (has submit_experiment.sh + .env):
    ./runbook.sh probe       0. embedding sanity
    ./runbook.sh manifest    1. build jobs.tsv     (BATCH=$BATCH)
    ./runbook.sh submit      2. run the pool       (PODS=$PODS WORKERS=$WORKERS)
    ./runbook.sh monitor     3. progress
       ... wait for jobs to finish ...

  LOCALLY (set RES=~/local/results):
    RES=~/local/results ./runbook.sh plot     4. ALL figures
    RES=~/local/results ./runbook.sh grade    5. paper tables (optional)

  THIS TASK'S NEW RUNS (default BATCH=ZY):
    Z = A0_nowm_honest_c100    single-seed, all-honest, NO watermark (lambda=0);
        plotted with honest_per_round -> trig_acc stays HIGH (vs A1 ~0).
    Y = J4_scope_graft_block2_{c36,c17}  single-seed J4 reproduction at classes 3,6 and 1,7.

  batch letters: A D E EA H K  (+ new Z Y). Combine, e.g. BATCH=EEAK. Batch size stays 16.
USAGE
    ;;
esac