#!/usr/bin/env bash
# =============================================================================
# runbook.sh 
#
#   ./runbook.sh help          # print the phase order
#   ./runbook.sh probe         # 0. embedding sanity check (gates the paper rows)
#   ./runbook.sh manifest      # 1. build jobs.tsv for the next batch
#   ./runbook.sh submit        # 2. run the pool (PODS x WORKERS)
#   ./runbook.sh monitor       # 3. watch progress 
#   disabled: ./runbook.sh fetch         # 4. copy result.json to a local dir for plotting
#   ./runbook.sh calibrate     # 5. recompute eta from honest families
#   ./runbook.sh plot          # 6. all figures
#   ./runbook.sh grade         # 7. paper_check tables (Table VII/IX + Table V gap)
#
# Knobs (env): BATCH, PAPER_OK, PODS, WORKERS, RES, OUT, FAST_DATA(1), DETERMINISM(0)
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"

BATCH="${BATCH:-EFHIV}"          # UN-RUN / to-fix families: E(E1,E2,E3-fixed) F(F3) H(H5) I V(V2).
                                # A and D are done (3 seeds). C is EXCLUDED (sin crash, see R14).
                                # J is the graft-coast suite (persistence/sawtooth/coast); run
                                # it alone with BATCH=J after the clients.py graft fix is in.
                                # K = DYNAMIC-submarine 1-seed tests (self-eta / derived margin /
                                #     dynamic warmup / all-dynamic+block2). Run with BATCH=K.
                                # EA = distribution-aware trigger assignment (non-IID fairness);
                                #     run WITH E in parallel via BATCH=EEA.
                                # To run the two NEW groups this task adds, together with E:
                                #     BATCH=EEAK ./runbook.sh manifest && BATCH=EEAK ./runbook.sh submit
                                # The pool SKIPS any family whose result.json EXISTS -- so the buggy
                                # E3 a01/a03/a10 will NOT be re-run unless you first delete their dirs:
                                #   rm -rf $RES/E3_*_niid_*_a0{1,3}_rep* $RES/E3_*_niid_*_a10_rep*
                                # (a01==a03 byte-identical last time; drop a03, re-run a01/a10 clean.)
PAPER_OK="${PAPER_OK:-1}"         # 1 = also build the probe-gated paper rows (F3/Table IX)
FAST_DATA="${FAST_DATA:-1}"       # 1 = GPU-resident loaders (kills DataLoader fork storms)
DETERMINISM="${DETERMINISM:-0}"   # 0 = cuDNN autotuner on (~1.3-2x; stat. identical over seeds)
PODS="${PODS:-2}"; WORKERS="${WORKERS:-6}"   # batch=16 resnet18 barely uses an A100, so the run is
                                # GPU-STARVED, not GPU-bound: throughput comes from packing MORE
                                # concurrent runs per pod, not from a bigger batch (which is frozen
                                # for comparability). With FAST_DATA on, 6-8 workers/A100 is usually
                                # the sweet spot; watch `nvidia-smi dmon -s u` and raise until util
                                # stops climbing. (Was 3 -- that under-fills the card.)
MPS="${MPS:-1}"                 # 1 = start CUDA MPS on each pod so the many small processes share
                                # the SM scheduler instead of context-switching (big win at batch 16)
RES="${RES:-/mnt/nfs/home/zu/results}"   # cluster results (submit) OR local dir (plot); override for local
OUT="${OUT:-$RES/figs}"           # output folder should be RES/figs by default
ALL="$RES/*/result.json"
HON=A1_honest_c100                # the honest calibration family for c100/10cl

PL="python ../scripts/plots.py"
DET="python ../scripts/detection.py"
ATH="python ../scripts/plot_all_thresholds.py"
PC="python ../scripts/paper_check.py"
PAIR="python ../scripts/plot_sameclass_pair.py"
run(){ echo "== $*"; eval "$*" || echo "   (skipped -- family may not exist yet)"; }

# ---------------------------------------------------------------------------
phase_probe(){
  # 0. Confirm the watermark still embeds on this commit before burning a batch.
  #    BER must fall over rounds and pmax must not be nan. Gate for all paper rows.
  echo ">>> PROBE: embedding sanity (watch run.log: ber_h should drop, pmax not nan)"
  NUM_WORKERS=0 FAMILY=probe_fix WM_BITS=2 WM_NUM_TRIGGERS=50 ROUNDS=25 \
      ./submit_experiment.sh 11 0
}

phase_manifest(){
  # 1. Build the manifest for the next batch. run_now.sh only adds new families
  echo ">>> MANIFEST: groups=[$BATCH] PAPER_OK=$PAPER_OK FAST_DATA=$FAST_DATA DETERMINISM=$DETERMINISM"
  rm -f jobs.tsv
  FAST_DATA="$FAST_DATA" DETERMINISM="$DETERMINISM" PAPER_OK="$PAPER_OK" ./run_now.sh "$BATCH"
  echo "   -> jobs.tsv built. Review the per-family counts above."
}

phase_submit(){
  # 2. Run the pool: PODS runai jobs (== GPUs), workers concurrent runs each.
  echo ">>> SUBMIT: $PODS pod(s) x $WORKERS workers, shared queue over jobs.tsv"
  [ -s jobs.tsv ] || { echo "!! no jobs.tsv -- run ./runbook.sh manifest first"; return 1; }
  if [ "$MPS" = "1" ]; then
    echo "   (MPS=1: start the daemon inside each pod BEFORE the workers, e.g. in the pod entrypoint:"
    echo "      export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-\$HOSTNAME; nvidia-cuda-mps-control -d )"
    echo "    verify with: echo get_default_active_thread_percentage | nvidia-cuda-mps-control"
  fi
  unset DRYRUN
  WORKERS="$WORKERS" PODS="$PODS" ./submit_pool.sh
  echo "   monitor with:  ./runbook.sh monitor"
}

phase_monitor(){
  # 3. Progress + the ONE integrity check that matters: did the AK twin apply?
  echo ">>> MONITOR"
  run "runai list jobs"
  echo "--- quick digest of whatever has landed ---"
  run "python scripts/resultio.py digest --in '$ALL'"
  echo "--- speed: seconds/round (last col) ---"
  run "for d in $RES/*/run.log; do echo \"\$d:\"; awk '\$2==\"R\" && \$3 ~ /^[0-9]/ {print \$3, \$NF}' \"\$d\" | tail -3; done"
  echo "   (live GPU utilisation: nvidia-smi dmon -s u   -- <30% util => launch/data-bound)"
}

phase_speedcheck(){
  # Optional: confirm fast_data + determinism-off are actually active and helping.
  echo ">>> SPEEDCHECK"
  echo "--- are the flags in the manifest? ---"
  run "grep -c -- '--fast_data' jobs.tsv; grep -c -- '--no_determinism' jobs.tsv"
  echo "--- did fast_data engage in a run? (look for the [fast_data] line in pod.log) ---"
  run "grep -h '\[fast_data\]' $RES/*/pod.log | sort | uniq -c | head"
  echo "--- seconds/round before vs after (compare an old A-family run.log to a new I-family) ---"
  run "for d in $RES/A1_honest_c100_rep0/run.log $RES/I_*_rep0/run.log; do [ -f \"\$d\" ] && { echo \"\$d:\"; awk '\$2==\"R\" && \$3 ~ /^[0-9]/ {s+=\$NF;n++} END{if(n)print \"  mean s/round =\", s/n}' \"\$d\"; }; done"
}

phase_validate_fastdata(){
  # Prove fast_data is distributionally equivalent before trusting the batch:
  # run one honest c100 seed the old way and the new way, compare the class floors.
  echo ">>> VALIDATE fast_data (1 seed, honest c100: old CPU vs new GPU path)"
  echo "   (a) confirm datasets.py uses RandomCrop(pad=4)+HFlip+Normalize for CIFAR --"
  echo "       fast_data replicates exactly that; a different old augmentation WOULD shift results."
  FAST_DATA=0 FAMILY=VAL_honest_cpu  ATTACK=none ROUNDS=50 ./submit_experiment.sh 14 0
  FAST_DATA=1 FAMILY=VAL_honest_fast ATTACK=none ROUNDS=50 ./submit_experiment.sh 14 0
  echo "   when both finish, the per-class floors should agree within seed noise:"
  echo "     python scripts/plots.py honest_lines --in '$ALL' --family VAL_honest_cpu  --tail 20 --out $OUT/val_cpu.png"
  echo "     python scripts/plots.py honest_lines --in '$ALL' --family VAL_honest_fast --tail 20 --out $OUT/val_fast.png"
  echo "   (also run: python -m faremark.fast_data \$DATA_ROOT  -- the built-in distribution self-test)"
}

# phase_fetch(){
#   # 4. Pull results to a LOCAL dir for plotting (plotting is offline). EDIT the
#   #    remote path to your cluster. Then re-run later phases with RES=<local>.
#   : "${REMOTE:?set REMOTE=user@cluster:/mnt/nfs/home/zu/results and RES=~/local/results}"
#   echo ">>> FETCH: $REMOTE -> $RES"
#   mkdir -p "$RES"
#   run "rsync -av --include='*/' --include='result.json' --include='pod.log' --exclude='*' '$REMOTE/' '$RES/'"
# }

phase_calibrate(){
  # 5. Recompute the frozen eta from each honest family (the value the timelines
  #    and separability use). The timelines also draw eta_loose (pooled) automatically.
  echo ">>> CALIBRATE eta from honest families"
  run "$DET calibrate --in '$ALL' --honest-family $HON --tail 20 --out $RES/eta_c100.json"
  run "$DET calibrate --in '$ALL' --honest-family E1_honest_niid_c100 --tail 20 --out $RES/eta_niid.json"
  run "$DET calibrate --in '$ALL' --honest-family F1_honest_nc200 --tail 20 --out $RES/eta_nc200.json"
  echo "   (eta tight is frozen 0.064; eta loose ~0.264 is drawn on every timeline)"
}

phase_plot(){
  mkdir -p "$OUT"
  echo ">>> PLOT -> $OUT"

  # --- A/D/E/F: the per-group standard figures (thresholds + timeline + sep) ---
  #     (C dropped -- sin-smoothing run failed, see R14; re-add 'C' once it's fixed)
  run "RES='$RES' OUT='$OUT' ./plot_groups.sh A D E F"

  # --- Class difficulty (the mechanism): floors, entropy/dominance vs accuracy ---
  run "$PL honest_lines   --in '$ALL' --family $HON --tail 20 --out $OUT/A1_class_floors.png"
  run "$PL class_probe    --in '$ALL' --family $HON --out $OUT"
  run "$PL eta_stability  --in '$ALL' --family $HON --out $OUT/A1_eta_stability"
  run "$ATH --in '$ALL' --family $HON --tail 20 --out $OUT/A1_thresholds"

  # --- ISOLATED same-class pairs (no shared-class conflict): honest A1 vs FR alone ---
  #     A4/AK put honest+FR on class 6 in ONE model (interference). Use A3's cid6 as
  #     the clean "FR alone on 6" and A1's cid6 as "honest alone on 6".
  run "$PAIR --honest_in '$RES/A1_honest_c100_rep*/result.json' --fr_in '$ALL' --family A3_reduced_c100_c36 --class 6 --out $OUT/iso_c6"
  run "$PAIR --honest_in '$RES/A1_honest_c100_rep*/result.json' --fr_in '$ALL' --family A3_reduced_c100_c36 --class 3 --out $OUT/iso_c3"
  run "$PAIR --honest_in '$RES/A1_honest_c100_rep*/result.json' --fr_in '$ALL' --family A2_reduced_c100_c17 --class 1 --out $OUT/iso_c1"
  run "$PAIR --honest_in '$RES/A1_honest_c100_rep*/result.json' --fr_in '$ALL' --family A2_reduced_c100_c17 --class 7 --out $OUT/iso_c7"
  # key-lottery counterpoints at the HARD class 6: A3's draw put FR ABOVE honest, but other
  # draws put it BELOW. A4 (own key) FR-c6~0.067 < honest 0.114 = cleaner; AK (same key) = the
  # controlled twin (only effort differs). Note A4/AK share class 6 in-model (not fully isolated).
  run "$PAIR --honest_in '$RES/A1_honest_c100_rep*/result.json' --fr_in '$ALL' --family A4_sameclass_c100_c6 --class 6 --out $OUT/iso_c6_A4_cleaner"
  run "$PAIR --honest_in '$RES/A1_honest_c100_rep*/result.json' --fr_in '$ALL' --family AK_sameclass_samekey_c6 --class 6 --out $OUT/iso_c6_AK_samekey"

  # --- E3 non-IID alpha sweep: reduced-vs-honest timeline per alpha (a03 dropped -- was a01's twin) ---
  for at in a01 a10; do
    run "$PL timeline --in '$ALL' --family E3_reduced_niid_c36_${at} \
         --honest_in '$ALL' --honest_family E3_honest_niid_c100_${at} --out $OUT/E3_${at}_timeline"
  done

  # --- H baselines: crude attacks the scheme was designed to catch (sanity) ---
  for fam in H3_prevmodel_c10 H4_gaussian_c10; do
    run "$DET separability --honest-in '$ALL' --honest-family H1_honest_c10 \
         --attack-in '$ALL' --attack-family $fam --tail 20 --per-class --emit $OUT/H_sep_${fam}.json"
  done

  # --- I adaptive-tap: timeline per family (two-eta lines drawn automatically) ---
  #     I families (SEEDS_I) + the new GROUP J graft suite (persistence / sawtooth / coast A-B).
  for fam in I0_smoke_always_cpc5_c36 \
             I_data_n0_c36 I_data_n1_c36 I_data_n5_c36 \
             I_when_threshold_c36 I_when_every_k_c36 \
             I_eta_oracle_c36 I_eta_self_c36 \
             I_coast_resend_c36 I_coast_decay_c36 I_maxcoast_m8_c36 \
             I_tight_eta0064_c36 \
             J0_gate_alwaystap_c36 \
             J1_persist_graft_p2_c36 J1_persist_graft_p3_c36 J1_persist_graft_p4_c36 \
             J1_persist_graft_p6_c36 J1_persist_graft_p12_c36 \
             J2_saw_graft_head_c36 J5_submarine_head_c36 \
             J3_coast_resend_p3_c36 J3_coast_decay_p3_c36 \
             J4_scope_graft_block_c36 J4_scope_graft_block2_c36; do
    run "$PL timeline --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON \
         --out $OUT/tap_${fam}"
  done

  # --- PER-FREE-RIDER split: one FIGURE per free-rider + same-class honest twin. ---
  #     The meeting's plot: each FR on its own axis, taps/coasts, server BER vs self-probe,
  #     with the honest GLOBAL mean AND the same-class honest twin (from $HON) on BOTH.
  #     Auto-generated for every submarine family (J2/J5 + the dynamic K-suite).
  for fam in J2_saw_graft_head_c36 J5_submarine_head_c36 J4_scope_graft_block2_c36 \
             K0_control_J2_c36 K1_selfeta_c36 K2_derivedmargin_c36 K3_dynwarmup_c36 \
             K4_alldyn_block2_c36; do
    run "$PL tap_perfr --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON \
         --out $OUT/tap_perfr_${fam}"
  done

  # --- K-SUITE timelines + dynamics (the dynamic-submarine 1-seed tests) ---
  for fam in K0_control_J2_c36 K1_selfeta_c36 K2_derivedmargin_c36 K3_dynwarmup_c36 \
             K4_alldyn_block2_c36; do
    run "$PL timeline --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON \
         --out $OUT/tap_${fam}"
    run "$PL tap_dynamics --in '$ALL' --family $fam --out $OUT/tap_dyn_${fam}"
  done

  # --- ACCURACY: global test acc, attack vs honest, + FR trigger-class acc. ---
  #     The 'Fig B' panels -- free-riders barely dent global accuracy while their own
  #     trigger class is the sacrificed cost. Auto-generated for the key attack families.
  for fam in J2_saw_graft_head_c36 J5_submarine_head_c36 \
             K0_control_J2_c36 K1_selfeta_c36 K4_alldyn_block2_c36 \
             A3_reduced_c100_c36 D1_reduced_c100_c36_n5 \
             E2_reduced_niid_c36 EA2_reduced_niid_distrib_c36; do
    run "$PL accuracy --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON \
         --out $OUT/accuracy_${fam}"
  done
  # non-IID accuracy uses the non-IID honest reference
  run "$PL accuracy --in '$ALL' --family E2_reduced_niid_c36 \
       --honest_in '$ALL' --honest_family E1_honest_niid_c100 --out $OUT/accuracy_E2_niid"

  # --- GPU-CYCLES SAVED: cumulative gpu_ms per round, each FR vs honest mean, ---
  #     + the running fraction-of-honest curve. Every free-rider family (with & without
  #     free-riders logs gpu_ms per round now, so the honest baseline is real).
  for fam in J2_saw_graft_head_c36 J5_submarine_head_c36 \
             K0_control_J2_c36 K1_selfeta_c36 K4_alldyn_block2_c36 \
             A3_reduced_c100_c36 D1_reduced_c100_c36_n5 \
             E2_reduced_niid_c36 EA2_reduced_niid_distrib_c36; do
    run "$PL gpu_savings --in '$ALL' --family $fam --out $OUT/gpu_savings_${fam}"
  done

  # --- NON-IID TRIGGER FAIRNESS: BER vs #trigger images held, round-robin vs ---
  #     distribution assignment overlaid. Shows whether starvation drives BER and
  #     whether distribution assignment removes it. Needs the wm_trigger_holdings field
  #     (runner patched) -- honest non-IID families carry it.
  run "$PL trigger_fairness --in '$ALL' --tail 20 \
       --families E1_honest_niid_c100 EA1_honest_niid_distrib_c100 \
       --out $OUT/trigger_fairness_niid"
  # per-alpha honest (round-robin only) as a fallback if EA hasn't run yet
  run "$PL trigger_fairness --in '$ALL' --family E1_honest_niid_c100 --tail 20 \
       --out $OUT/trigger_fairness_E1"

  # --- DIRICHLET reference heatmap (NEW): what the non-IID skew looks like per alpha. ---
  #     No result files needed (re-draws the datasets.py partition rule). --in is a placeholder.
  run "$PL dirichlet_dist --in '$ALL' --out $OUT/dirichlet_dist"

  # --- SAME-CLASS BER pair (iso_c*): the clean A4/AK replacement. Honest client from A1
  #     and the reduced FR from A2/A3, plotted as individual per-round BER lines on the
  #     SAME trigger class -- shows the FR mark is at least as clean as honest and the
  #     frozen eta flags the wrong one. (BER only; the iso_ACC accuracy version is
  #     `plots.py accuracy`.)
  SCP="python ../scripts/plot_sameclass_pair.py"
  run "$SCP --honest_in '$ALL' --fr_in '$ALL' --family A3_reduced_c100_c36 --class 6 --out $OUT/iso_A3_c6"
  run "$SCP --honest_in '$ALL' --fr_in '$ALL' --family A3_reduced_c100_c36 --class 3 --out $OUT/iso_A3_c3"
  run "$SCP --honest_in '$ALL' --fr_in '$ALL' --family A2_reduced_c100_c17 --class 1 --out $OUT/iso_A2_c1"
  run "$SCP --honest_in '$ALL' --fr_in '$ALL' --family A2_reduced_c100_c17 --class 7 --out $OUT/iso_A2_c7"
  # non-IID same-class pair (honest E1 vs reduced E2), + distribution variant (EA1 vs EA2)
  run "$SCP --honest_in '$ALL' --fr_in '$ALL' --family E2_reduced_niid_c36 --class 6 --out $OUT/iso_E2_c6"
  run "$SCP --honest_in '$ALL' --fr_in '$ALL' --family EA2_reduced_niid_distrib_c36 --class 6 --out $OUT/iso_EA2_c6"




  # --- plot: recall at a fixed honest FPR across every attack (one deployable eta) ---
  #     Crude c100 baseline H5 is the positive control (should light up); insiders stay ~0.
  run "$PL operating_point --in '$ALL' --honest_family $HON --tail 20        --families A2_reduced_c100_c17 A3_reduced_c100_c36 A4_sameclass_c100_c6                   AK_sameclass_samekey_c6 D1_reduced_c100_c36_n5 V2_tableV_attack_c36_tnm1                   H5_prevmodel_c100        --out $OUT/operating_point"

  # --- TAP DYNAMICS: FADE (rounds_between_taps) + RECOVERY (ber_drop_per_tap). ---
  #     Per-family trace for the runs where fade/recovery is the whole point, then the frontier.
  for fam in J2_saw_graft_head_c36 J5_submarine_head_c36 \
             J1_persist_graft_p6_c36 J1_persist_graft_p12_c36 \
             J3_coast_resend_p3_c36 J3_coast_decay_p3_c36 \
             J4_scope_graft_block_c36 J4_scope_graft_block2_c36 \
             I_when_threshold_c36 I_coast_resend_c36; do
    run "$PL tap_dynamics --in '$ALL' --family $fam --out $OUT/tap_dyn_${fam}"
  done
  run "$PL tap_dynamics --in '$ALL' --out $OUT/tap_frontier        --families I0_smoke_always_cpc5_c36 I_data_n0_c36 I_data_n1_c36 I_data_n5_c36                   I_when_threshold_c36 I_when_every_k_c36 I_eta_oracle_c36 I_eta_self_c36                   I_coast_resend_c36 I_coast_decay_c36 I_maxcoast_m8_c36 I_tight_eta0064_c36                   J0_gate_alwaystap_c36 J1_persist_graft_p2_c36 J1_persist_graft_p3_c36 J1_persist_graft_p4_c36                   J1_persist_graft_p6_c36 J1_persist_graft_p12_c36 J2_saw_graft_head_c36 J5_submarine_head_c36                   J3_coast_resend_p3_c36 J3_coast_decay_p3_c36 J4_scope_graft_block_c36 J4_scope_graft_block2_c36"

  # --- V2 Table V attack: FR BER vs #trigger-training-samples (overfit -> caught) ---
  #     TN values match run_now (10, 100, 500, and m1 = full trigger-class anchor).
  for tn in 10 100 500 m1; do
    run "$DET separability --honest-in '$ALL' --honest-family $HON \
         --attack-in '$ALL' --attack-family V2_tableV_attack_c36_tn${tn} \
         --tail 20 --per-class --emit $OUT/V2_sep_tn${tn}.json"
  done
  echo "   done. Headline numbers live in every *_sep.json:"
  echo "     overlap_coefficient          1.0 = honest & FR BER identical"
  echo "     best_threshold_balanced_error 0.5 = no threshold beats a coin"
}

phase_grade(){
  # 7. Paper reproduction tables.
  echo ">>> GRADE vs the FareMark paper"
  # Table IX capacity (cifar10, 50cl, client_train) + held-out twin => memorisation gap
  run "$PC --row t9 --in '$ALL' \
        --family F3_tableIX_c10_nc50 --heldout-family F3_tableIX_c10_nc50_heldout"
  # V1 Table VII / memorisation gap at N_T=50 (c100): client_train vs held-out class
  run "$PC --row c100 --in '$ALL' \
        --family V1_verify_client_train_nt50_c100 --heldout-family V1_verify_class_nt50_c100"
  # H fidelity: all-honest cifar10 matches Table I/II
  run "$PC --row c10 --in '$ALL' --family H1_honest_c10"
}

case "${1:-help}" in
  probe)     phase_probe ;;
  manifest)  phase_manifest ;;
  submit)    phase_submit ;;
  monitor)   phase_monitor ;;
  speedcheck) phase_speedcheck ;;
  validate)  phase_validate_fastdata ;;
  # fetch)     phase_fetch ;;
  calibrate) phase_calibrate ;;
  plot)      phase_plot ;;
  grade)     phase_grade ;;
  all-submit) phase_probe; phase_manifest; phase_submit ;;   # convenience: 0->1->2
  all-plot)   phase_calibrate; phase_plot; phase_grade ;;    # convenience: 5->6->7 (after fetch)
  *)
    cat <<USAGE
runbook.sh -- run phases in this order (wait for the cluster between 2 and 4):

  ON THE CLUSTER (has submit_experiment.sh + .env):
    ./runbook.sh probe       0. embedding sanity  (gates paper rows F3/Table IX)
    ./runbook.sh manifest    1. build jobs.tsv    (BATCH=$BATCH PAPER_OK=$PAPER_OK)
    ./runbook.sh submit      2. run the pool      (PODS=$PODS WORKERS=$WORKERS)
    ./runbook.sh monitor     3. progress 
       ... wait for jobs to finish ...

  LOCALLY (set RES=~/local/results):
    REMOTE=user@host:/mnt/nfs/home/zu/results RES=~/local/results ./runbook.sh fetch   4.
    RES=~/local/results ./runbook.sh calibrate   5.
    RES=~/local/results ./runbook.sh plot        6.
    RES=~/local/results ./runbook.sh grade        7.

  optional:   ./runbook.sh validate    fast_data A/B (old vs new floors agree within seed noise)
              ./runbook.sh speedcheck   confirm flags active + seconds/round before vs after
  shortcuts:  all-submit (0->1->2)   all-plot (5->6->7, after fetch)
  speed levers (on by default for this batch): FAST_DATA=1 (GPU loaders), DETERMINISM=0 (autotuner).
  batch size is NOT changed -- it is a hyperparameter that shifts the honest floor/eta, so keep 16
  for comparability with the A-H runs (see STATUS_AND_PLAN R6).
  groups this batch: BATCH=EFHIV -> E(E1,E2,E3-fixed), F3, H5, I*, V2 (A/D done, C excluded, J later).
  fast path: run just the stealth attack now with  MPS=1 WORKERS=8 SEEDS_I=0 BATCH=I ./runbook.sh submit
USAGE
    ;;
esac