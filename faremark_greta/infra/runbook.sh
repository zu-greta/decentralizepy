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
                                # The pool SKIPS any family whose result.json exists 
PAPER_OK="${PAPER_OK:-1}"         # 1 = also build the probe-gated paper rows (F3/Table IX)
FAST_DATA="${FAST_DATA:-1}"       # 1 = GPU-resident loaders (kills DataLoader fork storms)
DETERMINISM="${DETERMINISM:-0}"   # 0 = cuDNN autotuner on (~1.3-2x; stat. identical over seeds)
PODS="${PODS:-2}"; WORKERS="${WORKERS:-6}"   # batch=16 resnet18 barely uses an A100 - 6-8 workers per pod
MPS="${MPS:-1}"                 # 1 = start CUDA MPS on each pod so the many small processes share
                                # the SM scheduler instead of context-switching (big win at batch 16)
RES="${RES:-/mnt/nfs/home/zu/results}"   # cluster results (submit) OR local dir (plot); override for local
OUT="${OUT:-$RES/figs}"           # output folder should be RES/figs by default
ALL="$RES/*/result.json"
HON=A1_honest_c100                # the honest calibration family for c100/10cl
HONCLASS="${HONCLASS:-A1_honest_c100}"   # all-honest family for the per-client class-accuracy
                                  # check (class_acc). Default reuses A1; the `classacc`
                                  # subcommand makes a dedicated single-seed A0 run.

PL="python ../scripts/plots.py"
DET="python ../scripts/detection.py"
ATH="python ../scripts/plot_all_thresholds.py"
PC="python ../scripts/paper_check.py"
PAIR="python ../scripts/plot_sameclass_pair.py"
PHR="python ../scripts/plot_honest_per_round.py"   # per-ROUND honest BER + trig-acc (zero-acc check)
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
  # 3. Progress
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
  # =========================================================================
  # THESIS-RELEVANT PLOT SET:
  #   A  -- baseline HONEST-ONLY (per-class floors + why every eta fails)
  #   D  -- basic reduced free-riders (the +N price-of-invisibility spectrum)
  #   E  -- STARVED non-IID (round-robin: honest floored by empty trigger class)
  #   EA -- FAIR non-IID (distribution assignment: FR still matches same-class honest)
  #   K  -- the WORKING dynamic submarine (per-free-rider panels; the 2 FRs split)
  #   GPU-- effort/compute comparison for D/E/EA/K + the sharing-inflation check
  # =========================================================================

  # ===================== GROUP A -- honest baseline ONLY ====================
  # plot_groups.sh A does honest calibration + the reduced-attack pair/sep; honest baselines
  run "RES='$RES' OUT='$OUT' ./plot_groups.sh A"
  run "$PL honest_lines   --in '$ALL' --family $HON --tail 20 --out $OUT/A1_class_floors.png"   # per-class honest BER floors (the ceiling FRs hide under)
  run "$PL class_probe    --in '$ALL' --family $HON --out $OUT"                                  # WHY floors exist: entropy/dominance/trig-acc vs BER
  run "$PL class_difficulty --in '$ALL' --family $HON --out $OUT/A1_class_difficulty"            # per-class test-acc vs per-class BER correlation
  run "$PL eta_stability  --in '$ALL' --family $HON --out $OUT/A1_eta_stability"                 # seed variance of the calibrated eta
  run "$ATH --in '$ALL' --family $HON --tail 20 --out $OUT/A1_thresholds"                        # every candidate eta + the .md table (none separates)
  run "$ATH --in '$ALL' --family E1_honest_niid_c100 --tail 20 --out $OUT/E1_thresholds"          # non-IID: shows the tight rule needs ~24% honest FPR to be non-degenerate

  # --- per-client trigger-class accuracy check (all-honest run). One panel per
  #     client: its trigger-class test-acc vs the mean non-trigger-class acc vs global.
  run "$PL class_acc --in '$ALL' --family $HONCLASS --out $OUT/A0_class_acc"

  # --- per-ROUND honest BER + trigger-class accuracy (the "is the zero accuracy
  #     still there / is it suppression or starvation" time-series; complements the
  #     class_acc bar chart). Drawn for the IID honest (A1), the non-IID starved
  #     honest (E1), and the non-IID distribution honest (EA1). Prints the
  #     suppression-vs-starvation split per family.
  run "$PHR --in '$ALL' --family $HON --eta_tight 0.064 --eta_loose 0.264 --out $OUT/A1_honest_per_round"
  run "$PHR --in '$ALL' --family E1_honest_niid_c100  --eta_tight 0.161 --eta_loose 0.576 --out $OUT/E1_honest_per_round"
  run "$PHR --in '$ALL' --family EA1_honest_niid_distrib_c100 --eta_tight 0.161 --eta_loose 0.576 --out $OUT/EA1_honest_per_round"

  # --- reduced-attack timelines with FROZEN reference etas (fixes the stale
  #     'eta_loose=0.075' that older timelines drew; every figure now uses the same
  #     0.064 tight / 0.264 loose reference lines). A2 = easy classes, A3 = hard.
  run "$PL timeline --in '$ALL' --family A2_reduced_c100_c17 --honest_in '$ALL' --honest_family $HON --eta_tight 0.064 --eta_loose 0.264 --out $OUT/A2_easy_timeline"
  run "$PL timeline --in '$ALL' --family A3_reduced_c100_c36 --honest_in '$ALL' --honest_family $HON --eta_tight 0.064 --eta_loose 0.264 --out $OUT/A3_hard_timeline"

  # ===================== GROUP D -- basic reduced free-riders ================
  # plot_groups.sh D draws the D1 +N spectrum (price-of-invisibility) + sep.json.
  run "RES='$RES' OUT='$OUT' ./plot_groups.sh D"
  # same-class BER pair: reduced FR vs honest on the SAME class (medium 3, hard 6).
  SCP="python ../scripts/plot_sameclass_pair.py"
  run "$SCP --honest_in '$ALL' --fr_in '$ALL' --family A3_reduced_c100_c36 --class 3 --out $OUT/iso_A3_c3"
  run "$SCP --honest_in '$ALL' --fr_in '$ALL' --family A3_reduced_c100_c36 --class 6 --out $OUT/iso_A3_c6"

  # ===================== GROUP E -- STARVED non-IID =========================
  # plot_groups.sh E: honest non-IID floors + reduced-vs-honest timeline/sep.
  run "RES='$RES' OUT='$OUT' ./plot_groups.sh E"
  # alpha severity sweep (starvation gets worse as alpha shrinks).
  for at in a01 a10; do
    run "$PL timeline --in '$ALL' --family E3_reduced_niid_c36_${at} \
         --honest_in '$ALL' --honest_family E3_honest_niid_c100_${at} --out $OUT/E3_${at}_timeline"
  done
  # starvation itself: honest BER vs #trigger images held (round-robin).
  run "$PL trigger_fairness --in '$ALL' --family E1_honest_niid_c100 --tail 20 --out $OUT/trigger_fairness_E1"
  # same-class BER pair, non-IID starved (honest E1 vs reduced E2 on class 6).
  # iso pair MUST use the non-IID honest twin (E1), NOT the IID A1 -- else it compares
  # an IID honest client against a non-IID free-rider (mismatched partitions).
  run "$SCP --honest_in '$RES/E1_honest_niid_c100_rep*/result.json' --fr_in '$ALL' --family E2_reduced_niid_c36 --class 6 --out $OUT/iso_E2_c6"

  # ===================== GROUP EA -- FAIR non-IID ===========================
  # The distribution-aware fair comparison: each reduced FR vs the honest client
  # assigned the SAME trigger class. If the FR still matches the (now un-starved)
  # honest twin, removing starvation did NOT open a separating threshold.
  for pair in "EA2_reduced_niid_distrib_c36 EA1_honest_niid_distrib_c100"; do
    set -- $pair
    run "$PL ea_fair --in '$ALL' --family $1 --honest_family $2 --out $OUT/ea_fair_$1"
  done
  # round-robin (starved) vs distribution (fair) overlaid: does assignment remove starvation?
  run "$PL trigger_fairness --in '$ALL' --tail 20 \
       --families E1_honest_niid_c100 EA1_honest_niid_distrib_c100 --out $OUT/trigger_fairness_niid"
  # same-class BER pair under distribution assignment (honest EA1 vs reduced EA2, class 6).
  # iso pair MUST use the non-IID distribution-aware honest twin (EA1), NOT the IID A1.
  run "$SCP --honest_in '$RES/EA1_honest_niid_distrib_c100_rep*/result.json' --fr_in '$ALL' --family EA2_reduced_niid_distrib_c36 --class 6 --out $OUT/iso_EA2_c6"
  # what the non-IID skew looks like 
  run "$PL dirichlet_dist --in '$ALL' --out $OUT/dirichlet_dist"

  # ===================== GROUP K -- the submarine ===================
  # PER-FREE-RIDER split: ONE FIGURE PER FREE-RIDER (cid3 and cid6 on SEPARATE
  # panels) -- they do not tap together, so never collapse them. Each panel carries
  # the honest GLOBAL mean AND the same-class honest twin (from $HON) for the
  # "does it blend in on its own class" read. 
  for fam in K4_alldyn_block2_c36 ${K4B:+K4b_alldyn_block2_fulldata_c36}; do
    run "$PL tap_perfr --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON \
         --out $OUT/tap_perfr_${fam}"
  done
  # accuracy: the FR barely dents global test-acc while its own trigger class pays.
  for fam in K4_alldyn_block2_c36 K5_alldyn_full_c36 ${K4B:+K4b_alldyn_block2_fulldata_c36}; do
    run "$PL accuracy --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON \
         --out $OUT/accuracy_${fam}"
  done
  # ISOLATED same-class twin: submarine FR vs the honest client on the SAME class
  # (from A1). This is the "does it match the honest twin" money-read for the
  # submarine. cid3 -> class 3 (easy: matches/beats twin, sawtooth);
  # cid6 -> class 6 (hard: cleaner than reduced but stays ABOVE the twin -- the
  # data-limited hard-class finding).
  run "$PAIR --honest_in '$RES/${HON}_rep*/result.json' --fr_in '$RES/K4_alldyn_block2_c36_rep*/result.json' --class 3 --eta_tight 0.064 --eta_loose 0.264 --out $OUT/iso_K4_c3"
  run "$PAIR --honest_in '$RES/${HON}_rep*/result.json' --fr_in '$RES/K4_alldyn_block2_c36_rep*/result.json' --class 6 --eta_tight 0.064 --eta_loose 0.264 --out $OUT/iso_K4_c6"
  run "$PAIR --honest_in '$RES/${HON}_rep*/result.json' --fr_in '$RES/K5_alldyn_full_c36_rep*/result.json' --class 3 --eta_tight 0.064 --eta_loose 0.264 --out $OUT/iso_K5_c3"
  run "$PAIR --honest_in '$RES/${HON}_rep*/result.json' --fr_in '$RES/K5_alldyn_full_c36_rep*/result.json' --class 6 --eta_tight 0.064 --eta_loose 0.264 --out $OUT/iso_K5_c6"

  # ===================== GPU CYCLES -- effort comparison ====================
  # cumulative gpu_ms / samples per round, each FR vs the honest mean, for the
  # kept attack families (D, E, EA, K).
  for fam in D1_reduced_c100_c36_n5 E2_reduced_niid_c36 EA2_reduced_niid_distrib_c36 \
             K4_alldyn_block2_c36 K5_alldyn_full_c36 K5_alldyn_full_c36 ${K4B:+K4b_alldyn_block2_fulldata_c36}; do
    run "$PL gpu_savings --in '$ALL' --family $fam --out $OUT/gpu_savings_${fam}"
  done
  # sharing-inflation check: single-tenant (WORKERS=1) vs shared saved-% ratio.
  run "$PL gpu_inflation --in '$ALL' --family K4_alldyn_block2_c36 --out $OUT/gpu_inflation_K4_alldyn_block2_c36"

  # #########################################################################
  # ###  COMMENTED OUT -- not part of the current thesis-relevant set.    ###
  # ###  Uncomment a block to restore.)                                   ###
  # #########################################################################
  #
  # # --- plot_groups F (capacity, >clients-than-classes) ---
  # run "RES='$RES' OUT='$OUT' ./plot_groups.sh F"
  #
  # # --- ISOLATED same-class pairs via $PAIR (A4/AK key-lottery + A2 easy classes) ---
  # run "$PAIR --honest_in '$RES/A1_honest_c100_rep*/result.json' --fr_in '$ALL' --family A3_reduced_c100_c36 --class 6 --out $OUT/iso_c6"
  # run "$PAIR --honest_in '$RES/A1_honest_c100_rep*/result.json' --fr_in '$ALL' --family A3_reduced_c100_c36 --class 3 --out $OUT/iso_c3"
  # run "$PAIR --honest_in '$RES/A1_honest_c100_rep*/result.json' --fr_in '$ALL' --family A2_reduced_c100_c17 --class 1 --out $OUT/iso_c1"
  # run "$PAIR --honest_in '$RES/A1_honest_c100_rep*/result.json' --fr_in '$ALL' --family A2_reduced_c100_c17 --class 7 --out $OUT/iso_c7"
  # run "$PAIR --honest_in '$RES/A1_honest_c100_rep*/result.json' --fr_in '$ALL' --family A4_sameclass_c100_c6 --class 6 --out $OUT/iso_c6_A4_cleaner"
  # run "$PAIR --honest_in '$RES/A1_honest_c100_rep*/result.json' --fr_in '$ALL' --family AK_sameclass_samekey_c6 --class 6 --out $OUT/iso_c6_AK_samekey"
  # run "$SCP --honest_in '$ALL' --fr_in '$ALL' --family A2_reduced_c100_c17 --class 1 --out $OUT/iso_A2_c1"
  # run "$SCP --honest_in '$ALL' --fr_in '$ALL' --family A2_reduced_c100_c17 --class 7 --out $OUT/iso_A2_c7"
  # run "$SCP --honest_in '$ALL' --fr_in '$ALL' --family EA2b_reduced_niid_distrib_pin_c36 --class 6 --out $OUT/iso_EA2b_c6"
  #
  # # --- H crude-baseline separability (positive controls) ---
  # for fam in H3_prevmodel_c10 H4_gaussian_c10; do
  #   run "$DET separability --honest-in '$ALL' --honest-family H1_honest_c10 \
  #        --attack-in '$ALL' --attack-family $fam --tail 20 --per-class --emit $OUT/H_sep_${fam}.json"
  # done
  #
  # # --- I/J exploratory timelines (superseded by K) ---
  # for fam in I0_smoke_always_cpc5_c36 I_data_n0_c36 I_data_n1_c36 I_data_n5_c36 \
  #            I_when_threshold_c36 I_when_every_k_c36 I_eta_oracle_c36 I_eta_self_c36 \
  #            I_coast_resend_c36 I_coast_decay_c36 I_maxcoast_m8_c36 I_tight_eta0064_c36 \
  #            J0_gate_alwaystap_c36 J1_persist_graft_p2_c36 J1_persist_graft_p3_c36 \
  #            J1_persist_graft_p4_c36 J1_persist_graft_p6_c36 J1_persist_graft_p12_c36 \
  #            J2_saw_graft_head_c36 J5_submarine_head_c36 J3_coast_resend_p3_c36 \
  #            J3_coast_decay_p3_c36 J4_scope_graft_block_c36 J4_scope_graft_block2_c36; do
  #   run "$PL timeline --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --out $OUT/tap_${fam}"
  # done
  # # J-suite per-FR + K0-K3 ablations (K0=J2 control, K1 self-eta, K2 margin, K3 warmup)
  # for fam in J2_saw_graft_head_c36 J5_submarine_head_c36 J4_scope_graft_block2_c36 \
  #            K0_control_J2_c36 K1_selfeta_c36 K2_derivedmargin_c36 K3_dynwarmup_c36; do
  #   run "$PL tap_perfr --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --out $OUT/tap_perfr_${fam}"
  # done
  # # K-suite timelines (redundant with tap_perfr) + tap_dynamics (collapses the 2 FRs -> not wanted)
  # for fam in K0_control_J2_c36 K1_selfeta_c36 K2_derivedmargin_c36 K3_dynwarmup_c36 \
  #            K4_alldyn_block2_c36 K5_selfeta_derivedmargin_head_c36 K6_selfeta_tailfix_head_c36 K6_full_submarine_head_c36; do
  #   run "$PL timeline --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --out $OUT/tap_${fam}"
  #   run "$PL tap_dynamics --in '$ALL' --family $fam --out $OUT/tap_dyn_${fam}"
  # done
  # # accuracy for the non-K families (secondary)
  # for fam in J2_saw_graft_head_c36 J5_submarine_head_c36 K0_control_J2_c36 K1_selfeta_c36 \
  #            A3_reduced_c100_c36 D1_reduced_c100_c36_n5 E2_reduced_niid_c36 \
  #            EA2_reduced_niid_distrib_c36 EA2b_reduced_niid_distrib_pin_c36; do
  #   run "$PL accuracy --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --out $OUT/accuracy_${fam}"
  # done
  # run "$PL accuracy --in '$ALL' --family E2_reduced_niid_c36 --honest_in '$ALL' --honest_family E1_honest_niid_c100 --out $OUT/accuracy_E2_niid"
  # # gpu_savings for the non-kept families
  # for fam in J2_saw_graft_head_c36 J5_submarine_head_c36 K0_control_J2_c36 K1_selfeta_c36 \
  #            A3_reduced_c100_c36 EA2b_reduced_niid_distrib_pin_c36; do
  #   run "$PL gpu_savings --in '$ALL' --family $fam --out $OUT/gpu_savings_${fam}"
  # done
  # # gpu_inflation for the other candidate winners
  # for fam in K6_full_submarine_head_c36 J2_saw_graft_head_c36; do
  #   run "$PL gpu_inflation --in '$ALL' --family $fam --out $OUT/gpu_inflation_${fam}"
  # done
  #
  # --- operating_point: recall @ fixed honest FPR -- THE headline negative-result summary.
  #     Insider attacks (reduced A2/A3/D1 + submarine K4) sit near 0 recall at the honest-FPR
  #     where the crude positive control (H5) is caught at ~1.0. A4/AK removed; V2 not run.
  run "$PL operating_point --in '$ALL' --honest_family $HON --tail 20 \
       --families A2_reduced_c100_c17 A3_reduced_c100_c36 D1_reduced_c100_c36_n5 \
                 K4_alldyn_block2_c36 H5_prevmodel_c100 --out $OUT/operating_point"
  # positive-control separability (H5 crude FR should split cleanly from honest):
  run "$DET separability --honest-in '$ALL' --honest-family $HON \
       --attack-in '$ALL' --attack-family H5_prevmodel_c100 --tail 20 --per-class --emit $OUT/H5_sep.json"
  #
  # # --- tap_dynamics per-family + frontier (I/J fade/recovery exploration) ---
  # for fam in J2_saw_graft_head_c36 J5_submarine_head_c36 J1_persist_graft_p6_c36 \
  #            J1_persist_graft_p12_c36 J3_coast_resend_p3_c36 J3_coast_decay_p3_c36 \
  #            J4_scope_graft_block_c36 J4_scope_graft_block2_c36 I_when_threshold_c36 I_coast_resend_c36; do
  #   run "$PL tap_dynamics --in '$ALL' --family $fam --out $OUT/tap_dyn_${fam}"
  # done
  #
  # # --- V2 Table V positive control (trigger-sample overfit -> caught) ---
  # for tn in 10 100 500 m1; do
  #   run "$DET separability --honest-in '$ALL' --honest-family $HON \
  #        --attack-in '$ALL' --attack-family V2_tableV_attack_c36_tn${tn} \
  #        --tail 20 --per-class --emit $OUT/V2_sep_tn${tn}.json"
  # done

  echo "   done -> $OUT  (thesis-relevant set: A honest / D reduced / E starved-niid / EA fair-niid / K submarine / GPU)"
}

phase_classacc(){
  # Single-seed ALL-HONEST run + the per-client class-accuracy check.
  # Purpose: rule out "this client's watermark BER is high only because its trigger
  # class is intrinsically hard to classify". For each honest client it plots, on its
  # OWN panel: (a) test-acc on its trigger class, (b) mean test-acc on the other
  # (non-trigger) classes, (c) the global test-acc -- so a hard trigger-class draw is
  # visible as a low (a) bar regardless of the watermark.
  #
  #   ./runbook.sh classacc            # submit the single-seed honest run, then plot
  #   PLOT_ONLY=1 ./runbook.sh classacc  # skip the run, just (re)plot A0/$HONCLASS
  local FAM="A0_classacc_honest_c100"
  mkdir -p "$OUT"
  if [ -z "${PLOT_ONLY:-}" ]; then
    echo ">>> CLASSACC: single-seed all-honest run ($FAM, seed 0)"
    run "env ATTACK=none NUM_FREE_RIDERS=0 DS=c100 NUM_CLIENTS=10 ROUNDS=50 \
         FAMILY='$FAM' NOTE='A0 single-seed all-honest, per-client class-acc check' \
         ./submit_experiment.sh 14 0"
    echo "   (when it lands in \$RES, re-run with PLOT_ONLY=1 to plot, or just: ./runbook.sh classacc)"
  fi
  echo ">>> CLASSACC PLOT -> $OUT"
  # plot the dedicated A0 run if present, else fall back to $HONCLASS (=A1 by default).
  run "$PL class_acc --in '$ALL' --family $FAM     --out $OUT/A0_class_acc"
  run "$PL class_acc --in '$ALL' --family $HONCLASS --out $OUT/A1_class_acc"
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
  classacc)  phase_classacc ;;
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
    RES=~/local/results ./runbook.sh plot        6.   (thesis-relevant set: A/D/E/EA/K/GPU)
    RES=~/local/results ./runbook.sh grade        7.

  optional:   ./runbook.sh classacc   single-seed all-honest run + per-client trigger-class
                                       accuracy check (rules out class-difficulty confounds).
                                       PLOT_ONLY=1 to re-plot without re-running.
              ./runbook.sh validate    fast_data A/B (old vs new floors agree within seed noise)
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