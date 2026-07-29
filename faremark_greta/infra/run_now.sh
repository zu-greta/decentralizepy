#!/usr/bin/env bash
# =============================================================================
# run_now.sh -- builds jobs.tsv for the experiment plan (see STATUS_AND_PLAN.md).
#
#   ./run_now.sh              # build manifest for the default safe set (submits nothing)
#   ./run_now.sh A            # build ONLY group A (proven baseline, incl. AK)
#   ./run_now.sh ACDEFH       # everything except the probe-gated paper rows
#   PAPER_OK=1 ./run_now.sh ACDEFH   # ALSO build F3/Table-IX paper-repro rows
#   unset DRYRUN; WORKERS=6 PODS=2 ./submit_pool.sh
#
# Groups: A=proven baseline (+AK same-key)   C=smoothing   D=+N spectrum
#         E=non-iid (+alpha sweep)   F=capacity (+Table IX)   H=paper baselines
#         I=adaptive-tap single-knob sweeps   J=adaptive-tap multi-knob combos
#         V=trigger-sample mode + Table V     P=paper reproduction (probe-gated)
# NEXT BATCH (does not disturb the running A/C/D/E/F/H jobs -- new families only):
#         PAPER_OK=1 ./run_now.sh IJVF   then   unset DRYRUN; WORKERS=6 PODS=2 ./submit_pool.sh
#
# All families use the KNOWN-GOOD cifar100/10-client base unless the point is to
# vary it. NUM_WORKERS=0 (DataLoader worker churn, see prior analysis).
#
# ETA NOTE: WM_ETA_FIXED draws the dashed detection line on the timelines and
# fills the live FPR/recall columns. It does NOT decide separability -- the
# offline all-thresholds sweep (detection.py / plot_all_thresholds.py) sweeps
# every rule after the fact and is what the thesis cites. The frozen values below
# are each family's honest calibration; recompute from the 6-seed honest runs with
#   python scripts/detection.py calibrate --in 'results/A1_honest_c100_rep*/result.json' --tail 20
# =============================================================================
set -uo pipefail
export DRYRUN=1 JOBS_FILE="${JOBS_FILE:-jobs.tsv}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
WANT="${1:-ACDEFH}"                      # which groups to build; default = the safe set
PAPER_OK="${PAPER_OK:-0}"                # 1 => also build the probe-gated paper rows (F3/Table IX)
rm -f "$JOBS_FILE"
echo "== building $JOBS_FILE  groups=[$WANT]  PAPER_OK=$PAPER_OK  NUM_WORKERS=$NUM_WORKERS =="
has(){ [[ "$WANT" == *"$1"* ]]; }

# ---------------------------------------------------------------------------
# GROUP A -- proven baseline (cifar100, 10 clients). 
#   A1 honest x6   A2 reduced easy   A3 reduced hard   A4 sameclass (own key)
#   AK sameclass SAME KEY  <-- the controlled effort-only isolation
# Note: config 14 defaults num_free_riders=2, but FREE_RIDER_IDS pins the exact
# cids, so free_rider_ids WINS (resolve_free_riders) -> A4/AK have one free-rider.
# ---------------------------------------------------------------------------
# NOTE: group A run already done - just need to plot
if has A; then
  echo "   (group A done already)"
#   for s in 0 1 2 3 4 5; do
#     env ATTACK=none NUM_FREE_RIDERS=0 DS=c100 NUM_CLIENTS=10 ROUNDS=50 \
#         FAMILY="A1_honest_c100" NOTE="A1 honest baseline (known-good config)" \
#         ./submit_experiment.sh 14 "$s"
#   done
#   for s in 0 1 2; do
#     env ATTACK=reduced FREE_RIDER_IDS=1,7 AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 \
#         AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
#         FAMILY="A2_reduced_c100_c17" NOTE="A2 reduced +5 easy classes 1,7" \
#         ./submit_experiment.sh 14 "$s"
#   done
#   for s in 0 1 2; do
#     env ATTACK=reduced FREE_RIDER_IDS=3,6 AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 \
#         AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
#         FAMILY="A3_reduced_c100_c36" NOTE="A3 reduced +5 hard classes 3,6" \
#         ./submit_experiment.sh 14 "$s"
#   done
#   for s in 0 1 2; do
#     env ATTACK=reduced FREE_RIDER_IDS=0 TRIGGER_CLASS_MAP="0:6" AUTOP_COMMON_PER_CLASS=5 \
#         AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
#         FAMILY="A4_sameclass_c100_c6" NOTE="A4 FR shares class 6 with honest (own key)" \
#         ./submit_experiment.sh 14 "$s"
#   done
#   # ---- AK: SAME trigger class AND SAME key/message -----------------------
#   # trigger_class_map 0:6  -> FR cid0 sits on class 6 (with honest cid6)
#   # wm_key_twins    0:6    -> FR cid0 derives its key M and message B from cid6
#   for s in 0 1 2; do
#     env ATTACK=reduced FREE_RIDER_IDS=0 TRIGGER_CLASS_MAP="0:6" WM_KEY_TWINS="0:6" \
#         AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
#         WM_ETA_FIXED=0.064 ROUNDS=50 \
#         FAMILY="AK_sameclass_samekey_c6" NOTE="AK FR shares class 6 AND key/msg with honest cid6" \
#         ./submit_experiment.sh 14 "$s"
#   done
fi

# ---------------------------------------------------------------------------
# GROUP C -- smoothing function (does a different f() move the floors?)
#   C1 = sin smoothing.  C2 (entropy/dominance vs accuracy) is OFFLINE: it is
#   computed from the diagnostics already stored in the A1 result.json files by
#   the class_probe script -- no new run needed. See STATUS_AND_PLAN.md Group C.
# ---------------------------------------------------------------------------
# DISABLED pending the sin-smoothing crash fix 
if has C; then
  echo "   (C1 sin-smoothing DISABLED -- see STATUS_AND_PLAN R14; nothing queued for C)"
  # for s in 0 1 2; do
  #   env ATTACK=none NUM_FREE_RIDERS=0 WM_F=sin WM_ALPHA=1.5708 ROUNDS=50 \
  #       FAMILY="C1_honest_sin_c100" NOTE="C1 sin smoothing alpha=pi/2" \
  #       ./submit_experiment.sh 14 "$s"
  # done
fi

# ---------------------------------------------------------------------------
# GROUP D -- +N free-riding spectrum at the hard classes
#   N = images kept per common class; -1 = full shard (still a free-rider)
# ---------------------------------------------------------------------------
if has D; then
  echo "   (group D done already - +1/common class is the plateau)"
  # for N in -1 0 1 2 5 10 25 50; do
  #   for s in 0 1 2; do
  #     env ATTACK=reduced FREE_RIDER_IDS=3,6 AUTOP_COMMON_PER_CLASS=$N \
  #         AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
  #         FAMILY="D1_reduced_c100_c36_n${N}" NOTE="D1 +N spectrum N=$N" \
  #         ./submit_experiment.sh 14 "$s"
  #   done
  # done
fi

# ---------------------------------------------------------------------------
# GROUP E -- non-IID (Dirichlet)
#   E1 honest a=0.5   E2 reduced a=0.5   E3 reduced ALPHA SWEEP
# ---------------------------------------------------------------------------
if has E; then
  for s in 0 1 2; do
    env ATTACK=none NUM_FREE_RIDERS=0 PART=niid DIRICHLET_ALPHA=0.5 ROUNDS=50 \
        FAMILY="E1_honest_niid_c100" NOTE="E1 non-iid honest a=0.5" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in 0 1 2; do
    env ATTACK=reduced FREE_RIDER_IDS=3,6 PART=niid DIRICHLET_ALPHA=0.5 \
        AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
        WM_ETA_FIXED=0.161 ROUNDS=50 \
        FAMILY="E2_reduced_niid_c36" NOTE="E2 non-iid reduced hard a=0.5" \
        ./submit_experiment.sh 14 "$s"
  done
  # E3 -- severity sweep. a=0.5 already covered by E1/E2, so with {0.1, 1.0} skew levels (0.1 / 0.5 / 1.0)
  for A in 0.1 1.0; do
    ATAG="a$(printf '%s' "$A" | tr -d '.')"
    for s in 0 1 2; do
      env ATTACK=none NUM_FREE_RIDERS=0 PART=niid DIRICHLET_ALPHA=$A ROUNDS=50 \
          FAMILY="E3_honest_niid_c100_${ATAG}" NOTE="E3 non-iid honest alpha=$A" \
          ./submit_experiment.sh 14 "$s"
    done
    for s in 0 1 2; do
      env ATTACK=reduced FREE_RIDER_IDS=3,6 PART=niid DIRICHLET_ALPHA=$A \
          AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
          WM_ETA_FIXED=0.161 ROUNDS=50 \
          FAMILY="E3_reduced_niid_c36_${ATAG}" NOTE="E3 non-iid reduced hard alpha=$A" \
          ./submit_experiment.sh 14 "$s"
    done
  done
fi

# ---------------------------------------------------------------------------
# GROUP F -- more clients than classes (200 clients). needs more rounds to train.
#   F1 honest 200cl   F2 reduced 200cl shared 6,7   F3 Table IX (paper repro)
#   F3 is PROBE-GATED: it reproduces FareMark Table IX (cifar10, 50 clients,
#   client_train trigger mode -- verify-on-training-images = memorisation). It is
#   only built when PAPER_OK=1 
# ---------------------------------------------------------------------------
if has F; then
  for s in 0 1 2; do
    env ATTACK=none NUM_FREE_RIDERS=0 NUM_CLIENTS=200 ROUNDS=100 \
        FAMILY="F1_honest_nc200" NOTE="F1 200 clients honest, 100 rounds" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in 0 1 2; do
    env ATTACK=reduced NUM_CLIENTS=200 FREE_RIDER_IDS=106,107 AUTOP_COMMON_PER_CLASS=5 \
        AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.384 ROUNDS=100 \
        FAMILY="F2_reduced_nc200_c67" NOTE="F2 200cl reduced, shared classes 6,7" \
        ./submit_experiment.sh 14 "$s"
  done
  if [ "$PAPER_OK" = "1" ]; then
    for s in 0 1 2; do
      env ATTACK=none NUM_FREE_RIDERS=0 NUM_CLIENTS=50 \
          WM_TRIGGER_MODE=client_train WM_NUM_TRIGGERS=50 ROUNDS=50 \
          FAMILY="F3_tableIX_c10_nc50" NOTE="F3 Table IX cifar10 50cl client_train (paper)" \
          ./submit_experiment.sh 11 "$s"
    done
    for s in 0 1 2; do
      env ATTACK=none NUM_FREE_RIDERS=0 NUM_CLIENTS=50 \
          WM_TRIGGER_MODE=class WM_NUM_TRIGGERS=50 ROUNDS=50 \
          FAMILY="F3_tableIX_c10_nc50_heldout" NOTE="F3 Table IX held-out twin (generalisation)" \
          ./submit_experiment.sh 11 "$s"
    done
  else
    echo "   (F3/Table IX skipped -- set PAPER_OK=1 to build the paper-repro rows)"
  fi
fi

# ---------------------------------------------------------------------------
# GROUP H -- paper baselines 
#   H1 honest cifar10 (fidelity)          H3 previous-models FR (crude, caught)
#   H2 honest cifar100 == A1 (reference)  H4 gaussian-noise FR (crude, caught)
# ---------------------------------------------------------------------------
if has H; then
  echo "   (group H done already)"
  # for s in 0 1 2; do
  #   env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 \
  #       FAMILY="H1_honest_c10" NOTE="H1 fidelity: all-honest watermark cifar10" \
  #       ./submit_experiment.sh 11 "$s"
  # done
  # # H2 (honest cifar100) is identical to A1 -- do not rerun; cite A1_honest_c100.
  # for s in 0 1 2; do
  #   env ATTACK=previous_models NUM_FREE_RIDERS=2 WM_ETA_FIXED=0.25 ROUNDS=50 \
  #       FAMILY="H3_prevmodel_c10" NOTE="H3 crude FR: previous-models attack cifar10" \
  #       ./submit_experiment.sh 11 "$s"
  # done
  # for s in 0 1 2; do
  #   env ATTACK=gaussian NUM_FREE_RIDERS=2 NOISE_SIGMA=0.1 WM_ETA_FIXED=0.25 ROUNDS=50 \
  #       FAMILY="H4_gaussian_c10" NOTE="H4 crude FR: gaussian-noise attack cifar10" \
  #       ./submit_experiment.sh 11 "$s"
  # done
  # # H5: crude previous-models FR on CIFAR-100 -- the same dataset positive control
  # # for the operating-point plot (calibrated on A1_honest_c100). Should be caught.
  # for s in 0 1 2; do
  #   env ATTACK=previous_models FREE_RIDER_IDS=6 WM_ETA_FIXED=0.064 ROUNDS=50 \
  #       FAMILY="H5_prevmodel_c100" NOTE="H5 crude FR previous-models on c100 (money-plot control)" \
  #       ./submit_experiment.sh 14 "$s"
  # done
fi

# ---------------------------------------------------------------------------
# GROUP I -- ADAPTIVE-TAP single-knob sweeps (attack="adaptive_tap").
# Base: FR at hard classes 3,6, oracle eta = 0.064, warmup 12 / calib 4.
# SEEDS_I trims the seed count for these exploratory sweeps (bump to "0 1 2").
# ---------------------------------------------------------------------------
if has I; then
  SEEDS_I="${SEEDS_I:-0}"
  base="ATTACK=adaptive_tap FREE_RIDER_IDS=3,6 WM_ETA_FIXED=0.064 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 ROUNDS=50 \
        TAP_WHEN=threshold TAP_PERIOD=3 TAP_MARGIN=0.0 TAP_DATA_CPC=0 TAP_SCOPE=full TAP_ETA_SOURCE=oracle TAP_COAST_MODE=resend"
  # --- when to tap ---
  for W in threshold every_k; do
    for s in $SEEDS_I; do
      env $base TAP_WHEN=$W TAP_PERIOD=3 \
          FAMILY="I_when_${W}_c36" NOTE="I knob=when val=$W" ./submit_experiment.sh 14 "$s"
    done
  done
  # --- margin under eta ---
  for M in 0.0 0.10; do
    MT="m$(printf '%s' "$M" | tr -d '.')"
    for s in $SEEDS_I; do
      env $base TAP_MARGIN=$M \
          FAMILY="I_margin_${MT}_c36" NOTE="I knob=margin val=$M" ./submit_experiment.sh 14 "$s"
    done
  done
  # --- data per tap (cpc) ---
  # 0 (trigger-only), 1 (plateau) - from group D, -1 (full shard). 
  for D in 0 1 -1; do
    for s in $SEEDS_I; do
      env $base TAP_DATA_CPC=$D \
          FAMILY="I_data_n${D}_c36" NOTE="I knob=data_cpc val=$D" ./submit_experiment.sh 14 "$s"
    done
  done
  # --- model scope per tap ---
  for SC in full block2; do
    for s in $SEEDS_I; do
      env $base TAP_SCOPE=$SC \
          FAMILY="I_scope_${SC}_c36" NOTE="I knob=scope val=$SC" ./submit_experiment.sh 14 "$s"
    done
  done
  # --- threshold estimation: oracle vs self ---
  for ES in oracle self; do
    for s in $SEEDS_I; do
      env $base TAP_ETA_SOURCE=$ES TAP_ETA_K=3.0 \
          FAMILY="I_eta_${ES}_c36" NOTE="I knob=eta_source val=$ES" ./submit_experiment.sh 14 "$s"
    done
  done
  # --- how to free-ride between taps ---
  for CM in resend decay; do
    for s in $SEEDS_I; do
      env $base TAP_COAST_MODE=$CM \
          FAMILY="I_coast_${CM}_c36" NOTE="I knob=coast_mode val=$CM" ./submit_experiment.sh 14 "$s"
    done
  done
  # --- force-tap cap (max consecutive coasts) ---
  # NOTE: removed for now
  # for MC in 1 2 4 999; do
  #   for s in $SEEDS_I; do
  #     env $base TAP_MAX_COAST=$MC \
  #         FAMILY="I_maxcoast_${MC}_c36" NOTE="I knob=max_coast val=$MC" ./submit_experiment.sh 14 "$s"
  #   done
  # done
fi

# ---------------------------------------------------------------------------
# GROUP J -- ADAPTIVE-TAP multi-knob combos (move several at once) -> run after group I
# ---------------------------------------------------------------------------
# if has J; then
#   SEEDS_J="${SEEDS_J:-0 1 2}"
#   base="ATTACK=adaptive_tap FREE_RIDER_IDS=3,6 WM_ETA_FIXED=0.064 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 ROUNDS=50"
  # # J1: cheapest invisible attacker -- self eta, big margin, tiny data, head-only, coast hard
  # for s in $SEEDS_J; do
  #   env $base TAP_ETA_SOURCE=self TAP_MARGIN=0.05 TAP_DATA_CPC=0 TAP_SCOPE=head \
  #       TAP_WHEN=threshold TAP_MAX_COAST=6 TAP_COAST_MODE=resend \
  #       FAMILY="J1_cheapest_c36" NOTE="J1 self+margin.05+trigonly+head+coast6" ./submit_experiment.sh 14 "$s"
  # done
  # # J2: aggressive holder -- oracle eta, tap every round, full data+scope (upper bound on stealth cost)
  # for s in $SEEDS_J; do
  #   env $base TAP_ETA_SOURCE=oracle TAP_WHEN=always TAP_DATA_CPC=5 TAP_SCOPE=full \
  #       FAMILY="J2_holder_c36" NOTE="J2 oracle+always+cpc5+full" ./submit_experiment.sh 14 "$s"
  # done
  # # J3: periodic minimalist -- tap every 4 rounds, trigger-only, block scope
  # for s in $SEEDS_J; do
  #   env $base TAP_WHEN=every_k TAP_PERIOD=4 TAP_DATA_CPC=0 TAP_SCOPE=block \
  #       FAMILY="J3_periodic_c36" NOTE="J3 every4+trigonly+block" ./submit_experiment.sh 14 "$s"
  # done
  # # J4: decay-coaster -- resend own last tapped weights between taps, big margin
  # for s in $SEEDS_J; do
  #   env $base TAP_COAST_MODE=decay TAP_MARGIN=0.05 TAP_WHEN=threshold TAP_DATA_CPC=1 \
  #       FAMILY="J4_decaycoast_c36" NOTE="J4 decay+margin.05+cpc1" ./submit_experiment.sh 14 "$s"
  # done

  # NOTE: adjust this configuration after group I sweeps and pick the one with the best knobs to run on different trigger classes
  # J5: TODO:pick the best configuration: oracle eta, tap when threshold, large margin, data reduced +5, full scope, resend 
  # for s in $SEEDS_J; do
  #   env $base TAP_ETA_SOURCE=oracle TAP_WHEN=threshold TAP_MARGIN=0.05 TAP_DATA_CPC=5 \
  #       TAP_SCOPE=full TAP_COAST_MODE=resend \
  #       FAMILY="J5_conservative_c36" NOTE="J5 oracle+thresh+margin.05+cpc5+full+resend" ./submit_experiment.sh 14 "$s"
  # done
# fi

# ---------------------------------------------------------------------------
# GROUP V -- TRIGGER SAMPLES, two faces of the paper.
#
# V1 (SERVER side, honest, no free-riders): sweep HOW the server verifies --
#   client_train (verify imgs == train imgs = memorisation, paper Table IX/V
#   protocol), client (held-out per-client), class (held-out shared) -- and the
#   verifier budget N_T = wm_num_triggers in {1,10,50} (paper Table VII). This
#   is a MEASUREMENT of the memorisation gap = wm-acc(client_train) - wm-acc(class).
#
# V2 (ATTACK side, Table V free-rider): the reduced attacker but training on FEWER
#   than a full trigger class -- AUTOP_TRIGGER_TRAIN_N caps the trigger-class images
#   the FR trains on. This is the paper's "training-trigger-sample-only" free-rider
#   (Sec V-D4): with too few samples the mark overfits and fails the held-out bank.
#   TRAIN-side count (how many the FR learns from), independent of N_T (server read).
#   cpc=0 -> trigger-only training, exactly the paper's setting; verify on held-out
#   'class' bank so the overfit is exposed.
# ---------------------------------------------------------------------------
# -- V1: server verify-mode x N_T (honest) ---
# NOTE: V1 deferred - less important (already in paper). V2 (Table V attack) still runs.
if has V; then
  SEEDS_V="${SEEDS_V:-0 1 2}"
#   # --- V1: server verify-mode x N_T (honest) ---
#   for MODE in client_train client class; do
#     for NT in 1 10 50; do
#       for s in $SEEDS_V; do
#         env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 \
#             WM_TRIGGER_MODE=$MODE WM_NUM_TRIGGERS=$NT \
#             FAMILY="V1_verify_${MODE}_nt${NT}_c100" \
#             NOTE="V1 server verify mode=$MODE N_T=$NT (Table VII + memorisation gap)" \
#             ./submit_experiment.sh 14 "$s"
#       done
#     done
#   done
  # --- V2: Table V attack -- reduced FR trained on TN trigger images (cpc=0) ---
  # TN=-1 is the full-trigger-class anchor 
  for TN in 10 100 500 -1; do
    TNTAG="${TN/-/m}"                      # -1 -> m1 (no dash in family names)
    for s in $SEEDS_V; do
      env ATTACK=reduced FREE_RIDER_IDS=3,6 AUTOP_COMMON_PER_CLASS=0 \
          AUTOP_TRIGGER_TRAIN_N=$TN AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
          WM_TRIGGER_MODE=class WM_NUM_TRIGGERS=50 WM_ETA_FIXED=0.064 ROUNDS=50 \
          FAMILY="V2_tableV_attack_c36_tn${TNTAG}" \
          NOTE="V2 Table V FR: trigger-only, trained on $TN trigger imgs, verify held-out" \
          ./submit_experiment.sh 14 "$s"
    done
  done
fi

# ---------------------------------------------------------------------------
# GROUP P -- paper reproduction. RUN ONLY AFTER the embedding probe passes.
#   (kept commented so a stray ./run_now.sh P cannot burn the night on nan)
# ---------------------------------------------------------------------------
# if has P; then
#   ROW=c10  SEEDS='0 1 2' BALANCED=0 FAM=P_paper_c10  ./paper_check.sh submit
#   ROW=c100 SEEDS='0 1 2' BALANCED=0 FAM=P_paper_c100 ./paper_check.sh submit
#   ROW=t9   SEEDS='0 1 2' BALANCED=0 HELDOUT=1 FAM=P_paper_t9 ./paper_check.sh submit
# fi

N=$(grep -c . "$JOBS_FILE" 2>/dev/null || echo 0)
echo
echo "== $N runs queued  (groups: $WANT) =="
cut -f1 "$JOBS_FILE" | sed 's/_rep[0-9]*$//' | sort | uniq -c | sed 's/^/   /'
cat <<NEXT

Next:
    unset DRYRUN
    WORKERS=6 PODS=2 ./submit_pool.sh
    runai list jobs
NEXT