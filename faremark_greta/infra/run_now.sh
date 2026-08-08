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
#         E=non-iid (+alpha sweep)   EA=non-iid DISTRIBUTION assignment (fairness)
#         F=capacity (+Table IX)   H=paper baselines
#         I=adaptive-tap single-knob sweeps   J=adaptive-tap multi-knob combos
#         K=DYNAMIC-submarine 1-seed tests (self-eta / derived margin / dynamic warmup)
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
#   A1 honest x6   A2 reduced easy   A3 reduced hard   
# NOTE: (A4/AK = same-key free-rider removed - use differnet runs for no conflicts)
# Note: config 14 defaults num_free_riders=2, but FREE_RIDER_IDS pins the exact
# cids, so free_rider_ids WINS (resolve_free_riders) 
# ---------------------------------------------------------------------------
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
  echo "   (group E done already)"
  # SEEDS_E="${SEEDS_E:-0 1 2}"     # set SEEDS_E=0 for a 1-seed quick pass
  # for s in $SEEDS_E; do
  #   env ATTACK=none NUM_FREE_RIDERS=0 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 ROUNDS=50 \
  #       FAMILY="E1_honest_niid_c100" NOTE="E1 non-iid honest a=0.5" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # for s in $SEEDS_E; do
  #   env ATTACK=reduced FREE_RIDER_IDS=3,6 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 \
  #       AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
  #       WM_ETA_FIXED=0.161 ROUNDS=50 \
  #       FAMILY="E2_reduced_niid_c36" NOTE="E2 non-iid reduced hard a=0.5" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # # E3 -- severity sweep. a=0.5 already covered by E1/E2, so with {0.1, 1.0} skew levels (0.1 / 0.5 / 1.0)
  # for A in 0.1 1.0; do
  #   ATAG="a$(printf '%s' "$A" | tr -d '.')"
  #   for s in $SEEDS_E; do
  #     env ATTACK=none NUM_FREE_RIDERS=0 PARTITION=dirichlet DIRICHLET_ALPHA=$A ROUNDS=50 \
  #         FAMILY="E3_honest_niid_c100_${ATAG}" NOTE="E3 non-iid honest alpha=$A" \
  #         ./submit_experiment.sh 14 "$s"
  #   done
  #   for s in $SEEDS_E; do
  #     env ATTACK=reduced FREE_RIDER_IDS=3,6 PARTITION=dirichlet DIRICHLET_ALPHA=$A \
  #         AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
  #         WM_ETA_FIXED=0.161 ROUNDS=50 \
  #         FAMILY="E3_reduced_niid_c36_${ATAG}" NOTE="E3 non-iid reduced hard alpha=$A" \
  #         ./submit_experiment.sh 14 "$s"
  #   done
  # done
fi

# ---------------------------------------------------------------------------
# GROUP F -- more clients than classes (200 clients). needs more rounds to train.
#   F1 honest 200cl   F2 reduced 200cl shared 6,7   F3 Table IX (paper repro)
#   F3 is PROBE-GATED: it reproduces FareMark Table IX (cifar10, 50 clients,
#   client_train trigger mode -- verify-on-training-images = memorisation). It is
#   only built when PAPER_OK=1 
# ---------------------------------------------------------------------------
if has F; then
  echo "   (group F to do later -- 200 clients, 100 rounds, needs more time)"
  # for s in 0 1 2; do
  #   env ATTACK=none NUM_FREE_RIDERS=0 NUM_CLIENTS=200 ROUNDS=100 \
  #       FAMILY="F1_honest_nc200" NOTE="F1 200 clients honest, 100 rounds" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # for s in 0 1 2; do
  #   env ATTACK=reduced NUM_CLIENTS=200 FREE_RIDER_IDS=106,107 AUTOP_COMMON_PER_CLASS=5 \
  #       AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.384 ROUNDS=100 \
  #       FAMILY="F2_reduced_nc200_c67" NOTE="F2 200cl reduced, shared classes 6,7" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # if [ "$PAPER_OK" = "1" ]; then
  #   for s in 0 1 2; do
  #     env ATTACK=none NUM_FREE_RIDERS=0 NUM_CLIENTS=50 \
  #         WM_TRIGGER_MODE=client_train WM_NUM_TRIGGERS=50 ROUNDS=50 \
  #         FAMILY="F3_tableIX_c10_nc50" NOTE="F3 Table IX cifar10 50cl client_train (paper)" \
  #         ./submit_experiment.sh 11 "$s"
  #   done
  #   for s in 0 1 2; do
  #     env ATTACK=none NUM_FREE_RIDERS=0 NUM_CLIENTS=50 \
  #         WM_TRIGGER_MODE=class WM_NUM_TRIGGERS=50 ROUNDS=50 \
  #         FAMILY="F3_tableIX_c10_nc50_heldout" NOTE="F3 Table IX held-out twin (generalisation)" \
  #         ./submit_experiment.sh 11 "$s"
  #   done
  # else
  #   echo "   (F3/Table IX skipped -- set PAPER_OK=1 to build the paper-repro rows)"
  # fi
fi

# ---------------------------------------------------------------------------
# GROUP H -- paper baselines 
#   H1 honest cifar10 (fidelity)          H3 previous-models FR (crude, caught)
#   H2 honest cifar100 == A1 (reference)  H4 gaussian-noise FR (crude, caught)
# ---------------------------------------------------------------------------
if has H; then
  echo "   (H1-H4 done/cited via A1; H5 = previous-models FR on c100, the money-plot positive control, runs below)"
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
  # for the operating-point plot (calibrated on A1_honest_c100). Should be caught.
  for s in 0 1 2; do
    env ATTACK=previous_models NUM_FREE_RIDERS=2 FREE_RIDER_IDS=3,6 WM_ETA_FIXED=0.064 ROUNDS=50 \
        FAMILY="H5_prevmodel_c100" NOTE="H5 crude FR previous-models on c100 (money-plot control), cids 3,6 to match K4/D1" \
        ./submit_experiment.sh 14 "$s"
  done
fi

# ---------------------------------------------------------------------------
# GROUP I -- ADAPTIVE-TAP single-knob sweeps (attack="adaptive_tap").
# Base: FR at hard classes 3,6, oracle eta = 0.064, warmup 12 / calib 4.
# SEEDS_I trims the seed count for these exploratory sweeps (bump to "0 1 2").
# ---------------------------------------------------------------------------
if has I; then
  echo "   (group I adaptive-tap sweeps -- done see group JK)"
  SEEDS_I="${SEEDS_I:-0}"
  # =====================================================================
  # FIX (root cause of the flat-0.6 plots): the OLD base set TAP_DATA_CPC=0
  # (trigger-only) = FareMark Table V overfitting. Every knob-sweep config
  # inherited it, so they all pinned at ~0.6 no matter the knob. Group D
  # already showed cpc=0 -> BER 0.44, cpc>=1 -> ~0.11 plateau.
  #   * base now uses TAP_DATA_CPC=5  (the Group-D plateau recipe)
  #   * cpc=0 kept ONLY as a labeled positive control (expected caught)
  #   * aim under eta_loose=0.264 (beatable per-client threshold); a 0.064
  #     variant shows the HARD-CLASS FLOOR (class 6 ~0.22 can't go < 0.064
  #     -- that is the operating-point thesis, not an attack failure)
  #   * TAP_MARGIN=0.02 gives probe/defender headroom; TAP_MAX_COAST=4
  #     forces a re-tap so the mark can't silently drift over eta
  #   * TAP_PROBE_HOLDOUT=16 + the _prepare MIN_TRAIN_TRIG=8 cap => the tap
  #     trains on ~34 triggers (grep the trace: n_trigger_train ~34, NOT ~1)
  # =====================================================================
  # base="ATTACK=adaptive_tap FREE_RIDER_IDS=3,6 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 ROUNDS=50 \
  #       WM_ETA_FIXED=0.264 TAP_WHEN=threshold TAP_MARGIN=0.02 TAP_DATA_CPC=5 TAP_SCOPE=full \
  #       TAP_ETA_SOURCE=oracle TAP_COAST_MODE=resend TAP_MAX_COAST=4 TAP_PROBE_HOLDOUT=16"

  # # --- 0. SMOKE / GATE: always-tap at cpc=5 MUST reproduce Group D (mean ~0.13, NOT 0.6). ---
  # #     If this one is flat at 0.6, the tap EMBED path is still broken -- stop and inspect
  # #     the trace (n_trigger_train, ber_after) before trusting any sweep below.
  # for s in $SEEDS_I; do
  #   env $base TAP_WHEN=always \
  #       FAMILY="I0_smoke_always_cpc5_c36" NOTE="I0 GATE: always-tap cpc5 must == D reduced (~0.13)" \
  #       ./submit_experiment.sh 14 "$s"
  # done

  # # --- 1. DATA PER TAP (the effort dial): 0=Table V control (caught), 1=plateau edge, 5=plateau ---
  # for D in 0 1 5; do
  #   for s in $SEEDS_I; do
  #     env $base TAP_DATA_CPC=$D \
  #         FAMILY="I_data_n${D}_c36" NOTE="I data_cpc=$D (0 = Table V positive control)" \
  #         ./submit_experiment.sh 14 "$s"
  #   done
  # done

  # # --- 2. WHEN / duty-cycle: threshold (adaptive, cheapest) vs every_k(P=3) ---
  # for W in threshold every_k; do
  #   for s in $SEEDS_I; do
  #     env $base TAP_WHEN=$W TAP_PERIOD=3 \
  #         FAMILY="I_when_${W}_c36" NOTE="I when=$W" ./submit_experiment.sh 14 "$s"
  #   done
  # done

  # # --- 3. ETA SOURCE (realism): oracle (given the true eta) vs self (FR estimates it) ---
  # for ES in oracle self; do
  #   for s in $SEEDS_I; do
  #     env $base TAP_ETA_SOURCE=$ES TAP_ETA_K=3.0 \
  #         FAMILY="I_eta_${ES}_c36" NOTE="I eta_source=$ES" ./submit_experiment.sh 14 "$s"
  #   done
  # done

  # # --- 5. KEEP-THE-MARK-ALIVE BETWEEN TAPS (FedIPR/FedTracker persistence -> low duty cycle) ---
  # #     coast_mode=decay resends the FR's OWN last-tapped weights (mark fades slower than
  # #     resending the global). This is the paper-grounded lever for coasting longer per tap.
  # for CM in resend decay; do
  #   for s in $SEEDS_I; do
  #     env $base TAP_COAST_MODE=$CM \
  #         FAMILY="I_coast_${CM}_c36" NOTE="I coast_mode=$CM (decay = slower mark fade)" \
  #         ./submit_experiment.sh 14 "$s"
  #   done
  # done
  # # --- 6. HOW LAZY CAN IT BE: force a re-tap only every 8 coasts (probe mark persistence) ---
  # #     If the mark survives 8 coasts under eta, the duty cycle -> tiny = the cheap-stealth result.
  # for s in $SEEDS_I; do
  #   env $base TAP_MAX_COAST=8 \
  #       FAMILY="I_maxcoast_m8_c36" NOTE="I max_coast=8 (persistence / lowest duty cycle)" \
  #       ./submit_experiment.sh 14 "$s"
  # done

  # # --- 4. TIGHT-eta variant (operating point): shows the hard-class floor at eta=0.064 ---
  # #     class 3 evades (~0.037 < 0.064), class 6 cannot (~0.22 > 0.064) = the split the thesis predicts.
  # for s in $SEEDS_I; do
  #   env $base WM_ETA_FIXED=0.064 \
  #       FAMILY="I_tight_eta0064_c36" NOTE="I tight eta=0.064 (hard-class floor demo)" \
  #       ./submit_experiment.sh 14 "$s"
  # done
fi

# ---------------------------------------------------------------------------
# GROUP J -- graft-coast submarine: PERSISTENCE (how long the mark lasts untapped)
#            + RECOVERY (how fast one tap re-embeds) + the adaptive SAWTOOTH.
#            Run AFTER the clients.py graft + probe-target fix is in.  1 seed for now.
#
# LESSONS FROM GROUP I BAKED IN (do NOT repeat the BER~0.6 mistake):
#   * TAP_DATA_CPC=5  = the Group-D PLATEAU recipe. cpc=0 (trigger-only) is exactly what
#     pinned group I at ~0.6 (Table V overfitting) -- it is used ONLY as a labelled control,
#     never as a base. J0 is a GATE that must reproduce the ~0.11-0.13 plateau or you STOP.
#   * decision eta = AUTOP_ORACLE_ETA=0.264 (the beatable per-client LOOSE rule): the FR aims
#     just UNDER 0.264, not under the degenerate 0.064. WM_ETA_FIXED=0.064 only draws the tight
#     REFERENCE line; the loose 0.264 reference draws by default. BOTH are reference only --
#     they do not decide anything (the offline sweep is the verdict).
#   * coast_mode=graft + scope=head: the body follows the global (submission moves, no replay),
#     only the mark head is frozen, so the mark FADES GRADUALLY (a sawtooth) instead of dying in
#     1 round (resend) or never (decay). Requires the clients.py probe-target fix.
# ---------------------------------------------------------------------------
if has J; then
  SEEDS_J="${SEEDS_J:-0}"          # 1 seed for the exploratory pass; bump to "0 1 2" once shape confirmed
  jbase="ATTACK=adaptive_tap FREE_RIDER_IDS=3,6 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
         AUTOP_ORACLE_ETA=0.264 WM_ETA_FIXED=0.064 TAP_DATA_CPC=5 TAP_ETA_SOURCE=oracle \
         TAP_PROBE_HOLDOUT=16 ROUNDS=40 FAST_DATA=1"

  # --- J0 GATE: always-tap cpc=5 MUST reproduce the Group-D plateau (~0.11-0.13), NOT 0.6. ----
  #     If this is flat at ~0.6 the tap EMBED path is broken -> STOP and grep the trace
  #     (n_trigger_train should be ~tens, ber_after ~0.1) before trusting anything below.
  for s in $SEEDS_J; do
    env $jbase TAP_WHEN=always TAP_SCOPE=full TAP_COAST_MODE=resend \
        FAMILY="J0_gate_alwaystap_c36" \
        NOTE="J0 GATE always-tap cpc5 must == Group D (~0.13, NOT 0.6)" \
        ./submit_experiment.sh 14 "$s"
  done

  # --- J1 PERSISTENCE: how many rounds does the mark survive with NO tap? ---------------------
  #     every_k(P) taps ONCE then coasts P-1 rounds on graft. P are divisors of the warmup (12)
  #     so round 12 always taps and establishes the mark first:
  #       P=2 ->1 coast   P=3 ->2   P=4 ->3   P=6 ->5   P=12 ->11 coasts between taps.
  #     READ: on the coast rounds, how high does the FR BER climb? The largest P whose coast
  #     rounds stay < eta = the persistence limit = "the mark lasts ~this many rounds untapped".
  #     tap_dynamics gives fade_per_coast + stayed_below_target per P.
  for P in 2 3 4 6 12; do
    for s in $SEEDS_J; do
      env $jbase TAP_WHEN=every_k TAP_PERIOD=$P TAP_MAX_COAST=999 \
          TAP_SCOPE=head TAP_COAST_MODE=graft \
          FAMILY="J1_persist_graft_p${P}_c36" \
          NOTE="J1 persistence: graft, tap 1-in-$P (survives $((P-1)) coasts?)" \
          ./submit_experiment.sh 14 "$s"
    done
  done

  # --- J2 THE SAWTOOTH: adaptive threshold + graft. Tap ONLY when the fading mark nears eta. ---
  #     This is the headline run. From its trace, tap_dynamics extracts BOTH observables:
  #       rounds_between_taps = FADE TIME (how long one tap lasts before BER climbs to eta-margin)
  #       ber_drop_per_tap    = RECOVERY (how much BER a single re-embed buys back)
  #     TAP_MAX_COAST=12 lets it coast long; TAP_MARGIN=0.03 aims a hair under the 0.264 loose rule.
  for s in $SEEDS_J; do
    env $jbase TAP_WHEN=threshold TAP_MARGIN=0.03 TAP_MAX_COAST=12 \
        TAP_SCOPE=head TAP_COAST_MODE=graft \
        FAMILY="J2_saw_graft_head_c36" \
        NOTE="J2 adaptive sawtooth: graft coast, re-tap only near eta (fade+recovery)" \
        ./submit_experiment.sh 14 "$s"
  done

  # --- J3 COAST-MODE A/B at 1-in-3 (the fade-mechanism control; graft@P3 lives in J1). ---------
  #     resend -> mark dies in 1 coast (BER ~0.5-0.8 on coasts = CAUGHT). Also the COLD-RECOVERY
  #               probe: every tap re-embeds from a dead ~0.5 start, so ber_drop_per_tap here =
  #               how much ONE tap recovers from scratch.
  #     decay  -> mark frozen flat (evades, but a REPLAY: byte-identical submissions).
  #     (graft at P=3 is J1_persist_graft_p3_c36 -- compare the three side by side.)
  for CM in resend decay; do
    for s in $SEEDS_J; do
      env $jbase TAP_WHEN=every_k TAP_PERIOD=3 TAP_MAX_COAST=999 \
          TAP_SCOPE=head TAP_COAST_MODE=$CM \
          FAMILY="J3_coast_${CM}_p3_c36" \
          NOTE="J3 coast_mode=$CM at 1-in-3 (fade mechanism; resend=cold-recovery probe)" \
          ./submit_experiment.sh 14 "$s"
    done
  done

  # --- J4 GRAFT SCOPE: how much of the mark must be frozen for a usable fade rate? -------------
  #     head (last 2 params) = J2 (fastest fade). block (8) / block2 (20) freeze MORE of the
  #     head -> the mark survives longer per coast. If J2/J3 graft snaps to ~0.5 in one coast,
  #     block/block2 slow the fade; if graft is dead flat, go back to head.
  for SC in block block2; do
    for s in $SEEDS_J; do
      env $jbase TAP_WHEN=threshold TAP_MARGIN=0.03 TAP_MAX_COAST=12 \
          TAP_SCOPE=$SC TAP_COAST_MODE=graft \
          FAMILY="J4_scope_graft_${SC}_c36" \
          NOTE="J4 graft scope=$SC (slower fade than head; persistence-vs-cost tuning)" \
          ./submit_experiment.sh 14 "$s"
    done
  done
fi

# ---------------------------------------------------------------------------
# GROUP NOW -- 3-SEED RUN: the confirmed submarine (J2) + the tuned one (J5).
#   BATCH=NOW ./runbook.sh manifest && BATCH=NOW ./runbook.sh submit
#   3 seeds = repeats 0,1,2 = seeds 1000/1001/1002 (config.py: seed = base_seed + repeat).
#   'NOW' shares no letter with any group token (A C D E F H I J V), and has() is a SUBSTRING
#   match, so BATCH=NOW fires ONLY this block -- not the 11-family J suite.
# ---------------------------------------------------------------------------
if has NOW; then
  SEEDS_NOW="${SEEDS_NOW:-0 1 2}"
  # same recipe as group J: cpc5 taps, decision eta 0.264, reference lines 0.064/0.264,
  # warmup 12, 40 rounds, graft/head/threshold (the confirmed adaptive submarine).
  nbase="ATTACK=adaptive_tap FREE_RIDER_IDS=3,6 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
         AUTOP_ORACLE_ETA=0.264 WM_ETA_FIXED=0.064 TAP_DATA_CPC=5 TAP_ETA_SOURCE=oracle \
         TAP_SCOPE=head TAP_COAST_MODE=graft TAP_WHEN=threshold ROUNDS=40 FAST_DATA=1"

  # CONFIG A -- J2 reproduced at 3 seeds: the confirmed adaptive submarine (the headline table row).
  #   IDENTICAL knobs to the 1-seed J2 (margin 0.03, max_coast 12, holdout 16) so this IS J2 x3.
  #   NOTE: the pool skips a family/rep whose result.json already EXISTS (it does NOT check knobs),
  #   so DELETE the 1-seed rep0 first (see CLI) to get a clean 3-seed set under one recipe.
  for s in $SEEDS_NOW; do
    env $nbase TAP_MARGIN=0.03 TAP_MAX_COAST=12 TAP_PROBE_HOLDOUT=16 \
        FAMILY="J2_saw_graft_head_c36" \
        NOTE="J2 confirmed submarine @3seeds (graft/head/threshold, cpc5)" \
        ./submit_experiment.sh 14 "$s"
  done

  # CONFIG B -- J5 tuned submarine at 3 seeds: honest probe (holdout 48 -> ~25 real images, half the
  #   trigger class) + safety margin 0.05 + deep coast (max_coast 20). Aim: the EASY FR (cid3) coasts
  #   far (tap-fraction -> ~0) while the honest probe + margin keep the HARD FR (cid6) under eta_loose
  #   on EVERY coast peak across all 3 seeds. cpc stays 5 (confirmed to re-embed); a cpc {1,2} sweep is
  #   a separate 1-seed follow-up, not worth 3 seeds until the shape is confirmed.
  for s in $SEEDS_NOW; do
    env $nbase TAP_MARGIN=0.05 TAP_MAX_COAST=20 TAP_PROBE_HOLDOUT=48 \
        FAMILY="J5_submarine_head_c36" \
        NOTE="J5 tuned submarine @3seeds: honest probe + margin 0.05 + deep coast (max_coast 20)" \
        ./submit_experiment.sh 14 "$s"
  done
fi

# ---------------------------------------------------------------------------
# GROUP K -- DYNAMIC SUBMARINE (1-seed tests, one run per new dynamic feature).
#   BATCH=K ./runbook.sh manifest && BATCH=K ./runbook.sh submit
#   Each new dynamic mechanism added to clients.py gets ONE 1-seed run here that
#   toggles exactly that feature ON, everything else = the confirmed J2 recipe, so
#   a diff against K0 isolates the feature. All default to J2 behaviour when the
#   knob is unset, so these cannot disturb J2/J4/NOW.  Purpose: prove each dynamic
#   piece RUNS and does the intended thing before spending 3 seeds on it.
#   ('K' shares no letter with A C D E F H I J V NOW, so BATCH=K fires ONLY this.)
# ---------------------------------------------------------------------------
if has K; then
  SEEDS_K="${SEEDS_K:-0 1 2}"     # K4 is now the 3-seed free-riding headline (was 1-seed wiring tests)
  # NOTE: for the clean re-run, run submit with SMOOTH_EPS=1e-8 exported so the pod
  #       forwards it (submit_experiment.sh -e SMOOTH_EPS). Unset => legacy 1e-3.
  # the confirmed-J2 base; K runs flip ONE dynamic knob each on top of it.
  kbase="ATTACK=adaptive_tap FREE_RIDER_IDS=3,6 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
         AUTOP_ORACLE_ETA=0.264 WM_ETA_FIXED=0.064 TAP_DATA_CPC=5 \
         TAP_SCOPE=head TAP_COAST_MODE=graft TAP_WHEN=threshold TAP_PROBE_HOLDOUT=16 \
         TAP_MARGIN=0.03 TAP_MAX_COAST=12 ROUNDS=50 FAST_DATA=1"

  # --- K0/K1/K2/K3 ablations: DISABLED. These were the 1-seed feature-isolation runs;
  #     the sweep is done and K4 is the config to table. Re-enable (uncomment) only if
  #     you need the per-feature diff again.
  # for s in $SEEDS_K; do
  #   env $kbase TAP_ETA_SOURCE=oracle TAP_MARGIN_MODE=fixed TAP_WARMUP_MODE=fixed \
  #       FAMILY="K0_control_J2_c36" \
  #       NOTE="K0 control = J2 (oracle eta, fixed margin, fixed warmup) -- the diff baseline" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # for s in $SEEDS_K; do
  #   env $kbase TAP_ETA_SOURCE=self TAP_ETA_K=3.0 TAP_MARGIN_MODE=fixed TAP_WARMUP_MODE=fixed \
  #       FAMILY="K1_selfeta_c36" \
  #       NOTE="K1 self-estimated eta (mu+3sigma over own calib probes) -- removes the oracle crutch" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # for s in $SEEDS_K; do
  #   env $kbase TAP_ETA_SOURCE=oracle TAP_MARGIN_MODE=derived TAP_MARGIN_K=1.0 TAP_WARMUP_MODE=fixed \
  #       FAMILY="K2_derivedmargin_c36" \
  #       NOTE="K2 derived margin (eta - k*sigma) -- safety gap scales with estimation noise" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # for s in $SEEDS_K; do
  #   env $kbase TAP_ETA_SOURCE=oracle TAP_MARGIN_MODE=fixed \
  #       TAP_WARMUP_MODE=dynamic TAP_CONV_EPS=0.03 TAP_CONV_PATIENCE=2 \
  #       TAP_HONEST_MIN=6 TAP_WARMUP_CAP=15 \
  #       FAMILY="K3_dynwarmup_c36" \
  #       NOTE="K3 dynamic warmup (defect on own-probe convergence, per-class defect round)" \
  #       ./submit_experiment.sh 14 "$s"
  # done

  # --- K4 ALL-DYNAMIC + block2 sawtooth: THE free-riding submarine to table (3 seeds).
  for s in $SEEDS_K; do
    env $kbase TAP_SCOPE=block2 TAP_ETA_SOURCE=self TAP_ETA_K=3.0 \
        TAP_MARGIN_MODE=derived TAP_MARGIN_K=1.0 \
        TAP_WARMUP_MODE=dynamic TAP_CONV_EPS=0.03 TAP_CONV_PATIENCE=2 \
        TAP_HONEST_MIN=6 TAP_WARMUP_CAP=15 \
        TAP_MAX_COAST=6 TAP_GRAFT_DECAY=0.25 \
        FAMILY="K4_alldyn_block2_c36" \
        NOTE="K4 all-dynamic + block2 sawtooth (self-eta, derived margin, dynamic warmup, non-polluting)" \
        ./submit_experiment.sh 14 "$s"
  done

  # --- K5/K6 head configs: DISABLED. The re-uploaded tap_perfr plots show head taps ~100%
  #     on both classes (self-sufficient + evades, but saves NO compute -> not a free-rider
  #     in the effort sense). K4/block2 is the free-riding config we table. Re-enable only
  #     to quote K6/head for the "needs no oracle" point, or after a self-probe fix
  #     (raise the n_trig//2 cap in clients.py:_prepare + EMA-smooth) lets head coast.
  # for s in $SEEDS_K; do
  #   env $kbase TAP_SCOPE=head TAP_ETA_SOURCE=self TAP_ETA_K=3.0 \
  #       TAP_MARGIN_MODE=derived TAP_MARGIN_K=1.0 \
  #       TAP_MAX_COAST=6 TAP_GRAFT_DECAY=0.5 \
  #       FAMILY="K5_selfeta_derivedmargin_head_c36" \
  #       NOTE="K5 self-eta + derived margin, head scope, non-polluting (fixes K1 hard-class overshoot)" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # for s in $SEEDS_K; do
  #   env $kbase TAP_SCOPE=head TAP_ETA_SOURCE=self TAP_ETA_K=3.0 \
  #       TAP_MARGIN_MODE=derived TAP_MARGIN_K=1.0 \
  #       TAP_WARMUP_MODE=dynamic TAP_CONV_EPS=0.03 TAP_CONV_PATIENCE=2 \
  #       TAP_HONEST_MIN=6 TAP_WARMUP_CAP=15 \
  #       TAP_MAX_COAST=6 TAP_GRAFT_DECAY=0.5 \
  #       FAMILY="K6_full_submarine_head_c36" \
  #       NOTE="K6 COMPLETE submarine: fully dynamic (self-eta+derived margin+dynamic warmup) + head + tail-fix (no pollution)" \
  #       ./submit_experiment.sh 14 "$s"
  # done
fi

# ---------------------------------------------------------------------------
# GROUP EA -- DISTRIBUTION-AWARE TRIGGER ASSIGNMENT (non-IID fairness fix).
#   Runs ALONGSIDE Group E so the two assignment policies are directly comparable.
#   The server assigns each client a trigger class it HOLDS a lot of (max-count
#   matching) instead of the blind cid%n. Same alpha=0.5 as E1/E2. Enable together:
#     BATCH=EEA ./runbook.sh manifest && BATCH=EEA ./runbook.sh submit
#   (E and EA run in parallel in the same pool.)
# ---------------------------------------------------------------------------
if has EA; then
  SEEDS_EA="${SEEDS_EA:-0 1 2}"
  # EA1 honest, distribution assignment, a=0.5 -- the fair counterpart to E1.
  for s in $SEEDS_EA; do
    env ATTACK=none NUM_FREE_RIDERS=0 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 \
        WM_TRIGGER_ASSIGN=distribution ROUNDS=50 \
        FAMILY="EA1_honest_niid_distrib_c100" \
        NOTE="EA1 non-iid honest a=0.5, DISTRIBUTION trigger assignment (fairness fix)" \
        ./submit_experiment.sh 14 "$s"
  done
  # EA2 reduced FR, distribution assignment -- the REALISTIC/fair-to-server counterpart to E2.
  #   IMPORTANT: the server does NOT know who is a free-rider, so the free-riders are
  #   assigned trigger classes by the SAME distribution rule as everyone else. We set
  #   FREE_RIDER_IDS=3,6 (which CLIENTS free-ride) but do NOT pin their trigger CLASS, so
  #   cid3/cid6 also get a class they hold a lot of. Consequence: the FR's trigger class
  #   varies by seed (it depends on the dirichlet draw), which is exactly the fair, no-
  #   special-treatment scenario. Answers: "with fully fair assignment for everyone, does
  #   the FR still hide?"
  for s in $SEEDS_EA; do
    env ATTACK=reduced FREE_RIDER_IDS=3,6 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 \
        WM_TRIGGER_ASSIGN=distribution \
        AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
        WM_ETA_FIXED=0.161 ROUNDS=50 \
        FAMILY="EA2_reduced_niid_distrib_c36" \
        NOTE="EA2 non-iid reduced a=0.5, DISTRIBUTION assignment for ALL incl. free-riders (fair to server)" \
        ./submit_experiment.sh 14 "$s"
  done
  # EA2b CONTROLLED variant: pin the free-riders to classes 3 (medium) & 6 (hard) in BOTH
  #   E2 and EA2b via trigger_class_map, and let ONLY the honest clients be distribution-
  #   assigned. This isolates the honest-floor effect at FIXED free-rider difficulty, so the
  #   E2 <-> EA2b comparison is apples-to-apples (same FR classes, only honest assignment
  #   changes). trigger_class_map pins cid3->3, cid6->6 as forced reservations; the matcher
  #   assigns the honest clients around them.
  for s in $SEEDS_EA; do
    env ATTACK=reduced FREE_RIDER_IDS=3,6 TRIGGER_CLASS_MAP="3:3,6:6" \
        PARTITION=dirichlet DIRICHLET_ALPHA=0.5 \
        WM_TRIGGER_ASSIGN=distribution \
        AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
        WM_ETA_FIXED=0.161 ROUNDS=50 \
        FAMILY="EA2b_reduced_niid_distrib_pin_c36" \
        NOTE="EA2b non-iid reduced a=0.5, DISTRIBUTION for honest, FR pinned to cls 3&6 (controlled vs E2)" \
        ./submit_experiment.sh 14 "$s"
  done
fi

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