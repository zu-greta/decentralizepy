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
#         P=paper reproduction (ONLY after the embedding probe passes)
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
# GROUP A -- proven baseline (cifar100, 10 clients). THIS CONFIG IS KNOWN GOOD.
#   A1 honest x6   A2 reduced easy   A3 reduced hard   A4 sameclass (own key)
#   AK sameclass SAME KEY  <-- the controlled effort-only isolation
# Note: config 14 defaults num_free_riders=2, but FREE_RIDER_IDS pins the exact
# cids, so free_rider_ids WINS (resolve_free_riders) -> A4/AK have ONE free-rider.
# ---------------------------------------------------------------------------
if has A; then
  # for s in 0 1 2 3 4 5; do
  #   env ATTACK=none NUM_FREE_RIDERS=0 DS=c100 NUM_CLIENTS=10 ROUNDS=50 \
  #       FAMILY="A1_honest_c100" NOTE="A1 honest baseline (known-good config)" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # for s in 0 1 2; do
  #   env ATTACK=reduced FREE_RIDER_IDS=1,7 AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 \
  #       AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
  #       FAMILY="A2_reduced_c100_c17" NOTE="A2 reduced +5 easy classes 1,7" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # for s in 0 1 2; do
  #   env ATTACK=reduced FREE_RIDER_IDS=3,6 AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 \
  #       AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
  #       FAMILY="A3_reduced_c100_c36" NOTE="A3 reduced +5 hard classes 3,6" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # for s in 0 1 2; do
  #   env ATTACK=reduced FREE_RIDER_IDS=0 TRIGGER_CLASS_MAP="0:6" AUTOP_COMMON_PER_CLASS=5 \
  #       AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
  #       FAMILY="A4_sameclass_c100_c6" NOTE="A4 FR shares class 6 with honest (own key)" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # ---- AK: SAME trigger class AND SAME key/message -----------------------
  # trigger_class_map 0:6  -> FR cid0 sits on class 6 (with honest cid6)
  # wm_key_twins    0:6    -> FR cid0 derives its key M and message B from cid6
  for s in 0 1 2; do
    env ATTACK=reduced FREE_RIDER_IDS=0 TRIGGER_CLASS_MAP="0:6" WM_KEY_TWINS="0:6" \
        AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
        WM_ETA_FIXED=0.064 ROUNDS=50 \
        FAMILY="AK_sameclass_samekey_c6" NOTE="AK FR shares class 6 AND key/msg with honest cid6" \
        ./submit_experiment.sh 14 "$s"
  done
fi

# ---------------------------------------------------------------------------
# GROUP C -- smoothing function (does a different f() move the floors?)
#   C1 = sin smoothing.  C2 (entropy/dominance vs accuracy) is OFFLINE from the  
#   diagnostics in the A1 result.json files 
# ---------------------------------------------------------------------------
if has C; then
  for s in 0 1 2; do
    env ATTACK=none NUM_FREE_RIDERS=0 WM_F=sin WM_ALPHA=1.5708 ROUNDS=50 \
        FAMILY="C1_honest_sin_c100" NOTE="C1 sin smoothing alpha=pi/2" \
        ./submit_experiment.sh 14 "$s"
  done
fi

# ---------------------------------------------------------------------------
# GROUP D -- +N free-riding spectrum at the hard classes
#   N = images kept per common class; -1 = full shard (still a free-rider)
# ---------------------------------------------------------------------------
if has D; then
  for N in -1 0 1 2 5 10 25 50; do
    for s in 0 1 2; do
      env ATTACK=reduced FREE_RIDER_IDS=3,6 AUTOP_COMMON_PER_CLASS=$N \
          AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
          FAMILY="D1_reduced_c100_c36_n${N}" NOTE="D1 +N spectrum N=$N" \
          ./submit_experiment.sh 14 "$s"
    done
  done
fi

# ---------------------------------------------------------------------------
# GROUP E -- non-IID (Dirichlet)
#   E1 honest a=0.5   E2 reduced a=0.5   E3 reduced ALPHA SWEEP
#   Each alpha has its OWN honest floor, so E3 pairs a honest run with a reduced
#   run at each alpha (so eta can be recalibrated per-alpha offline). The frozen
#   line here is a placeholder (0.161); the verdict comes from the offline sweep.
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
  # E3 -- severity sweep. a=0.5 already covered by E1/E2.
  for A in 0.1 0.3 1.0; do
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
# GROUP F -- more clients than classes (200 clients). NEEDS MORE ROUNDS to train.
#   F1 honest 200cl   F2 reduced 200cl shared 6,7   F3 Table IX (paper repro)
#   F3 is PROBE-GATED: it reproduces FareMark Table IX (cifar10, 50 clients,
#   client_train trigger mode -- verify-on-training-images = memorisation). It is
#   only built when PAPER_OK=1 so a stray run cannot burn the night on an
#   un-validated embedding. It also builds a held-out twin (class mode) so
#   paper_check.py can report the memorisation-vs-generalisation gap.
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
# GROUP H -- paper baselines (the "our reimplementation matches, and the crude
# attacks it was designed for ARE caught" evidence). This is the credibility
# floor: it shows the detector works exactly as the paper claims against dumb
# free-riders, so the reduced/same-class non-separability is a real limitation,
# not a broken build.
#   H1 honest cifar10 (fidelity)          H3 previous-models FR (crude, caught)
#   H2 honest cifar100 == A1 (reference)  H4 gaussian-noise FR (crude, caught)
#   Table IX capacity baseline lives in F3 (PAPER_OK=1).
# ---------------------------------------------------------------------------
if has H; then
  for s in 0 1 2; do
    env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 \
        FAMILY="H1_honest_c10" NOTE="H1 fidelity: all-honest watermark cifar10" \
        ./submit_experiment.sh 11 "$s"
  done
  # H2 (honest cifar100) is identical to A1 -- do not rerun; cite A1_honest_c100.
  for s in 0 1 2; do
    env ATTACK=previous_models NUM_FREE_RIDERS=2 WM_ETA_FIXED=0.25 ROUNDS=50 \
        FAMILY="H3_prevmodel_c10" NOTE="H3 crude FR: previous-models attack cifar10" \
        ./submit_experiment.sh 11 "$s"
  done
  for s in 0 1 2; do
    env ATTACK=gaussian NUM_FREE_RIDERS=2 NOISE_SIGMA=0.1 WM_ETA_FIXED=0.25 ROUNDS=50 \
        FAMILY="H4_gaussian_c10" NOTE="H4 crude FR: gaussian-noise attack cifar10" \
        ./submit_experiment.sh 11 "$s"
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

FIRST TIME after the code fix, run the probe instead:
    NUM_WORKERS=0 FAMILY=probe_fix WM_BITS=2 WM_NUM_TRIGGERS=50 ROUNDS=25 ./submit_experiment.sh 11 0
    # confirm BER falls and pmax is not nan BEFORE launching the batch

AK CHECK (do this once the pool is running): confirm the twin actually applied --
    grep -h wm_key_twins \$RESULTS/AK_sameclass_samekey_c6_rep*/pod.log
    # every AK pod.log MUST show:  wm_key_twins   ->   0:6
    # if it shows None or is absent, the twin did not reach the run -- stop and fix.
NEXT