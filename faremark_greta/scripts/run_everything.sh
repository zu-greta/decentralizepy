#!/usr/bin/env bash
# =============================================================================
# run_everything.sh  --  fire the whole thesis matrix through run_all, in STAGES.
#
# NOTHING here waits on the cluster. Each phase submits jobs (WAIT=0, like run_all)
# and returns immediately. Phases:
#
#   submit    fire EVERYTHING now (honest + attacks), no waiting.  <-- fire-and-forget
#   honest    submit only the honest jobs
#   attacks   (after honest done) calibrate the REAL eta + submit attacks
#   plot      (after all done, results local) calibrate + separability tables + figures
#
# Recommended when you just want to launch and walk away:
#     ./run_everything.sh submit
#     # ... let the whole cluster batch run ...  then scp results to local, then:
#     RES=~/local/results ./run_everything.sh plot
#
# WHY 'submit' can fire attacks before honest finishes:
#   The reduced / sameclass attackers do NOT use eta -- they train on reduced data
#   every round regardless. Eta is only used by the server's LIVE flagging
#   (wm_fpr / wm_fr_recall / flagged), which separability.py recomputes offline from
#   the logged per-client BER. So the attacks only need SOME eta value at submit time
#   to pass run_all's bookkeeping; the logged BER is identical to a real-eta run.
#   'submit' passes a provisional PROV_ETA (default 0.065, your calibrated ballpark);
#   the 'plot' phase then calibrates the REAL eta and the analysis uses that.
#
# All runs land in run_all's flat results dir, tagged by a UNIQUE family per leg.
# Run submit/honest/attacks from the dir with submit_experiment.sh + .env; run plot
# from your local repo root with RES pointing at the scp'd results.
#
# KNOBS: HONEST_SEEDS('0 1 2 3 4 5')  ATTACK_SEEDS('0 1 2')  BALANCED(0)  CAP_NC(200)
#        CAP10_NC(50)  DO_PLOTS(1)  PROV_ETA(0.065)  LEGS(all)  RES(plot: where results are)
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RA="$HERE/run_all.sh"

HONEST_SEEDS="${HONEST_SEEDS:-0 1 2 3 4 5}"
ATTACK_SEEDS="${ATTACK_SEEDS:-0 1 2}"
export BALANCED="${BALANCED:-0}"
CAP_NC="${CAP_NC:-200}"
CAP10_NC="${CAP10_NC:-50}"
DO_PLOTS="${DO_PLOTS:-1}"
PROV_ETA="${PROV_ETA:-0.065}"                # provisional eta for 'submit' (recomputed at 'plot')
LEGS="${LEGS:-sanity10 sanity100 iid balanced noniid noniid_a01 noniid_a1 noniid_a100 sin bits20 classes capacity capacity_cv capacity_paper capacity10 capacity10_paper}"
CLASS_MAP="0:9,1:19,2:29,3:39,4:49,5:59,6:69,7:79,8:89,9:99"
MAX_INFLIGHT="${MAX_INFLIGHT:-0}"    # 0 = fire everything (no waiting, default).
                                     # >0 = keep at most N of YOUR jobs running/pending;
                                     #      polls `runai list jobs` and sleeps. Your project
                                     #      deserved quota is small -- MAX_INFLIGHT=3 keeps
                                     #      you inside it (non-preemptible, no complaints).
THROTTLE_POLL="${THROTTLE_POLL:-60}"

inflight(){   # count your Running/Pending jobs; echo -1 if runai is unavailable/unparseable
  command -v runai >/dev/null 2>&1 || { echo -1; return; }
  local n
  n=$(runai list jobs 2>/dev/null | awk 'NR>1 && ($0 ~ /Running|Pending|ContainerCreating/)' | wc -l) || { echo -1; return; }
  echo "${n:--1}"
}

throttle(){   # block until inflight < MAX_INFLIGHT (no-op when MAX_INFLIGHT=0)
  [ "$MAX_INFLIGHT" -gt 0 ] 2>/dev/null || return 0
  local n
  while :; do
    n=$(inflight)
    [ "$n" -lt 0 ] && return 0                      # cannot measure -> do not block
    [ "$n" -lt "$MAX_INFLIGHT" ] && return 0
    echo "    [throttle] $n job(s) in flight >= MAX_INFLIGHT=$MAX_INFLIGHT, waiting ${THROTTLE_POLL}s..."
    sleep "$THROTTLE_POLL"
  done
}

nH=$(echo $HONEST_SEEDS | wc -w)
nA=$(echo $ATTACK_SEEDS | wc -w)

# ---- per-leg config: HENV (honest/calibrate env) + ATKS ("extra_env|target") ----
set_leg(){
  HENV=(); ATKS=()
  case "$1" in
    iid)        HENV=(DS=c100);                                ATKS=("POS=1,7|reduced" "POS=3,6|reduced" "SC_FR=0 SC_CLASS=6|sameclass") ;;
    balanced)   HENV=(DS=c100 BALANCED=1 VTAG=bal);            ATKS=("POS=3,6|reduced" "SC_FR=0 SC_CLASS=6|sameclass") ;;
    noniid)     HENV=(DS=c100 PART=niid);                      ATKS=("POS=3,6|reduced" "SC_FR=0 SC_CLASS=6|sameclass") ;;
    # --- Dirichlet alpha sweep: how non-IID severity moves the honest floor & eta ---
    # small alpha = severe skew (a client may hold FEW/NO images of its own trigger class);
    # large alpha -> IID. alpha=0.5 is the 'noniid' leg above (the FL benchmark default).
    noniid_a01) HENV=(DS=c100 PART=niid DIRICHLET_ALPHA=0.1);   ATKS=("POS=3,6|reduced") ;;
    noniid_a1)  HENV=(DS=c100 PART=niid DIRICHLET_ALPHA=1.0);   ATKS=("POS=3,6|reduced") ;;
    noniid_a100) HENV=(DS=c100 PART=niid DIRICHLET_ALPHA=100);  ATKS=("POS=3,6|reduced") ;;
    sin)        HENV=(DS=c100 WMF=sin);                        ATKS=("POS=3,6|reduced") ;;
    bits20)     HENV=(DS=c100 BITS=20);                        ATKS=("POS=1,7|reduced" "POS=3,6|reduced") ;;
    classes)    HENV=(DS=c100 VTAG=spread TCMAP="$CLASS_MAP"); ATKS=("POS=3,6|reduced") ;;
    # --- paper sanity rows (all honest, 10 seeds; compare with paper_check.sh) ---
    sanity10)   HENV=(DS=c10 VTAG=sanity);                     ATKS=() ;;
    sanity100)  HENV=(DS=c100 VTAG=sanity NUM_CLIENTS=100);    ATKS=() ;;
    # --- capacity: all three verifier trigger-image modes, both datasets ---
    capacity)   HENV=(DS=c100 VTAG=nc200 NUM_CLIENTS="$CAP_NC"); ATKS=("POS=106,107|reduced") ;;
    capacity_cv) HENV=(DS=c100 VTAG=nc200 NUM_CLIENTS="$CAP_NC" TRIGMODE=client WM_NUM_TRIGGERS=50)
                 ATKS=("POS=106,107|reduced") ;;
    capacity_paper) HENV=(DS=c100 VTAG=nc200 NUM_CLIENTS="$CAP_NC" TRIGMODE=client_train WM_NUM_TRIGGERS=50)
                    ATKS=("POS=106,107|reduced") ;;
    capacity10) HENV=(DS=c10 NUM_CLIENTS="$CAP10_NC");         ATKS=("POS=16,17|reduced") ;;
    capacity10_paper) HENV=(DS=c10 VTAG=tmtrain NUM_CLIENTS="$CAP10_NC" TRIGMODE=client_train WM_NUM_TRIGGERS=50)
                      ATKS=("POS=16,17|reduced") ;;
    *) echo "  unknown leg '$1'"; return 1 ;;
  esac
}

run_leg(){   # $1=leg  $2=phase
  set_leg "$1" || return 0
  local a extra target
  case "$2" in
    honest)
      local sd="$HONEST_SEEDS"
      case "$1" in sanity10|sanity100) sd="${SANITY_SEEDS:-0 1 2 3 4 5 6 7 8 9}" ;; esac
      throttle; env "${HENV[@]}" SEEDS="$sd" bash "$RA" honest
      ;;
    attacks)
      env "${HENV[@]}" bash "$RA" calibrate
      for a in "${ATKS[@]}"; do extra="${a%|*}"; target="${a#*|}"
        throttle; env "${HENV[@]}" $extra SEEDS="$ATTACK_SEEDS" bash "$RA" "$target"
      done
      ;;
    submit)   # honest + attacks together, provisional eta, no waiting
      local sd2="$HONEST_SEEDS"
      case "$1" in sanity10|sanity100) sd2="${SANITY_SEEDS:-0 1 2 3 4 5 6 7 8 9}" ;; esac
      throttle; env "${HENV[@]}" SEEDS="$sd2" bash "$RA" honest
      for a in "${ATKS[@]}"; do extra="${a%|*}"; target="${a#*|}"
        throttle; env "${HENV[@]}" $extra USE_FIXED_ETA=1 FIXED_ETA="$PROV_ETA" \
            SEEDS="$ATTACK_SEEDS" bash "$RA" "$target"
      done
      ;;
    plot)     # local: calibrate the REAL eta, then tables + figures
      env "${HENV[@]}" bash "$RA" calibrate      || true
      env "${HENV[@]}" bash "$RA" separability   || true
      [ "$DO_PLOTS" = "1" ] && { env "${HENV[@]}" bash "$RA" PLOTALL || true; }
      ;;
  esac
}

# ============================== DRIVER =======================================
PHASE="${1:-help}"
case "$PHASE" in
  count)
    tot=0
    printf "%-14s %8s %8s %8s\n" leg honest attacks total
    for leg in $LEGS; do
      set_leg "$leg" || continue
      h=$nH
      case "$leg" in sanity10|sanity100) h=$(echo ${SANITY_SEEDS:-0 1 2 3 4 5 6 7 8 9} | wc -w) ;; esac
      a=$(( ${#ATKS[@]} * nA )); t=$((h+a)); tot=$((tot+t))
      printf "%-14s %8d %8d %8d\n" "$leg" "$h" "$a" "$t"
    done
    echo "-----------------------------------------------"
    printf "%-14s %26d GPU-jobs (1 GPU each)\n" TOTAL "$tot"
    echo
    echo "context: project sacs-zu DESERVED = 3 GPUs (guaranteed, non-preemptible)."
    echo "         jobs beyond 3 still run if the cluster is idle, but are PREEMPTIBLE."
    echo "         each job here requests -g 1, so #jobs == #GPUs."
    echo
    echo "  runai list projects   # DESERVED vs ALLOCATED right now"
    echo "  runai list jobs       # what you already have running/queued"
    echo
    echo "stay inside quota automatically:   MAX_INFLIGHT=3 ./run_everything.sh submit"
    echo "or submit a few legs at a time:    LEGS=\"iid noniid\" ./run_everything.sh submit"
    ;;
  submit|honest|attacks|plot)
    echo "=== run_everything: phase=$PHASE  legs=[$LEGS]  honest=[$HONEST_SEEDS] attack=[$ATTACK_SEEDS] BALANCED=$BALANCED${PHASE:+ } $([ "$PHASE" = submit ] && echo "PROV_ETA=$PROV_ETA") ==="
    for leg in $LEGS; do
      echo "########## $leg / $PHASE ##########"
      run_leg "$leg" "$PHASE"
    done
    echo "=== phase '$PHASE' submitted for [$LEGS] ==="
    case "$PHASE" in
      submit)  echo "everything is firing. when the cluster is done, scp results and: RES=<local> ./run_everything.sh plot" ;;
      honest)  echo "next: wait for honest jobs, then ./run_everything.sh attacks" ;;
      attacks) echo "next: wait for attack jobs, scp results, then RES=<local> ./run_everything.sh plot" ;;
      plot)    echo "tables + figures written under each family's figs/ in \$RES=${RES:-<run_all default>}" ;;
    esac
    ;;
  *)
    cat <<USAGE
usage: ./run_everything.sh <count|submit|honest|attacks|plot>
  count    print how many GPU-jobs each phase would submit (submits NOTHING)
  submit   fire EVERYTHING now (honest + attacks w/ provisional eta), no waiting
  honest   submit only honest jobs
  attacks  (after honest done) calibrate real eta + submit attacks
  plot     (results local) calibrate + separability tables + figures; set RES=<local>
knobs: LEGS HONEST_SEEDS ATTACK_SEEDS BALANCED CAP_NC CAP10_NC DO_PLOTS PROV_ETA
       MAX_INFLIGHT=N  cap concurrent jobs (0=off/default; 3 = your deserved quota)
       RUNAI_EXTRA="--node-pools <p>"  pin GPU type (cluster is heterogeneous)
legs:  sanity10 sanity100 iid balanced noniid noniid_a01 noniid_a1 noniid_a100
       sin bits20 classes capacity capacity_cv capacity_paper capacity10 capacity10_paper
Run submit/honest/attacks from the dir with submit_experiment.sh + .env;
run plot from your local repo root with RES pointing at the scp'd results.
USAGE
    exit 1 ;;
esac