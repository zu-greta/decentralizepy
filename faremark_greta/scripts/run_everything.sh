#!/usr/bin/env bash
# =============================================================================
# run_everything.sh  --  fire the whole thesis matrix through run_all, in stages.
# 
#     ./run_everything.sh honest      # submit every leg's honest jobs, return
#     # ... wait for them to finish on the cluster (runai / kubectl) ...
#     ./run_everything.sh attacks     # calibrate each leg's eta + submit its attacks, return
#     # ... wait ... then scp the results dir to local ...
#     RES=~/local/results ./run_everything.sh plot     # local: separability tables + figures
#
# LEGS (override with LEGS="iid sin"):
#   iid        CIFAR-100 IID, default m, UNBALANCED keys; reduced 1,7 + 3,6 + sameclass(0->6)
#   balanced   as iid but BALANCED keys (VTAG=bal)  -> unbalanced-vs-balanced comparison
#   noniid     CIFAR-100 Dirichlet(0.5); reduced 3,6 + sameclass
#   sin        CIFAR-100 IID, sin() smoothing (Eq.9); reduced 3,6
#   bits20     CIFAR-100 IID, m=20 bits; reduced 1,7 + 3,6
#   classes    CIFAR-100 IID, trigger classes 9,19,..,99 (VTAG=spread); reduced -> classes 39,69
#   capacity   CIFAR-100 IID, CAP_NC clients (VTAG=nc200, classes SHARED); reduced 106,107
#   capacity10 CIFAR-10  IID, CAP10_NC clients; reduced 16,17
#
# KNOBS: HONEST_SEEDS('0 1 2 3 4 5')  ATTACK_SEEDS('0 1 2')  BALANCED(0)  CAP_NC(200)
#        CAP10_NC(50)  DO_PLOTS(1)  LEGS(all)  RES(plot phase: where the results are)
#        SKIP_CALIBRATE(0)  PROVISIONAL_ETA(0.064)  
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RA="$HERE/run_all.sh"                       # run_all lives next to this script

HONEST_SEEDS="${HONEST_SEEDS:-0 1 2 3 4 5}"
ATTACK_SEEDS="${ATTACK_SEEDS:-0 1 2}"
export BALANCED="${BALANCED:-0}"
CAP_NC="${CAP_NC:-200}"
CAP10_NC="${CAP10_NC:-50}"
DO_PLOTS="${DO_PLOTS:-1}"
LEGS="${LEGS:-iid balanced noniid sin bits20 classes capacity capacity10}"
CLASS_MAP="0:9,1:19,2:29,3:39,4:49,5:59,6:69,7:79,8:89,9:99"

# optional: skip calibrate during attack, use provisional eta, and compute threshold during plot
SKIP_CALIBRATE="${SKIP_CALIBRATE:-0}"
PROVISIONAL_ETA="${PROVISIONAL_ETA:-0.064}"

# --------------------------------------------------------------------------- #
#  one function per leg; $1 = phase (honest | attacks | plot)                  #
#  env is set inline per run_all call and kept IDENTICAL across honest/attacks #
#  so families / eta files line up.                                           #
# --------------------------------------------------------------------------- #
_plot(){ # $@ = env prefix for run_all
  # If we skipped calibrate during attacks, run it now to get correct thresholds
  if [ "$SKIP_CALIBRATE" = "1" ]; then
    echo "Plot phase: running calibrate (because SKIP_CALIBRATE=1)"
    env "$@" bash "$RA" calibrate
  fi
  env "$@" bash "$RA" separability || true
  [ "$DO_PLOTS" = "1" ] && { env "$@" bash "$RA" PLOTALL || true; }
}

leg_iid(){ case "$1" in
  honest)  env SEEDS="$HONEST_SEEDS" DS=c100 bash "$RA" honest ;;
  attacks)
    if [ "$SKIP_CALIBRATE" = "1" ]; then
      echo "iid attacks: using provisional ETA=$PROVISIONAL_ETA (skip calibrate)"
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c100 POS=1,7 bash "$RA" reduced
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c100 POS=3,6 bash "$RA" reduced
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c100 SC_FR=0 SC_CLASS=6 bash "$RA" sameclass
    else
      env DS=c100 bash "$RA" calibrate
      env SEEDS="$ATTACK_SEEDS" DS=c100 POS=1,7 bash "$RA" reduced
      env SEEDS="$ATTACK_SEEDS" DS=c100 POS=3,6 bash "$RA" reduced
      env SEEDS="$ATTACK_SEEDS" DS=c100 SC_FR=0 SC_CLASS=6 bash "$RA" sameclass
    fi
    ;;
  plot)    _plot DS=c100 ;;
esac; }

leg_balanced(){ case "$1" in
  honest)  env SEEDS="$HONEST_SEEDS" DS=c100 BALANCED=1 VTAG=bal bash "$RA" honest ;;
  attacks)
    if [ "$SKIP_CALIBRATE" = "1" ]; then
      echo "balanced attacks: using provisional ETA=$PROVISIONAL_ETA (skip calibrate)"
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c100 BALANCED=1 VTAG=bal POS=3,6 bash "$RA" reduced
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c100 BALANCED=1 VTAG=bal SC_FR=0 SC_CLASS=6 bash "$RA" sameclass
    else
      env DS=c100 BALANCED=1 VTAG=bal bash "$RA" calibrate
      env SEEDS="$ATTACK_SEEDS" DS=c100 BALANCED=1 VTAG=bal POS=3,6 bash "$RA" reduced
      env SEEDS="$ATTACK_SEEDS" DS=c100 BALANCED=1 VTAG=bal SC_FR=0 SC_CLASS=6 bash "$RA" sameclass
    fi
    ;;
  plot)    _plot DS=c100 VTAG=bal ;;
esac; }

leg_noniid(){ case "$1" in
  honest)  env SEEDS="$HONEST_SEEDS" DS=c100 PART=niid bash "$RA" honest ;;
  attacks)
    if [ "$SKIP_CALIBRATE" = "1" ]; then
      echo "noniid attacks: using provisional ETA=$PROVISIONAL_ETA (skip calibrate)"
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c100 PART=niid POS=3,6 bash "$RA" reduced
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c100 PART=niid SC_FR=0 SC_CLASS=6 bash "$RA" sameclass
    else
      env DS=c100 PART=niid bash "$RA" calibrate
      env SEEDS="$ATTACK_SEEDS" DS=c100 PART=niid POS=3,6 bash "$RA" reduced
      env SEEDS="$ATTACK_SEEDS" DS=c100 PART=niid SC_FR=0 SC_CLASS=6 bash "$RA" sameclass
    fi
    ;;
  plot)    _plot DS=c100 PART=niid ;;
esac; }

leg_sin(){ case "$1" in
  honest)  env SEEDS="$HONEST_SEEDS" DS=c100 WMF=sin bash "$RA" honest ;;
  attacks)
    if [ "$SKIP_CALIBRATE" = "1" ]; then
      echo "sin attacks: using provisional ETA=$PROVISIONAL_ETA (skip calibrate)"
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c100 WMF=sin POS=3,6 bash "$RA" reduced
    else
      env DS=c100 WMF=sin bash "$RA" calibrate
      env SEEDS="$ATTACK_SEEDS" DS=c100 WMF=sin POS=3,6 bash "$RA" reduced
    fi
    ;;
  plot)    _plot DS=c100 WMF=sin ;;
esac; }

leg_bits20(){ case "$1" in
  honest)  env SEEDS="$HONEST_SEEDS" DS=c100 BITS=20 bash "$RA" honest ;;
  attacks)
    if [ "$SKIP_CALIBRATE" = "1" ]; then
      echo "bits20 attacks: using provisional ETA=$PROVISIONAL_ETA (skip calibrate)"
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c100 BITS=20 POS=1,7 bash "$RA" reduced
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c100 BITS=20 POS=3,6 bash "$RA" reduced
    else
      env DS=c100 BITS=20 bash "$RA" calibrate
      env SEEDS="$ATTACK_SEEDS" DS=c100 BITS=20 POS=1,7 bash "$RA" reduced
      env SEEDS="$ATTACK_SEEDS" DS=c100 BITS=20 POS=3,6 bash "$RA" reduced
    fi
    ;;
  plot)    _plot DS=c100 BITS=20 ;;
esac; }

leg_classes(){ case "$1" in
  honest)  env SEEDS="$HONEST_SEEDS" DS=c100 VTAG=spread TCMAP="$CLASS_MAP" bash "$RA" honest ;;
  attacks)
    if [ "$SKIP_CALIBRATE" = "1" ]; then
      echo "classes attacks: using provisional ETA=$PROVISIONAL_ETA (skip calibrate)"
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c100 VTAG=spread TCMAP="$CLASS_MAP" POS=3,6 bash "$RA" reduced
    else
      env DS=c100 VTAG=spread bash "$RA" calibrate
      env SEEDS="$ATTACK_SEEDS" DS=c100 VTAG=spread TCMAP="$CLASS_MAP" POS=3,6 bash "$RA" reduced
    fi
    ;;
  plot)    _plot DS=c100 VTAG=spread ;;
esac; }

leg_capacity(){ case "$1" in
  honest)  env SEEDS="$HONEST_SEEDS" DS=c100 VTAG=nc200 NUM_CLIENTS="$CAP_NC" bash "$RA" honest ;;
  attacks)
    if [ "$SKIP_CALIBRATE" = "1" ]; then
      echo "capacity attacks: using provisional ETA=$PROVISIONAL_ETA (skip calibrate)"
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c100 VTAG=nc200 NUM_CLIENTS="$CAP_NC" POS=106,107 bash "$RA" reduced
    else
      env DS=c100 VTAG=nc200 NUM_CLIENTS="$CAP_NC" bash "$RA" calibrate
      env SEEDS="$ATTACK_SEEDS" DS=c100 VTAG=nc200 NUM_CLIENTS="$CAP_NC" POS=106,107 bash "$RA" reduced
    fi
    ;;
  plot)    _plot DS=c100 VTAG=nc200 ;;
esac; }

leg_capacity10(){ case "$1" in
  honest)  env SEEDS="$HONEST_SEEDS" DS=c10 NUM_CLIENTS="$CAP10_NC" bash "$RA" honest ;;
  attacks)
    if [ "$SKIP_CALIBRATE" = "1" ]; then
      echo "capacity10 attacks: using provisional ETA=$PROVISIONAL_ETA (skip calibrate)"
      ETA="$PROVISIONAL_ETA" env SEEDS="$ATTACK_SEEDS" DS=c10 NUM_CLIENTS="$CAP10_NC" POS=16,17 bash "$RA" reduced
    else
      env DS=c10 NUM_CLIENTS="$CAP10_NC" bash "$RA" calibrate
      env SEEDS="$ATTACK_SEEDS" DS=c10 NUM_CLIENTS="$CAP10_NC" POS=16,17 bash "$RA" reduced
    fi
    ;;
  plot)    _plot DS=c10 ;;
esac; }

# ============================== DRIVER =======================================
PHASE="${1:-help}"
case "$PHASE" in
  honest|attacks|plot)
    echo "=== run_everything: phase=$PHASE  legs=[$LEGS]  honest_seeds=[$HONEST_SEEDS] attack_seeds=[$ATTACK_SEEDS] BALANCED=$BALANCED SKIP_CALIBRATE=$SKIP_CALIBRATE PROVISIONAL_ETA=$PROVISIONAL_ETA ==="
    for leg in $LEGS; do
      echo "########## $leg / $PHASE ##########"
      "leg_$leg" "$PHASE"
    done
    echo "=== phase '$PHASE' submitted for [$LEGS] ==="
    case "$PHASE" in
      honest)  echo "next: wait for the honest jobs to finish, then ./run_everything.sh attacks" ;;
      attacks)
        if [ "$SKIP_CALIBRATE" = "1" ]; then
          echo "next: (provisional mode) wait for attacks, scp results, then run: RES=<local> ./run_everything.sh plot"
        else
          echo "next: wait for the attack jobs, then scp results to local and: RES=<local> ./run_everything.sh plot"
        fi
        ;;
      plot)    echo "figures + tables written under each family's figs/ (in \$RES=${RES:-run_all default})" ;;
    esac
    ;;
  *)
    cat <<USAGE
usage: ./run_everything.sh <honest|attacks|plot>
  honest   submit every leg's honest jobs (fire-and-forget), then STOP
  attacks  (after honest done) calibrate each leg's eta + submit its attacks, then STOP
           (if SKIP_CALIBRATE=1, skip calibrate and use PROVISIONAL_ETA instead)
  plot     (after scp to local) separability tables + figures; set RES=<local results dir>
           (if SKIP_CALIBRATE=1, automatically runs calibrate first to compute thresholds)
knobs: LEGS HONEST_SEEDS ATTACK_SEEDS BALANCED CAP_NC CAP10_NC DO_PLOTS
       SKIP_CALIBRATE (0/1) PROVISIONAL_ETA (default 0.064)
legs:  iid balanced noniid sin bits20 classes capacity capacity10
Run honest/attacks from the dir with submit_experiment.sh + .env; run plot from your
local repo root with RES pointing at the scp'd results.
USAGE
    exit 1 ;;
esac