#!/usr/bin/env bash
# =============================================================================
# paper_check.sh -- Table IX + check setup
#
#   PAPER TARGET (Table IX, capacity analysis):
#       ResNet-18 / CIFAR-10 / 50 clients / 50 training rounds / all honest
#       watermark detection accuracy = 95.78 %
#       main-task classification acc = 88.42 %
#
#   ./paper_check.sh submit          # fire the runs (3 seeds, parallel)
#   ./paper_check.sh check           # compare against the paper's numbers
#   RES=~/local/results ./paper_check.sh check     # locally
#
# PAPER SETTINGS:
#   ResNet-18, CIFAR-10, lr 0.01, batch 16, 5 local epochs, 50 rounds   (config 11)
#   50 clients  -> cid % 10  -> exactly 5 clients per trigger class     (Table IX row)
#   ATTACK=none -> all honest (no free-riders)                          ("all participants")
#   N_T = 50 trigger samples per client                                 ("Each client utilize 50")
#   WM_TRIGGER_MODE=client_train -> trigger-sample consistency:
#         "the trigger samples used during testing are identical to those
#          employed in training"                                        (Section V-F3)
#   UNBALANCED keys (BALANCED=0) -> the paper's random +/-1 projection matrix M
#
# TODO check: BIT LENGTH
#   CIFAR-10 has n=10 classes, so the code picks m = max(2, n//10) = 2 bits, l = n/m = 5.
#   With random (unbalanced) key rows of length 5, P(a row is all-same-sign) = 2^(1-5)
#   = 6.25% of bits are structurally unembeddable and sit at ~50% error, giving
#       expected BER floor        ~ 0.031
#       watermark accuracy ceiling ~ 96.9 %
#   The paper's 95.78% sits just under that ceiling -- consistent. If you instead see
#   ~75%, the run used m=5 (l=2, 50% unembeddable); if ~50%, m=10 (l=1, all stuck).
#   Override with WM_BITS=<m> to test that directly.
#
# KNOBS: SEEDS('0 1 2')  NC(50)  ROUNDS(50)  NT(50)  MODE(client_train)  WM_BITS()
#        HELDOUT=1  -> also submit the held-out-bank twin (MODE=class) for comparison
#        RES  -> where result.json live (check phase); default the cluster results dir
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"   # so `check` can find ../scripts/paper_check.py

ROW="${ROW:-t9}"          # t9 | c10 | c100   (which paper row to check)
case "$ROW" in
  t9)   PAPER_WM=95.78; PAPER_ACC=88.42; DEF_NC=50;  DEF_CFG=11; DEF_MODE=client_train
        ROWDESC="Table IX  ResNet-18 / CIFAR-10 / 50 clients (capacity)" ;;
  c10)  PAPER_WM=99.72; PAPER_ACC=90.78; DEF_NC=10;  DEF_CFG=11; DEF_MODE=class
        ROWDESC="Table I+II  ResNet-18 / CIFAR-10 / 10 clients" ;;
  c100) PAPER_WM=99.71; PAPER_ACC=75.31; DEF_NC=100; DEF_CFG=14; DEF_MODE=class
        ROWDESC="Table I+II  ResNet-18 / CIFAR-100 / 100 clients" ;;
  *) echo "ROW must be t9 | c10 | c100"; exit 1 ;;
esac

CMD="${1:-help}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
NC="${NC:-$DEF_NC}"
ROUNDS="${ROUNDS:-50}"
NT="${NT:-50}"
MODE="${MODE:-$DEF_MODE}"
WM_BITS="${WM_BITS:-}"
HELDOUT="${HELDOUT:-0}"
CFG="${CFG:-$DEF_CFG}"
RES="${RES:-/mnt/nfs/home/zu/results}"
FAM="${FAM:-paper_${ROW}_nc${NC}_${MODE}}"
FAM_HO="paper_${ROW}_nc${NC}_class"


submit_one(){   # $1=family  $2=trigger mode
  local fam="$1" mode="$2" s
  echo "== [$ROWDESC] submitting $fam  (clients=$NC, rounds=$ROUNDS, N_T=$NT, mode=$mode, seeds: $SEEDS)"
  for s in $SEEDS; do
    env ATTACK=none NUM_CLIENTS="$NC" ROUNDS="$ROUNDS" \
        WM_NUM_TRIGGERS="$NT" WM_TRIGGER_MODE="$mode" \
        ${WM_BITS:+WM_BITS=$WM_BITS} \
        FAMILY="$fam" \
        NOTE="paper Table IX check: resnet18/cifar10/${NC} clients/all honest/mode=$mode" \
        WAIT=0 ./submit_experiment.sh "$CFG" "$s"
  done
}

case "$CMD" in
  submit)
    submit_one "$FAM" "$MODE"
    [ "$HELDOUT" = "1" ] && submit_one "$FAM_HO" "class"
    echo
    echo "submitted. when the jobs finish:  ./paper_check.sh check"
    echo "  (or scp results locally and:  RES=<local dir> ./paper_check.sh check)"
    ;;

  check)
    python3 "${HERE:-.}/../scripts/paper_check.py" \
        --row "$ROW" --in "$RES/*/result.json" \
        --family "$FAM" ${HELDOUT:+--heldout-family "$FAM_HO"} \
        --clients "$NC" --nt "$NT" --rounds "$ROUNDS"
    ;;

  *)
    cat <<USAGE
usage: ROW=<t9|c10|c100> ./paper_check.sh <submit|check>
  ROW=c10   Table I+II  CIFAR-10  10 clients   (wm 99.72 / acc 90.78)
  ROW=c100  Table I+II  CIFAR-100 100 clients  (wm 99.71 / acc 75.31)
  ROW=t9    Table IX    CIFAR-10  50 clients   (wm 95.78 / acc 88.42)
  submit   fire the run for $ROW (ResNet-18 / CIFAR-10 / ${NC} clients / all honest)
  check    compare the finished runs against the paper (wm ${PAPER_WM}%, acc ${PAPER_ACC}%)

knobs: SEEDS('0 1 2')  NC(50)  ROUNDS(50)  NT(50)  MODE(client_train)  WM_BITS()
       HELDOUT=1  also run the held-out-bank twin (memorisation-vs-generalisation gap)
       RES=<dir>  where to look for result.json during 'check'
USAGE
    exit 1 ;;
esac