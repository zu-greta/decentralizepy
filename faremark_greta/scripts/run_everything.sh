#!/usr/bin/env bash
# =============================================================================
# run_everything.sh  --  ONE command, the whole thesis matrix.
#
#   BASE=/mnt/nfs/home/zu/results/thesis_$(date +%F) ./run_everything.sh
#
# Each "leg" lands in its OWN results subdir (BASE/<leg>) with its own frozen eta
# and its own figs/, so plotting never mixes legs. Within a leg the flow is:
#     honest (parallel jobs)  ->  wait  ->  calibrate eta
#   ->  attacks (parallel jobs)  ->  wait  ->  separability tables + all figures
# The waits watch the leg's RES dir for result.json files (jobs are fired with
# WAIT=0 so the cluster runs them in parallel), so the single command walks the
# whole matrix start to finish and you come back to complete results + plots.
#
# LEGS (override with LEGS="iid sin"):
#   iid        CIFAR-100 IID, default m, UNBALANCED keys, reduced easy(1,7)+hard(3,6)+sameclass(0->6)
#   balanced   same as iid but BALANCED keys -> direct unbalanced-vs-balanced comparison (F6 control)
#   noniid     CIFAR-100 Dirichlet(0.5), reduced hard(3,6) + sameclass(0->6)
#   sin        CIFAR-100 IID, sin() smoothing (paper Eq.9), reduced hard(3,6)
#   bits20     CIFAR-100 IID, m=20 bits, reduced easy(1,7) + hard(3,6)
#   capacity   CIFAR-100 IID, CAP_NC clients (>100 -> classes SHARED); FR on classes 6,7 (nat. overlap)
#   capacity10 CIFAR-10 IID, CAP10_NC clients (>10 -> classes shared); FR on classes 6,7 (comparison)
#
# KNOBS:
#   BASE           parent results dir (default results/thesis_<date>)
#   HONEST_SEEDS   default '0 1 2 3 4 5'  (more = stabler eta; >=3 always)
#   ATTACK_SEEDS   default '0 1 2'        (>=3)
#   BALANCED       default 0 (paper-faithful unbalanced keys). The 'balanced' leg forces 1.
#   CAP_NC         CIFAR-100 capacity clients (default 200 = 2/class)
#   CAP10_NC       CIFAR-10  capacity clients (default 50  = 5/class)
#   DO_PLOTS       default 1 (make figures inline). Set 0 if you scp results and plot locally.
#   POLL_TIMEOUT   max seconds to wait at each barrier (default 10800 = 3h)
# =============================================================================
set -uo pipefail

BASE="${BASE:-/mnt/nfs/home/zu/results/thesis_$(date +%Y%m%d)}"
HONEST_SEEDS="${HONEST_SEEDS:-0 1 2 3 4 5}"
ATTACK_SEEDS="${ATTACK_SEEDS:-0 1 2}"
export BALANCED="${BALANCED:-0}"
CAP_NC="${CAP_NC:-200}"
CAP10_NC="${CAP10_NC:-50}"
DO_PLOTS="${DO_PLOTS:-1}"
LEGS="${LEGS:-iid balanced noniid sin bits20 capacity capacity10}"
POLL_TIMEOUT="${POLL_TIMEOUT:-10800}"

nH=$(echo $HONEST_SEEDS | wc -w)
nA=$(echo $ATTACK_SEEDS | wc -w)

# ---- barrier: wait until $2 result.json exist under $1 (or timeout) ----------
wait_for(){
  local RES="$1" want="$2" have=0 waited=0
  echo "  ...waiting for $want result.json in $RES"
  while :; do
    have=$(ls "$RES"/*/result.json 2>/dev/null | wc -l)
    [ "$have" -ge "$want" ] && { echo "  barrier reached: $have/$want"; return 0; }
    if [ "$waited" -ge "$POLL_TIMEOUT" ]; then
      echo "  !! timeout ($((POLL_TIMEOUT/60))min) at $have/$want -- proceeding with what finished"
      return 0
    fi
    sleep 30; waited=$((waited+30))
    [ $((waited % 300)) -lt 30 ] && echo "  still waiting: $have/$want ($((waited/60))min)"
  done
}

analyze(){   # $1=RES  rest=run_all env prefix (DS=.. PART=.. WMF=.. BITS=..)
  local RES="$1"; shift
  env RES="$RES" "$@" ./run_all.sh separability || true       # numpy tables (cheap, always)
  if [ "$DO_PLOTS" = "1" ]; then
    env RES="$RES" "$@" ./run_all.sh PLOTALL || true          # matplotlib figures
    echo "  leg done -> tables & figures in $RES/figs/"
  else
    echo "  leg done -> tables in $RES/figs/ (DO_PLOTS=0: scp $RES and plot locally)"
  fi
}

# ============================== LEGS =========================================
leg_iid(){
  local RES="$BASE/c100_iid"; mkdir -p "$RES"
  echo "########## LEG iid -> $RES ##########"
  env RES="$RES" SEEDS="$HONEST_SEEDS" DS=c100 ./run_all.sh honest
  wait_for "$RES" "$nH"
  env RES="$RES" DS=c100 ./run_all.sh calibrate
  env RES="$RES" SEEDS="$ATTACK_SEEDS" DS=c100 POS=1,7 ./run_all.sh reduced
  env RES="$RES" SEEDS="$ATTACK_SEEDS" DS=c100 POS=3,6 ./run_all.sh reduced
  env RES="$RES" SEEDS="$ATTACK_SEEDS" DS=c100 SC_FR=0 SC_CLASS=6 ./run_all.sh sameclass
  wait_for "$RES" $((nH + 3*nA))
  analyze "$RES" DS=c100
}

leg_balanced(){
  # identical to leg_iid but with BALANCED keys, so you can put c100_iid (unbalanced)
  # next to c100_iid_bal (balanced) and see how much of the honest-BER spread / overlap
  # was the same-sign-row artifact (F6) vs real class difficulty.
  local RES="$BASE/c100_iid_bal"; mkdir -p "$RES"
  echo "########## LEG balanced (BALANCED=1) -> $RES ##########"
  env RES="$RES" BALANCED=1 SEEDS="$HONEST_SEEDS" DS=c100 ./run_all.sh honest
  wait_for "$RES" "$nH"
  env RES="$RES" BALANCED=1 DS=c100 ./run_all.sh calibrate
  env RES="$RES" BALANCED=1 SEEDS="$ATTACK_SEEDS" DS=c100 POS=3,6 ./run_all.sh reduced
  env RES="$RES" BALANCED=1 SEEDS="$ATTACK_SEEDS" DS=c100 SC_FR=0 SC_CLASS=6 ./run_all.sh sameclass
  wait_for "$RES" $((nH + 2*nA))
  analyze "$RES" DS=c100
}

leg_noniid(){
  local RES="$BASE/c100_niid"; mkdir -p "$RES"
  echo "########## LEG noniid -> $RES ##########"
  env RES="$RES" SEEDS="$HONEST_SEEDS" DS=c100 PART=niid ./run_all.sh honest
  wait_for "$RES" "$nH"
  env RES="$RES" DS=c100 PART=niid ./run_all.sh calibrate
  env RES="$RES" SEEDS="$ATTACK_SEEDS" DS=c100 PART=niid POS=3,6 ./run_all.sh reduced
  env RES="$RES" SEEDS="$ATTACK_SEEDS" DS=c100 PART=niid SC_FR=0 SC_CLASS=6 ./run_all.sh sameclass
  wait_for "$RES" $((nH + 2*nA))
  analyze "$RES" DS=c100 PART=niid
}

leg_sin(){
  local RES="$BASE/c100_sin"; mkdir -p "$RES"
  echo "########## LEG sin -> $RES ##########"
  env RES="$RES" SEEDS="$HONEST_SEEDS" DS=c100 WMF=sin ./run_all.sh honest
  wait_for "$RES" "$nH"
  env RES="$RES" DS=c100 WMF=sin ./run_all.sh calibrate
  env RES="$RES" SEEDS="$ATTACK_SEEDS" DS=c100 WMF=sin POS=3,6 ./run_all.sh reduced
  wait_for "$RES" $((nH + nA))
  analyze "$RES" DS=c100 WMF=sin
}

leg_bits20(){
  local RES="$BASE/c100_b20"; mkdir -p "$RES"
  echo "########## LEG bits20 -> $RES ##########"
  env RES="$RES" SEEDS="$HONEST_SEEDS" DS=c100 BITS=20 ./run_all.sh honest
  wait_for "$RES" "$nH"
  env RES="$RES" DS=c100 BITS=20 ./run_all.sh calibrate
  env RES="$RES" SEEDS="$ATTACK_SEEDS" DS=c100 BITS=20 POS=1,7 ./run_all.sh reduced
  env RES="$RES" SEEDS="$ATTACK_SEEDS" DS=c100 BITS=20 POS=3,6 ./run_all.sh reduced
  wait_for "$RES" $((nH + 2*nA))
  analyze "$RES" DS=c100 BITS=20
}

leg_capacity(){
  local RES="$BASE/c100_cap"; mkdir -p "$RES"
  echo "########## LEG capacity ($CAP_NC clients) -> $RES ##########"
  # >100 clients on CIFAR-100 -> cid%100 forces classes to be SHARED. With CAP_NC=200
  # every class has 2 clients; making cid 106/107 free-ride pits them against the honest
  # clients on classes 6/7 (natural same-class overlap). NUM_CLIENTS is exported so both
  # honest and reduced legs see it.
  export NUM_CLIENTS="$CAP_NC"
  env RES="$RES" SEEDS="$HONEST_SEEDS" DS=c100 ./run_all.sh honest
  wait_for "$RES" "$nH"
  env RES="$RES" DS=c100 ./run_all.sh calibrate
  env RES="$RES" SEEDS="$ATTACK_SEEDS" DS=c100 POS=106,107 ./run_all.sh reduced
  wait_for "$RES" $((nH + nA))
  analyze "$RES" DS=c100
  unset NUM_CLIENTS
}

leg_capacity10(){
  local RES="$BASE/c10_cap"; mkdir -p "$RES"
  echo "########## LEG capacity10 (CIFAR-10, $CAP10_NC clients) -> $RES ##########"
  # CIFAR-10 comparison for capacity: >10 clients -> classes shared, but each client keeps
  # MUCH more data than the CIFAR-100 case (50k/50 = 1000 imgs, ~100 trigger imgs/client), so
  # embedding is NOT data-starved -> isolates the sharing effect from the thin-data confound.
  # With CAP10_NC=50 every class has 5 clients; FR cid 16/17 share classes 6/7 with honest 6/7.
  export NUM_CLIENTS="$CAP10_NC"
  env RES="$RES" SEEDS="$HONEST_SEEDS" DS=c10 ./run_all.sh honest
  wait_for "$RES" "$nH"
  env RES="$RES" DS=c10 ./run_all.sh calibrate
  env RES="$RES" SEEDS="$ATTACK_SEEDS" DS=c10 POS=16,17 ./run_all.sh reduced
  wait_for "$RES" $((nH + nA))
  analyze "$RES" DS=c10
  unset NUM_CLIENTS
}

# ============================== DRIVER =======================================
echo "=== run_everything: BASE=$BASE  legs=[$LEGS]  honest_seeds=[$HONEST_SEEDS] attack_seeds=[$ATTACK_SEEDS]  BALANCED=$BALANCED ==="
mkdir -p "$BASE"
for leg in $LEGS; do
  case "$leg" in
    iid)        leg_iid ;;
    balanced)   leg_balanced ;;
    noniid)     leg_noniid ;;
    sin)        leg_sin ;;
    bits20)     leg_bits20 ;;
    capacity)   leg_capacity ;;
    capacity10) leg_capacity10 ;;
    *) echo "unknown leg '$leg' (want: iid balanced noniid sin bits20 capacity capacity10)";;
  esac
done

echo
echo "=== ALL LEGS DONE ==="
echo "results tree:"
for leg in $LEGS; do
  case "$leg" in
    iid) d="$BASE/c100_iid";; balanced) d="$BASE/c100_iid_bal";;
    noniid) d="$BASE/c100_niid";; sin) d="$BASE/c100_sin";;
    bits20) d="$BASE/c100_b20";; capacity) d="$BASE/c100_cap";;
    capacity10) d="$BASE/c10_cap";; *) continue;;
  esac
  echo "  $leg -> $d/figs/   (eta in $d/eta_*.json)"
done