#!/usr/bin/env bash
# =============================================================================
# table9_check.sh -- reproduce ONE row of the paper's Table IX and check our setup.
#
#   PAPER TARGET (Table IX, capacity analysis):
#       ResNet-18 / CIFAR-10 / 50 clients / 50 training rounds / all honest
#       watermark detection accuracy = 95.78 %
#       main-task classification acc = 88.42 %
#
# Standalone: needs only submit_experiment.sh + .env (same dir), like run_all.
# Nothing waits -- 'submit' fires and returns; run 'check' when the jobs are done.
#
#   ./table9_check.sh submit          # fire the runs (3 seeds, parallel)
#   ./table9_check.sh check           # compare against the paper's numbers
#   RES=~/local/results ./table9_check.sh check     # ...after scp'ing results local
#
# PAPER-FAITHFUL SETTINGS USED (paper Section V-A + V-F3):
#   ResNet-18, CIFAR-10, lr 0.01, batch 16, 5 local epochs, 50 rounds   (config 11)
#   50 clients  -> cid % 10  -> exactly 5 clients per trigger class     (Table IX row)
#   ATTACK=none -> all honest (no free-riders)                          ("all participants")
#   N_T = 50 trigger samples per client                                 ("Each client utilize 50")
#   WM_TRIGGER_MODE=client_train -> trigger-sample consistency:
#         "the trigger samples used during testing are identical to those
#          employed in training"                                        (Section V-F3)
#   UNBALANCED keys (BALANCED=0) -> the paper's random +/-1 projection matrix M
#
# BIT LENGTH NOTE (read this before interpreting the number):
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

CMD="${1:-help}"
SEEDS="${SEEDS:-0 1 2}"
NC="${NC:-50}"
ROUNDS="${ROUNDS:-50}"
NT="${NT:-50}"
MODE="${MODE:-client_train}"
WM_BITS="${WM_BITS:-}"
HELDOUT="${HELDOUT:-0}"
CFG=11                                   # wm_resnet18_cifar10 (resnet18/cifar10/50 rounds/5 epochs)
RES="${RES:-/mnt/nfs/home/zu/results}"
FAM="${FAM:-table9_c10_nc${NC}_${MODE}}"
FAM_HO="table9_c10_nc${NC}_class"

PAPER_WM=95.78
PAPER_ACC=88.42

submit_one(){   # $1=family  $2=trigger mode
  local fam="$1" mode="$2" s
  echo "== submitting $fam  (clients=$NC, rounds=$ROUNDS, N_T=$NT, mode=$mode, seeds: $SEEDS)"
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
    echo "submitted. when the jobs finish:  ./table9_check.sh check"
    echo "  (or scp results locally and:  RES=<local dir> ./table9_check.sh check)"
    ;;

  check)
    python3 - "$RES" "$FAM" "$FAM_HO" "$PAPER_WM" "$PAPER_ACC" "$NC" "$NT" "$ROUNDS" << 'PY'
import glob, json, os, sys
res, fam, fam_ho, p_wm, p_acc, nc, nt, rounds = sys.argv[1:9]
p_wm, p_acc, nc, nt, rounds = float(p_wm), float(p_acc), int(nc), int(nt), int(rounds)

def load(res):
    out=[]
    for f in sorted(glob.glob(os.path.join(res,"*","result.json"))):
        try: out.append((f,json.load(open(f))))
        except Exception: pass
    return out

def famof(r): return (r.get("manifest",{}) or {}).get("family")

def honest_ber(r, tail=1):
    """mean BER over honest clients; tail=1 -> final round only."""
    vals=[]
    for h in r.get("history",[])[-tail:]:
        for p in (h.get("wm_per_client") or []):
            if not p.get("is_free_rider"): vals.append(p["ber"])
    return sum(vals)/len(vals) if vals else None

def acc(r, tail=1):
    xs=[h.get("test_acc") for h in r.get("history",[])[-tail:] if h.get("test_acc") is not None]
    return sum(xs)/len(xs) if xs else None

def clients_per_class(r):
    h=r.get("history",[])
    if not h: return None,None
    pcs=h[-1].get("wm_per_client") or []
    from collections import Counter
    c=Counter(int(p["trigger_class"]) for p in pcs)
    return len(pcs), (min(c.values()), max(c.values())) if c else None

runs=load(res)
if not runs:
    sys.exit(f"no result.json under {res} -- wrong RES? (looked in {res}/*/result.json)")

def report(target_fam, label, want_mode):
    sel=[(f,r) for f,r in runs if famof(r)==target_fam]
    if not sel:
        print(f"\n(no runs yet for family '{target_fam}')"); return None
    print(f"\n=== {label} ===")
    print(f"family: {target_fam}   seeds found: {len(sel)}")
    r0=sel[0][1]; cfg=r0.get("config",{}) or {}
    n_cl, cpc = clients_per_class(r0)
    m=r0.get("wm_bits_m"); l=r0.get("wm_group_size_l"); un=r0.get("wm_unembeddable_frac")
    print("\nSETUP (vs paper)")
    def row(name,got,want):
        ok = "ok " if (want is None or str(got)==str(want)) else "!! "
        print(f"  {ok}{name:<26} {str(got):<12} paper: {want}")
    row("clients", n_cl, nc)
    row("clients per trigger class", (f"{cpc[0]}-{cpc[1]}" if cpc and cpc[0]!=cpc[1] else (cpc[0] if cpc else "?")), 5)
    row("trigger samples N_T", cfg.get("wm_num_triggers"), nt)
    row("rounds", cfg.get("rounds"), rounds)
    row("local epochs", cfg.get("local_epochs"), 5)
    row("lr / batch", f"{cfg.get('lr')} / {cfg.get('batch_size')}", "0.01 / 16")
    row("free-riders", len(r0.get("free_rider_indices") or []), 0)
    row("trigger mode", cfg.get("wm_trigger_mode") or "class", want_mode)
    if m:
        ceil = 100*(1-0.5*un) if un is not None else None
        print(f"     watermark bits m={m}, group l={l}, unembeddable rows={un}")
        if ceil is not None:
            print(f"     -> structural wm-accuracy ceiling ~ {ceil:.2f}%  (paper reports {p_wm})")

    # metrics: final round (paper reports after 50 rounds) + converged tail
    fin_wm=[100*(1-honest_ber(r,1)) for _,r in sel if honest_ber(r,1) is not None]
    fin_ac=[acc(r,1) for _,r in sel if acc(r,1) is not None]
    t_wm=[100*(1-honest_ber(r,10)) for _,r in sel if honest_ber(r,10) is not None]
    t_ac=[acc(r,10) for _,r in sel if acc(r,10) is not None]
    mean=lambda v: sum(v)/len(v) if v else float("nan")
    sd=lambda v:(sum((x-mean(v))**2 for x in v)/len(v))**0.5 if len(v)>1 else 0.0

    print(f"\nRESULTS (mean over {len(sel)} seed(s))")
    print(f"  {'metric':<24}{'paper':>8}{'yours':>9}{'+/-':>7}{'diff':>8}   {'tail-10':>8}")
    for name,fv,tv,paper in (("watermark accuracy %",fin_wm,t_wm,p_wm),
                             ("classification acc %",fin_ac,t_ac,p_acc)):
        mv=mean(fv); d=mv-paper
        print(f"  {name:<24}{paper:>8.2f}{mv:>9.2f}{sd(fv):>7.2f}{d:>+8.2f}   {mean(tv):>8.2f}")

    ok_wm=abs(mean(fin_wm)-p_wm)<=2.0; ok_ac=abs(mean(fin_ac)-p_acc)<=2.0
    print(f"\n  VERDICT: watermark {'MATCH' if ok_wm else 'OFF'} | "
          f"classification {'MATCH' if ok_ac else 'OFF'}  (tolerance +/-2pp)")
    if not ok_wm and want_mode == "client_train":
        print("   - wm far below? check m/l above: m=5(l=2)->~75% ceiling, m=10(l=1)->~50%.")
        print("   - wm ~50%? the mark is not embedding at all (check wm_lambda, trigger mode).")
    if not ok_wm and want_mode == "class":
        print("   - (expected: this is the held-out control, NOT the paper's protocol.)")
    if not ok_ac:
        print("   - acc low? 50 clients on CIFAR-10 = 1000 imgs each; FedAvg needs the full 50 rounds.")
    return dict(wm=mean(fin_wm), acc=mean(fin_ac))

a=report(fam, "PAPER-FAITHFUL (trigger-sample consistency, V-F3)", "client_train")
b=report(fam_ho, "HELD-OUT BANK (generalisation control)", "class")
if a and b:
    print(f"\n=== memorisation gap ===")
    print(f"  watermark acc: paper-mode {a['wm']:.2f}%  vs held-out {b['wm']:.2f}%"
          f"   -> gap {a['wm']-b['wm']:+.2f} pp")
    print("  a large positive gap = the mark is memorised on the client's own trigger images,")
    print("  not a generalising property of the class (paper's own Table V caveat).")
PY
    ;;

  *)
    cat <<USAGE
usage: ./table9_check.sh <submit|check>
  submit   fire the Table IX run (ResNet-18 / CIFAR-10 / ${NC} clients / all honest)
  check    compare the finished runs against the paper (wm ${PAPER_WM}%, acc ${PAPER_ACC}%)

knobs: SEEDS('0 1 2')  NC(50)  ROUNDS(50)  NT(50)  MODE(client_train)  WM_BITS()
       HELDOUT=1  also run the held-out-bank twin (memorisation-vs-generalisation gap)
       RES=<dir>  where to look for result.json during 'check'
USAGE
    exit 1 ;;
esac
