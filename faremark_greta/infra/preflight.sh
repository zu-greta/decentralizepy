#!/usr/bin/env bash
# =============================================================================
# preflight.sh -- run this BEFORE submit_pool.sh. Takes ~20 seconds.
#
#   ./preflight.sh
#
# Catches the failure that would cost you the whole weekend: the pods do NOT run
# the code in this directory. They `git clone` from GitHub. So every Python fix
# must be COMMITTED AND PUSHED, or 20 runs will silently execute the old code.
#
# The shell scripts (run_now/submit_pool/submit_experiment) DO run locally and
# only need to be present here.
# =============================================================================
set -uo pipefail
FAIL=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=1; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$1"; }

echo "== 1. local scripts (these run on THIS machine) =="
for f in .env submit_experiment.sh submit_pool.sh run_now.sh; do
  [ -f "$f" ] && ok "$f present" || bad "$f MISSING"
done
grep -q 'DRYRUN.*= *"1"' submit_experiment.sh \
  && ok "submit_experiment.sh has the DRYRUN manifest branch" \
  || bad "submit_experiment.sh is the OLD one (no DRYRUN branch)"
grep -q 'RUN_TAG="${FAMILY}${USER_TAG}_rep${REPEAT}"$' submit_experiment.sh \
  && ok "RUN_TAG is deterministic (no timestamp) -- resume will work" \
  || bad "RUN_TAG still has a timestamp: a resumed pod will re-run EVERYTHING"
grep -q 'NUM_WORKERS' submit_experiment.sh \
  && ok "NUM_WORKERS -> --num_workers hook present" \
  || bad "NUM_WORKERS hook missing: you will get the slow 2-worker default"

echo
echo "== 2. THE BIG ONE: does the code the PODS will clone contain the fixes? =="
REPO="${GIT_REPO:-https://github.com/zu-greta/decentralizepy.git}"
BRANCH="${GIT_BRANCH:-main}"
PKG="${PKG_SUBDIR:-faremark_greta}"
TMP=$(mktemp -d)
echo "  cloning $REPO ($BRANCH) exactly as the pod will..."
if git clone --depth 1 --branch "$BRANCH" "$REPO" "$TMP/repo" >/dev/null 2>&1; then
  SHA=$(git -C "$TMP/repo" rev-parse --short HEAD)
  ok "cloned, HEAD = $SHA"
  WM="$TMP/repo/$PKG/faremark/watermark.py"
  if [ -f "$WM" ]; then
    ok "found $PKG/faremark/watermark.py"
    grep -q 'SMOOTH_EPS'      "$WM" && ok "  eps fix pushed (SMOOTH_EPS)"            || bad "  eps fix NOT pushed"
    grep -q 'smoothing_gain'  "$WM" && ok "  sin fix pushed (smoothing_gain guard)"  || bad "  sin fix NOT pushed"
    grep -q 'num_bits < 4'    "$WM" && ok "  make_bits fix pushed (m=1 constant)"    || bad "  make_bits fix NOT pushed -- R2 (CIFAR-10 m=1) WILL BE MEANINGLESS"
    grep -q 'balanced: bool = False' "$WM" && ok "  make_key default = False"        || bad "  make_key default still True"
    EPSDEF=$(sed -n 's/.*SMOOTH_EPS", *"\([^"]*\)".*/\1/p' "$WM" | head -1)
    case "$EPSDEF" in
      1e-3) ok "  SMOOTH_EPS defaults to 1e-3 (legacy) -- correct for THIS batch" ;;
      "")   bad "  could not read the SMOOTH_EPS default from watermark.py" ;;
      *)    warn "  SMOOTH_EPS defaults to '$EPSDEF' -- new runs will NOT match your existing families" ;;
    esac
  else
    bad "watermark.py not found at $PKG/faremark/watermark.py in the repo"
  fi
  RE="$TMP/repo/$PKG/scripts/run_experiment.py"
  [ -f "$RE" ] && { grep -q 'num_workers' "$RE" && ok "run_experiment.py accepts --num_workers" \
                    || bad "run_experiment.py has no --num_workers: NUM_WORKERS=0 will crash argparse"; }
  # uncommitted local edits?
  if git rev-parse --git-dir >/dev/null 2>&1; then
    DIRTY=$(git status --porcelain 2>/dev/null)
    # The pod clones and runs faremark/ + scripts/. infra/ runs on THIS machine,
    # so uncommitted infra edits are expected and harmless.
    POD_DIRTY=$(grep -E '(faremark|scripts)/' <<< "$DIRTY" | grep -v '/infra/' || true)
    INF_DIRTY=$(grep -E '/infra/' <<< "$DIRTY" || true)
    if [ -n "$POD_DIRTY" ]; then
      bad "uncommitted changes to code the PODS RUN -- push these or they are ignored:"
      sed 's/^/       /' <<< "$POD_DIRTY"
    else
      ok "no uncommitted changes to pod-executed code (faremark/, scripts/)"
    fi
    [ -n "$INF_DIRTY" ] && printf '  \033[36mINFO\033[0m local-only infra edits (run here, never cloned -- fine):\n%s\n' \
      "$(sed 's/^/       /' <<< "$INF_DIRTY")"
  fi
else
  bad "could not clone $REPO -- check network / credentials"
fi
rm -rf "$TMP"

echo
echo "== 3. manifest =="
if DRYRUN=1 JOBS_FILE=/tmp/_pf.tsv ./run_now.sh >/dev/null 2>&1; then
  N=$(grep -c . /tmp/_pf.tsv)
  [ "$N" -eq 20 ] && ok "run_now.sh builds $N rows" || warn "run_now.sh builds $N rows (expected 20)"
  NW=$(grep -c -- '--num_workers 0' /tmp/_pf.tsv)
  [ "$NW" -eq "$N" ] && ok "all $N rows carry --num_workers 0" || bad "only $NW/$N rows have --num_workers 0"
  grep -q -- '--wm_bits 1' /tmp/_pf.tsv && ok "R0/R2 carry --wm_bits 1" || bad "no --wm_bits 1 row: the CIFAR-10 rows are misconfigured"
  echo "  families:"; cut -f1 /tmp/_pf.tsv | sed 's/_rep[0-9]*$//' | sort -u | sed 's/^/       /'
  rm -f /tmp/_pf.tsv
else
  bad "run_now.sh failed to build a manifest"
fi

echo
echo "== 4. node pools =="
NP=$(runai list node-pools 2>/dev/null | sed '/deprecat/d;/^$/d')
if grep -qi 'Showing jobs' <<< "$NP" || [ -z "$NP" ]; then
  warn "'runai list node-pools' is not supported by this CLI (it fell through to the job list)."
  warn "=> do NOT pass POOLS. Launch without pinning; see the command below."
  PIN=0
else
  ok "node pools available:"; sed 's/^/       /' <<< "$NP"
  PIN=1
fi

echo
echo "== 5. results dir / already-done runs =="
RES_DIR="${MOUNT:-}/home/zu/results"
if [ -n "${MOUNT:-}" ] && [ -d "$RES_DIR" ]; then
  ok "results dir reachable: $RES_DIR ($(find "$RES_DIR" -name result.json 2>/dev/null | wc -l) existing runs)"
else
  warn "cannot see \$MOUNT/home/zu/results from here (normal if the PVC is pod-only)"
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "== ALL CHECKS PASSED =="
  echo "  ./run_now.sh"
  if [ "${PIN:-0}" = "1" ]; then
    echo '  POOLS="<80GB-pool> <40GB-pool>" WORKERS_LIST="6 3" PODS=2 ./submit_pool.sh'
  else
    echo "  WORKERS=3 PODS=2 ./submit_pool.sh"
    echo "     ^ no pool pinning available, so you cannot know which pod gets the"
    echo "       40GB card. WORKERS=3 is the value that is safe on either."
  fi
  echo "  runai list jobs        # expect exactly 2"
else
  echo "== FIX THE FAILURES ABOVE BEFORE LAUNCHING =="
  echo "   Most likely cause: Python changes are not committed+pushed."
  echo "   git add faremark/watermark.py && git commit -m 'watermark fixes' && git push"
fi
exit "$FAIL"