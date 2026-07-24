#!/usr/bin/env bash
# =============================================================================
# submit_pool.sh -- run the whole experiment matrix on a FIXED number of pods.
#
# Instead of "1 runai job per run" (153 pods), this submits exactly PODS jobs.
# Each pod carries its own slice of the manifest and replays it with WORKERS
# concurrent runs on its single GPU. Two pods = 2 GPUs = 2*WORKERS runs at once.
#
#   ./submit_pool.sh                 # PODS=2, WORKERS=6, reads ./jobs.tsv
#   PODS=1 ./submit_pool.sh          # if quota drops further
#
# HETEROGENEOUS GPUs (e.g. one A100-80 + one A100-40) -- assign per pod:
#   POOLS="a100-80 a100-40" WORKERS_LIST="6 4" PODS=2 ./submit_pool.sh
# Pod i gets POOLS[i] and WORKERS_LIST[i]; anything unset falls back to
# RUNAI_EXTRA / WORKERS. Pods pull from a SHARED queue and claim jobs
# atomically, so a slower GPU simply completes fewer runs -- no pod sits idle
# waiting for the other, and you never have to guess the split.
#
# Build the manifest first, from your existing leg definitions:
#   rm -f jobs.tsv
#   DRYRUN=1 ./run_everything.sh submit
#
# WHY THIS IS SAFE TO LEAVE UNATTENDED
#   * RUN_TAG is deterministic, so a pod skips any run whose result.json exists.
#     Resubmit the pool after a preemption and it resumes instead of restarting.
#   * exit 2 (accuracy outside the config band) is treated as success -- it is
#     normal for every attack run and result.json is already written.
#   * A run that dies for any other reason is logged and the pod moves on; it
#     never takes the pod down.
#   * Datasets are downloaded once per pod under a lock before any worker starts.
# =============================================================================
set -uo pipefail

PODS="${PODS:-2}"                  # <-- number of runai jobs == number of GPUs
WORKERS="${WORKERS:-6}"            # default concurrent runs INSIDE each pod
read -r -a _POOLS   <<< "${POOLS:-}"        # optional per-pod node-pool
read -r -a _WORKERS <<< "${WORKERS_LIST:-}" # optional per-pod worker count
JOBS_FILE="${JOBS_FILE:-jobs.tsv}"
POOL_TAG="${POOL_TAG:-pool$(date +%m%d%H%M)}"

if [ -f .env ]; then set -a; source .env; set +a
else echo "Error: .env file not found!"; exit 1; fi

[ -s "$JOBS_FILE" ] || {
  echo "no manifest at $JOBS_FILE. Build it first:"
  echo "    rm -f $JOBS_FILE && DRYRUN=1 ./run_everything.sh submit"
  exit 1; }

GIT_REPO="https://github.com/zu-greta/decentralizepy.git"
GIT_BRANCH="${GIT_BRANCH:-main}"
PKG_SUBDIR="faremark_greta"
SCRIPT="${SCRIPT:-scripts/run_experiment.py}"

TOTAL=$(grep -cve '^[[:space:]]*$' "$JOBS_FILE")

# --- validate node-pool names up front -------------------------------------
# runai rejects an unknown pool per-job, so without this you get PODS failures
# in a row and (before the exit-code fix below) a cheerful "submitted" banner.
if [ "${#_POOLS[@]}" -gt 0 ]; then
  AVAIL=$(runai list node-pools 2>/dev/null | awk 'NR>1{print $1}' | grep -v '^$')
  if [ -z "$AVAIL" ]; then
    echo "!! could not read 'runai list node-pools' -- skipping validation, submissions may fail"
  else
    for pl in "${_POOLS[@]}"; do
      grep -qx -- "$pl" <<< "$AVAIL" || {
        echo "!! node-pool '$pl' does not exist on this cluster."
        echo "   available pools:"; sed 's/^/     /' <<< "$AVAIL"
        echo
        echo "   Either use a real name, or drop POOLS entirely and let the"
        echo "   scheduler place the pods:"
        echo "       WORKERS=3 PODS=2 ./submit_pool.sh"
        echo "   (WORKERS=3 is the safe uniform value when you do not know which"
        echo "    pod lands on the 40GB card -- see the memory note at the bottom.)"
        exit 1; }
    done
    echo "node-pools validated: ${_POOLS[*]}"
  fi
fi

echo "=== pool $POOL_TAG: $TOTAL runs -> $PODS pod(s), shared queue ==="

# Every pod gets the FULL manifest and claims rows atomically from a shared
# directory on the PVC. With mismatched GPUs this matters: a static split would
# leave the fast pod idle while the slow one finished its half.
FULL_B64=$(base64 -w0 < "$JOBS_FILE")
SUBMITTED=0

for ((i=0; i<PODS; i++)); do
  POD_POOL="${_POOLS[i]:-}"
  POD_WORKERS="${_WORKERS[i]:-$WORKERS}"
  POD_EXTRA="${RUNAI_EXTRA:-}"
  [ -n "$POD_POOL" ] && POD_EXTRA="$POD_EXTRA --node-pools $POD_POOL"
  JOB_NAME="faremark-${POOL_TAG}-w${i}"
  echo "--- $JOB_NAME : pool=${POD_POOL:-<any>} workers=$POD_WORKERS (shared queue of $TOTAL)"

  if runai submit "$JOB_NAME" \
    --project "$PROJECT" -g 1 --image "$IMAGE" --pvc "$PVC:$MOUNT" \
    ${POD_EXTRA:-} \
    --run-as-uid "$USER_UID" --run-as-gid "$USER_GID" --memory "$MEMORY" \
    -e "SHARD_B64=$FULL_B64" -e "WORKERS=$POD_WORKERS" -e "SHARD_ID=$i" \
    -e "POOL_TAG=$POOL_TAG" \
    -e "RESULTS_ROOT=${MOUNT}/home/zu/results" -e "DATA_ROOT=${MOUNT}/home/zu/data" \
    -e "GIT_REPO=$GIT_REPO" -e "GIT_BRANCH=$GIT_BRANCH" \
    -e "PKG_SUBDIR=$PKG_SUBDIR" -e "SCRIPT=$SCRIPT" \
    --command -- bash -c '
      set -uo pipefail
      export USER=zu
      mkdir -p "$RESULTS_ROOT" "$DATA_ROOT" "$RESULTS_ROOT/.poollogs"
      exec > >(tee "$RESULTS_ROOT/.poollogs/pool_w${SHARD_ID}.log") 2>&1

      echo "================================================================"
      echo "== POOL WORKER $SHARD_ID =="
      printf "  %-18s %s\n" "started (UTC)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf "  %-18s %s\n" "node"          "${NODE_NAME:-unknown}"
      printf "  %-18s %s\n" "workers"       "$WORKERS"
      nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed "s/^/  gpu: /"

      # ---- code ---------------------------------------------------------
      rm -rf /tmp/decentralizepy
      git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO" /tmp/decentralizepy 2>&1 | sed "s/^/  /"
      [ -d "/tmp/decentralizepy/$PKG_SUBDIR" ] || { echo "ERROR: $PKG_SUBDIR missing"; exit 3; }
      GIT_COMMIT="$(git -C /tmp/decentralizepy rev-parse HEAD 2>/dev/null || echo unknown)"
      export GIT_COMMIT GIT_BRANCH
      printf "  %-18s %s\n" "commit" "$GIT_COMMIT"
      export PYTHONPATH="/tmp/decentralizepy/$PKG_SUBDIR"
      cd "/tmp/decentralizepy/$PKG_SUBDIR"

      # keep N processes from each grabbing every core
      export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

      # ---- datasets, ONCE, under a lock ----------------------------------
      # Six workers calling download=True against an empty volume race and leave
      # a half-written archive. This is the classic overnight-run killer.
      LOCK="$DATA_ROOT/.dl.lock"
      for t in $(seq 1 120); do mkdir "$LOCK" 2>/dev/null && break; sleep 10; done
      python - <<PY
import os
from torchvision import datasets as d
r = os.environ["DATA_ROOT"]
for f in (d.CIFAR10, d.CIFAR100):
    f(r, train=True, download=True); f(r, train=False, download=True)
print("  datasets ready")
PY
      rmdir "$LOCK" 2>/dev/null

      # ---- manifest ------------------------------------------------------
      echo "$SHARD_B64" | base64 -d > /tmp/shard.tsv
      N=$(grep -c . /tmp/shard.tsv)
      echo "  shard: $N runs"
      echo "================================================================"

      # mkdir is atomic on POSIX, so two pods can never take the same row.
      CLAIMS="$RESULTS_ROOT/.claims_${POOL_TAG}"
      mkdir -p "$CLAIMS"

      run_one() {
        local tag="$1" cfg="$2" rep="$3" extra="$4" note="$5"
        local out="$RESULTS_ROOT/$tag"
        if [ -s "$out/result.json" ]; then return 0; fi
        mkdir "$CLAIMS/$tag" 2>/dev/null || return 0     # another pod has it
        mkdir -p "$out"
        local t0=$SECONDS
        echo "START $tag"
        local arr=($extra)
        [ -n "$note" ] && arr+=(--manifest_note "$note")
        python -u "$SCRIPT" --config_idx "$cfg" --repeat "$rep" --device cuda \
               --output_dir "$out" --data_root "$DATA_ROOT" "${arr[@]}" \
               > "$out/pod_run.log" 2>&1
        local rc=$?
        # exit 2 = accuracy outside the config band. NORMAL for attack runs and
        # result.json is written before the exit. Not a failure.
        case "$rc" in
          0|2) echo "DONE  $tag rc=$rc $((SECONDS-t0))s" ;;
          *)   echo "FAIL  $tag rc=$rc $((SECONDS-t0))s -- see $out/pod_run.log"
               rmdir "$CLAIMS/$tag" 2>/dev/null          # release for a retry
               grep -qi "out of memory" "$out/pod_run.log" 2>/dev/null && \
                 echo "        ^ OOM: lower WORKERS for this pod (WORKERS_LIST)" ;;
        esac
        return 0
      }

      # ---- drain the shard, WORKERS at a time ----------------------------
      while IFS=$"\t" read -r tag cfg rep extra note; do
        [ -z "${tag:-}" ] && continue
        while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 5; done
        run_one "$tag" "$cfg" "$rep" "$extra" "$note" &
        sleep 3          # stagger cuDNN autotune / CUDA context creation
      done < /tmp/shard.tsv
      wait

      DONE=$(cut -f1 /tmp/shard.tsv | while read -r t; do [ -s "$RESULTS_ROOT/$t/result.json" ] && echo x; done | wc -l)
      echo "  (shared queue: this pod took whatever it could claim)"
      echo "================================================================"
      echo "== POOL WORKER $SHARD_ID FINISHED: $DONE/$N complete =="
      printf "  %-18s %s\n" "finished (UTC)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "================================================================"
      sync; sleep 2
      exit 0
    '
  then
    SUBMITTED=$((SUBMITTED+1))
    echo "    submitted OK"
  else
    echo "    !! runai submit FAILED for $JOB_NAME (see the error above)"
  fi
done

echo
if [ "$SUBMITTED" -eq 0 ]; then
  cat <<EOF
=== NOTHING WAS SUBMITTED ($SUBMITTED/$PODS succeeded) ===
Fix the errors above and rerun. Nothing is running; no results were touched.
  runai list node-pools     # the valid POOLS values
  runai list jobs           # confirm: should show no faremark-$POOL_TAG jobs
EOF
  exit 1
fi

if [ "$SUBMITTED" -lt "$PODS" ]; then
  echo "=== PARTIAL: only $SUBMITTED/$PODS pods submitted ==="
  echo "The queue is shared, so the pod(s) that did start will still drain all"
  echo "$TOTAL runs -- just slower. Rerun with the SAME POOL_TAG to add the rest:"
  echo "  POOL_TAG=$POOL_TAG ./submit_pool.sh"
else
  echo "=== $SUBMITTED/$PODS pods submitted ==="
fi

cat <<EOF

  runai list jobs                                  # expect $SUBMITTED faremark-$POOL_TAG job(s)
  kubectl logs -n $NAMESPACE -l release=faremark-${POOL_TAG}-w0 -f
  ls ${MOUNT}/home/zu/results/.poollogs/           # per-pod progress logs

Resume after a preemption -- safe, skips finished runs:
  POOL_TAG=$POOL_TAG ./submit_pool.sh
(reuse the SAME POOL_TAG so the claim directory is reused)

MEMORY NOTE: Eq.14 keeps a model copy PER CLIENT, so a 200-client run needs
~9 GB on its own. Six concurrent would want ~57 GB -- fine on an 80 GB card,
an OOM on a 40 GB one. If you cannot pin node-pools and therefore do not know
which pod lands where, use a uniform WORKERS=3. Watch .poollogs for "OOM".
EOF
exit 0