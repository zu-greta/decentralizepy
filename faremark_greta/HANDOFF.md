# HANDOFF — start-here for the next conversation

Paste this whole file into a new chat, then upload the files listed in §4.

---

## 1. One-paragraph context

We are stress-testing **FareMark** (Li et al., IEEE IoT-J 2025) — a box-free,
output-space **watermark** free-rider detector for federated learning. Each honest
client embeds a private m-bit watermark on its trigger class's softmax; the server
extracts it on N_T=50 held-out trigger images, computes a **bit-error-rate (BER)**,
and flags a client if **BER >= η**, where **η = μ + 3σ over honest clients' BER**.
Setup: CIFAR-100, ResNet-18, 10 clients, 2 free-riders, 50 rounds, 5 local epochs,
batch 16, paper-faithful (random keys, full softmax, cumulative μ+3σ), 3 seeds. We
built an adaptive free-rider ("autopilot") that behaves honestly until η is
calibrated, then coasts/taps to hold its mark under η at minimum cost. **Goal:
characterize when FareMark can be evaded / where it fails.**

## 2. The key finding (already established)

The attack does NOT cleanly evade under plain IID — and that is the result. A
free-rider training identically to an honest client (`honest_clone` control) lands on
its **position's BER floor** (~0.10-0.20 for hard positions, ~0.00 for easy ones).
The **floor is position-dependent**: position = the (trigger_class, key, bits) triple
assigned by `cid` (server-assigned, not attacker-chosen). The detector's η is
calibrated on the **per-round MEAN** honest BER (tight, ~0.09) but applied to
**individual** clients (spread), so:

- an honest client at a hard position is flagged (~30% false-positive rate), AND
- the free-rider on a hard position is flagged too.

**The dilemma:** any η tight enough to catch an embedding free-rider also
false-positives honest hard-position clients; any η loose enough to spare them lets
the free-rider pass. FareMark's security reduces to η-calibration. The headline is a
**false-positive / threshold-calibration limitation**, not "cheap evasion."

## 3. Where things stand (see STATUS.md for provenance)

- ESTABLISHED: floor is structural (honest-clone); honest FPR (~30%); effort meter
  fixed (~1.0x); data ablation reproduces Table V (+5/class ~0.10 at ~14% cost);
  easy-class selection ruled out by threat model.
- OPEN / TO MEASURE: the two-η dilemma end-to-end (per-client η arm not yet run —
  `plot_tests.test1_fpr` computes both); eta-poisoning with position logging;
  non-IID reported under both η definitions (α=0.5 "win" was a frozen-window artifact).
- CODE STATE: cleaned to autopilot-only. Positions can be pinned via
  `free_rider_ids`. A 3-test harness (`run_tests.sh` + `plot_tests.py`) replaces the
  old sweep. Smoke test passed (pipeline wiring correct).

## 4. Files to upload to the new conversation

**Docs (read first, in this order):**
- `HANDOFF.md` (this file), `STATUS.md`, `STORYLINE.md`, `CODE_MAP.md`.

**Core code (needed for any code work):**
- `config.py`, `attacks.py`, `attacks_adaptive.py`, `wm_client.py`, `wm_verify.py`,
  `watermark.py`, `client.py`, `server.py`, `datasets.py`, `compute_meter.py`,
  `thresholds.py`, `utils.py`, `manifest.py`.
- (if touching model/robustness) `models.py`, `robustness.py` — NOT YET REVIEWED by Claude.

**Runners / plots:**
- `run_experiment.py`, `submit_experiment.sh`, `run_tests.sh`, `plot_tests.py`.
- (reproduction extras, optional) `aggregate_results.py`, `submit_sweep.sh`, `run_robustness.py`.

**Results:**
- The latest `result.json` files (or the figs) from the cleaned `run_tests.sh` run,
  once available. (The pre-cleanup figures were position-confounded — see §5.)

**Reference:**
- The FareMark paper PDF.

## 5. Ground rules for the next Claude

- Be honest about negative results: "the attack fails under a fair IID η" IS the
  finding (it's a false-positive limitation of the paper).
- The free-rider stays MODULAR: it reuses honest key/bits/λ/α/β/memory/
  `_local_train_wm` verbatim; only the control flow differs. `honest_clone` removes
  even that. Don't silently diverge the embedding path.
- Judge evasion against a fair, converged/frozen η — never the swingy cumulative one.
- Distinguish **per-round-mean η** (tight, as implemented) from **per-client η**
  (loose) — this distinction is the whole result. Don't conflate them.
- Position confounds everything with only 2 FRs: always pin `free_rider_ids` and
  average over >=3 seeds before quoting a number.
- Trigger class is SERVER-ASSIGNED (`cid % num_classes`); the attacker cannot choose
  an easy class. Easy-class evasion is off the table under the paper's threat model.

## 6. Immediate next steps

1. Run cleaned Test 1/2/3 (pinned POS_A/POS_B, 3 seeds); confirm the honest FPR and
   MEASURE both η definitions (the dilemma) from Test 1.
2. eta-poisoning re-run with per-FR position logging.
3. Non-IID reported under both frozen and converged η.
4. Design direction: a per-position / per-client-calibrated threshold that resolves
   the dilemma — the natural next contribution and the likely thesis chapter.

## 7. CLI cheat-sheet

```
# smoke (one seed, 3 rounds):
ATTACK=autopilot AUTOP_ORACLE_ETA=0.09 AUTOP_SCOPE=full AUTOP_COMMON_PER_CLASS=5 \
FREE_RIDER_IDS=3,6 ROUNDS=3 AUTOP_HONEST_UNTIL=2 AUTOP_WARMUP_CAP=2 \
FAMILY=smoke NOTE=smoke WAIT=1 ./submit_experiment.sh 14 0

# full sweep (3 seeds, 3 tests):
SEEDS="0 1 2" ./run_tests.sh

# plot:
RES=/mnt/nfs/home/zu/results ./run_tests.sh PLOT

# list configs:
python scripts/run_experiment.py --list_configs      # autopilot = idx 14
```
