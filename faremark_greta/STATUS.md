# STATUS — analysis, results, conclusions 

Standard setup unless noted: **CIFAR-100, ResNet-18, 10 clients, 50 rounds, 5 local epochs, batch 16, m=10 watermark bits, N_T=50, lambda=5, beta=0.6, alpha=0.4, config 14 (`submarine_resnet18_cifar100`).** Detector flags client i iff `BER_i >= eta`.

**THRESHOLD (canonical, frozen):** `eta = mu + 3*sigma` where `m_r` = mean BER over
clients in round r, `mu = mean_r(m_r)`, `sigma = std_r(m_r)`, over the converged tail.
Calibrated ONCE on honest-only multi-seed runs (`threshold.py calibrate`), written to
`eta_calibrated.json`, and reused for every experiment via `WM_ETA_FIXED`. The live
per-round calc in the server is commented out.

---

## The finding - TODO

...

---

## Concepts - TODO

- **BER floor** = lowest BER a client can reach for its assigned watermark, no matter
  how much it trains. Set by the (trigger_class, key, bits) "position", not by effort.
- **Why a position is hard:** the watermark lives in the smoothed softmax TAIL. A class
  the model predicts confidently (high `pmax`, low `entropy`) has no tail to shape ->
  some bits stuck -> BER floors above 0. Logged now via `pmax`/`entropy`/`dominance`
  (server, wm_verify) and `wm_loss`/`trig_train_acc` (client, wm_client.wm_stats).
- **Round-mean vs per-client BER:** eta is built from round-means (tight, sigma shrunk
  by ~sqrt(N)); it is applied to individual clients (wide). That mismatch is the flaw.

---

## Results and provenance - TODO

- **R1. Floor is structural, not the embedder** — `autop_honest_clone=1` control:
  an FR embedding via the exact honest path plateaus at the same floor as the submarine.
- **R2. Honest false positives at hard positions (headline)** — all-honest run
  (`honest_iid`): a tight round-mean eta flags honest clients at hard trigger classes.
  Strongest, most defensible result. Measure with `plots.py honest_fpr` / `thresholds`.
- **R3. Effort meter fixed** — probe forward-passes counted via `record_forward_only`,
  not `record_batch`, so effort ratios are ~100% for full-shard taps, not inflated.
- **R4. Data ablation** — triggers-only overfits (BER ~0.5), +5/class reaches the floor
  cheaply, full shard = 100% effort. `autop_common_per_class in {0,5,-1}`.
- **R5. The dilemma** — with the canonical tight eta, honest hard-position FPR is high;
  loosening eta to fix FPR lets the FR pass. Show both sides with `thresholds` panel (b).
- **R6. Easy-class selection off the table** — `trigger_class = cid % num_classes`
  (server-assigned); the FR inherits its position.

---

## EXPERIMENTS TO RUN (TODO) — one knob at a time

All IID, config 14, from `scripts/`. Run the block, wait for jobs, then plot.

### STEP 0 — Threshold (do this first)
- **Setup:** all-honest, multi-seed (10 recommended), no free-riders.
- **CLI:**
  ```
  SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_all.sh honest
  # wait for all jobs, then:
  ./run_all.sh calibrate                        # -> $RES/eta_calibrated.json
  RES=$RES ./run_all.sh PLOTALL                 # (or just the honest-family plots)
  # confirm the numbers + the assumption BEFORE any submarine run:
  python scripts/threshold.py verify --in "$RES/*/result.json" --honest-family honest_iid --eta-file "$RES/eta_calibrated.json"
  python scripts/plots.py class_difficulty --in "$RES/*/result.json" --family honest_iid --out "$RES/figs"
  ```
- **For:** produce the ONE frozen eta + prove it's the right line.
- **Expect:** `eta_calibrated.json` with `eta` ~0.09-0.12 (tail); per-seed etas within
  a few thousandths; `thresholds_honest_iid.png` shows the derivation and the honest
  FPR at that eta; `positions`/`class_dynamics` show which classes are hard and WHY
  (high `pmax`/`wm_loss`). `class_difficulty` correlates BER with per-class test
  accuracy/loss (Pearson r) -> confirms harder class ids are fuzzier-boundary classes.

> ALWAYS use >=3 seeds for every experiment (run_all.sh default SEEDS="0 1 2").

### KNOB 1 — position (hard vs easy)  [start here after Step 0]
- **Setup:** submarine, tap every round, +5/common, full scope; vary only the FR's
  trigger positions. Everything else fixed. Uses the frozen eta.
- **CLI:**
  ```
  POS=3,6 ./run_all.sh tap_every      # hard positions
  POS=1,7 ./run_all.sh tap_every      # easy positions   (re-run same family or tag it)
  RES=$RES ./run_all.sh PLOTALL
  ```
- **For:** the central hypothesis — evasion is position-dependent.
- **Expect:** hard-position FR sits near its floor (may be >= eta -> caught, but an
  honest client at that class is ALSO flagged); easy-position FR reaches BER ~0 -> hides.

### KNOB 2 — data per tap (`autop_common_per_class`)
- **Setup:** submarine, tap every round, full scope, fixed hard positions; sweep the
  data used per tap.
- **CLI:** `CPC_HOPS="0 5 -1"` style — run three families or:
  ```
  for cpc in 0 5 -1; do
    env AUTOP_COMMON_PER_CLASS=$cpc ATTACK=submarine FREE_RIDER_IDS=3,6 \
        AUTOP_SCOPE=full WM_ETA_FIXED=$(python threshold.py ... ) \
        FAMILY=tap_cpc${cpc}_iid ROUNDS=50 ./submit_experiment.sh 14 0
  done
  ```
  (or add a `tap_data` function to run_all.sh mirroring `tap_every`).
- **For:** the effort floor — how little data reaches the mark.
- **Expect:** triggers-only (0) overfits -> high BER -> caught; +5/class reaches the
  floor at ~a third of honest effort; full shard (-1) = floor at 100% effort.

### KNOB 3 — coast vs tap-every (`autop_stay_min`)
- **Setup:** submarine, +5/common, full scope, fixed positions; toggle coasting.
- **CLI:** `./run_all.sh tap_stay` (coast) vs `./run_all.sh tap_every` (tap always).
- **For:** validate the submarine coast mechanism + measure the effort saving.
- **Expect:** `tap_stay` shows `coast` actions in the timeline between `tap`s, lower
  total image-passes, BER held just under `target = eta - margin0 - safety`.

> Later knobs (park for now): `autop_scope` (block2/head), `autop_warmup_mode`
> (dynamic), non-IID (`PARTITION=dirichlet`), `calib_on_all` (eta poisoning).

---

## Current conclusion

Under IID with the fair frozen eta, the submarine free-rider is indistinguishable from
an honest client at a hard position — and the threshold that catches one flags the
other. FareMark's security reduces to eta-calibration, which is fragile (tight-vs-loose
dilemma; untrusted pool; slow convergence under heterogeneity). The next contribution
is a per-position / per-client-calibrated threshold that resolves the dilemma.