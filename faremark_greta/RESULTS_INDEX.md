# Results summary

## Thresholds
- **η tight = 0.064:** threshold at μ+3σ over the *round-mean* honest BER (run and calibrated as average over 10 seeds)
- **η loose = 0.264:** threshold at μ+3σ over *per-client* honest BER 

---

## Group A — baseline, isolation, and the operating point  (3 seeds; A1 honest = 6 seeds)

### [operating_point.png](results/groups/figs/operating_point.png) (+ operating_point.md) — the headline figure

**What it is.** Every other plot shows honest and free-rider BER *overlap*. This one turns that
overlap into the question a defender actually asks: *if I pick a threshold that only false-alarms
on a small fraction of honest clients, what fraction of free-riders do I catch?*

**Each term:**
- **FPR budget (false-positive rate):** the fraction of *honest* clients you're willing to wrongly
  flag. No operator can accuse many honest participants, so the budget is small — 1 %, 5 %, 10 %.
- **η per budget:** the threshold producing that honest FPR. (At the 1 % budget the only way to
  false-alarm ≤ 1 % is η = 1.0 = flag *nobody*, because honest BER is spread and quantised — so
  recall is 0 there by construction.)
- **recall:** the fraction of *free-riders* actually caught at that η. **This is the number you
  want near 1.0** (catch the cheaters).
- **global vs per-class oracle:** *global* = one η for everyone — what a real server must deploy,
  since it can't know a client's trigger class in advance. *per-class oracle* = a separate η
  calibrated on the honest clients of the free-rider's *own* class — a cheat the server cannot
  actually perform, shown only as a generous upper bound.
- **target 0.9 line:** a useful detector should catch ≥ 90 % of free-riders.

**What's happening / why it matters.** At any usable FPR (≤ 5 %) insider recall is **≤ 0.17** —
almost nothing caught — and **no bar reaches 0.9.** The same-class insiders (A4/AK) show 33–62 %
recall only at the 10 % budget, but that is a **class-averaging illusion**: the global 10 % FPR is
spread over quiet easy classes while the insider's own class is flagged far more. The per-class
oracle exposes this — it collapses A4/AK recall to **0.00**. Conclusion: **no single deployable
threshold separates insiders from honest clients at a usable operating point.** This is the whole
thesis in one figure. *(Missing: the crude H5 positive-control bar, which should sit near 1.0.)*

### [AK_samekey_timeline.png](results/groups/figs/AK_samekey_timeline.png) — effort-only, controlled

Free-rider vs its honest **twin** on class 6, with the **same key, message, and class** — so the
*only* difference is effort (the free-rider trains on 31 % of the data). Over 3 seeds the two BER
curves oscillate together in overlapping bands → doing less work opens **no detectable gap**.
*NOTE: AK puts the two clients in one run sharing a trigger class — for a version with guaranteed
no in-model conflict, see the iso plots below.*

### [iso_c1.png](results/groups/figs/iso_c1.png), [iso_c7.png](results/groups/figs/iso_c7.png) — easy classes, isolated

"Isolated" = honest (from A1) and free-rider (from A2) read on the **same** trigger class but from
**separate runs**, each the only client on that class -> no two clients fighting over one class
inside one model. At the *easy* classes 1 and 7 the free-rider's mark drops to **0.00 and stays
there** — *cleaner than the honest client* — so no threshold separates them.

### [iso_c3.png](results/groups/figs/iso_c3.png) — medium class, isolated

Class 3 (medium difficulty): free-rider ≈ 0.037 vs honest ≈ 0.057 — tangled, free-rider slightly
cleaner. Inseparable.

### [iso_c6.png](results/groups/figs/iso_c6.png) — hard class, isolated (one lottery outcome)

Class 6 (hard): here the free-rider (≈ 0.22) sits *above* honest (≈ 0.114). This is the one place
the free-rider looks noisier — but which direction it goes is a **key lottery** (a different key
draw flips it), shown next.

### [iso_c6_A4_cleaner.png](results/groups/figs/iso_c6_A4_cleaner.png) — hard class, the other lottery outcome

Same hard class 6, a different key draw (A4): the free-rider (≈ 0.067) is now *below* honest
(≈ 0.114) — **cleaner**. Together with iso_c6 this shows the honest-vs-free-rider ordering at the
hard class is set by the random key, not by who did the work. *(Caveat: A4/AK share class 6 in one
model, so not as strictly isolated as the A2/A3 pairs.)*

## Group D — +N free-riding spectrum: effort vs detectability  (3 seeds; +50 = 2 seeds)

### [D1_spectrum.png](results/groups/figs/D1_spectrum.png) 

**Settings.** Two free-riders sit on class 3 (easy — honest BER floor ~0.057) and class 6 (hard — honest floor ~0.114). The x-axis is the free-rider's data budget: from trigger-images only through +N random images per common class, to a full honest shard (100 % effort - basically an honest client). The y-axis is the free-rider's
converged BER. Top panel = BER over rounds per budget; bottom panel = converged value per budget
with error bars.

**How it proves the point.**
1. **Trigger-only overfits and is caught** (BER ≈ 0.44, above both η lines) — the positive control: a lazy free-rider is detectable - as per the paper's table V.
2. But adding just +1 real image per class (~24 % of honest effort) collapses BER to a flat plateau (~0.11–0.13) that every larger budget also sits on.
3. That plateau sits below η loose (0.264) and stays around the range of honest client floor on those classes

### [D1_sep_n5.json](results/groups/figs/D1_sep_n5.json) — per-class separability at +5/class (≡ A3)

Numbers behind the +5 point. **Class 3:** FR 0.037 vs honest 0.057 (inseparable). **Class 6:** FR 0.220 vs honest 0.114 — catchable, but only at a **40 % honest false-positive rate** (unusable).

## Group E — non-IID (realistic label skew)  (⚠ prior runs were secretly IID — see bug note)

**What non-IID means and why it matters.** Real federated learning is **non-IID**: clients don't
hold identical class mixes — each has a skewed subset. The Dirichlet parameter **α controls the skew: low α (0.1) = extreme skew** (each client sees few classes), **high α (1.0) ≈ balanced / near-IID.** 

**Where η = 0.161 comes from.** It is **not** recalibrated at plot time — it is a frozen constant
`WM_ETA_FIXED=0.161` that the E2/E3 runs were *launched* with (run_now.sh), logged every round, and
the plot reads back. That constant is meant to be μ+3σ over the round-mean honest BER of the
non-IID honest family **`E1_honest_niid_c100`** (α=0.5) — the value `runbook.sh calibrate`
recomputes. ⚠ E1 hasn't been re-confirmed in this batch, so 0.161 is currently **unverified**: run
E1 and `detection.py calibrate --honest-family E1_honest_niid_c100 --tail 20` to check it lands
≈0.161 (else update the constant, or redraw with `--eta_tight <calibrated>`).

> **⚠ BUG (now fixed) — the earlier E runs were secretly IID, so α did nothing.** `run_now.sh` set
> `PART=niid`, but `submit_experiment.sh` reads the env var **`PARTITION`** (and `niid` isn't a
> valid value; the choices are `iid|dirichlet|noniid`). So `--partition` was never passed and
> `cfg.partition` stayed at the config-14 default **`iid`** — which ignores `dirichlet_alpha`
> entirely. That, not an int-cast, is why α=0.1 and α=0.3 came out identical. Fixed to
> `PARTITION=dirichlet` in all four E lines. **Runtime check:** in `run.log`, the "client shards"
> line must now show **uneven** shard sizes (Dirichlet); the old IID runs showed equal ~5000/5000.
> **Re-run E from scratch** (`rm -rf $RES/E1_* $RES/E2_* $RES/E3_*`), since every prior E result
> is IID-mislabelled.

**Does non-IID make free-riding *easier*? No — and that's the point.** Skew raises the honest BER
floor, the free-rider's BER, **and** the threshold η all together, so on absolute level it is
symmetric (the free-rider is noisier too). Detection depends on **separation** — the gap between
honest and free-rider relative to their spread — not on absolute BER. Since everything shifts up in
lockstep, the free-rider stays **inside the honest band**, exactly as under IID → still
indistinguishable. So E3's value is **defensive**: it rules out the rebuttal *"your IID
non-separability is a lab artifact; real heterogeneous FL would pull them apart."* It doesn't.
*(Whether non-IID is strictly worse for the detector — honest variance could inflate η faster than
the FR mean rises, widening the band to hide in — is a plausible but UNVERIFIED hypothesis. Settle
it by computing per-class separability E2-vs-E1 and comparing OVL / best-balanced-error to
`D1_sep_n5`; if OVL stays ~1 and balanced error ~0.5, the honest claim is "separability stays at
chance under non-IID, same as IID.")*

### [E3_a01_timeline.png](results/groups/figs/E3_a01_timeline.png), [E3_a10_timeline.png](results/groups/figs/E3_a10_timeline.png)

Reduced free-rider (30 % data) under strong skew (α ≈ 0.1) and near-IID (α = 1.0). In both, η rises
to 0.161 and the free-rider rides at ~0.11–0.13

## Group I — adaptive-tap "submarine"  🔧 ROOT CAUSE FOUND (base was trigger-only) — fixed, re-run

**Read of the six plotted configs** (tap_I_coast_decay / coast_resend / eta_oracle / eta_self /
margin_m010 / scope_block2, all c36, seed 1000): free-rider mean BER drops to ~0.10–0.15 during
warmup+calibration, then jumps to ~0.60 at defection (round 12) and stays pinned, knob-invariant.

**Root cause (from the code audit) — NOT primarily the probe-holdout starvation.** The
`run_now.sh` Group-I *base* set **`TAP_DATA_CPC=0` (trigger-only)**. All six plotted families are
knob-sweeps that inherit that base, so every one trains the tap on trigger-class images **with no
common-class images** — exactly FareMark's Table V overfitting regime. Group D already quantified
this: cpc=0 → BER ≈ 0.44, cpc=1 → ≈ 0.11 plateau. Intermittent tapping + the self-probe holdout
push the trigger-only case from 0.44 up to ~0.60. This fully explains the **knob-invariance**: the
swept knobs (coast_mode / margin / scope / eta_source) never touch the data amount, so all six sit
in the same overfit regime. The `_prepare` probe-holdout starvation is a *secondary* compounding
factor and is **already fixed** in the current `clients.py` (`MIN_TRAIN_TRIG=8`, holdout capped at
`n_trig//2`, `tap_probe_holdout` default 16 → tap trains on ~34 of ~50 triggers).

**Fix (in the patched `run_now.sh`):**
- **base `TAP_DATA_CPC=5`** (the Group-D plateau recipe: all triggers + 5 images/common class).
  `cpc=0` is kept ONLY as a labeled positive control (`I_data_n0_c36`, expected caught).
- **base aims under `WM_ETA_FIXED=0.264` (η_loose, beatable).** A single `I_tight_eta0064_c36`
  variant aims under η_tight=0.064 to show the **hard-class floor**: class 3 evades (~0.037<0.064),
  class 6 cannot (~0.22>0.064). That split is the operating-point thesis, not an attack failure.
- **`TAP_MARGIN=0.02`** (probe/defender headroom) and **`TAP_MAX_COAST=4`** (forced re-tap so the
  mark can't silently drift over η).
- **Gate first:** `I0_smoke_always_cpc5_c36` (always-tap, cpc=5) must reproduce Group D
  (mean BER ~0.13, NOT 0.6). If it is still 0.6, the tap *embed* path is broken beyond cpc — stop
  and inspect the trace (`n_trigger_train` should be ~34, `ber_after` should fall).

**Group E α-transmission bug: verified FIXED end-to-end** in this code. `run_now.sh` sets
`PARTITION=dirichlet DIRICHLET_ALPHA=$A` → `submit_experiment.sh` emits `--partition`/
`--dirichlet_alpha` → `run_experiment.py` `_OVERRIDABLE` copies them onto `cfg` → `build_data(...,
dirichlet_alpha=cfg.dirichlet_alpha)` → `dirichlet_partition`. `--partition` argparse accepts
`dirichlet`. So α now reaches the run. **Sanity after launch:** `run.log` "client shards" must show
**uneven** sizes (Dirichlet); the old IID runs showed equal ~5000/5000.

### Minimal knobs for the corrected Group I  (1 seed each; 9 families)
- **I0_smoke_always_cpc5** — GATE. always-tap, cpc=5. Must ≈ D (~0.13).
- **I_data_n{0,1,5}** — the effort dial. 0 = Table V control (caught), 1 = plateau edge, 5 = plateau.
- **I_when_{threshold,every_k}** — duty cycle. threshold = adaptive/cheapest; every_k(P=3) = fixed period.
- **I_eta_{oracle,self}** — realism. oracle = given the true η; self = FR estimates η from its own calib BER.
- **I_tight_eta0064** — operating point. same attack, aimed under η_tight; shows the class-3/class-6 split.
Base (all inherit): `adaptive_tap`, FR cids 3,6, warmup 12 / calib 4, `TAP_DATA_CPC=5`,
`TAP_MARGIN=0.02`, `TAP_MAX_COAST=4`, `TAP_SCOPE=full`, `TAP_COAST_MODE=resend`,
`TAP_ETA_SOURCE=oracle`, `WM_ETA_FIXED=0.264`.

**Expected post-fix signature:** free-rider BER falls to ~0.11–0.22 after defection and **sawtooths
just under η** (tap down → coast up → tap), NOT a flat 0.60. The cheapest evading config (lowest
duty cycle at cpc that stays under η_loose) is the constructive result; `I_data_n0` and the class-6
line under η_tight are the "caught" controls.

### Run it (I + E in parallel)
```bash
# 0. clear bug-poisoned I and IID-mislabelled E results so the pool re-runs them
rm -rf $RES/I_*_c36_rep* $RES/I0_*_c36_rep* $RES/E1_* $RES/E2_* $RES/E3_*

# 1. (recommended) gate the tap embed FIRST — 1 quick run, ~5 min
SEEDS_I=0 BATCH=I ./runbook.sh manifest          # builds I incl. I0 gate
MPS=1 WORKERS=8 PODS=2 ./runbook.sh submit
#    watch I0_smoke run.log: ber_fr must drop to ~0.13, NOT sit at 0.6

# 2. once the gate passes, build I (1 seed) + E (3 seeds) together and submit
MPS=1 WORKERS=8 PODS=2 SEEDS_I=0 BATCH=IE ./runbook.sh manifest
MPS=1 WORKERS=8 PODS=2 ./runbook.sh submit
./runbook.sh monitor
# E check: run.log "client shards" shows UNEVEN sizes (Dirichlet)
# I check: trace n_trigger_train ~34 (not ~1); sawtooth under eta on the timelines

# 3. when done: recalibrate eta (E1 -> confirm ~0.161) and plot
RES=$RES OUT=$OUT ./runbook.sh calibrate
RES=$RES OUT=$OUT ./runbook.sh plot
```
*(Cosmetic, unfixed: the `cpc=-1.0` in tap titles reads `autop_common_per_class`, not the tap's
real `tap_data_cpc` — a plots.py label bug, harmless to the BER data.)*
---

## Experiments to run — priority order

1. **Group I — adaptive-tap "submarine"  → FIX + RE-RUN FIRST** (1 seed, `SEEDS_I=0 BATCH=I`). The
   current runs are invalid (probe-holdout starvation bug — see the Group I section). Apply the
   `_prepare` one-line fix, `rm -rf $RES/I_*_c36_rep*`, re-run. Then `tap_*` timelines / `tap_frontier`
   should show the mark hugging just under η. This is the biggest missing piece of the thesis.
2. **H5 — money-plot positive control** (crude previous-models FR on c100; cheap, 3 seeds). Adds
   the "detector *can* catch crude attacks" bar so `operating_point` shows the contrast. `BATCH=H`.
3. **V2 — Table V trigger-sample attack** (`tn10/100/500/m1`; cheap). Overfit → caught; the
   attack-side complement to D's trigger-only point. `BATCH=V`.
4. **E1 / E2 / E3 — non-IID (now that the partition bug is fixed).** All prior E runs were secretly
   IID (`PART`≠`PARTITION`), so re-run from scratch: `rm -rf $RES/E1_* $RES/E2_* $RES/E3_*`. Verify
   `run.log` "client shards" shows **uneven** sizes (Dirichlet). E1/E2 give the honest floor / η;
   E3 {0.1, 1.0} shows the α effect. Priority-bumped per your request.
5. **J5 — best-knob adaptive-tap combo.** Compose *after* analysing group I (pick the winning
   knobs), run at 3 seeds. Uncomment the J block in run_now.
6. **F1 / F2 — capacity (200 clients).** Most expensive per run; 2 seeds fine. F3 / paper-repro
   (Table IX/VII) lowest priority — thesis fidelity, not your contribution.
7. **C1 — sin smoothing.** BLOCKED on the crash fix (R14); ablation, do last.

### Run the stealth attack now
```bash
runai delete job <pool-job-name>            # stop the slow pool
for d in "$RES"/*/result.json; do python -c "import json,sys;json.load(open(sys.argv[1]))" "$d" \
  2>/dev/null || rm -rf "$(dirname "$d")"; done   # drop any half-written result.json
MPS=1 WORKERS=8 PODS=2 SEEDS_I=0 BATCH=I ./runbook.sh manifest
MPS=1 WORKERS=8 PODS=2 SEEDS_I=0 BATCH=I ./runbook.sh submit
./runbook.sh monitor
# when done:  RES=$RES OUT=$OUT ./runbook.sh plot   (I-family plot list now matches run_now)
```