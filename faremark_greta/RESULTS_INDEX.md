# Results summary

## How to read these plots (terms used everywhere)

- **BER (bit-error-rate):** when the server checks a client's model with that client's secret
  watermark key, BER = the fraction of watermark bits recovered *wrong*. **0 = the mark is
  perfectly present** (the client trained and carried the watermark); **0.5 = a coin flip = no
  detectable mark.** Low BER = strong watermark = the client did the work.
- **The detection idea being tested:** a free-rider that doesn't really train should carry a
  *weak* mark → *high* BER. So the server tries to catch free-riders by flagging any client whose
  BER sits above a threshold **η**. The thesis is that this fails, because honest clients on hard
  classes *also* have high BER, so no single η cleanly separates cheaters from honest clients.
- **η tight = 0.064:** threshold at μ+3σ over the *round-mean* honest BER — the value the paper's
  live detector approximates. It is low, so it also flags many honest clients (high false alarms).
- **η loose = 0.264:** threshold at μ+3σ over *per-client* honest BER (the statistically honest
  version) — the **low-false-alarm line a real server would actually deploy.**
- A free-rider whose BER stays **below** a line is **not flagged** by that line.
- On timelines: yellow = forced-honest warmup; green = the calibration window where η is frozen;
  grey dashed = the round the client starts free-riding.

Plot files live in `results/groups/figs/`. Full narrative + numbers are in STATUS_AND_PLAN.md
(sections R7–R14).

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
**separate runs**, each the only client on that class → no two clients fighting over one class
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

### [D1_spectrum.png](results/groups/figs/D1_spectrum.png) — the price-of-invisibility curve

> **Plot updated:** the misleading honest floor is removed; the two deployable thresholds
> (η tight 0.064, η loose 0.264) are drawn instead, and titles simplified. **Re-render** with the
> new `plots.py` (`./runbook.sh plot`, or the `sweep` line with `--eta_tight 0.064 --eta_loose
> 0.264`). Layout preview:
> [D1_spectrum_two_thresholds_demo_1seed.png](results/groups/figs/D1_spectrum_two_thresholds_demo_1seed.png).

**Settings, and why they were chosen.** Two free-riders sit on **class 3 (easy** — honest BER floor
~0.057**) and class 6 (hard** — honest floor ~0.114**)**. That pair is chosen deliberately to
**span the difficulty range**, so one experiment shows both regimes at once. The x-axis is the
free-rider's **data budget**: from *trigger-images only* (`triggers only`, the laziest attack),
through *+N real images per class*, to a *full honest shard* (`full shard`, the 100 %-effort
anchor — still a free-rider, so you can see maximal effort). The y-axis is the free-rider's
converged BER. Top panel = BER over rounds per budget; bottom panel = converged value per budget
with error bars.

**How it proves the point.**
1. **Trigger-only overfits and is caught** (BER ≈ 0.44, above both η lines) — the positive control:
   a genuinely lazy free-rider *is* detectable.
2. But adding just **+1 real image per class (~24 % of honest effort)** collapses BER to a **flat
   plateau (~0.11–0.13)** that *every* larger budget — including a full shard — also sits on.
   **Watermark strength is decoupled from effort** above a trivial threshold: a client doing a
   quarter of the work looks identical to one doing all of it.
3. That plateau sits **below η loose (0.264)** — the line a server would actually deploy — so the
   free-rider is **not flagged.** (It's above η tight 0.064, but η tight also flags many honest
   clients, so it isn't usable — see the operating point.)

### [D1_sep_n5.json](results/groups/figs/D1_sep_n5.json) — per-class separability at +5/class (≡ A3)

Numbers behind the +5 point. **Class 3:** FR 0.037 vs honest 0.057, best balanced error **0.500** =
*no threshold beats a coin* (inseparable). **Class 6:** FR 0.220 vs honest 0.114 — catchable, but
only at a **40 % honest false-positive rate** (unusable).

## Group E — non-IID (realistic label skew)  (3 seeds; α-sweep needs the int-cast fix)

**What non-IID means and why it matters.** Real federated learning is **non-IID**: clients don't
hold identical class mixes — each has a skewed subset. The Dirichlet parameter **α controls the
skew: low α (0.1) = extreme skew** (each client sees few classes), **high α (1.0) ≈ balanced /
near-IID.** Groups A and D use the clean IID case; group E asks whether the result survives the
realistic skewed case.

**Does non-IID help your argument? Yes — it strengthens it.** Under skew, honest clients' own
watermarks embed *less cleanly*, so the **honest BER floor widens** and the detection threshold
**η must rise** (0.064 → **0.161**) to keep false alarms down. A higher η = **more room for the
free-rider to hide under.** So non-IID makes detection *harder*, not easier — and it shows the
"insiders are invisible" result is **not an artifact of the clean lab setting**; it holds, and
worsens, in the conditions real deployments actually face.

### [E3_a01_timeline.png](results/groups/figs/E3_a01_timeline.png), [E3_a10_timeline.png](results/groups/figs/E3_a10_timeline.png)

Reduced free-rider (30 % data) under strong skew (α ≈ 0.1) and near-IID (α = 1.0). In both, η rises
to 0.161 and the free-rider rides at ~0.11–0.13 — **below η the entire time → never flagged.**
⚠ `E3_a03` came out byte-identical to `a01` (α likely int-cast, so 0.1 ≡ 0.3); it's dropped. You
have two valid non-IID points, enough for the claim above; a full α *severity gradient* needs the
int-cast fixed and a clean re-run.

---

## Experiments to run — priority order

1. **Group I — adaptive-tap "submarine"  → RUNNING NOW** (1 seed, `SEEDS_I=0 BATCH=I`). The novel
   stealth attack: coasts under η, re-embeds only when it nears the line. → `tap_*` timelines,
   `tap_frontier`, `tap_dyn_*`. Biggest missing piece of the thesis.
2. **H5 — money-plot positive control** (crude previous-models FR on c100; cheap, 3 seeds). Adds
   the "detector *can* catch crude attacks" bar so `operating_point` shows the contrast. `BATCH=H`.
3. **V2 — Table V trigger-sample attack** (`tn10/100/500/m1`; cheap). Overfit → caught; the
   attack-side complement to D's trigger-only point. `BATCH=V`.
4. **E1 / E2 — non-IID anchors** (honest + reduced at α = 0.5). Give the honest floor / η the E3
   timelines recalibrate against. Re-run E3 clean *only* for the severity gradient (fix the α
   int-cast first; delete stale E3 dirs).
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