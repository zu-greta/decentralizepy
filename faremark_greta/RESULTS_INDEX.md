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

## Group E — non-IID  (1 seed; α∈{0.1,0.5,1.0}, CIFAR-100, 10 clients, seed 1000)

**How the non-IID split is built (`datasets.py::dirichlet_partition`, Hsu et al. 2019 label-skew).**
For each class `c`: take all of `c`'s training samples, and draw a proportion vector
`props ~ Dirichlet(α · ones(K))` over the `K=10` clients; hand client `k` the fraction `props[k]` of
class `c`. Repeat for all 100 classes; concatenate per client; shuffle. **Small α -> concentrated**
(one client hogs each class = severe skew); **large α -> uniform ≈ IID.** Crucially, the
trigger-class assignment (`cid i -> class i`) is drawn **independently** of this partition, so a
client is usually *not* the one that got the bulk of its own trigger class -> it is **data-starved on the very class it must watermark.** That coupling is the whole story below.

**E1 honest floors (α=0.5)** [E1_class_floors.png](results/groups/figs/E1_class_floors.png). Honest per-class BER floors are wildly heterogeneous: cls7/8 ≈ 0.00 (embeds perfectly) but **cls3 = 0.60, cls6 = 0.195, cls1 = 0.14**. Some trigger classes simply cannot be watermarked by their assigned client under skew — *for honest clients*.

**E2 reduced vs E1 honest (α=0.5), per-class separability** [E2_niid_sep.json](results/groups/figs/E2_niid_sep.json), [E2_niid_timeline.png](results/groups/figs/E2_niid_timeline.png).
This is the finding:
- **class 3: honest 0.600, FR 0.600, overlap 1.0, best balanced-error 0.50** — a literal coin flip.
  The free-rider on class 3 is **identical** to an honest client on class 3.
- class 6: honest 0.195 ± 0.02, FR 0.245 ± 0.07, best balanced-error 0.225 — barely separable.
- GLOBAL (pool all clients): overlap 0.28, best balanced-error 0.14, Youden η=0.15 gives recall 0.95
  at **FPR 0.23**. That "0.14" looks catchable, but it is a **pooling artifact**: it mixes
  easy-class honest clients (BER ≈ 0) with the FR. To actually flag a class-3 FR the server must set
  η below 0.60, which flags **every** honest class-3/class-6/class-1 client → 23% FPR. Unusable.

**α sweep** [E3_a01_timeline.png](results/groups/figs/E3_a01_timeline.png) (α=0.1), [E3_a10_timeline.png](results/groups/figs/E3_a10_timeline.png) (α=1.0).
- **α=0.1 (extreme skew):** honest mean ≈ 0.37 ≈ FR mean ≈ 0.40, η_loose = 0.43 — total overlap,
  zero separation. Skew so severe *nobody* embeds; the FR vanishes into the honest cloud.
- **α=1.0 (near-IID):** honest mean 0.07 (most classes embed) **but** cls3 floor 0.55, cls6 floor
  0.89, FR mean ≈ 0.70. warning: **single-seed bad-draw artifact** — at α=1.0 the trigger-class floors
  should be *low*; this draw starved clients 3 & 6 of their trigger class. Do NOT quote the α=1.0
  numbers; they are one unlucky Dirichlet realization.

**Conclusion (direction solid, exact floors need seeds).** Non-IID does **not** rescue the detector —
it breaks it *further*. Skew raises the honest floor on each client's trigger class (because that
client rarely holds much of it), and the free-rider hides among honest clients who **also** can't
embed on hard classes. Per-class, honest and FR coincide (class 3: balanced-error 0.50). This kills
the "your IID non-separability is a lab artifact; real heterogeneous FL would separate them"
rebuttal — heterogeneous FL is *worse*. **CAVEAT: 1 seed.** Which classes are worst is
draw-dependent; the *mechanism* (trigger-class starvation -> camouflage) is seed-robust, the specific
per-class floors are not. Re-run at 3 seeds before any table.

## Group I — adaptive-tap "submarine" (1 seed, 30 rounds; I0=50)

**The device fix + `TAP_DATA_CPC=5` worked — no more flat-0.60 pin.** Free-riders now embed and evade.
But the sweep also delivers a clean **negative** result on the "lazy submarine" premise.

**Data dial — WORKS exactly as predicted** [tap_I_data_n0/n1/n5]:
- cpc=0 (trigger-only): FR → **0.60, caught** — the Table V positive control. ✓
- cpc=1: FR → 0.10–0.20, **evades** η_loose 0.264 (≈38% effort).
- cpc=5: FR → 0.10–0.15, **evades** (≈43% effort). cid3 ≈ 0.00, cid6 ≈ 0.20.
- I0 gate (always-tap cpc=5, 50 rounds): FR mean 0.10, **evades**, 31% effort. ✓

**Duty-cycle knobs — the mark fades too fast to coast** [tap_dyn_*, tap_I_when_every_k]:
- `when=threshold`, `eta=oracle`, `eta=self`, `coast=resend`: all **evade**, but every one **taps
  100% of rounds** (tap_dyn: "tap fraction 100%", "a tap lasts ~1.0 rounds", drop/tap ≈ 0.52). The
  self-probe reads BER ≈ 0.5–0.6 *before each round's tap* because the mark has already faded → it
  re-taps every round. Threshold degenerates to always-tap; it is NOT adaptive in practice.
- `when=every_k` (P=3, tap once, coast twice): **sawtooth** — during the 2 coast rounds the
  *defender-measured* FR BER shoots to **0.6–0.8** (well above η) and is **caught** on those rounds.
  The mark decays essentially completely in ~1 aggregation round.

**Verdict — is there hope?** Split:
- **The evasion attack: YES.** A free-rider doing ~40% of an honest client's work, re-embedding every
  round (cpc≥1), sits under η_loose and evades; a lazy trigger-only (cpc=0) is caught. This *is* the
  constructive companion to Groups A/D, with a live attacker.
- **The "lazy / low-duty-cycle submarine": NO (as currently built).** FareMark's output-layer mark is
  so fragile under aggregation that it fully decays in one coasted round, so the FR *must* re-embed
  every round. The adaptive tap therefore collapses to the constant `reduced` attack — no effort
  savings from adaptivity. `coast_mode=decay` (re-inject own last-tapped weights during coast) is the
  one untested lever that could slow the fade; it's Group J's hopeful shot.

**Knob status:** working → `data_cpc` (effort dial), `when=always` (evades). Degenerate/negative →
`when=threshold` (= always in practice), `when=every_k` (caught on coasts), `eta_source` &
`coast_mode` (indistinguishable while everything taps 100%). The `cpc=-1.0` in titles is the cosmetic
`autop_common_per_class` mislabel; real value is `tap_data_cpc` (0/1/5).

---

## Experiments to run — priority order

1. **Group I — adaptive-tap "submarine"** (1 seed each; 12 families). 
2. **E1 / E2 / E3 — non-IID.** E1/E2 give the honest floor / η; E3 {0.1, 1.0} shows the α effect. 
3. **H5 — positive control** (crude previous-models FR on c100; cheap, 3 seeds). Adds
   the "detector *can* catch crude attacks" bar so `operating_point` shows the contrast. `BATCH=H`.
4. **V2 — Table V trigger-sample attack** (`tn10/100/500/m1`; cheap). Overfit -> caught; the
   attack-side complement to D's trigger-only point. `BATCH=V`. 
5. **J5 — best-knob adaptive-tap combo.** Compose *after* analysing group I (pick the winning
   knobs), run at 3 seeds. Uncomment the J block in run_now.
6. **F1 / F2 — capacity (200 clients).** Most expensive per run; 2 seeds. F3 / paper-repro
   (Table IX/VII) lowest priority — thesis fidelity, not your contribution.
7. **C1 — sin smoothing.** BLOCKED on the crash fix (R14); ablation, do last.

### Run the stealth attack now
```bash
runai delete job <pool-job-name>            # stop the slow pool
for d in "$RES"/*/result.json; do python -c "import json,sys;json.load(open(sys.argv[1]))" "$d" \
  2>/dev/null || rm -rf "$(dirname "$d")"; done   # drop any half-written result.json
MPS=1 WORKERS=4 PODS=2 SEEDS_I=0 BATCH=I ./runbook.sh manifest
MPS=1 WORKERS=4 PODS=2 SEEDS_I=0 BATCH=I ./runbook.sh submit
./runbook.sh monitor
# when done:  RES=$RES OUT=$OUT ./runbook.sh plot   (I-family plot list now matches run_now)
```