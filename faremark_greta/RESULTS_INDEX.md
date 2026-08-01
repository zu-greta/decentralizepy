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

## Group E — non-IID  (**3 seeds** rerun; α∈{0.1, 0.5, 1.0}, CIFAR-100, 10 clients)

> **Updated to the 3-seed reruns (was 1 seed).** The direction is unchanged — non-IID erodes
> separability, it does not rescue it — but the numbers are *milder and now stable*: at 3 seeds the
> class-3 honest floor is **0.255**, not the 1-seed 0.60, and the α=1.0 draw is no longer a lone
> outlier. Where a per-class OVL/balanced-error is quoted below it still needs the 3-seed
> `E2_niid_sep.json` re-emitted (the JSON was 1-seed); the timeline/floor reads below are 3-seed.

**How the non-IID split is built (`datasets.py::dirichlet_partition`, Hsu et al. 2019 label-skew).**
For each class `c`: take all of `c`'s training samples, and draw a proportion vector
`props ~ Dirichlet(α · ones(K))` over the `K=10` clients; hand client `k` the fraction `props[k]` of
class `c`. Repeat for all 100 classes; concatenate per client; shuffle. **Small α -> concentrated**
(one client hogs each class = severe skew); **large α -> uniform ≈ IID** (but Dirichlet(1.0) still
has high variance, so even α=1.0 is *not* equal shards). Crucially, the trigger-class assignment
(`cid i -> class i`) is drawn **independently** of this partition, so a client is usually *not* the
one that got the bulk of its own trigger class -> it is **data-starved on the very class it must
watermark.** That coupling is the whole story below.

**E1 honest floors (α=0.5, 3 seeds)** [E1_class_floors.png](results/groups/figs/E1_class_floors.png).
Honest per-class BER floors are still **wildly heterogeneous** but milder than the 1-seed draw:
cls2 **0.007**, cls7 **0.018**, cls8 **0.020** (embed near-perfectly) rising to cls6 **0.167**, cls0
**0.168**, cls1 **0.187**, cls3 **0.255** (hardest). Span **0.007 → 0.255** (was 0.00 → 0.60 at 1
seed). Some trigger classes simply cannot be watermarked by their assigned client under skew — *for
honest clients* — which is the ceiling the free-rider hides under.

**E1 thresholds (α=0.5, 3 seeds, m=10)** [E1_thresholds.png](results/groups/figs/E1_thresholds.png).
Every candidate η recalibrated on the non-IID honest tail: **coded 0.150 (FPR 24%)**, pooled 0.182
(24%), trimmed-10% 0.318 (9%), p95 0.400 (9%), σ-clip 0.415 (4%), median+3·MAD 0.545 (4%), **loose
0.550 (4%)**, p99 0.600 (4%). Note the skew lifts the tight/coded rule to **0.150 — now *above* 1/m
= 0.100, so no longer degenerate** (unlike IID's 0.064), *but its honest FPR is 24 %*: the only way
to make the tight rule non-degenerate under skew is to accept flagging a quarter of honest clients.
The frozen line used on the E2/E3 timelines (`WM_ETA_FIXED=0.161`) sits right in this coded/pooled band.

**E2 reduced vs E1 honest (α=0.5, 3 seeds)** [E2_niid_timeline.png](results/groups/figs/E2_niid_timeline.png).
η tight (frozen, used) **0.161**, η loose (pooled) **0.182**. Honest floors at the FR's own classes:
**cls3 0.26, cls6 0.17**. In the converged tail the **global** honest mean drops to ≈0.07–0.10 while
the reduced FR rides ≈**0.18–0.20** — i.e. the FR sits **inside the honest-floor band (0.17–0.26) for
its own trigger classes** and *above both η lines*. So the frozen 0.161 does "catch" the FR — but
only by simultaneously flagging every honest client on cls3 (floor 0.26) and cls6 (0.17), which is
exactly the **24% honest FPR** E1 attaches to η≈0.16. Per-class, honest and FR coincide; the global
honest mean of 0.07 is a **pooling artifact** (easy classes drag it down). *(Exact OVL /
balanced-error need the 3-seed `E2_niid_sep.json`.)*

**α sweep** [E3_a01_timeline.png](results/groups/figs/E3_a01_timeline.png) (α=0.1, 3 seeds),
[E3_a10_timeline.png](results/groups/figs/E3_a10_timeline.png) (α=1.0, 3 seeds).
- **α=0.1 (extreme skew):** η tight 0.161, **η loose 0.521**; honest floors cls3 0.31 / cls6 0.35.
  Honest mean ≈0.25–0.28, FR mean ≈0.31–0.33 — **total overlap**, both riding the honest-floor line.
  Skew so severe *nobody* embeds well; the FR vanishes into the honest cloud. η loose 0.521 is above
  everything (flags no one); η tight 0.161 is below everything (flags everyone).
- **α=1.0 (near-IID):** η tight 0.161, **η loose 0.330**; honest floors cls3 0.27 / cls6 0.45. The
  **global** honest mean is low (≈0.05–0.08, most classes embed) **but the FR's own classes are still
  starved** (Dirichlet(1.0) ≠ equal shards), so their floors are 0.27/0.45 — *higher* than the IID
  floors (0.057/0.114). The FR rides ≈0.32: **above η tight 0.161, below η loose 0.330, inside the
  0.27–0.45 honest-floor band** of its own classes. **No longer a "bad-draw artifact"** (it held over
  3 seeds); it is the starvation mechanism showing that even near-IID skew lifts the FR's own-class
  floor above IID.

**Conclusion (3 seeds — direction *and* now the numbers).** Non-IID does **not** rescue the detector;
it erodes it further. Label skew raises the honest floor on each client's trigger class (that client
rarely holds much of it), so the reduced free-rider hides among honest clients who **also** can't
embed on hard classes. Per-class, honest and FR BER coincide across all three α; the only η that
flags the FR also flags a large fraction of honest clients (E1: FPR 24 % at η≈0.16). This kills the
"your IID non-separability is a lab artifact; real heterogeneous FL would separate them" rebuttal —
heterogeneous FL is *worse*. **The mechanism (trigger-class starvation → camouflage) is now
seed-robust; which specific classes are worst still varies with the draw.** Remaining: re-emit
`E2_niid_sep.json` at 3 seeds for the exact per-class OVL/balanced-error table.

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
- **The "lazy / low-duty-cycle submarine": PARTLY — see Group J.** Under `coast_mode=resend` (the
  Group I default: coast by re-submitting the received global) FareMark's output-layer mark fully
  decays in one coasted round, so the FR *must* re-embed every round and the tap collapses to the
  constant `reduced` attack — no savings. **Group J tested the untested lever, `coast_mode=decay`
  (re-submit the FR's OWN last-tapped weights during coast), and it flips the failure the other way:
  the mark no longer fades at all — the FR coasts flat at BER ≈ 0.10 under η_loose at near-zero
  marginal compute.** That evades the BER detector but is a *replay* (byte-identical submissions), so
  it trades the BER test for a staleness/liveness test and produces **no sawtooth**. The genuine
  low-duty submarine needs a *middle* coast mode (`graft`/`follow`: global body + frozen mark head)
  that lets the mark decay *gradually* while the submission still tracks the global — that is the
  next build (see Group J verdict + STATUS N4).

**Knob status:** working → `data_cpc` (effort dial), `when=always` (evades). Degenerate/negative →
`when=threshold` (= always in practice), `when=every_k` (caught on coasts), `eta_source` &
`coast_mode` (indistinguishable while everything taps 100%). The `cpc=-1.0` in titles is the cosmetic
`autop_common_per_class` mislabel; real value is `tap_data_cpc` (0/1/5).

---

## Group J — decay-coast "submarine" (1 seed, seed=1000, 30 rounds; hard classes 3,6)

**What J is.** Group I swept adaptive-tap knobs one at a time and found the tap collapses to
always-tap under the default `coast_mode=resend`. Group J moves several knobs at once to test the
**one lever I left untested — `coast_mode=decay`** — plus two duty cycles. All three J runs share:
`attack=adaptive_tap`, FR on classes 3 & 6, warmup 12 / calib 4, **`tap_data_cpc=5`** (≈31% effort
*per tap round*, the Group-D plateau recipe), `tap_scope=full`, `tap_eta_source=oracle`,
`WM_ETA_FIXED=0.264`, `tap_coast_mode=decay`, `ROUNDS=30`. They differ only in **when** they tap.

> **Title caveat (answers the `cpc` question).** The plot titles read `cpc=-1.0`. That is the
> **`autop_common_per_class`** field at its config-14 default (−1 = full shard) — a knob the
> `adaptive_tap` attacker **never reads**. The J launch lines set **`TAP_DATA_CPC=5`** and do *not*
> set `AUTOP_COMMON_PER_CLASS`, so the real per-tap data budget is **5 common images/class (~31%
> effort), not a full shard.** The title is mislabelling an inert field (same bug flagged in Group I).
> Fix: have `plots.py timeline` format the label from `tap_data_cpc` when `attack==adaptive_tap`.

### [tap_J2_threshold_decay_c36.png](results/groups/figs/tap_J2_threshold_decay_c36.png) — threshold + decay

`tap_when=threshold`, `tap_margin=0.02`, `tap_max_coast=6`. FR mean settles **flat at ≈0.10**, well
under both η lines (0.264), and **evades** — but it shows a **tap ▽ on every attack round** and
never coasts. **Why it taps while the plotted BER is low (the second question):** the threshold
decision is made on `ber_before` = the probe BER of the *received global model*, not the BER of what
the FR will submit. The FR's mark does not survive one aggregation round, so the received global
reads BER ≈ 0.5–0.6 → 0.5 > (η−margin)=0.244 → **tap, every round.** Meanwhile the plotted
"free-rider mean BER" ≈ 0.10 is what the **server reads from the FR's freshly re-embedded *submitted*
model** — a *different* BER from the one driving the tap decision. So there are two BERs in play and
the threshold logic is watching the wrong one. This is the exact `when=threshold` degeneracy from
Group I, unchanged by `decay`. **It is a probe-target bug, not an attack failure:** `when=threshold`
must probe the *coast-candidate submission* (what BER the model it is about to send would read), not
the raw global.

### [tap_J3_everyk3_decay_c36.png](results/groups/figs/tap_J3_everyk3_decay_c36.png) — tap 1-in-3 + decay

`tap_when=every_k`, `tap_period=3` (tap once, **coast ▢ twice**). The forced schedule overrides the
degenerate threshold, so coasts actually happen. Result: FR mean **flat at ≈0.10 across every
tap *and* coast round** — cid6 rides ≈0.20, cid3 ≈0.00, mean ≈0.10, all under η. **The direct A/B
against the Group-I `resend` every_k run** (which shot to **0.6–0.8** on its coast rounds and was
caught): swapping `resend`→`decay` **removes the sawtooth entirely.** Because `decay` re-submits the
FR's *own last-tapped full-model weights* verbatim, every coast submission is identical to the last
tap → BER never moves → flat line, no teeth. So `decay` doesn't *slow* the fade, it *abolishes* it.

### [tap_J4_everyk2_decay_c36.png](results/groups/figs/tap_J4_everyk2_decay_c36.png) — tap 1-in-2 + decay

`tap_period=2` (one coast between taps). Identical picture: **flat ≈0.10, evades, no sawtooth.**
Fewer coasts than J3, same flat trace — confirming the flatness is `decay`'s doing, independent of
duty cycle.

**Effort-metric artifact (important, read before quoting the inset).** The insets read **43% (J2) /
39% (J3) / 40% (J4)** "data used." Those numbers barely move with duty cycle because they are
**dominated by the 12-round forced-honest warmup**: an honest client trains all 30 rounds, the FR
trains 12 full warmup rounds + a handful of cheap tap rounds, so the ratio ≈ 12/30 ≈ 40% *before the
attack phase contributes anything*. The submarine's actual **marginal attack-phase cost is ~0** (a
decay coast trains on nothing; a cpc=5 tap is tiny). The aggregate inset **hides** the very saving
the attack is about. → Report **attack-phase-only effort** (rounds 13–30) separately, or shrink
`honest_until`, before making a "cheap submarine" claim.

**Verdict — J splits cleanly, like I:**
- **`decay` as a BER-evasion: YES, and stronger than expected.** It holds BER ≈ 0.10 under η_loose at
  arbitrarily low duty cycle (you can coast indefinitely at ~0 cost). This is a *better* evasion than
  a sawtooth — no teeth to get caught on.
- **`decay` as a *stealthy* submarine: NO.** Re-submitting stored weights every coast round is a
  **replay**: byte-identical (or stale, global-ignoring) submissions. It defeats the BER test only by
  handing the server a trivial **staleness / gradient-liveness** tell instead. It is not the
  honest-looking, work-tracking client the threat model (N1) describes.
- **The sawtooth still doesn't exist**, because both shipped coast modes are extremes: `resend` =
  instant full decay (teeth too big, caught — Group I); `decay` = zero decay (flat, replay-detectable
  — Group J). **Neither produces a controlled ramp.**

**The fix (Group J's real deliverable → next build).** Add a *middle* coast mode
**`graft`/`follow`**: on a coast, submit **the received global body + the FR's frozen last-tapped
output-layer (mark) head.** Then the body tracks the global every round (submission moves, looks
alive, no replay tell) while the frozen head's projected bits **degrade gradually as the drifting
features flow through it → a genuine BER sawtooth.** Pair it with `tap_scope=head` (cheap head-only
re-embed) and the fixed `when=threshold` probe (measure the *graft candidate's* BER) so the FR
re-taps only when a tooth approaches η−margin. That is the low-duty, replay-free submarine the thesis
wants. **Blocked on a small `clients.py` patch (new coast mode) + the probe-target fix.**

**Knob status (updated).** working → `data_cpc`, `when=always`, `coast_mode=resend` (as a *fast-fade*
control), **`coast_mode=decay` (flat evasion, but replay-detectable)**. Degenerate/negative →
`when=threshold` (probes the wrong model → always-tap), `when=every_k`+`resend` (caught on coasts).
Missing → **`coast_mode=graft`** (the sawtooth lever, not yet implemented). **CAVEAT: J is 1 seed
(seed=1000), 30 rounds — direction only; re-run the winning combo at 3 seeds before any table.**

---

## Experiments to run — priority order

0. **DONE:** Group I (12 families, 1 seed) and Group J (J2/J3/J4, 1 seed) — analysed above.
1. **`coast_mode=graft` build + J5 sawtooth run (TOP PRIORITY — the missing result).** Implement the
   global-body + frozen-mark-head coast mode and fix the `when=threshold` probe target (measure the
   coast-candidate submission, not the raw global). Then run J5 = `graft` + `tap_scope=head` +
   `when=threshold` at classes 3,6. Expect the first real sawtooth (evades on the down-teeth, submission
   tracks the global so no replay tell). 1 seed to confirm the shape, then 3 seeds. **Needs the
   `clients.py` + `plots.py` patch below.**
2. **E1 / E2 / E3 — non-IID.** E1/E2 give the honest floor / η; E3 {0.1, 1.0} shows the α effect.
   (E1/E2 α=0.5 now plotted — see updated Group E; E3 α-sweep still 1 seed.)
3. **H5 — positive control** (crude previous-models FR on c100; cheap, 3 seeds). Adds
   the "detector *can* catch crude attacks" bar so `operating_point` shows the contrast. `BATCH=H`.
4. **V2 — Table V trigger-sample attack** (`tn10/100/500/m1`; cheap). Overfit -> caught; the
   attack-side complement to D's trigger-only point. `BATCH=V`.
5. **Attack-phase effort metric.** Add an inset/field that reports FR effort over rounds 13–30 only
   (offline from `compute.per_client[fr].per_round`) so the submarine's ~0 marginal cost is visible
   and the warmup-dominated 40% artifact stops understating it.
6. **F1 / F2 — capacity (200 clients).** Most expensive per run; 2 seeds. F3 / paper-repro
   (Table IX/VII) lowest priority — thesis fidelity, not your contribution.
7. **C1 — sin smoothing.** BLOCKED on the crash fix (R14); ablation, do last.

### Confirm the two-BER story from the existing J2 trace (offline, no rerun)
```bash
# ber_before (probe on received global) should be ~0.5 while the SUBMITTED/after BER is ~0.10 →
# proves the threshold is watching the wrong model.
python - <<'PY'
import json, glob
f = sorted(glob.glob("results/J2_threshold_decay_c36_*/result.json"))[0]
r = json.load(open(f))
fr = r["compute"]["per_client"]["3"]["trace"]   # cid 3; try "6" too
for row in fr[-12:]:
    print(row.get("round"), row.get("action"),
          "before=%.3f"%row.get("ber_before",float('nan')),
          "after=%.3f"%row.get("ber_after",float('nan')),
          "target=%.3f"%row.get("target",float('nan')))
PY
```

### Next build → the graft-coast J suite (persistence + recovery + sawtooth), 1 seed
```bash
# The clients.py graft + probe-target fix is in; the J block in run_now.sh is the new suite:
#   J0 gate (always-tap cpc5 == Group-D plateau, else STOP)
#   J1 persistence sweep (every_k P=2,3,4,6,12 graft/head -> how many coasts the mark survives)
#   J2 the adaptive sawtooth (threshold+graft/head -> fade time + recovery)
#   J3 coast A/B (resend=cold-recovery+caught, decay=flat replay; graft@P3 = J1_p3)
#   J4 graft scope (block, block2 -> slower fade than head)
SEEDS_J=0 BATCH=J PODS=1 WORKERS=6 ./runbook.sh manifest   # build jobs.tsv (11 families, 1 seed)
SEEDS_J=0 BATCH=J PODS=1 WORKERS=6 ./runbook.sh submit      # run the pool
./runbook.sh monitor
# when done (RES/OUT point at your results dir):
RES=~/local/results OUT=~/local/results/figs ./runbook.sh plot
#   -> tap_J*_c36 timelines (BER sawtooth vs the 0.064/0.264 reference lines) and
#      tap_dyn_* (rounds_between_taps = FADE TIME, ber_drop_per_tap = RECOVERY, fade_per_coast)
```
Read order after it lands: **J0 must sit at ~0.11–0.13** (if flat ~0.6 the embed path is broken —
grep the trace for `n_trigger_train ~tens` and `ber_after ~0.1`, do not trust the rest). Then **J1**
tells you the persistence limit (largest P whose coast rounds stay under η), **J2** is the sawtooth
with fade+recovery annotated, **J3** is the resend-vs-decay-vs-graft(=J1_p3) mechanism triptych, and
**J4** tunes the fade rate if graft snaps too fast (→ block/block2) or stays flat (→ head).