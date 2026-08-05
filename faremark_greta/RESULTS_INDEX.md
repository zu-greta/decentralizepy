# Results index 

**Claim** For output-layer (box-free) watermarking, no deployable threshold η separates
honest clients from free-riders, and an adaptive free-rider can stay under any such η at a fraction of the honest compute.

## reference thresholds drawn on the timelines
- **η tight = 0.064** — μ+3σ over the *round-mean* honest BER, frozen from a 10-seed honest calibration.
- **η loose = 0.264** — μ+3σ over *per-client* honest BER, frozen from a 10-seed honest calibration. 

---

## Group A — baselines  (3 seeds; A1 honest = 6 seeds)

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
threshold separates insiders from honest clients at a usable operating point.** 

### [AK_samekey_timeline.png](results/groups/figs/AK_samekey_timeline.png) — effort-only, controlled

Free-rider vs its honest **twin** on class 6, with the **same key, message, and class** — so the
*only* difference is effort (the free-rider trains on 31 % of the data). Over 3 seeds the two BER
curves oscillate together in overlapping bands -> doing less work opens **no detectable gap**.
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

### [iso_acc_c6.png](results/figs/iso_acc_c6.png), [iso_acc_c7.png](results/figs/iso_acc_c7.png) — BER ≠ trigger-class accuracy 

**The apparent paradox.** On the same isolated pairs, the free-rider has the *lower* BER (cleaner mark)
**and** the *higher* trigger-class classification accuracy, while the honest client has the *higher* BER
**and** ~**0** trigger-class accuracy (Fig A of each). Both runs reach the same ~72 % global test
accuracy (Fig B). This is not a contradiction — **BER and trigger-class accuracy are orthogonal**, and
FareMark's own construction decouples them:
- BER measures the *sign-alignment of the smoothed softmax tail* with a secret key (Eq. 13/15), **not**
  argmax correctness. The mark is carried by the tail *shape*.
- The scheme **requires suppressing the trigger class's own probability below 0.5** to embed at all
  (paper Eq. 4–6, dominance constraint Eq. 10, Fig. 6). So a watermarked model is *designed* not to be
  confidently correct on its trigger class — low trigger accuracy is expected, not a bug.
- The honest client (hard class 6, ~50 trigger images diluted in a 5000-image shard, 10-way aggregation)
  **over-suppresses** the trigger class to acc 0 while the hard-class tail stays unstable → BER floors at
  ~0.11. The free-rider (concentrated reduced shard, aggressive per-round re-embed) fits *both* objectives
  on its trigger class → a cleaner mark (~0.067) *and* some recovered trigger accuracy.

**Takeaway.** BER tracks neither effort nor classification quality — the lower-effort FR gets the cleaner
mark *and* classifies the trigger class better than the honest client. This is consistent with FareMark's
mechanism, but the honest→0 trigger-class collapse is a **cost the paper never reports** (it publishes
only main-task accuracy — Table I — which barely moves because sacrificing ~10 of 100 classes is lost in
the 100-class average; Fig B here shows exactly that). 

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

**non-IID split (`datasets.py::dirichlet_partition`, Hsu et al. 2019 label-skew).**
For each class `c`: take all of `c`'s training samples, and draw a proportion vector
`props ~ Dirichlet(α · ones(K))` over the `K=10` clients; hand client `k` the fraction `props[k]` of
class `c`. Repeat for all 100 classes; concatenate per client; shuffle. **Small α -> concentrated**
(one client hogs each class = severe skew); **large α -> uniform ≈ IID** (but Dirichlet(1.0) still
has high variance, so even α=1.0 is *not* equal shards). Crucially, the trigger-class assignment
(`cid i -> class i`) is drawn **independently** of this partition, so a client is usually *not* the
one that got the bulk of its own trigger class -> it is **data-starved on the very class it must
watermark.** That coupling is the whole story below.

**E1 honest floors (α=0.5, 3 seeds)** [E1_class_floors.png](results/groups/figs_2/E1_class_floors.png).
Honest per-class BER floors are still **wildly heterogeneous** but milder than the 1-seed draw:
cls2 **0.007**, cls7 **0.018**, cls8 **0.020** (embed near-perfectly) rising to cls6 **0.167**, cls0
**0.168**, cls1 **0.187**, cls3 **0.255** (hardest). Span **0.007 → 0.255** (was 0.00 → 0.60 at 1
seed). Some trigger classes simply cannot be watermarked by their assigned client under skew — *for
honest clients* — which is the ceiling the free-rider hides under.

**E1 thresholds (α=0.5, 3 seeds, m=10)** [E1_thresholds.png](results/groups/figs_2/E1_thresholds.png).
Every candidate η recalibrated on the non-IID honest tail: **coded 0.150 (FPR 24%)**, pooled 0.182
(24%), trimmed-10% 0.318 (9%), p95 0.400 (9%), σ-clip 0.415 (4%), median+3·MAD 0.545 (4%), **loose
0.550 (4%)**, p99 0.600 (4%). Note the skew lifts the tight/coded rule to **0.150 — now *above* 1/m
= 0.100, so no longer degenerate** (unlike IID's 0.064), *but its honest FPR is 24 %*: the only way
to make the tight rule non-degenerate under skew is to accept flagging a quarter of honest clients.
The frozen line used on the E2/E3 timelines (`WM_ETA_FIXED=0.161`) sits right in this coded/pooled band.

**E2 reduced vs E1 honest (α=0.5, 3 seeds)** [E2_niid_timeline.png](results/groups/figs_2/E2_niid_timeline.png).
η tight (frozen, used) **0.161**, η loose (pooled) **0.182**. Honest floors at the FR's own classes:
**cls3 0.26, cls6 0.17**. In the converged tail the **global** honest mean drops to ≈0.07–0.10 while
the reduced FR rides ≈**0.18–0.20** — i.e. the FR sits **inside the honest-floor band (0.17–0.26) for
its own trigger classes** and *above both η lines*. So the frozen 0.161 does "catch" the FR — but
only by simultaneously flagging every honest client on cls3 (floor 0.26) and cls6 (0.17), which is
exactly the **24% honest FPR** E1 attaches to η≈0.16. Per-class, honest and FR coincide; the global
honest mean of 0.07 is a **pooling artifact** (easy classes drag it down).

**E2 separability — the exact 3-seed numbers** (`E2_niid_sep.json`, 3 seeds, tail 20; this closes the
last E item). **OVL = overlap coefficient** = the shared area of the honest-BER and free-rider-BER
histograms on the same axis, `Σ_bins min(p_honest, p_FR)` (Weitzman's coefficient; equivalently
`1 − total-variation distance`). **1.0 = the two distributions are identical → no test can tell them
apart; 0 = disjoint → a perfect threshold exists.** `best_threshold_balanced_error` is the *lowest*
balanced error **any** η achieves — even an oracle η that peeks at the free-riders — so **0.50 = no
threshold beats a coin.**

| view | honest BER | FR BER | OVL | best balanced-error | reading |
|---|---|---|---|---|---|
| **class 3** (FR's own) | 0.255 | **0.222** | 0.667 | **0.500** | FR *cleaner* than honest; **inseparable** |
| **class 6** (FR's own) | 0.167 | **0.143** | 0.783 | **0.500** | FR *cleaner* than honest; **inseparable** |
| GLOBAL (server pools all) | 0.109 | 0.183 | 0.690 | 0.388 | looks weakly separable (bal-acc 0.61) — **the illusion** |

**Why this table is the payoff.** It converts "the curves overlap" into two threshold-independent
numbers, and they are decisive: **at each of the free-rider's own trigger classes the best possible
threshold scores balanced-error 0.50 — a coin flip — and the free-rider's BER is actually *lower* than
honest** (0.222 vs 0.255; 0.143 vs 0.167). The only place any separation appears is the **GLOBAL** row
(balanced-error 0.388), and that is precisely the pooling illusion: the server, comparing one free-rider
against *all* honest clients (most on easy classes with BER ≈ 0), sees a gap that **does not exist at the
class where detection actually has to happen.** A server can only act on the global view, so it will
believe it can separate — the per-class rows prove it cannot. 

**α sweep** [E3_a01_timeline.png](results/groups/figs_2/E3_a01_timeline.png) (α=0.1, 3 seeds),
[E3_a10_timeline.png](results/groups/figs_2/E3_a10_timeline.png) (α=1.0, 3 seeds).
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
heterogeneous FL is *worse*.

## Group I — adaptive-tap "submarine" (1 seed, 30–50 rounds; I0=50)

> **Status:** the `resend`/`decay` limitations below are resolved by the shipped **`graft`** coast mode
> (Group J) and superseded at 3 seeds 

**The device fix + `TAP_DATA_CPC=5` worked — no more flat-0.60 pin.** Free-riders now embed and evade.
But the sweep also delivers a clean **negative** result on the "lazy submarine" premise.

**Data dial — works as predicted** [tap_I_data_n0/n1/n5]:
- cpc=0 (trigger-only): FR → **0.60, caught** — the Table V positive control. 
- cpc=1: FR → 0.10–0.20, **evades** η_loose 0.264 (≈38% effort).
- cpc=5: FR → 0.10–0.15, **evades** (≈43% effort). cid3 ≈ 0.00, cid6 ≈ 0.20.
- I0 gate (always-tap cpc=5, 50 rounds): FR mean 0.10, **evades**, 31% effort. 

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
  graft build (now shipped — see Group NOW).

**Knob status:** working → `data_cpc` (effort dial), `when=always` (evades). Degenerate/negative →
`when=threshold` (= always in practice), `when=every_k` (caught on coasts), `eta_source` &
`coast_mode` (indistinguishable while everything taps 100%). The `cpc=-1.0` in titles is the cosmetic
`autop_common_per_class` mislabel; real value is `tap_data_cpc` (0/1/5).

---

## Group J — graft-coast submarine (1 seed, seed=1000, 40 rounds; hard classes 3,6)

**What J is now.** The `clients.py` graft coast + probe-target fix landed, and the run_now J block is
the new suite: **J0** gate, **J1** persistence sweep, **J2** the adaptive sawtooth, **J3** coast A/B,
**J4** graft scope. All share `attack=adaptive_tap`, FR on classes 3 & 6, warmup 12 / calib 4,
**`tap_data_cpc=5`**, decision η = `AUTOP_ORACLE_ETA=0.264` (the beatable loose rule; the FR aims just
under it), and the hardcoded reference lines η tight 0.064 / η loose 0.264. **Headline: graft is a
real, gradually-fading coast — a genuine adaptive submarine exists (J2), not just a re-skinned Group-D
reduced attack — but its effort win is modest and it has a self-probe blind spot on the hard class.**

### [tap_J0_gate_alwaystap_c36.png](results/groups/figs/tap_J0_gate_alwaystap_c36.png) — the gate PASSED

Always-tap, cpc=5. FR mean settles at **~0.10–0.15** (honest ~0.04), **above η tight 0.064 but under
η loose 0.264 → evades the loose rule**, exactly the Group-D plateau. **This is the check that the
Group-I `cpc=0` BER≈0.6 bug is gone** — the embed path works, so everything below is trustworthy.
Tap fraction 100% (trains every round) = the reduced-attack baseline the submarine must beat on cost.

### [tap_J1_persist_graft_p2/3/4/6/12_c36.png] + [tap_dyn_J1_*] — persistence: how long the mark lasts untapped

`every_k(P)` graft/head: tap once, coast `P-1` rounds, repeat. **This is the "how long does it last"
answer, and it's the good news + the wall in one sweep:**
- **Graft fades *gradually*** (unlike decay = never, resend = 1 round): the coast BER ramps up smoothly
  each round. So the sawtooth is real.
- **But it fades fast enough that only ~1 coast stays safely under η_loose.** P=2 (1 coast) rides
  ~0.10 ↔ 0.20 (under 0.264). P=3/4 peak at ~0.30–0.35; P=6 peaks 0.30–0.49; P=12 climbs to a ~0.35
  plateau — **all crossing η_loose 0.264 on the coast rounds → caught on those rounds** (server-measured
  mean).
- **The self-probe is optimistic vs the server.** `tap_dyn_J1_p12` reads the FR's own probe at ~0.20
  (< target 0.244, "fade +0.022/coast, a tap lasts 12") — i.e. the FR *thinks* it's safe — while the
  timeline (server's 50-image bank, mean over cid3+cid6) shows it crossing 0.264. The gap is the hard
  class **cid6**: on the server's images its mark drifts up faster than the FR's 16-image self-probe
  reports. **A threshold submarine that trusts its own probe will coast too long and get caught.**

### [tap_J2_saw_graft_head_c36.png](results/groups/figs/tap_J2_saw_graft_head_c36.png) + [tap_dyn_J2_*] — THE adaptive submarine

Threshold + graft/head — the headline. `tap_dyn`: **tap fraction 36%** (vs 100% for J0/reduced), a tap
lasts ~5.2 rounds, fade +0.002/coast. The FR coasts at ~0.10–0.20 under target 0.234, taps only when it
drifts up, and the **submission tracks the global every round (graft body = fresh global) — not a
replay.** Server timeline: mostly 0.15–0.20, a few peaks near 0.30, otherwise under η_loose. **This is a
genuine adaptive submarine: it does ~1/3 the training of the reduced attack while staying (mostly)
hidden.** Two caveats: (1) the ~36% saving is **masked in the headline "30% data used"** because the
12-round honest warmup dominates that ratio — report attack-phase-only effort to see the win; (2) a few
coast peaks touch 0.30 (the cid6 self-probe optimism above), so a small safety margin is needed to keep
every peak under η.

### [tap_J3_coast_decay/resend_p3_c36.png] + [tap_dyn_J3_*] — the fade-mechanism control

At a fixed 1-in-3 duty cycle, the two extremes bracket graft:
- **`resend`**: `tap_dyn` drop/tap **0.500**, BER-before flat at **0.60** — the mark dies completely in
  one coast. Timeline = the giant 0.10 ↔ 0.70 sawtooth, **caught on every coast round.**
- **`decay`**: `tap_dyn` fade **+0.000/coast**, BER-before flat at **0.0** — the mark never fades because
  the FR re-submits its own stored weights verbatim. Timeline flat ~0.10–0.15, **evades — but it is a
  replay** (byte-identical submissions), defeating the BER test only by handing the server a trivial
  staleness tell. **Graft is the only middle that both evades *and* keeps the submission live.**

### [tap_J4_scope_graft_block/block2_c36.png] + [tap_dyn_J4_*] — scope tunes the fade rate

Freezing more of the head changes how fast the graft mark fades:
- **`head`** (J2, 2 params): slowest fade (+0.002), **lowest tap fraction 36%**, shallow sawtooth —
  holds the mark best.
- **`block`** (8 params): tap fraction 57%, holds ~0.10–0.15, a tap lasts ~5.
- **`block2`** (20 params): **biggest, cleanest sawtooth** (fade +0.025/coast, drop/tap 0.204) but taps
  **59%** and its peaks reach ~0.30 (cross η_loose). Counter-intuitively, freezing *more* fades *faster*:
  a deeper frozen block interfaces with the drifting fresh body below it, so the mark degrades quicker
  than the pure output-layer head, whose input features drift smoothly. **Takeaway: `head` is the
  cheapest hidden scope; `block2` is the most *visible* sawtooth but not the stealthiest.**

### [tap_frontier.png](results/groups/figs/tap_frontier.png) — the effort/persistence map

x = tap fraction (compute spent), y = rounds a tap lasts (persistence); **upper-left = cheaper +
longer-lived.** J0 sits bottom-right (1.0, 1) = the reduced baseline. The scheduled graft points climb
the left wall — **J1_p12 (0.10, 12)** and **J1_p6 (0.17, 6)** are cheapest and longest-lived *but* their
coast peaks cross η on the server, so their "persistence" is mechanical, not safe. **J2 (0.36, 5)** is the
adaptive sweet spot that stays (mostly) under η. The gap between J1_p12's *mechanical* 12-round
persistence and J2's *safe* ~5 is exactly the self-probe-vs-server gap to close.

### Verdict — is this a real submarine, or just Group-D reduced?

**A real adaptive submarine exists: J2 (graft/head/threshold).** It taps 36% of rounds (reduced taps
100%), the mark fades *gradually* so it can genuinely coast, and every submission tracks the global
(not the decay replay). That is categorically more than "reduced with a schedule." **But** the win is
currently modest (≈64% fewer attack-phase taps, hidden by the warmup-dominated headline metric) and
bounded by two things: the mark only survives ~1–2 coasts under η_loose before the hard class crosses,
and the FR's self-probe under-reads what the server measures on cid6. **The detection-impossibility
thesis is untouched either way**: to keep its coast peaks under a *usable* η the submarine must hold the
mark near the honest floor of its class — i.e. it is doing enough real work to be indistinguishable from
an honest client there, and any η that catches its peaks also flags honest cls6 clients.

### Next steps → J5, a stronger submarine (1 seed, then 3)

1. **Exploit the per-FR asymmetry (biggest lever).** The threshold is *already* per-FR — each free-rider
   is its own instance deciding on its own probe (`clients.py:256` loop). cid3 (easy) already coasts more
   than cid6 (hard); what caps the win is (a) cid6's probe under-reads the server, so it coasts into a
   catch, and (b) `tap_max_coast`/`margin` too tight for cid3 to fully exploit its headroom. Fix both:
   larger `tap_max_coast`, `tap_margin`≈0.06, and an honest probe (next item) → cid3's tap fraction →
   ~0, average well below J2's 36%.
2. **Close the self-probe/server gap.** Raise `tap_probe_holdout` (more than 16 images) and/or aim a
   safety margin below η (larger `tap_margin`, ~0.06) so the threshold accounts for the probe being
   optimistic on cid6. This is what stops J2's peaks from touching 0.30. problem: cannot be too high or too many held out and the trigger samples are starved - cannot embed
3. **Cheaper taps.** Graft holds the mark, so sweep `tap_data_cpc ∈ {1,2}` with graft/head — if cpc=1
   still re-embeds, each tap costs ~half of cpc=5.
4. **Attack-phase-only effort metric.** Report effort over rounds 13–40 (drop the warmup) so the tap-
   fraction win is visible instead of the warmup-dominated ~30%.
5. **Investigate the round-36–40 tail spikes** (honest *and* FR jump to ~0.5 in p12 / J4_block / J2):
   looks like a late aggregation/memory-update instability at this seed, not the attack — confirm it
   vanishes at other seeds before it contaminates a table.

**CAVEAT: J is 1 seed (seed=1000), 40 rounds.** Shapes and orderings only; re-run J2 + J5 at 3 seeds
(1000/1001/1002) before any table.

---

## Seed analysis — what a run's seed actually randomizes

The CLI "seed" is the **repeat index** (the trailing `0` in `./submit_experiment.sh 14 0`).
**`config.py:228-229`** `seed_for(cfg, repeat) = cfg.base_seed + repeat`, with **`base_seed=1000`**
(`config.py:22`), so `S = 1000 + repeat` — **consecutive integers** (3 seeds = 1000/1001/1002). Every
stream below is a deterministic function of `S`, forked by a fixed offset so the streams stay
independent. `--no_determinism` (your runbook default) only flips cuDNN's autotuner; it consumes no RNG,
so all draws stay reproducible.

| # | What it randomizes | Theory (what varies over seeds) | Code — `file:line` | IID? | Non-IID extra |
|---|---|---|---|---|---|
| 0 | **Master seed** `S=1000+repeat` | the one integer everything derives from | `config.py:229` (`seed_for`), `config.py:22` (`base_seed`), used at `run_experiment.py:283-284` | — | — |
| 1 | **Global RNG seeding** | fixes model init, augmentation samples, and every generator-less `torch.rand*` | `utils.py:23-26` (`set_seed`: python/np/torch/cuda) | ✓ | ✓ |
| 2 | **IID shard assignment** | which samples land in each client (subset per client); class balance stays uniform → floors barely move | `datasets.py:69-70` (`np.random.default_rng(S).permutation`) | ✓ | — |
| 3 | **Dirichlet label skew** | **the whole non-IID skew pattern** — who gets what fraction of each class; decides whether a client is starved on its *own* trigger class. This is why E-group per-class floors move across seeds while the *mechanism* is stable | `datasets.py:83, 88-89` (`default_rng(S)`; `rng.shuffle`; `rng.dirichlet(α·1_K)`) | — | ✓ **(the big one)** |
| 4 | **Per-client minibatch order** | SGD shuffle per client per epoch → optimization trajectory / gradient noise | `datasets.py:124` (`torch.Generator().manual_seed(S+cid)`) | ✓ | ✓ |
| 5 | **Watermark key M (the "key lottery")** | the secret ±1 projection (paper Eq.1/Fig.5). At small `l`, an unlucky key has same-sign rows → structurally unembeddable → a per-client BER floor *from the key alone* (`P(same-sign)=2^{1-l}`). Flips the honest-vs-FR ordering at a hard class | `clients.py:261` → `watermark.py:158-165`; floor math `watermark.py:168-180` | ✓ | ✓ |
| 6 | **Watermark target bits B** | the secret message; balanced so a random guesser sits at BER 0.5 (what separates honest from FR) | `clients.py:263` → `watermark.py:206-211` (offset `+7919` decouples bits from key) | ✓ | ✓ |
| 7 | **Verification trigger images** | which held-out test images the server extracts the mark from (tests generalisation, not memorisation) | `wm_verify.py:52/85` (`Generator().manual_seed(S)`), seeded at `run_experiment.py:344/347` | ✓ | ✓ |
| 8 | **Model weight init** | the optimization starting point → a different minimum & per-class difficulty realisation; a *major* floor-variance driver | `run_experiment.py:316` (`build_model`, draws the global RNG from #1) | ✓ | ✓ |
| 9 | **Reduced / adaptive_tap common-image sampling** | which `+N` common-class images the FR trains on each freeride round (the `cpc` budget); sampled once | `clients.py:574, 579` (`torch.randperm`, **global** RNG) | ✓ | ✓ |

**Not seeded / frozen (so seeds are comparable):**
- **Trigger-class assignment** `cid→class` — deterministic round-robin (`clients.py:257`), unless `TRIGGER_CLASS_MAP` is set. In non-IID this is the crux: the class is *fixed* while the Dirichlet split (#3) is *seeded*, so the seed controls the starvation coupling.
- **Free-rider identity** — with `FREE_RIDER_IDS=3,6` the FR set is fixed (`clients.py:226`); only the fallback `choose_free_riders` (`clients.py:472`, `random.Random(S).sample`) is seeded.
- **Detection thresholds** (the 0.064/0.264 reference lines) — calibrated offline, injected as constants.
- **Gaussian-noise FR** (`clients.py:450`) — seeded by a hardcoded `1234 + cid*1000 + round`, **not by S** — so its fake-gradient noise is identical across your experiment seeds. If you ever want seed-varied Gaussian FRs, add `+ S` there.

**Two caveats worth knowing:**
1. **Consecutive-seed aliasing in the dataloader.** Because #4 uses `S + cid`, client `c`'s minibatch order at seed `S` equals client `c+1`'s at seed `S-1` (e.g. `(1000, cid1) == (1001, cid0)`). Different clients hold different shards so it's harmless in practice, but it means your 3 seeds are *not* fully independent in the shuffle stream. If you want strict independence, change the offset to e.g. `S*1000 + cid`.
2. **What actually moves the numbers across seeds:** (a) model init (#8), (b) the per-client key/bits lottery (#5/#6), and — **only in non-IID** — (c) the Dirichlet draw (#3). (a)/(b) exist in IID too and explain the modest per-class jitter there; (c) is what makes the non-IID per-class floors genuinely draw-dependent — which is exactly why the E-group caveat ("mechanism seed-robust, exact floors not") holds and why 3 seeds is the floor for any non-IID table.


---

## Group NOW — the 3-seed submarine (J2×3 + J5×3): setup, mechanism, verdict

*Derived from the run artifacts. **J2 is now complete at 3 seeds** (the `tap_perfr_J2` 3-seed plot);
J5 rep0 crashed at R39 (no result.json). The single-seed (rep0) trace is kept below for the
round-by-round mechanism; the 3-seed numbers are in §4. (Generic substrate & code: STATUS_AND_PLAN
§5/§8; the full knob table is STATUS_AND_PLAN §10.8.)*

**3-seed headline (J2, from `tap_perfr_J2` over seeds 1000/1001/1002):**
- **cid3 (easy, class 3): tap-fraction 10 %, tail server-BER 0.13 → evades η_loose 0.264.**
- **cid6 (hard, class 6): tap-fraction 43 %, tail server-BER 0.22 → evades η_loose 0.264.**
- Both under the operating line ⇒ **the submarine evades on both classes at 3 seeds.** cid6 taps less than
  rep0 alone suggested (43 % vs the 66 % single-seed) — rep0 was a pessimistic-probe seed. The compute
  win is still asymmetric (cid3 cheap, cid6 ≈ reduced-attack cost), for the probe reason in §4.3.

---

## 0. What NOW actually is, and how it launches

`BATCH=NOW ./runbook.sh manifest && BATCH=NOW ./runbook.sh submit` fires **only** the NOW block
(`run_now.sh:407`; `has()` is a substring match and `NOW` shares no letter with the group tokens
`A C D E F H I J V`, so the 11-family J suite does **not** re-run — `run_now.sh:404`). It emits **6
runs**: two families × three seeds (`SEEDS_NOW="0 1 2"` → seeds **1000/1001/1002** via
`seed_for = base_seed + repeat`, `config.py:228-229`, `base_seed=1000` `config.py:22`).

Shared base for both families (`run_now.sh:411-413`):

```
ATTACK=adaptive_tap  FREE_RIDER_IDS=3,6  AUTOP_HONEST_UNTIL=12  AUTOP_CALIB_ROUNDS=4
AUTOP_ORACLE_ETA=0.264  WM_ETA_FIXED=0.064  TAP_DATA_CPC=5  TAP_ETA_SOURCE=oracle
TAP_SCOPE=head  TAP_COAST_MODE=graft  TAP_WHEN=threshold  ROUNDS=40  FAST_DATA=1
```

The two families differ only in the **trust/asymmetry** knobs (the J5 "Next steps" levers):

| Family | `TAP_MARGIN` | `TAP_MAX_COAST` | `TAP_PROBE_HOLDOUT` | Intent | Source |
|---|---|---|---|---|---|
| **J2** (confirmed) | 0.03 | 12 | 16 | reproduce the 1-seed J2 at 3 seeds | `run_now.sh:420` |
| **J5** (tuned) | 0.05 | 20 | 48 | honest probe + safety margin + deep coast | `run_now.sh:432` |

Everything routes through **CONFIG_IDX 14 = `submarine_resnet18_cifar100`** (`config.py:208-213`). Config
14's own `attack="submarine"` default is **dead/overridden** — every run sets `ATTACK=adaptive_tap` via
env, mapped env→`--flag`→`_OVERRIDABLE`. The literal `submarine`/`autopilot` attacker is commented out
(`clients.py:280-305, 937-960`); **`adaptive_tap` is the stand-in submarine that actually ran.** This
matters for the meeting: when we say "the submarine," we mean `AdaptiveTapFreeRider`
(`clients.py:737-934`), not the disabled full `SubmarineFreeRider` with dynamic warmup + own-η
estimation.

---

## 1. The fixed training substrate — and the batch-16 / steps-per-epoch / compute story

*(This is the section to have memorised. The whole "free-riding is cheap" claim lives here.)*

**Substrate (identical A/D/E/I/J/NOW).** ResNet-18 (32×32 CIFAR variant), CIFAR-100 (50 000 train /
10 000 test, 100 classes), **N=10 clients**, **IID equal shards** (`datasets.py:iid_partition`, 5000
imgs/client), **batch 16** (`config.py:19` → `datasets.py:116`), **local_epochs = 5**
(`config.py:17`), SGD lr 0.01 / mom 0.9 / wd 5e-4
(`config.py:18,20,21`; optimizer built in `clients.py:75-78` and `:132-133`), sample-weighted FedAvg
(`server.py:20-40`; = plain mean under equal IID shards), augmentation RandomCrop(32,pad4)+HFlip then
normalize (`datasets.py:_build_transforms`), 40 rounds.

### 1.1 The step count per round — where batch 16 bites

A **local epoch** is one full pass over the client's loader. With batch 16 the number of **SGD steps
per epoch = ⌈images / 16⌉**, and each `produce_update` runs `local_epochs = 5` of them
(`clients.py:141-142` honest-WM loop; `:79-80` plain honest). So per round:

| Who trains | images/round | steps/epoch (⌈n/16⌉) | × epochs | **SGD steps/round** | sample-passes/round | % of honest |
|---|---|---|---|---|---|---|
| **Honest / warmup FR** (full shard) | 5000 | 313 | 5 | **1565** | 25 000 | 100% |
| **cpc=5 tap** (J2/J5) | ~533 | 34 | 5 | **170** | 2 665 | **10.7%** |
| cpc=2 tap | ~236 | 15 | 5 | 75 | 1 180 | 4.7% |
| cpc=1 tap | ~137 | 9 | 5 | 45 | 685 | 2.7% |
| cpc=0 (trigger-only) | ~38 | 3 | 5 | 15 | 190 | 0.8% |
| **coast** (graft/decay/resend) | 0 | 0 | 0 | **0** | 0 | 0% |

The reduced-image count for cpc=5 is **~38 trigger images** (50 in the IID shard, minus ~12 held out
for the self-probe) **+ 5 images × 99 common classes = 495**, ≈ 533 total
(`clients.py:_prepare` 527-585; holdout logic 552-560; common sampling 568-580). Confirmed against the
run: `wm_stats[12].n_trigger_samples = 190 = 38 trig × 5 epochs` (JSON), and `reduced_n ≈ 533` in the
tap trace.

**Why this reduces cost — the mechanism, precisely.** A tap freezes the loader to the reduced
TensorDataset (`clients.py:804-806`) and, with `scope=head`, freezes every parameter except the last
2 tensors (`_freeze_scope` 778-787, `_SCOPE_KEEP={"head":2}` 744). So a tap is **170 SGD steps on a
2-tensor head** vs an honest **1565 steps on the full network** — cheaper in both step count (~11×) and
per-step FLOPs (backbone frozen ⇒ no backward through most layers). A **coast is literally zero
training**: `_do_coast` returns a state dict and calls the meter with `trained=False`
(`clients.py:861-872`) — no forward, no backward.

### 1.2 The compute the run actually recorded (J2 rep0, the number to bring)

From `result.json → compute.summary`:

```
honest_mean_samples = 1 000 000   (= 40 × 25 000  ✓)
fr_mean_samples     =   305 600
effort_ratio_samples = 0.3056     (≈ 31% of an honest client)
effort_ratio_gpu     = 0.3048     (GPU-ms ratio agrees)
```

Per-FR (JSON `compute.per_client`): **cid3 = 285 660 samples, duty 0.375; cid6 = 325 540, duty 0.75.**
These reconcile exactly:
- **Warmup dominates.** Rounds 1–11 are forced-honest full-shard training = **11 × 25 000 = 275 000**
  samples for *both* FRs, before the attack even starts.
- **Attack-phase taps are tiny.** cid3 taps 4× (R12,25,38,39) × 2 665 = 10 660 → 275 000 + 10 660 =
  **285 660** ✓. cid6 taps 19× × 2 665 = 50 635 → **325 635** ✓.

**The headline "31% effort" is a warmup artifact.** Subtract the warmup: attack-phase FR samples
(mean) = 305 600 − 275 000 = **30 600 over 29 freeride rounds = 4.2%** of what an honest client does in
that same phase. Bring **both** numbers to the meeting and lead with the attack-phase one — the 31%
*understates* how cheap the marginal attack is (it's ~4%), which is the point, not a weakness.

### 1.3 Why runs are cheap to parallelise (the throughput model)

Batch 16 + ResNet-18 **"barely uses an A100"** (`runbook.sh:30`) — the card is **GPU-starved, not
GPU-bound**. Speed therefore comes from **concurrency, not a bigger batch** (batch is frozen at 16 for
cross-run comparability). Levers, all validated statistically identical to the slow path
(`runbook.sh:109-110`):
- **`PODS×WORKERS = 2×6 = 12` concurrent runs/GPU** sharing the SM scheduler via **CUDA MPS**
  (`MPS=1`, `runbook.sh:36`; PODS/WORKERS `:30`).
- **`FAST_DATA=1`** = GPU-resident loaders, kills DataLoader fork storms — *not* a data reduction
  (log confirms: `[fast_data] GPU-resident loaders active`).
- **`DETERMINISM=0`** = cuDNN autotuner on, ~1.3–2× (`utils.py:set_seed` flips
  `cudnn.deterministic/benchmark`; consumes no RNG, so seeds stay reproducible).

NOW is 6 runs → a few concurrent slots, comfortably overnight (log: ~300–400 s/round, 40 rounds ≈
3.5–4.5 h/run).

---
## 2-3. How the attack works (theory + line-by-line)

Moved to **STATUS_AND_PLAN §4** (the FareMark watermark scheme) and **§5.4** (the `AdaptiveTapFreeRider`
walkthrough — phase gate, the frozen-η oracle, the three coast modes, the tap/coast decision that probes
the *coast candidate* not the raw global, scope-freezing, and why the memory buffer only advances on
taps). The one fact to keep in hand while reading the trace below: with `coast_mode=graft` the FR submits
a **fresh global body + its frozen last-tapped mark head**, so the mark fades *gradually* (a real
sawtooth) while every submission still tracks the global — and the threshold re-taps only when its own
probe drifts above `target = η − margin`.

---

## 4. Does it work? — J2 rep0 evidence (the trace is the whole story)

**Per-round tap/coast, from `compute.per_client[cid].trace`:**

**cid3 (the easy free-rider — the submarine working as designed):**
```
R12 TAP  before=0.600 after=0.000     ← embeds cleanly
R13–24 COAST  before=0.200 (12 straight coasts, held under target 0.234)  ← graft holds the mark
R25 TAP  before=0.200 → coasts R26–37 at before=0.100
R38 TAP  before=0.200 after=0.500  ⚠   R39 TAP before=0.300 after=0.600  ⚠   ← taps make it WORSE
R40 COAST before=0.100
```
cid3 taps **4/29** freeride rounds (~14%), coasts to the `max_coast=12` cap twice. **This is a genuine
gradually-fading graft coast** — exactly the sawtooth the design predicted.

**cid6 (the hard free-rider — the submarine degenerating):**
```
R12 TAP before=0.800 after=0.200
R13–28 TAP EVERY ROUND  before pinned at 0.300 > target 0.234  ← threshold says "tap" 16× in a row
R29–38 COAST before=0.200 (finally under target)
R39–40 TAP before=0.400
```
cid6 taps **19/29** (~66%). Because its probe never gets under 0.234 until R29, the threshold rule
**collapses to reduced/always-tap for the hard class.** cid6's duty (0.75) ≈ the reduced baseline.

**Two findings the meeting needs:**
1. **The saving is asymmetric and lives entirely on the easy class.** cid3 coasts; cid6 works nearly
   as hard as an honest client. The "36%/26%" headline tap-fraction on the `tap_dyn_J2` plot is the
   **cid3-only** curve (its `ber_before` trace matches the orange line in `tap_J2_saw_graft_head_c36.png`
   exactly). Don't quote a single tap-fraction — quote **per-cid duty from the JSON** (0.375 / 0.75).
   *(Gotcha: RESULTS_INDEX says J2 "tap fraction 36%", the PNG title says "26%", the JSON duty-cycles are
   0.375/0.75 — three different denominators. Pin down which one before it goes in a table.)*
2. **The self-probe under-reads the server on the hard class, and the tail taps fail.** cid3's R38/R39
   taps *raise* `ber_after` to 0.5/0.6 — the embed didn't take. This coincides with the **R33–40
   instability** (see J5 below). A threshold submarine that trusts its own probe coasts on optimistic
   reads and taps into a catch.

3. **On the hard class the taps DO work — but the self-probe is too noisy to know it, so cid6 taps for
   nothing.** (Corrected reading; the `tap_perfr` plot shows it directly.) You must separate the FR's
   **self-probe** (`ber_before`, on 16 held-out images) from the **server's** BER (`wm_per_client.ber`, on
   50 test images — the thing that actually gets flagged). On cid6 they diverge: the self-probe reads
   **~0.30 every round** while the **server reads ~0.20** (attack-phase means: probe 0.290, server
   **0.217**). So (a) the taps genuinely hold cid6's *server* BER at ~0.20 — they are **not** failing;
   (b) but because the 16-image probe over-reads at 0.30 > target 0.234, the threshold rule fires **every
   round** → cid6 taps ~66 % and saves **no** compute. The failure is a **probe-calibration** problem
   (pessimistic on the hard class), not a caught mark.
   **Net (corrected):** at the server, **both** free-riders **evade** η_loose 0.264 — cid3 tail 0.190,
   cid6 tail **0.217** (both < 0.264). cid6's 0.217 is exactly the plain reduced-attack value for class 6
   (A3/D1: FR ≈ 0.22), so **on the hard class the submarine ≡ the reduced attack in both cost and BER** —
   no better, no worse — and catching it there still needs ~40 % honest FPR (A3/D1). The compute win is
   entirely cid3 (14 % taps). *(cid3's R38/R39 server BER spikes to 0.5 while its probe reads 0.2 — the
   opposite gap, and that one is the R33–40 instability, not steady state.)*

**Detection numbers (JSON top-level), and why "recall 1.0" is a trap:**
```
wm_eta_used = 0.064   wm_fr_recall = 1.0   wm_fpr = 0.3875   wm_benign_ber = 0.0775   wm_fr_ber = 0.185
```
The reported recall 1.0 is computed at **η_tight = 0.064, which is < 1/m = 0.10 → degenerate**
(`watermark.py:detected` docstring 287-290: any η in (0,1/m) is the same "zero bits wrong" test). At
that η the detector flags **39% of honest clients every round** (`wm_fpr=0.3875`). The "catch" is not a
catch — it's flagging 4 of 10 honest participants. **At the only usable operating line, η_loose =
0.264:** FR server-BER 0.185 < 0.264 → **evades**, honest 0.0775 → clean → **inseparable**. The FR sits
between the honest floors of its own classes (cls3 0.057 / cls6 0.114) and η_loose — precisely the
Group-A/D non-separability, now with a live attacker.

---

## 5. J5 — "why is it all coast?", the crashed config, and the plotting

J5 = J2 + honest probe (`TAP_PROBE_HOLDOUT=48`), safety margin **0.05** (target 0.214), deep coast
`TAP_MAX_COAST=20`. Aim (`run_now.sh:426-430`): let cid3 coast to the cap while the honest probe + wider
margin stop cid6 coasting into a catch.

### 5.1 Why J5 "looks like it never taps, just coasts" — it's the per-cid asymmetry
Read from the J5 log's phase-change annotations (which only print on a *change*, so a single "TAP@12"
means it tapped every round until the next printed change):
- **cid3 (easy): 2 taps in 28 freeride rounds (~7 %).** HONEST(1)→CALIB(8)→TAP(12)→COAST(13–32)→
  TAP(33)→COAST(35–39). Essentially pure coast. **This is the "I'm not tapping" you're seeing.**
- **cid6 (hard): 21 taps in 28 (~75 %).** TAP(**12–28**)→COAST(29)→TAP(30–31)→COAST(32–33)→TAP(34–35)→
  COAST(36–39). Almost the full reduced attack.

So J5 is **not** "not tapping" — it's tapping **hard on cid6 and barely on cid3**. Three knobs pushed
cid3 into near-total coast: (a) `margin=0.05` lowers the target to 0.214 so the decision `probe > target`
fires less; (b) `max_coast=20` lets it coast up to 20 rounds before a forced tap (in a 28-round attack
phase that's ~1 forced tap); (c) `holdout=48` (capped by `_prepare` to `n_trig//2 ≈ 25` real images)
makes cid3's probe read lower/steadier. On cid6 none of this helps because its mark won't embed under
target at all (same wall as J2 §4.3), so the threshold keeps firing. **The average tap-fraction is a
mirage** — if a plot/summary shows a single "majority" or cid3-weighted number, it reads ~coast; the
truth is a 7 %/75 % split. **Always report per-cid.**

### 5.2 Make the plotting correct (this is a real reporting bug, not just a nuance)
- **J5 has no `result.json`** (see §5.3), so any `tap_dynamics` plot for J5 is reading **partial or empty
  trace** → it can legitimately show "≈0 taps". Re-run to completion before plotting J5 at all.
- **`tap_dynamics` collapses the two FRs** — the `tap_dyn_J*` headline ("tap fraction 26 %/36 %") is a
  single representative/majority curve; for J2 it matches **cid3 only** (§4), which is why J looked like
  "all coast". **Fixed:** use the new **`plots.py tap_perfr`** command — one panel per free-rider cid,
  server BER vs self-probe, tap ▼ / coast ▢ markers, attack-phase tap-fraction per cid, evade/caught
  verdict per class. (See STATUS_AND_PLAN §10.7 for the exact CLI.) It confirms cid3 14 % / cid6 66 % and
  shows the self-probe sitting **above** the server line on cid6 — the over-reading that forces the
  wasteful tapping. Retire the single-family `tap_dynamics` trace for J; keep `tap_dynamics --families …`
  only for the multi-family stealth frontier.
- **To verify tapping directly (no plot needed):** run the two-BER snippet in "Experiments to run" on
  each cid — count `action=='tap'` over freeride rounds and check `ber_after`. That's the ground truth.

### 5.3 The crashed config — what happened to the second NOW run
NOW launched **6 runs** (J2×3 + J5×3). Of the two families, **J2 rep0 completed** (has `result.json`);
**J5 rep0 did not** — its log ends at **R39/40** (last line `R 39 … 70.34`, 01:22:40) with **no R40 and no
`result.json`**. Because `result.json` (compute + detection summary) is written only at the very end, the
run died in/after R39 during finalization — most likely a pod eviction/timeout (R39 itself completed
cleanly at 70.34 % acc), possibly nudged by the instability just before it. **Consequence: there is no J5
compute/detection summary to quote — only the log.** What the J5 log *does* tell us:
- The tuning worked **directionally**: cid3 coasts far more than in J2 (2 taps vs 4), cid6 unchanged (~75 %).
- **The R33–35 instability is severe here:** test acc **crashes 69.3 → 24.3 %** at R33, honest BER spikes
  **0.037 → 0.338 → 0.438** (R33–35), FPR 0.75/0.875, then recovers to ~70 %/0.037 by R36. Coincides with
  cid3 TAP@33 and cid6 TAP@34 — a suspected late **aggregation / Eq.14 memory** instability, **not** the
  attack. It must vanish (or be fixed) at seeds 1001/1002 before any J5 number is tabled, because it sits
  in the converged tail we read the verdict from.

**Bottom line on J5:** design change confirmed (cid3 → near-zero taps), but the run is **incomplete and
contaminated** — re-run all three seeds to completion, and it still does nothing for the hard class cid6.

---

## 6. What's missing for a *complete* working submarine

A "complete" submarine would coast on **both** classes, at an η it **estimated itself**, with a stable
tail. Concretely, in priority order:

1. **Honest η estimation (the biggest gap).** J2/J5 use `eta_source="oracle"` → the FR is *given*
   η=0.264. The self path exists (`_resolve_eta` 794-800: μ+k·σ over `_calib_bers`) but is **untested at
   3 seeds**. A real server never publishes η. Until the self-estimate is shown to (a) land near 0.264
   from the FR's own 4-round calib window and (b) still evade, the attack has an oracle crutch. **Raise
   this first — it's the honest-vs-cheating line.**
2. **Close the self-probe/server gap on the hard class.** cid6's 16-image (J2) / ~24-image (J5) probe
   under-reads the server's 50-image bank, so cid6 either taps every round (J2) or risks coasting into a
   catch. Holdout 48 helps but doesn't close it. Without this, the submarine on hard classes **is** the
   reduced attack — no saving.
3. **Resolve the R33–40 instability.** Confirm it's seed-specific / a memory-buffer artifact and fix or
   exclude it. It currently corrupts the converged-tail reads that produce the verdict.
4. **Attack-phase-only effort metric.** The pipeline logs per-round samples
   (`compute.per_client[cid].per_round`); emit an effort number over R13–40 only. The headline 31%
   *understates* the ~4% marginal cost — reporting it makes the win visible **and** honest.
5. **Cheaper taps.** Graft holds the mark, so sweep `tap_data_cpc ∈ {1,2}` under graft/head. cpc=1 =
   685 passes/tap (2.7% of honest) — if it still re-embeds, each tap halves again.
6. **Three seeds, both families.** Only **J2 rep0** has a result.json; **J5 rep0** is a partial log.
   Nothing here is tableable yet — the NOW batch is meant to produce 1000/1001/1002 for both.

**The verdict that survives all of the above (say it plainly in the meeting).** A real adaptive submarine
*exists* (J2/cid3: gradual graft fade, genuine coast, live submissions — categorically more than
"reduced with a schedule"). **But it does not threaten the impossibility thesis; it confirms it.** To
keep every coast peak under a *usable* η, the mark must hold near the **honest floor of its class** — i.e.
the FR is doing enough real work to be **indistinguishable from an honest client there**, and any η low
enough to catch its peaks also flags honest cls6 clients (the same 39% FPR at η_tight). The submarine
makes free-riding *cheaper on easy classes*; it does not make it *detectable*. Output-layer,
BER-thresholded free-rider detection still has no deployable operating point.

---

## 7. Gotchas to raise before anyone quotes a NOW number

*(Extends the Group-J gotchas list.)*
- **`adaptive_tap` ≠ the "submarine" class.** The full `SubmarineFreeRider` (dynamic warmup, own-η) is
  commented out (`clients.py:937-960`); NOW ran the simpler `AdaptiveTapFreeRider`. Don't claim dynamic
  warmup results.
- **Oracle η is a gift.** `AUTOP_ORACLE_ETA=0.264` hands the FR the true loose threshold. Any evasion
  claim carries this asterisk until `eta_source="self"` is run.
- **η_tight 0.064 < 1/m 0.10 is degenerate** — `recall=1.0 / FPR=0.39` is flagging 39% of honest
  clients, not detecting. The real line is **η_loose 0.264**, under which the FR evades.
- **Effort headline is warmup-dominated** (~90% of FR compute is the 11 forced-honest rounds). Marginal
  attack cost is ~4%, not 31%. Bring both.
- **Tap-fraction is ambiguous** (index 36% vs PNG 26% vs JSON duty 0.375/0.75). Use per-cid duty.
- **cid6 degenerates to always-tap in J2** — the saving is a cid3 phenomenon. Report per-class.
- **R33–40 instability is unresolved** and sits in the tail we read. Confirm across seeds first.
- **Only J2 rep0 has a result.json; J5 rep0 log is partial (ends R39).** Nothing is 3-seed yet.
- **The pool skips a family by directory existence, not knob match** (`runbook.sh:24-27`) — delete stale