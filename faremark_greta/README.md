# FareMark — reproduction + limitations study

Re-implementation and limitations analysis of **FareMark: Model-Watermark-Driven
Free-Rider Detection in Federated Learning** (Li et al., IEEE IoT-J 12(18), 2025).
Centralized FedAvg simulated on one GPU, with a per-client output-layer watermark loss,
a memory-enhanced update (Eq. 14), and server-side verification (Eq. 15–16).

**Read order:** this file → `REFERENCE.md` (threshold maths, CLI, experiment matrix) →
`RUNPLAN.md` (what to run right now) → `faremark_analysis.md` (full results review).

---

## Goal

Show experimentally that under the paper's own setup **and** under extensions
(non-IID, adaptive free-riders, more clients than classes), **no threshold separates
honest clients from free-riders.**

Two independent lines of attack, both now supported by data:

1. **The threshold is unusable.** `η = μ + 3σ` is calibrated on a mean over *N* clients
   (spread σ/√N) and applied to individual clients (spread σ). Measured across seven
   configurations it delivers **0.07σ–0.51σ of headroom, never 3σ**, producing 7–53%
   honest false-positive rates where a 3-sigma limit promises 0.13%. Independently,
   whenever the mark works well η falls below `1/m` and the rule degenerates to
   "flag if ≥1 bit wrong", making the calibrated value irrelevant.
2. **The quantity itself carries no contribution signal.** At a fixed trigger class, a
   free-rider spending ~30% of an honest client's compute achieves **equal or better**
   BER. Six per-class comparisons reach `best_threshold_balanced_error = 0.500` — no
   threshold anywhere beats a coin flip.

### What we now claim, and what we retracted

The original thesis was: *a confidently-predicted class has a flat, structureless tail,
so its bits are decided by noise and its BER floors above zero — intrinsically.*
**That mechanism is refuted by our own balanced-key runs** (STATUS F9): with zero-sum
key rows the mark embeds perfectly on *every* class, so the model can create tail
structure at negligible cost. The floor is a **key-row-bias × class-peakiness
interaction**, not a property of reading from the softmax.

The replacement claim is quantitative and survives: BER is a mean over *m* bits, so it
carries binomial noise `√(p(1−p)/m)`; the disjoint grouping forces `m·l = n`, hence
**m ≤ n**. Discriminative power is therefore capped by the number of classes in the
dataset. On CIFAR-10 that cap is fatal (F6).

---

## Layout

```
faremark/
  clients.py           every client class:
                       PART 1  honest FedAvg Client
                       PART 2  WatermarkClient (Eq.11-12 + Eq.14) + factory
                       PART 3a crude baselines Eq.17/18 + FR selection + build_clients
                       PART 3b reduced/tap attackers [+ DISABLED submarine]
  watermark.py         Eq.1-16 math -- leaf, used by clients.py AND wm_verify.py
                       *** PATCHED: sin smoothing, SMOOTH_EPS, make_key default ***
  wm_verify.py         server: extract -> BER -> frozen eta -> flag + diagnostics
  server.py            FedAvg aggregation + round loop
  runlog.py            all run.log formatting
  compute_meter.py     per-client effort accounting (gpu_ms / samples / flops)
  config.py datasets.py models.py manifest.py utils.py plotstyle.py
  fast_data.py         [OPTIONAL, NOT WIRED] GPU-resident loader, see PERFORMANCE
  robustness.py        [UNWIRED] finetune/prune/quantize -- paper V-E, not run
scripts/
  run_experiment.py    one (config, repeat) -> result.json
  resultio.py          result.json data contract (load/select/BER extraction)
  detection.py         calibrate | verify | separability
  plots.py             all plotting (subcommands, see REFERENCE 3a-F)
  plot_all_thresholds.py  *** NEW: every threshold rule on one timeline + .md table ***
  paper_check.py       grade runs vs the paper's published rows
infra/
  run_now.sh           *** NEW: THE script. Builds jobs.tsv for the current batch ***
  plot_now.sh          *** NEW: makes exactly the four figure groups, nothing else ***
  submit_pool.sh       *** NEW: submits exactly PODS worker jobs that drain jobs.tsv ***
  submit_experiment.sh *** PATCHED: deterministic RUN_TAG + DRYRUN manifest mode ***
  run_everything.sh    the full matrix in legs (still works; superseded by run_now.sh)
  run_all.sh           one leg: honest -> calibrate -> attacks -> separability -> PLOTALL
  paper_check.sh       submit/grade the paper reproduction rows
```

---

## Standard setup

CIFAR-100, ResNet-18, 10 clients, 50 rounds, 5 local epochs, batch 16, N_T=50, λ=5,
β=0.6, α=0.4, **unbalanced (paper-exact random ±1) keys**, config 14.
Trigger class = `cid % n`; trigger class is **included** in the projection
(`exclude_col = None` in `clients.py`) — this is paper-faithful.

**Bit count `m`, now derived rather than guessed.** A same-sign key row is stuck
forever, capping watermark accuracy at `100·(1 − 0.5·2^(1−l))`, `l = n//m`. Matching
that against the paper's own tables:

| dataset | m | l | stuck | ceiling | paper reports | use? |
|---|---|---|---|---|---|---|
| CIFAR-100 | **10** (code default) | 10 | 0.20% | 99.90% | 99.71 | ✅ **yes** |
| CIFAR-100 | 20 | 5 | 6.25% | 96.88% | — | ✗ cannot be faithful |
| CIFAR-10 | 2 (code default) | 5 | 6.25% | 96.88% | 99.72 | ❌ 3pp short |
| CIFAR-10 | **1** | 10 | 0.20% | 99.90% | 99.72 | ✅ **yes** (needs the `make_bits` fix, F12) |

So on CIFAR-10 the paper's headline number **forces a 1-bit watermark**, at which point
BER ∈ {0,1} is a single coin flip. That is a finding, not a configuration choice.

Threshold: `η = mean over seeds of (μ_s + 3σ_s)` over per-round mean-over-clients honest
BER, last 20 rounds; frozen and injected as `WM_ETA_FIXED`.
CIFAR-100 / 10 clients: **η = 0.063** (per-seed 0.017–0.115, std ≈ 40%).

---

## Quickstart

```bash
# 1. on the cluster login node, in infra/
./run_now.sh                    # builds jobs.tsv, submits NOTHING
wc -l jobs.tsv                  # expect 20
unset DRYRUN
RUNAI_EXTRA="--node-pools a100-80" PODS=2 WORKERS=6 ./submit_pool.sh
runai list jobs                 # must show exactly 2 -- then walk away

# 2. when done
scp -r <cluster>:$MOUNT/home/zu/results ~/local/results
RES=~/local/results ./plot_now.sh
```

`run_now.sh` and `submit_pool.sh` replace the old "one RunAI job per run" model:
one pod now drains many runs with `WORKERS` concurrent processes on its single GPU.
`RUN_TAG` is deterministic, so a preempted pool resumes instead of restarting —
just rerun `submit_pool.sh`. See REFERENCE §3a for every other command.

---

## Key results so far

Full write-up in `faremark_analysis.md`; findings F1–F14 in STATUS below.

**Solid:**
- **F8** the paper's `μ+3σ` rule delivers **0.07σ–0.51σ**, never 3σ. FPR 7–53%.
- **F9** whenever the mark works well, `η < 1/m` and the detector degenerates to
  "≥1 bit wrong" — the calibrated value does nothing. 4 of 8 legs.
- **F4** at a fixed trigger class a +5 free-rider on ~30% effort matches or beats
  honest BER; six per-class comparisons hit best-balanced-error **0.500**.
- **F13** this **directly refutes the paper's Table V**, which claims a
  trigger-sample-only free-rider fails to generalise.
- **F1** per-class difficulty is real and generalises (classes 9,19,…,99 reproduce
  classes 0–9 to 3 s.f.), driven by softmax peakiness, not accuracy.
- **F6** the grouping `m·l = n` makes CIFAR-10 unworkable in every configuration.
- **F7** honest FPR grows with client count: 27% at N=10, **53% at N=200**.

**Retracted or restated:**
- **F3 (restated)** honest BER is **not** bimodal — it is a near-binomial discrete
  decay (73/19.5/7/0.5% at 0/1/2/3 wrong bits). The high FPR is real; the mechanism
  is F9, which is stronger.
- **F9-mech (retracted)** the floor is **not** intrinsic to output-layer watermarking.
- **F10** most honest BER spread is bit-sampling noise, not class difficulty.

---

## Faithfulness caveats to state in any writeup

- The paper **does not state m**. We now derive it from their reported accuracies
  (above) rather than guessing: CIFAR-100 m=10, CIFAR-10 m=1.
- The paper's grouping is disjoint (`m ≤ n`); its "50/400 bits" refers to the FedIPR
  baseline, not FareMark.
- Trigger class is **included** in the projection — paper-faithful, confirmed in code.
- The paper is **internally inconsistent on training length**: §V-A says 2 local epochs
  / 100 global rounds, §V-B and §V-C say 5 epochs / 50 rounds. We follow the tables
  (5/50).
- N_T is **100** for Table I/II (§V-C) and **50** for Tables III and IX. Match per row.
- The paper averages **ten** repeats (§V-A1). We use 3–6. State the discrepancy.
- **§IV-D3 never says whether μ,σ are over clients, over rounds, or over round-means.**
  Those readings differ by √N. The ambiguity is itself a finding (F8).

---

## Which result files to keep

Everything lives flat in `$RES/<RUN_TAG>/`, one directory per run, containing
`result.json` (the data), `run.log` (the experiment record) and `pod.log` (the
environment record, including `env.git_commit`).

| family | status | keep? |
|---|---|---|
| `honest_c100_bdef_iid` (6 seeds) | the baseline; F8/F9/F3 rest on it | ✅ **keep** — still the honest reference for plot group (d) |
| `reduced_c100_bdef_iid_c17` / `_c36` | F4, F13 | ✅ keep |
| `sameclass_c100_bdef_iid_c6` | the headline non-separability slice | ✅ keep |
| `honest_c100_bdef_niid` + its attacks | non-IID leg | ⚠️ keep, **do not quote** until `n_trigger_samples` is logged and the starvation population is split out (F11) |
| `honest_c100_bdef_bal_iid` + attacks | balanced-key control, OVL 1.000 | ✅ keep — this is the "even at your own operating point" control |
| `honest_c100_bdef_spread_iid` + attack | generality of F1 | ✅ keep |
| `honest_c10_bdef_iid` (50 clients) + `reduced_c1617` | F6, capacity10 | ✅ keep |
| `honest_c100_bdef_nc200_iid` (3 seeds) | F7 (53% FPR) | ✅ keep, being topped up to 6 seeds by R5 |
| `honest_c100_b20_iid` + attacks (m=20) | **superseded** — m=20 caps at 96.88%, below the paper's own number, so it can never be a faithful configuration | 🗄️ **archive**, do not use in the writeup |
| `honest_c100_bdef_sin_iid` + attack | **invalid** — sin performed no smoothing (F12) | 🗑️ **delete**, rerun after the fix |

**One global caveat:** every existing CIFAR-100 run used `eps = 1e-3` in `smooth()`,
which erases 39–57% of the tail contrast (F12). The *structural* conclusions (F8, F9,
F4, F6, F7) do not depend on exact BER values and stand. The exact floors will move
when `SMOOTH_EPS=1e-8` is adopted. **The patched `watermark.py` defaults to the legacy
1e-3 precisely so this weekend's batch stays comparable with what you already have** —
switch only for a clean full re-run, and never mix the two inside one family.

---

## Performance note

Runs are ~781,250 optimizer steps each (50,000 images × 5 epochs × 50 rounds ÷ batch 16)
**regardless of client count** — more clients split the same data. Two pods × 6 workers
≈ 3.5–4.7 runs/hour.

Before optimising anything: your DataLoaders use `num_workers=2` without
`persistent_workers`, so workers are forked and killed on every iterator —
**2,500 fork/teardown cycles per run**. Time one run with `--num_workers 0` first; at
32×32 and batch 16 it may well win. `fast_data.py` (GPU-resident loader, ~2–4× more)
is written but **not wired**, and it re-rolls the augmentation RNG, so adopt it only at
a clean family boundary.

---
---
---

# STORYLINE

## Plan

### 1. Reproduce the FareMark method — ✅ done
Eq. 1–16 implemented in `watermark.py`, embedding in `clients.py`, verification in
`wm_verify.py`. Three deviations found and corrected this cycle: the `sin` branch
(F12), `SMOOTH_EPS` (F12), and `make_key`'s default disagreeing with the config.

### 2. Reproduce their results — 🔁 **in flight (R1–R4), the biggest historical gap**
Not one run to date matched any published row, and the paper's own crude free-riders
had never been fired. Without this, "the detector fails" has no counterpart showing it
ever worked in our code. R1 (CIFAR-100/100 clients), R2 (CIFAR-10/m=1) target Table
I+II; R3/R4 fire Eq. 17/18. **If R3/R4 are not cleanly caught, stop everything.**

### 3. Limitations — ✅ done
Threshold underspecification (F8, F9), per-class difficulty (F1), data regime (F11).

### 4. Prove non-separability — ✅ done, and it refutes Table V
Any threshold, any setting → FPR ↔ recall trade. Six per-class comparisons at
best-balanced-error 0.500; where a free-rider *is* catchable it costs 33–67% FPR among
honest clients at the same class. The paper's §V-D4 explicitly predicts this attacker
fails; it does not (F13).

### 5. Stress test — ⚠️ partial
non-IID ✅ (blocked on F11), free-rider fraction ❌, bits ✅ (m=20 archived, F14),
oversubscription ✅ honest / 🔁 attack in flight (R5, R6), adaptive free-rider 🔁 (R7).

### 6. Argue output-layer watermarking is impossible — ⚠️ **restated**
The tail-structure argument is refuted (F9). The surviving argument is the capacity
inequality: separating two BER populations differing by Δ with balanced error < ε needs
`Δ ≳ 2·z_ε·√(p(1−p)/m)` with `m ≤ n`, so discriminative power is bounded by the number
of classes. Quantitative, provable, and independent of softmax-tail intuition.

### 7. Hint at a solution — ❌ not touched
Per-position calibration · high-entropy triggers · reading off the output layer.

## Contribution arc
Reproduce → find the threshold is underspecified and self-defeating → show a cheap
free-rider matches honest BER at the same trigger class → show this holds under
non-IID, oversubscription and an adaptive attacker → bound the achievable
discrimination by the class count.

## HYPOTHESIS: why some positions are hard — **corrected**

Write the projection for bit *k* as

```
z_k  =  s_k · f̄   +   Σ_j M_kj · δ_j        s_k = Σ_j M_kj ,  δ_j = f(p_j) − f̄
       └─ bias ─┘     └─ shapeable part ─┘
```

With **unbalanced** keys `s_k ≠ 0`, so a bias term proportional to the group's average
smoothed probability competes with the shapeable part. In a peaky class the tail values
are nearly identical, the δ's are small, the bias wins, and the bit is effectively
fixed — reproducing the entropy correlation of F1. With **balanced** keys `s_k = 0`, the
bias vanishes and an arbitrarily small tail asymmetry suffices, which the model can
manufacture at negligible accuracy cost. Hence BER → 0 for every class (F9).

**So difficulty is a key–class interaction, not an intrinsic softmax property.**

## Attacks used to make the argument

| attack | what it does | role |
|---|---|---|
| `previous_models` (Eq. 17) | resubmits last round's model | **paper's own baseline** — must be caught cleanly, else the pipeline is broken |
| `gaussian` (Eq. 18) | prior global + Gaussian noise | same |
| `reduced` (+N) | honest until round 12, then trains only on all trigger images + N per common class. ~30% of honest image-passes | the main attacker. Trigger-enriched batches fire L_wm ~9× more often, so it embeds **harder** than honest |
| `sameclass` | `reduced`, pinned onto a trigger class an honest client already holds | the clean comparison: identical position, everything else equal |
| `tap_oracle` | given the true η, coasts (zero compute) while safely under it, taps when the mark decays | the adaptive attacker (R7) |

---

# STATUS

## Findings

### F1 — Per-class difficulty is real, driven by peakiness, and generalises ✅
Floors span 0.000–0.105 on CIFAR-100/10 clients. Entropy r = −0.67, dominance
r = +0.65, test accuracy r = −0.05 in the 10-client regime. Independent class sets
(9,19,…,99) reproduce η, mean, σ and FPR to 3 s.f. In the 200-client regime accuracy
correlates at r = −0.46 — not a contradiction: there the model is starved, so peakiness
and accuracy become the same variable.

### F2 — The threshold is seed-unstable ✅
Per-seed η 0.017–0.115, std ≈ 40% of its own value. A threshold whose calibration noise
is 40% cannot support a fixed detection policy. Cause: the key `M` and bits `B` are
redrawn per seed, so each seed asks a *different question*, not the same one more
precisely (REFERENCE §3).

### F3 — High honest FPR ✅ **(mechanism restated)**
27% (N=10), 17% (m=20), 53% (N=200). **The distribution is not bimodal** — panel (b)
of the thresholds figure shows 73/19.5/7/0.5% at 0/1/2/3 wrong bits, close to
Binomial(10, 0.036). The bimodality claim is withdrawn; see F9.

### F4 — The `reduced` attacker: non-separability, demonstrated ✅
Per class, `best_threshold_balanced_error = 0.500` in six comparisons — no threshold
beats a coin. In 8 of 18 the free-rider's mark is *cleaner* than an honest client's at
the identical trigger class. Where it is catchable, recall 1.0 costs 33–67% FPR at that
class. `sameclass_c6`: honest 0.105 vs free-rider **0.042** on 31% of the effort.

### F5 — Trigger-enrichment: less data embeds the mark *harder* ✅
The reduced shard is concentrated on the trigger class, so L_wm fires on ~9% of batches
instead of ~1%. This is why the free-rider is often *better* than honest. Frame it as
**"BER measures whether you trained on the trigger class, and that is the cheap part"**,
not as "free-riders are undetectable" — which invites the reply that the attacker simply
specialised.

### F6 — CIFAR-10 cannot support the scheme ✅
`m·l = n` forces: m=2 → BER ∈ {0,½,1}, threshold meaningless; m=5 → 50% stuck bits;
m=10 → 100% stuck. m=1 reaches the paper's number but makes BER a single coin flip.
There is no good setting, and CIFAR-10 is the paper's primary dataset.

### F7 — Honest FPR grows with client count ✅
27% at N=10 → 53% at N=200, matching the `3/√N` headroom prediction (F8). The scheme
degrades in exactly the direction real FL deployments go.

### F8 — `μ+3σ` delivers 0.07σ–0.51σ, never 3σ ✅ **(headline)**
η is built from `m_r` (mean over N clients, spread σ/√N) and applied to individuals
(spread σ). Measured `(η − μ)/σ_per-client`: iid 0.42, spread 0.42, bits20 0.31,
noniid 0.26, capacity10 0.23, sin 0.51, capacity(N=200) **0.07**. A Shewhart 3σ limit
promises ~0.13% false alarms; this gives 7–53%. Provable in two lines, independent of
any experiment.

### F9 — The better the watermark works, the more meaningless the threshold ✅ **(headline)**
BER only takes multiples of `1/m`. If the mark is good, μ→0 and σ→0, so η→0 and in
particular **η < 1/m** — at which point `flag iff BER ≥ η` is exactly `flag iff ≥1 bit
wrong` and every η in `(0, 1/m)` is the same detector. True in iid (0.063 vs 0.100),
spread, capacity10 (0.068 vs 0.500) and balanced (η = 0, flags 100%). The calibration
procedure is well-defined only when the scheme it protects is malfunctioning.

### F10 — Most honest BER spread is bit-sampling noise ✅
Predicted binomial σ vs observed: CIFAR-10 0.134/0.131 (**all** noise), iid
0.059/0.064, bits20 0.080/0.086. Only non-IID shows large genuine excess. The
per-class effect is second-order on top of a quantisation noise floor.

### F11 — Non-IID leg is contaminated ⚠️ **blocked**
Floors 0.007–0.297 with σ 0.179 > mean 0.115, far above the binomial prediction of
0.101. But under Dirichlet a client can be assigned trigger class *c* while holding
almost no images of *c* → BER ≈ 0.5 for a reason unrelated to the watermark.
**Log `n_trigger_samples` in `wm_verify.py` and split the honest population by it
before quoting a single non-IID number.**

### F12 — Three real bugs in `watermark.py` ✅ fixed
1. **`sin` performed no smoothing.** `torch.sin(alpha*p)` with the shared `wm_alpha=0.4`
   spans only `[0, 0.4]` radians, where sin is linear. Tail amplification: power 4.87,
   sin@0.4 **1.01 (the identity)**, sin@π/2 1.23. So `p_max` still dominated, Eq. 10's
   `f(max)/Σf < 0.5` was violated, and BER sat at chance and never descended. The fix
   validates α, rejects α > π/2 (**non-monotone**) and rejects any α with gain < 1.10.
   Second-order finding: even at its best, Eq. 9 amplifies ×1.23 against Eq. 8's ×4.87 —
   **the paper presents them as interchangeable and they are not.**
2. **`eps = 1e-3` erased the CIFAR-100 tail.** Tail probabilities are themselves ~1e-3,
   so 39–57% of the contrast between tail entries was destroyed before projection. Now
   `SMOOTH_EPS`, **defaulting to the legacy 1e-3** so in-flight families stay
   comparable; export `SMOOTH_EPS=1e-8` for a clean full re-run.
3. **`make_bits` returned a constant message at m=1.** `half = 1//2 = 0` gave
   `B = [0]` for every client, so the "secret" was a constant and a free-rider guessing
   0 scored 100%. m=1 is exactly what CIFAR-10 needs to reach the paper's 99.72%, so the
   CIFAR-10 reproduction was **impossible** before this fix. Now uniform for m < 4.
   Related faithfulness note: exact bit-balancing deviates from the paper at *every* m —
   at m=10 we always emit exactly five 1s where the paper draws uniformly ("we randomly
   set the watermark to be embedded", §V-A1). It lowers a random guesser's BER variance
   without moving its mean of 0.5.

### F13 — Our central result refutes the paper's Table V ✅ **(headline)**
§V-D4: *"if an attacker trains solely on a small number of trigger samples, the
watermark could not be detected … because the embedded watermark becomes overfitted to
those specific samples and cannot be generalized to other trigger-class samples."*
Our `reduced` attacker is exactly that, verified on a **held-out** bank (`class` mode) —
the strictest reading — and it matches or beats honest BER. **Lead the writeup with this.**

### F14 — Table IX contradicts Table V ✅
The capacity result holds only because §V-F3 enforces *trigger sample consistency*
("the trigger samples used during testing are identical to those employed in training").
That is precisely the memorisation Table V calls a failure mode. The paper's scaling
claim rests on the thing it elsewhere says does not generalise.

## What is proven vs to-be-shown

| claim | state |
|---|---|
| `μ+3σ` delivers ≤0.51σ in every configuration | ✅ proven, analytic + 7 legs |
| η < 1/m makes the threshold degenerate | ✅ proven, analytic + 4 legs |
| free-rider on ~30% effort matches honest BER at the same class | ✅ proven, 6 comparisons at 0.500 |
| catching a free-rider costs 33–67% FPR at its class | ✅ proven |
| per-class difficulty is real and generalises | ✅ proven |
| honest FPR grows with N | ✅ proven (27% → 53%) |
| CIFAR-10 cannot support the scheme | ✅ proven from `m·l = n` |
| refutes Table V | ✅ proven, pending R3/R4 confirming the pipeline |
| our code reproduces the paper's rows | 🔁 **in flight (R1–R4)** |
| non-IID worsens separability | ⚠️ blocked on F11 |
| discrimination is bounded by class count | ⚠️ inequality stated, not formalised |
| output-layer watermarking impossible in general | ⚠️ restated, empirical only |

## Immediate next
See `RUNPLAN.md`. In order: **R1–R8** (this batch) → log `n_trigger_samples` (F11) →
rerun `sin` with `WM_ALPHA=1.5708` → the +N sweep (`N=-1` anchor) → clean full re-run
with `SMOOTH_EPS=1e-8`.

---

# CODEMAP

Legend: **[WIRED]** in the pipeline · **[NEW]** added this cycle · **[PATCHED]** changed
this cycle · **[UNWIRED]** exists but not called.

## 1. Watermark math — `faremark/watermark.py` [WIRED] [PATCHED]

| step | function | paper | notes |
|---|---|---|---|
| smoothing f(p) | `smooth(p, kind, alpha, eps)` | Eq. 7–9 | **[PATCHED]** `eps` → module const `SMOOTH_EPS` (env-switchable, default legacy 1e-3); `sin` branch validates α ∈ (0, π/2] and rejects gain < 1.10 |
| how much f() actually smooths | `smoothing_gain(kind, alpha)` | — | **[NEW]** 1.0 = f does nothing. Check any (kind, α) before spending GPU |
| secret ±1 key M [m,l] | `make_key(m, l, seed, balanced)` | §IV-A | **[PATCHED]** default now `False`, matching `config.wm_balanced_keys` and the paper |
| stuck-row fraction | `unembeddable_fraction` | diagnostic | `P = 2^(1−l)`; logged as `wm_unembeddable_frac`. Ceiling = `1 − 0.5·P` |
| target bits B | `make_bits` | Eq. 2 | balanced 0/1, so a random model sits at BER 0.5 |
| group size l = n//m | `grouping` | §IV-A | `m ≤ n` — the bit ceiling (F6) |
| project → per-bit z | `project_logits(..., exclude)` | Eq. 1/13 | `exclude=None` = full softmax, **paper-faithful** |
| embed loss | `watermark_loss` | Eq. 11–12 | BCE(z, B) |
| extract | `extract_bits` | Eq. 15 | mean z over N_T, then sign |
| BER | `bit_error_rate` | Eq. 16 | |
| flag test | `detected(ber, eta)` | Eq. 16 | `ber < eta`. Docstring now spells out the η=0 and η<1/m degeneracies (F9) |
| dominance ratio | `dominance_ratio` | Eq. 6/10 | want < 0.5; the diagnostic that exposed the sin bug |

## 2. Honest client + factory — `faremark/clients.py` [WIRED]
- `WatermarkClient.produce_update` → `_local_train_wm` (`L = CE + λ·wm_loss` on trigger
  images) → `_memory_update` (Eq. 14 `W = β(memory+Δ) + (1−β)·global`).
- `build_watermarked_clients`: `trigger_class = cid % n`; `m = cfg.wm_bits or max(2, n//10)`;
  `l = n//m`; `exclude_col = None`; `key = make_key(..., balanced=cfg.wm_balanced_keys)`
  seeded `seed + 1000·cid + 1`. Dispatches free-rider slots by `cfg.attack`.
- **Open edit:** if you adopt `fast_data.py`, the reduced attacker must call
  `self.loader.subset(idx)` instead of building a fresh CPU `DataLoader`.

## 3. Attackers — `faremark/clients.py` PART 3
`previous_models` / `gaussian` (Eq. 17/18) · `reduced` (+N, the main one) ·
`tap_oracle` (coast/tap adaptive) · `submarine` **[DISABLED — warmup bug]**.

## 4. Detector — `faremark/wm_verify.py` [WIRED]
`WatermarkRegistry` (cid → trigger_class, key, bits, kind, alpha, exclude);
`build_trigger_bank` (`class` mode, held-out, shared per class),
`build_trigger_bank_per_client` (`client`, disjoint held-out slices),
`build_trigger_bank_from_train` (`client_train`, the client's own training images —
paper §V-F3). **All three are used only inside `verify_hook`, never in training**, so
one training run can be extracted three ways (saves 18 jobs on the capacity legs).
Per round emits `wm_benign_ber{,_p90,_max}`, `wm_fr_ber`, `wm_fpr`, `wm_fr_recall`,
`wm_eta_round`, `wm_eta_source`, `wm_flagged_cids`, and `wm_per_client[]` with
`{cid, trigger_class, ber, is_free_rider, flagged, pmax, entropy, dominance, trig_acc}`.
**Missing:** `n_trigger_samples` per client per round — required for F11.

## 5. Analysis — `scripts/detection.py` [WIRED]
`calibrate` (freeze η) · `verify` (confirm attack runs used it) · `separability`
(the rule-independent tables: 9 rules + OVL + best-threshold balanced error).
**Open edit:** emit `rule=degenerate` instead of `fpr=1.0, recall=1.0` when the honest
support is a point mass.

## 6. Orchestration — `scripts/run_experiment.py` [WIRED]
Every CLI flag overrides the matching `cfg` field. Writes `result.json`.
**Exit code 2 = accuracy outside `expected_acc`** — EXPECTED for attack runs;
`result.json` is written before the exit. `submit_pool.sh` treats 0 and 2 as success.

## 7. Plotting — `scripts/plots.py` [WIRED] + `plot_all_thresholds.py` [NEW]
Subcommands: `thresholds`, `class_difficulty`, `class_probe`, `class_dynamics`,
`positions`, `fidelity`, `timeline`, `honest_lines`, `separability`, `sweep`,
`honest_fpr`, `sanity`, and legacy `threshold`/`frontier`/`scorecard`/`test_data`.
`plot_all_thresholds.py` **[NEW]** draws *every* honest-only rule on one BER-vs-round
timeline plus a red `1/m` line, and emits a `.md` table giving each rule's η, how it was
computed in prose, its honest FPR, its **headroom in σ**, and whether it is degenerate.
This is the figure `plots.py thresholds` could not make.

**Open plot fixes:** the green "USED eta" line is described in `thresholds`' title but
never drawn; `timeline` prefers `config.wm_eta_fixed` over the eta file, so figures show
the provisional 0.050; `honest_lines` bands extend below 0; the 200-client legend covers
60% of the canvas.

## 8. Runners — `infra/`
| script | role |
|---|---|
| `run_now.sh` **[NEW]** | builds `jobs.tsv` for the current batch (R1–R8). Submits nothing |
| `submit_pool.sh` **[NEW]** | submits exactly `PODS` worker jobs; each drains its shard with `WORKERS` concurrent runs. Resume-safe |
| `plot_now.sh` **[NEW]** | local; makes exactly the four figure groups |
| `submit_experiment.sh` **[PATCHED]** | deterministic `RUN_TAG` (no timestamp — required for resume) + `DRYRUN=1` manifest mode |
| `run_all.sh`, `run_everything.sh`, `paper_check.sh` | unchanged; still work for ad-hoc legs |

## Data contract — `result.json`
Authoritative in `scripts/resultio.py`. Top level: `schema_version`, `config`,
`manifest{family, sweep_var, sweep_level}`, `summary{}`, `env{git_commit, torch, host}`,
`free_rider_indices`, `final_acc`, `best_acc`, `correctness_pass`, `per_class`,
`compute`, `history[]`. Quick look:

```bash
python scripts/resultio.py digest   --in 'results/*/result.json'
python scripts/resultio.py contract --in results/<run>/result.json
```