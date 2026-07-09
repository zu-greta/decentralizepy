> TODO: update the documentation files




> **READ STATUS.md FIRST.** This file may contain older framing. Current state (preliminary, pending the `run_full_sweep.sh` results): the three *coast* attacks (submarine/memory_exploit) are NOT cleanly cheap+evasive+harmless — they hit a trilemma (cheap / keep-mark / don't-poison: pick two). A 4th attack `reembed` (output-layer head fine-tune on the fresh global) is implemented but not yet run; it targets the actual weak point. Read `wm_fr_ber` + `final_acc`, not `wm_fr_recall`.

# ADAPTIVE ATTACKS — status, threat model, implementation, experiment catalog

The authoritative reference for the effort-minimizing free-rider work (submarine,
memory-exploit) against FareMark's output-space watermark detector. Read this
before a progress meeting: §2 (threat model) and §3 (implementation) are what
reviewers probe; §5 is the experiment catalog with a fixed analysis template.

---

## 1. Where we are (updated after the 50-round validation)

**Milestone reached:** the full pipeline runs end-to-end on the A100 — attacks
slot into the loop, compute is metered, runs are self-describing, traces are
logged. The 50-round validation fixed the horizon problem (honest BER converges
to ~0.03, η settles to ~0.3, accuracy ~66%) and surfaced the *real* mechanism.

**The key finding (validated, and central to the thesis):** there is **no cheap
shortcut to a mark that generalizes to the server's test triggers.** A
trigger-enriched burst embeds fast on the attacker's own samples but overfits
them and does *not* transfer (the validation showed warmup BER stalling at
0.2–0.4 and the server catching it) — this is exactly the paper's Table V
mechanism. The only thing that generalizes is **full-shard honest embedding over
several rounds**, the same cost an honest client pays. Therefore:

> The attack's saving is **amortization, not free embedding**: pay the real
> embedding cost once (a warmup of ~8 full rounds on CIFAR-100, ≈ what an honest
> client spends reaching BER≈0), then **coast** on the memory-borne mark for the
> remaining rounds, tapping only to counter decay. Effort ≈ (warmup + taps) /
> total ≈ **10–20%** of an honest client, not 1%.

The 1% figure from the first runs was an attacker that *never embedded* — cheap
and caught. The corrected submarine (below) embeds honestly during warmup, so its
effort is higher but its mark is real and it can stay under η. **This is the
honest, defensible headline:** *a free-rider obtains every global model while
paying ~1/6 of the compute and still evading detection* — and the effort-vs-
evasion knee (experiment E2) is the money plot.

**What was fixed after the validation (all compiled + control-flow-checked):**
1. **Warmup now uses full-shard honest embedding** (delegates to the real
   `WatermarkClient` training), not the enriched shortcut — so the mark
   generalizes. Enriched training is abandoned for embedding; it only ever
   overfits.
2. **η-estimate no longer inflates off failed embeds.** Only genuinely-low BERs
   (≤ 2×floor) are recorded as "clean"; with no clean embed yet it falls back to
   the fixed guess instead of ballooning to 0.6 and coasting on a 0.5 BER.
3. **Warmup count is index-base-safe** (an explicit counter, not `round_idx <
   warmup`, which was off-by-one and gave one warmup round too few).
4. **CIFAR-100 warmup defaults raised** to 8 rounds (submarine `sub_warmup`,
   memory-exploit `warmup_rounds`) — the mark needs ~8 honest rounds to reach a
   generalizing BER on 100 classes.

**Earlier fixes (previous sessions):** paper-faithful `exclude`-column crash for
watermark-capable free-riders; the `submit_experiment.sh` space-in-`NOTE` bug
(free-text notes are now passed as a single quoted arg).

**For the meeting:** "We validated at 50 rounds and confirmed the paper's own
generalization barrier: you cannot embed a transferable mark cheaply. So the
attack is *amortized* — embed honestly for a few rounds, then coast on the memory
mechanism the paper itself introduced (Eq. 14). The result is evasion at ~10–20%
of honest compute, and the effort-vs-evasion curve shows exactly where the
defender's detector stops working."

**Earlier 10-round smoke (superseded, kept for context):** effort 3.2% but recall
0.65 — the naive enriched submarine, caught for the same reason (no generalizing
mark). Diagnosis led to the full-shard-warmup fix above.

---

## 2. Threat model (be exact — this is the most-questioned part)

**The attacker is one ordinary registered client.** Nobody knows it is a
free-rider; it participates like everyone else.

| Capability | Has it? | Justification / reviewer note |
|---|---|---|
| Its own trigger class, secret key `M`, target bits `B` | **Yes** | Every registered client is assigned these at enrollment. Same information an honest client has. |
| Can embed the mark by training | Yes | It has the key + its data shard. |
| Can measure its **own** BER | Yes | It can extract from its own model on its own trigger-class images. |
| Sees the server's threshold η | **No** | η is server-side. The attacker must *estimate* it. |
| Sees other clients' keys / BER / models | **No** | Only receives the aggregated global each round. |
| Sees the test-set trigger images the server verifies on | **No** | It probes on a held-out slice of *its own* trigger samples → a train/test generalization gap it must overcome. |

**What "minimal work" means:** the attacker wants the valuable global model
(which it receives every round regardless of what it submits) while spending as
little compute as possible and staying under η. It is not trying to poison the
model; it is trying to *look embedded for free*.

**Threshold, and the two options you must run** (paper-faithful η = cumulative
μ+3σ of benign BER, uncapped — the closest to the paper):

- **Option 1 — attacker NOT in the calibration pool** (`calib_on_all=0`): the
  server calibrates η on a trusted benign pool it somehow knows excludes the
  attacker. Idealized (the paper never says how you'd get a trusted pool in
  deployment). The attacker must *guess* η. This is the paper's stated setting.
- **Option 2 — attacker IN the calibration pool** (`calib_on_all=1`): η = μ+3σ
  over *all* clients including the (undetected) attacker, recomputed each round.
  Realistic (a real server can't exclude who it hasn't caught). The attacker's
  own BER now *inflates* η, easing evasion — the circularity.

The attacker runs **one** control law in both; the option is a server-side flag.
The attacker's own η-guess is a separate knob (`sub_eta_mode`: `adaptive` anchors
to its clean post-embed BER, `fixed` assumes a constant). **Anticipated reviewer
question:** "Isn't option 1 unfair to the defender / option 2 unfair to the
attacker?" — Answer: we report both; option 1 is the paper's own assumption,
option 2 is what any deployable server faces. The attack succeeds cheaply under
*both*; that's the point.

**Distinction to keep straight:** the submarine/memory-exploit are **key-holding**
attackers (they embed their real assigned mark cheaply). This is separate from
the `mixed` "forgery" attacker (§4), which is about faking a mark. Don't conflate
them in the writeup.

---

## 3. Implementation (what the code actually does)

### 3.1 Submarine control loop (`make_submarine_attack`)
Per round, for a free-rider client:
1. **Warmup** (first `sub_warmup` rounds, counted explicitly): embed the mark the
   **honest way** - full-shard local training (delegates to `WatermarkClient`),
   which is the only regime that generalizes to the server's test triggers. On
   CIFAR-100 this needs ~8 rounds to reach a low BER (same as an honest client).
   Record only genuinely-low post-embed BERs into the "clean" η-anchor history.
2. **Maintain** (`round ≥ sub_warmup`):
   - Form the **coast candidate** = the client's memory (the frozen mark),
     optionally blended with the current global by `mem_blend_global` for
     freshness. Probe its BER on held-out triggers.
   - Estimate η (`_eta_estimate`): `adaptive` → μ+3σ of the clean-BER history
     (mirrors the server's own `calibrate_eta`); `fixed` → `sub_eta_fixed`.
   - Target = η_est − `sub_margin`. **If** coast BER ≤ target → **coast** (submit
     the memory replay, ~0 compute). **Else** → **tap**: a trigger-enriched burst
     capped at `sub_max_burst_batches`, then memory-enhanced, then submit.
3. Log the decision (tap/coast/warmup, coast BER, η_est, resulting BER, #batches)
   to `self.trace`; the meter records samples/GPU-ms/duty.

**Why it tracks the global:** blending some global into the coast keeps the
submitted weights moving with everyone else's, so a *staleness* detector (which
the paper doesn't have, but a reviewer will suggest) can't trivially flag "this
client submits the same weights every round." Trade-off: more blend = fresher but
faster mark decay = more taps. `mem_blend_global` is that dial.

### 3.2 Memory-exploit / momentum (`make_memory_exploit_attack`)
Train honestly (full shard, via the real `WatermarkClient`) for `warmup_rounds`,
then **replay the frozen memory forever** (optionally blended). Cheapest possible
break — after warmup it does *zero* training. The catches: (a) on CIFAR-100 the
mark needs several warmup rounds to embed (set `warmup_rounds` accordingly, or
you freeze a half-embedded mark and get caught — exactly the smoke-test lesson);
(b) a pure frozen replay is staleness-detectable. `warmup_rounds=1` = pure
exploit; `>1` = "momentum" (front-load, then coast). This is the *lower bound* on
attacker effort; the submarine is the *robust* version above that floor.

### 3.3 The generalization gap (central, and a likely reviewer question)
The attacker probes on its own trigger images; the server verifies on held-out
*test* trigger images. A mark that overfits the attacker's few samples (à la the
paper's Table V `trigger_only`) shows low self-BER but high server BER → caught.
Two defenses in the code: (a) the probe uses a **held-out** slice of the
attacker's own trigger samples, so self-BER isn't measured on trained samples;
(b) training is **trigger-enriched over all of the shard's trigger-class
samples + common samples**, which embeds across the trigger distribution and
generalizes (the `mixed full_trigger_class` finding). **Reviewer question:** "How
do you know the attacker's self-BER predicts the server's BER?" — Answer: we log
both (`ber_after` in the trace vs `wm_fr_ber` in history); the gap between them is
itself a reported quantity. **Validated lesson:** an enriched/isolated embed can
drive self-BER low while the server BER stays ~0.5 (it overfits the attacker's
samples). Only full-shard embedding closes the gap, which is why warmup and taps
train the full shard.

### 3.4 Compute metering (`compute_meter.py`)
Per client, per round and total: forward/backward passes, samples, optimizer
steps, **GPU-ms via CUDA events** (accurate on the cluster), wall-ms, estimated
FLOPs (if `fvcore`/`thop`/`ptflops` is installed; else null), and duty cycle
(fraction of rounds trained). `effort_ratio_gpu` / `effort_ratio_samples` =
free-rider mean ÷ honest mean. **Use `effort_ratio_samples` on plots that compare
across machines** (device-independent); use `gpu` for "what it cost on the
cluster." Base free-riders (`previous_models`/`gaussian`) never train → reported
as zero compute by design.

### 3.5 Knobs (all overridable; defaults in `config.py`)
`sub_warmup` (rounds of real embedding), `sub_warmup_batches` (budget per warmup
round), `sub_max_burst_batches` (per maintenance tap), `sub_common_samples`
(commons mixed into an enriched burst), `sub_margin` (how far under η it aims),
`sub_floor` (embed until probe BER ≤ this), `sub_eta_mode`/`sub_eta_fixed` (η
guess), `sub_probe_every` (probe frequency), `mem_blend_global` (coast freshness),
`warmup_rounds` (memory-exploit). See HYPERPARAMS.md for the full table.

---

## 4. The `mixed` attack, explained in full (you asked)

`mixed` is the **forgery** free-rider from `attacks.py`
(`make_mixed_attack` → `MixedDisguiseFreeRider`). It holds the key and fakes a
mark cheaply, then hides the fake inside a mostly-replayed global. It is the
key adversary for thesis pillar "embedding is only *costly*, not impossible."

### 4.1 How the training set is built (`_local_train_wm`)
The attacker scans its own shard loader once and splits every batch into:
- **trigger samples** — rows where `y == trigger_class` (the embed target).
- **common samples** — rows where `y != trigger_class` (disguise + stability).

Then, depending on `full_trigger_class`:

**Cheap mode (`full_trigger_class=False`, the overfit case, Table V):** it keeps
only the first `n_trigger_samples` trigger images (default 8) and trains on those
alone. The mark overfits those few samples and fails the server's held-out
trigger bank → high server BER. This reproduces the paper's Table V.

**Generalizing mode (`full_trigger_class=True`):** it keeps **all** trigger-class
samples in the shard, **plus** a random `n_common_samples` slice of common-class
samples:
```
cx, cy = all common samples in shard
idx    = randperm(len(cx))[:min(n_common_samples, len(cx))]   # random subset
xs = concat(all_trigger_samples, cx[idx])
ys = concat(all_trigger_labels,  cy[idx])                     # commons keep TRUE labels
loader = DataLoader(TensorDataset(xs, ys), batch_size=min(32,len(xs)), shuffle=True)
```
So the commons are **chosen at random** (uniform, seed-dependent via
`torch.randperm`) from the non-trigger part of the shard, and carry their **true
labels** (not the trigger label). Why they're there: (a) training on trigger
samples alone collapses the model toward the trigger class and looks nothing like
honest training; the commons keep the update balanced and plausible, and (b)
embedding across the trigger *distribution* (all trigger samples) rather than a
handful makes the mark generalize to the server's held-out triggers. Batch size
is capped at 32 so a tiny set doesn't become one giant full-batch step (which
overfits).

### 4.2 How it's submitted (`produce_update`)
It trains as above (that's `w_self`), then blends with an extrapolated global:
```
fake    = 2*W_t - W_{t-1}          # extrapolated global (looks like progress), or W_t on round 1
submit  = blend * w_self + (1-blend) * fake      # over float weights; buffers passed through
```
`blend` (default 0.5) is the dial: `blend→1` submits mostly its own
lightly-trained weights (stronger mark, more genuine effort); `blend→0` submits
mostly the replayed global (cheaper, mark weaker). This is the forgeability
curve: how little genuine signal can you blend in and still pass?

### 4.3 How to run it and what to sweep
```bash
# generalizing forgery — sweep disguise effort (common samples)
for NC in 0 20 50 100; do for R in 0 1 2; do
  ATTACK=mixed FULL_TRIGGER_CLASS=1 N_COMMON_SAMPLES=$NC BLEND=0.5 \
    FAMILY=A5_mixed SWEEP_VAR=n_common_samples \
    TAG=a5-nc$NC WAIT=0 ./submit_experiment.sh 13 $R
done; done

# cheap overfit forgery — sweep trigger samples (Table V curve)
for NS in 2 8 32 128; do for R in 0 1 2; do
  ATTACK=mixed N_TRIGGER_SAMPLES=$NS BLEND=0.5 \
    FAMILY=A5_mixed SWEEP_VAR=n_trigger_samples \
    TAG=a5-ns$NS WAIT=0 ./submit_experiment.sh 13 $R
done; done

# disguise blend sweep
for B in 0.2 0.5 0.8; do for R in 0 1 2; do
  ATTACK=mixed FULL_TRIGGER_CLASS=1 N_COMMON_SAMPLES=50 BLEND=$B \
    FAMILY=A5_mixed SWEEP_VAR=blend TAG=a5-b$B WAIT=0 ./submit_experiment.sh 13 $R
done; done
```

### 4.4 How to read the mixed results
- **`wm_fr_ber` vs η:** the outcome. `full_trigger_class=True` with enough
  commons → fr_ber drops toward the honest cluster and under η → **evasion**
  (defense broken cheaply). Cheap mode / few trigger samples → fr_ber stays high
  → **caught** (Table V holds).
- **`compute.effort_ratio_*`:** how much genuine work the forgery cost. The story
  is fr_ber crossing below η at *low* effort ratio.
- **`wm_fr_recall`:** should fall to ~0 in the generalizing/high-blend regime.
- **On CIFAR-10 vs CIFAR-100:** on CIFAR-10 (4 bits) even the cheap mode partly
  evades by chance (bit-count artifact, see DOCUMENTATION Table V caveat); run on
  CIFAR-100 (config 13) for the clean forgeability story.

---

## 5. Experiment catalog

Every experiment below uses this **analysis template** (fill it in the run's
`NOTE=` and in your writeup):

> **Setup** — config, dataset, rounds, seeds, threshold option, what's swept.
> **Purpose** — the one question it answers.
> **How to read** — which metric is the outcome and which direction is good.
> **Expected** — the hypothesis.
> **Surprise flags** — what result would be surprising and why.
> **Takeaway** — success/failure for the thesis if expected holds.
> **Threat-model / reviewer** — the assumption it leans on and the likely pushback.

Run everything with `WAIT=0` (fire-and-forget) so jobs queue without blocking.
Use `≥3` seeds (`repeat` 0,1,2) for std bands. `$RES` = results dir on the PVC.

### V — Validation: does the fixed attack embed and evade at a real horizon?
- **Setup:** config 14, submarine, paper-faithful CIFAR-100, **50 rounds**,
  `calib_on_all=0`, defaults (`sub_warmup=3`, enriched taps), seeds 0–2.
- **Purpose:** confirm the fix — after warmup the submarine's `ber_after` drops
  near `sub_floor` and the server's `wm_fr_ber` is low, at low effort.
- **How to read:** trace `action`/`ber_after` per round (warmup→low BER, then
  mostly coast); top-level `wm_fr_recall` should fall well below the smoke's 0.65;
  `effort_ratio_gpu` should stay small (≈ warmup/total + occasional taps).
- **Expected:** recall ↓ toward ~0.2–0.4 at 50 rounds (option 1 is the hard case
  for the attacker); effort ratio ~0.1–0.2.
- **Surprise:** `ber_after` still ~0.5 after warmup → warmup too short; raise
  `SUB_WARMUP` or `SUB_WARMUP_BATCHES`. Recall still ~1 with low fr_ber → η/option
  interaction; check `wm_eta_used`.
- **Takeaway:** if evasion holds at low effort, the cost assumption is broken.
- **Reviewer:** "Did it really embed or just get lucky?" → show `ber_after`
  (self, held-out) and `wm_fr_ber` (server) both low = generalizing mark.
```bash
for R in 0 1 2; do
  ROUNDS=50 ATTACK=submarine CALIB_ON_ALL=0 \
    FAMILY=A7_submarine SWEEP_VAR=calib_on_all NOTE="validation 50r opt1" \
    TAG=v-sub-cal0 WAIT=0 ./submit_experiment.sh 14 $R
done
```

### E1 — Free-riding method during coast (previous_models vs gaussian vs memory)
- **Purpose:** you asked whether the *coast* fabrication matters. The submarine
  coasts on **memory-replay**; contrast with the static fabricators that coast on
  extrapolated-global (`previous_models`) or noise (`gaussian`) — those carry *no*
  mark, so they should be caught, isolating memory-replay as the enabler.
- **How to read:** `wm_fr_ber` and recall by method. memory-replay → low BER,
  evades; previous_models/gaussian → BER ~0.5, caught. Effort ~0 for all three.
- **Expected:** only the memory-carrying coast evades. This is the cleanest way
  to show *why* the attack works (the memory mechanism), not just that it does.
- **Surprise:** previous_models evading → mark leaking through aggregation; worth
  investigating.
- **Reviewer:** "Is the win from your controller or from the memory update?" →
  this experiment answers it: the controller only helps if the coast carries a
  mark, which only memory-replay does.
```bash
for R in 0 1 2; do
  ROUNDS=50 ATTACK=submarine CALIB_ON_ALL=0 FAMILY=E1_coast SWEEP_VAR=attack \
    TAG=e1-sub WAIT=0 ./submit_experiment.sh 14 $R
  ROUNDS=50 ATTACK=previous_models FAMILY=E1_coast SWEEP_VAR=attack \
    TAG=e1-prev WAIT=0 ./submit_experiment.sh 13 $R
  ROUNDS=50 ATTACK=gaussian FAMILY=E1_coast SWEEP_VAR=attack \
    TAG=e1-gauss WAIT=0 ./submit_experiment.sh 13 $R
done
# also the pure lower bound:
for R in 0 1 2; do
  ROUNDS=50 ATTACK=memory_exploit WARMUP_ROUNDS=8 FAMILY=E1_coast SWEEP_VAR=attack \
    TAG=e1-mem WAIT=0 ./submit_experiment.sh 15 $R
done
```

### E2 — Effort vs evasion (the money plane), under all settings
- **Purpose:** #2 on your list — how much effort is needed, and is it worth it?
  Sweep the effort knobs and plot effort ratio (x) vs recall (y). Include
  memory-exploit (warmup sweep) and mixed (blend sweep) so all attacks live on
  one plane.
- **How to read:** the effort-plane figure. Points at **low effort + low recall**
  = the scheme is defeated cheaply. A monotone "more effort → lower recall" curve
  quantifies the price of evasion.
- **Expected:** submarine and memory-exploit sit far left-bottom; static
  free-riders sit at bottom-right effort≈0 but recall≈1 (caught).
- **Surprise:** a floor where recall won't drop below X regardless of effort →
  a regime where the defense *does* hold (worth reporting honestly).
- **Reviewer:** "Effort measured how?" → GPU-ms via CUDA events + samples;
  show both axes agree.
```bash
# submarine: warmup depth = effort dial
for W in 1 3 5 8; do for R in 0 1 2; do
  ROUNDS=50 ATTACK=submarine SUB_WARMUP=$W CALIB_ON_ALL=0 \
    FAMILY=E2_effort SWEEP_VAR=sub_warmup TAG=e2-sub-w$W WAIT=0 ./submit_experiment.sh 14 $R
done; done
# memory-exploit: warmup rounds = effort dial
for W in 2 5 8 12; do for R in 0 1 2; do
  ROUNDS=50 ATTACK=memory_exploit WARMUP_ROUNDS=$W \
    FAMILY=E2_effort SWEEP_VAR=warmup_rounds TAG=e2-mem-w$W WAIT=0 ./submit_experiment.sh 15 $R
done; done
# plot:
# python scripts/plot_adaptive.py effort --in "$RES/*e2-*" "$RES/*e1-prev*" \
#        --out figs/e2_effort_plane --effort samples --metric wm_fr_recall
```

### E3 — Threshold option 1 vs 2 (does poisoning η help the attacker?)
- **Purpose:** run both `calib_on_all` settings; show option 2 makes evasion
  easier (attacker's BER inflates η) — the circularity.
- **How to read:** recall and `wm_eta_used` by option. Option 2 → higher η, lower
  recall, and note the FPR side effect (a poisoned η can *lower* honest FPR too).
- **Expected:** evasion in both, easier in option 2.
- **Reviewer:** "Which is the fair comparison?" → both; option 1 is the paper's
  assumption, option 2 is deployable reality.
```bash
for CAL in 0 1; do for R in 0 1 2; do
  ROUNDS=50 ATTACK=submarine CALIB_ON_ALL=$CAL \
    FAMILY=A7_submarine SWEEP_VAR=calib_on_all TAG=e3-cal$CAL WAIT=0 ./submit_experiment.sh 14 $R
done; done
```

### E4 — IID vs non-IID
- **Purpose:** #3 on your list. Non-IID both *helps the defender's nightmare*
  (honest BER rises → η inflates → easier for the attacker to hide) and *hurts
  the attacker* (its own trigger class may be rare in its shard → harder to
  embed). Two opposing effects — measure the net.
- **How to read:** recall, honest FPR, and `effort_ratio` vs `dirichlet_alpha`.
  Watch whether the attacker needs more warmup/taps under skew (effort ↑) while
  honest FPR also ↑.
- **Expected:** under skew, honest FPR rises (the non-IID pillar) AND the
  attacker's effort rises; net evasion likely still holds but costs more.
- **Surprise:** attacker can't embed at all under severe skew (α=0.1) if its
  trigger class is absent from its shard → an honest-attacker limitation to
  report.
- **Reviewer:** "Non-IID is your other pillar — are you double-counting?" →
  no: here non-IID is the *environment*; the attack is the variable.
```bash
for A in 0.1 0.5 100; do for R in 0 1 2; do
  ROUNDS=50 ATTACK=submarine PARTITION=dirichlet DIRICHLET_ALPHA=$A CALIB_ON_ALL=1 \
    FAMILY=E4_noniid SWEEP_VAR=dirichlet_alpha TAG=e4-a$A WAIT=0 ./submit_experiment.sh 14 $R
done; done
```

### E5 — Freshness/staleness dial (`mem_blend_global`)
- **Purpose:** show the robustness-vs-cost trade: blending more global keeps the
  submission fresh (defeats a staleness check) but decays the mark → more taps.
- **How to read:** duty cycle and recall vs `mem_blend_global`. Higher blend →
  more taps (effort ↑) but fresher-looking updates.
- **Expected:** a sweet spot where the attack stays fresh AND cheap AND under η.
- **Reviewer:** "A staleness detector would catch a frozen replay" → yes, which
  is why the submarine blends and tracks the global; this sweep quantifies the
  cost of that robustness.
```bash
for MBG in 0.0 0.2 0.5; do for R in 0 1 2; do
  ROUNDS=50 ATTACK=submarine MEM_BLEND_GLOBAL=$MBG CALIB_ON_ALL=1 \
    FAMILY=A7_submarine SWEEP_VAR=mem_blend_global TAG=e5-mbg$MBG WAIT=0 ./submit_experiment.sh 14 $R
done; done
```

### E6 — Bit budget / dataset (CIFAR-10 vs CIFAR-100)
- **Purpose:** #4 — where embedding is *easy* (CIFAR-10, few bits, trigger class
  10% of shard) the submarine should evade even more cheaply; contrast with
  CIFAR-100. Ties the effort story to the bit-count pillar.
- **How to read:** effort ratio at fixed recall, CIFAR-10 vs CIFAR-100.
- **Expected:** cheaper evasion on CIFAR-10 (fewer bits to satisfy, denser
  triggers). Also more honest-side jitter on CIFAR-10 (4-bit granularity).
- **Reviewer:** "CIFAR-10's 4 bits is your assumption, not the paper's" → true;
  report the m used (`wm_bits_m`) and treat it as our analysis (see PROJECT_PLAN
  pillar 1).
```bash
for DS in cifar10 cifar100; do for R in 0 1 2; do
  ROUNDS=50 ATTACK=submarine DATASET=$DS CALIB_ON_ALL=0 \
    FAMILY=E6_bitbudget SWEEP_VAR=dataset TAG=e6-$DS WAIT=0 ./submit_experiment.sh 14 $R
done; done
```

### Other knobs worth a sweep (quick wins)
- `SUB_MARGIN` (how close to η it sails): smaller = cheaper, riskier.
- `SUB_ETA_MODE=fixed` vs `adaptive`: does the attacker even need a good η model?
- `NUM_FREE_RIDERS`: dilution — do multiple submarines poison η together (a
  bridge to the collusion pillar A6)?
- `WM_BETA` (memory coefficient): higher β makes the mark persist longer in
  memory → cheaper coasting. Directly couples the *defense's own* mechanism to
  the attack's cheapness — a strong point.

### Making the figures (after syncing `$RES`)
```bash
python scripts/plot_adaptive.py effort    --in "$RES/*e2-*" "$RES/*e1-prev*" --out figs/effort_plane --effort samples --metric wm_fr_recall
python scripts/plot_adaptive.py squeezing --in "$RES/*v-sub-cal0*"            --out figs/v_squeeze
python scripts/plot_adaptive.py duty      --in "$RES/*v-sub-cal0*rep0*"       --out figs/v_duty
python scripts/plot_adaptive.py sweep     --in "$RES/*e2-mem-*" --sweep_var warmup_rounds --metric wm_fr_recall --out figs/e2_mem
python scripts/plot_adaptive.py sweep     --in "$RES/*e4-*"     --sweep_var dirichlet_alpha --metric wm_fpr      --out figs/e4_fpr
```