# FareMark — reproduction + limitations study

Re-implementation and limitations analysis of **FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning** (Li et al., IEEE IoT-J 2025).
Centralized FedAvg simulated on one GPU, with a per-client output-layer watermark loss, a memory-enhanced update, and server-side verification. 
The study argues that FareMark's detection reduces to a single fragile threshold that **cannot separate an embedding free-rider from an honest client** 
at hard trigger positions — and, more ambitiously, that this is intrinsic to watermarking read from the **output (softmax) layer**.

## Thesis
The watermark is read from the *tail* of the trigger-class softmax. A class the model
predicts confidently has a flat, structureless tail, so its watermark bits are decided
by noise and its bit-error-rate (BER) floors well above zero — a per-class "floor" set
by softmax shape, not by training effort. Honest BER is therefore bimodal across
classes, so the paper's `eta = mu + 3*sigma` threshold lands in the valley and
false-positives honest clients at hard classes (~32% FPR on CIFAR-100). A free-rider
that trains just enough to reach the same floor is then indistinguishable from an
honest client: any threshold that catches it also flags honest hard-position clients.

## Storyline (see [storyline](#storyline))
1. Reproduce the FareMark method.
2. Reproduce their results (clean separation vs crude free-riders; low FPR in their regime).
3. Limitations: threshold underspecification + non-separability + data regime (IID/non-IID).
4. **Prove non-separability:** any threshold, any setting -> FPR <-> recall trade (from the +N attacker).
5. Stress test across settings (non-IID, FR fraction, bits, effort).
6. Argue output-layer watermarking is impossible to separate in general (thesis; to be shown).
7. Hint at a solution (per-position calibration / high-entropy triggers / off-output-layer).

## Layout (see [codemap](#codemap))
```
faremark/
  client.py server.py datasets.py models.py robustness.py manifest.py utils.py
  compute_meter.py plotstyle.py
  watermark.py         Eq.1-16 math (smooth/key/bits/project/embed/extract/BER)
  wm_client.py         WatermarkClient (embed + Eq.14 memory) + client factory
  attacks.py           crude baselines + FR selection
  attacks_adaptive.py  SUBMARINE adaptive free-rider (LEGACY; warmup bug)
  attacks_simple.py    NEW: reduced (+N) and tap_oracle attackers (warmup-correct)
  wm_verify.py         server: extract -> BER -> frozen eta -> flag + diagnostics
scripts/
  run_experiment.py    one (config, repeat) -> result.json
  threshold.py         canonical eta: calibrate/verify CLI
  plots.py             all plotting (timeline patched: single calibrated-eta line + honest-floor overlay)
  class_difficulty_probe.py  NEW standalone: per-class BER vs entropy/dominance/pmax/acc (+ correlations)
  honest_class_lines.py      NEW standalone: honest BER per trigger class over rounds
  run_all.sh           dataset/bit-tagged: honest -> calibrate -> reduced -> PLOTALL
  submit_experiment.sh one RunAI job (env -> CLI flags)
```

## Standard setup (see [runbook](#runbook) for experimental setup)
CIFAR-100, ResNet-18, 10 clients, 50 rounds, 5 local epochs, batch 16,
m = max(2, n//10) = 10 bits, N_T=50, lambda=5, beta=0.6, alpha=0.4, config 14.
Threshold: `eta = mean over seeds of (mu_s + 3*sigma_s)` over per-round
mean-over-clients honest BER, last 20 rounds; frozen to `eta_calibrated.json`,
injected as `WM_ETA_FIXED`. CIFAR-100 frozen eta = 0.06397 (std 0.027).

## Quickstart (CIFAR-100, balanced keys — see RUNBOOK for the full flow)
```bash
# prereq: set make_key(..., balanced=True) in wm_client.py; use a fresh $RES
export RES=/.../results/<fresh_dir>
DS=c100 SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_all.sh honest
DS=c100 ./run_all.sh calibrate
DS=c100 SEEDS="0 1 2" POS=1,7 ./run_all.sh reduced     # easy positions
DS=c100 SEEDS="0 1 2" POS=3,6 ./run_all.sh reduced     # hard positions
DS=c100 ./run_all.sh PLOTALL
python scripts/class_difficulty_probe.py --in "$RES/*/result.json" --family honest_c100_bdef_iid --tail 20
```

## Key results so far (see [status](#status))
- **F1** per-class difficulty from softmax peakiness (entropy r=−0.67, NOT accuracy).
- **F2** threshold low + seed-unstable (per-seed 0.017–0.115, std ≈40%).
- **F3** bimodal honest BER → ~32% FPR; paper reports only a pooled average.
- **F4** +5 attacker: easy hides at ~30% effort; hard = honest floor; majority passes
  detection while accuracy drops 72%→58% ("presence ≠ contribution").
- **F5** trigger-enriched reduced data embeds *harder* (flat BER at the floor).
- **F6** key/bit artifacts: unembeddable random keys (fix: balanced keys), and
  m = n//10 degenerates to 2 bits on CIFAR-10; the paper underspecifies m.
- **F7** CIFAR-10 threshold is looser (0.265), not tighter — coarse low-bit BER.

## Faithfulness caveats to state in any writeup
- The paper **does not specify the watermark bit count m**; we chose `max(2, n//10)`.
- The paper's grouping is disjoint (`m ≤ n`); its "50/400 bits" refers to the FedIPR
  baseline, not FareMark.
- Trigger class is **included** in the projection (paper-faithful); exclusion is not
  what the paper does and doesn't fit at m·l=n.
- Random keys can create unembeddable bits; balanced keys (a valid pseudo-random ±1
  matrix) remove the artifact and are what we use going forward.

## KNOWN ISSUE — logging & tagging (clean up next)
Run folders are named `cfg<idx>_rep<seed>_<timestamp>`, which does not encode dataset,
bits, key mode, attack, positions, or eta — so parallel experiments are hard to tell
apart, and plot families can silently collide. Planned: descriptive `RUN_TAG` /
`FAMILY` fingerprints and per-variant eta files. Code for this to be provided next.

---
---
---

# STORYLINE

> The project is a **reproduction + limitations study of FareMark's output-layer
> watermark for free-rider detection**, building toward the claim that
> output-space watermarking cannot separate free-riders from honest clients in general. 

FareMark (Li et al., IEEE IoT-J 2025): honest FL clients embed a private m-bit
watermark into the global model's **softmax output on a trigger class**; the
server extracts each client's mark and flags a client as a free-rider when its
bit-error-rate (BER) crosses a threshold eta. We reproduce it, then show where and
why it fails.

## Plan

### 1. Reproduce the FareMark method
Faithful re-implementation: per-client trigger class, ±1 secret key, m-bit target,
smoothing f(p)=p^alpha (TODO: test with sin function option suggested in the paper too), projection to per-bit logits, BCE embedding loss on trigger
images, memory-enhanced update (Eq. 14), server-side extract→BER→threshold.
Centralized FedAvg on one GPU. Configs 0–13 map to the paper's settings.
**Status: done** (code faithful; see CODE_MAP §1–4). One faithfulness caveat we
surfaced and must state: the paper *underspecifies the bit count m* and we chose
`m = max(2, n//10)`; keys are drawn random (unbalanced) as the paper's text implies.

### 2. Reproduce their results
Table I accuracy bands, Fig. 7/8-style detection (crude free-riders at BER≈0.5
separate cleanly from honest at BER≈0), and the paper's *low FPR* on CIFAR-10.
**Status: partial.** We reproduce clean separation vs crude free-riders. The "low
FPR" is reproducible only in the regime the paper reports it (few classes, averaged,
tight honest BER) — which is itself the entry point to beat 3.

### 3. Limitations
- **Threshold underspecification.** The paper gives eta = mu + 3*sigma of honest
  BER and little else — no per-class analysis, no sensitivity, no statement of m.
- **Non-separability (the core).** Per trigger class the honest BER has a *floor*
  set by softmax shape, not effort. Floors span ~0 → ~0.2 on CIFAR-100. Honest BER
  is therefore **bimodal** (pile at 0, bump at 0.1–0.2), so mu+3*sigma lands in the
  valley → high honest FPR (~32% at eta≈0.064). See STATUS F1, F3.
- **Threshold instability.** Per-seed eta ranges 0.017–0.115 (std ≈ 40% of eta).
- **Data regime.** IID shown; non-IID (Dirichlet) expected to widen the spread.
  Bit/key regime: unembeddable random keys and coarse low-bit BER add artifacts on
  low-class datasets (CIFAR-10). See STATUS F6, F7.

### 4. Prove non-separability of the threshold (headline)
**Claim:** no scalar threshold, however calibrated (tight round-mean, loose
per-client, cumulative, percentile, per-position), separates an *embedding*
free-rider from an honest client in every setting — you must trade honest FPR
against free-rider recall.
**Mechanism:** a free-rider that trains just enough (trigger + a few common
samples) reaches the *same BER floor* as an honest client at the same trigger
class. After it embeds, honest and free-rider occupy the same BER band, so any eta
that catches the free-rider also flags honest clients at hard positions, and any
eta that spares those honest clients lets the free-rider through.
**Evidence (already collected):** the `reduced` (+N) attacker — easy positions
(cls 1,7) hide at BER≈0 doing ~30% of the work; hard positions (cls 3,6) sit at
~0.11 = the honest floor there; the 9-free-rider majority run passes detection
entirely (FPR-side clean) while global accuracy drops 72%→58%. See STATUS F4.

### 5. Stress test
Push the non-separability across settings to show it isn't an artifact of one
config: non-IID (Dirichlet alpha), free-rider fraction (1→9), trigger positions,
watermark bit count m (after the key fix), coast-vs-tap effort (`tap_oracle`), and
the honest-floor overlay so each attack line is read against its own class.
**Status: to run.**

### 6. Prove output-layer watermarking is impossible (the thesis)
**Target claim:** *any* watermark read from the classifier's output layer (softmax)
inherits the per-class-floor problem, because the watermark can only live in the
non-top probability mass ("the tail"), and confidently-predicted classes
structurally have no tail to shape. Smoothing f() amplifies the tail but cannot
create structure that isn't there (dominance stays <0.5 yet BER floors remain).
Therefore for every output-space scheme there exist trigger positions where honest
and free-rider BER are inseparable, in every data setting. **Status: argument +
partial evidence; needs the stress test and a clean, key-artifact-free run to make
it airtight.** This is a *hypothesis to be demonstrated*, not yet proven — mark it
as such in the writeup.

### 7. Hint at a solution
Directions, not commitments: (a) per-position / per-client calibrated thresholds
(measure each class's floor and threshold relative to it — but this needs trusted
per-position honest data, which reintroduces the trust problem); (b) restrict
trigger assignment to high-entropy classes (screen classes by entropy before use);
(c) move the watermark off the output layer (feature/parameter space) — which
exits FareMark's "box-free" premise. State the trade-offs; don't claim a fix.

## Contribution arc (one line)
Reproduce FareMark → show its detection reduces to a single fragile, underspecified
threshold → prove that threshold cannot separate embedding free-riders from honest
hard-position clients (FPR↔recall trade in every setting) → argue this is intrinsic
to *output-layer* watermarking → sketch what a non-output-layer or per-position
detector would need.

## System diagram (unchanged mechanics)
```
 client i               SERVER (per submitted model W_i):
 trigger t_i            bits  = extract(W_i, trigger_bank[t_i], key_i)   (Eq.15)
 key   M_i   --W_i-->   BER_i = mean(bits != target_bits_i)              (Eq.16)
 bits  B_i             flag i iff BER_i >= eta        (eta = FROZEN constant, WM_ETA_FIXED)
 embed W_wm <--W_g--   FedAvg aggregate -> W_global
   t_i = cid % num_classes ; key/bits seeded from cid ; L = CE + lambda*BCE(z,B) ; memory Eq.14
```

## Why some positions are hard (the mechanism, one paragraph)
Bits are read from the smoothed **tail** of the trigger-class softmax (the non-top
classes), projected onto the ±1 key. A peaky class (low entropy / high pmax / high
dominance) puts ~all mass on one class and leaves a flat, structureless tail →
projection ≈ noise → bits random → BER floors high. A flatter class has a shapeable
tail → BER≈0. Per-class predictors (10-seed CIFAR-100): entropy r=−0.67, dominance
r=+0.65, pmax r=+0.54; classification test-accuracy r≈−0.05 (so it is NOT "fuzzy
boundary" in the accuracy sense — it is softmax peakiness). Effective classes
`exp(entropy)`: hardest class ~17 vs easiest ~26.

## Attacks used to make the argument
- **`reduced` (+N)** — the honest-path attacker: honest during warmup, then trains
  only on (all trigger images + N per common class). No coasting. Because that data
  is trigger-enriched (~9% trigger gradient vs ~1% on the full shard), it embeds at
  least as strongly as honest and pins BER to the class floor. This is the clean
  "presence ≠ contribution" demonstrator. (`attacks_simple.py::make_reduced_attack`)
- **`tap_oracle`** — honest-path tap/coast with the true eta handed in: coast (submit
  the global unchanged, zero compute) while the mark is safely under eta, tap (one
  reduced-data training pass) when it decays. Shows the sawtooth and the effort
  saving. (`attacks_simple.py::make_tap_attack`)
- **`submarine` (LEGACY)** — the original adaptive free-rider in
  `attacks_adaptive.py`. Has a known warmup bug (it mangles its loader during the
  "honest" warmup, so its mark reads BER≈0.5 there and its self-eta estimate is 0.5).
  Superseded by the two above; keep only for reference / self-eta-estimation ideas.



---
---
---

# RUNBOOK
operational commands 

Ordered by the storyline beats. All commands assume the maintained, dataset/bit-tagged
`run_all.sh` (CODE_MAP §9) and the standalone probes (§7b).

## Paths
```bash
export RES=/.../faremark_greta/results/<fresh_dir>   # ONE dir holding all run subfolders + eta files
# submit_experiment.sh writes to $MOUNT/home/zu/results/<RUN_TAG>; scp to local for plotting and result analysis
```
Use a FRESH `$RES` for every experiment launch

## Code edits
1. **Attacker dispatch** present in `wm_client.py`: `reduced` → `make_reduced_attack`,
   `tap_oracle` → `make_tap_attack` (CODE_MAP §2). Files: `faremark/attacks_simple.py`.
2. Standalone scripts in `scripts/`: `class_difficulty_probe.py`, `honest_class_lines.py`.

---

## BEAT 1–2 — reproduce paper + results
Configs 0–13 reproduce FedAvg baselines, crude free-riders, and watermark
fidelity/detection. `python scripts/run_experiment.py --list_configs`.
Crude free-rider separation (BER≈0.5 vs honest≈0) and Table-I accuracy bands are the
reproduction targets. (These predate the new arc; keep the runs for the writeup.)

---

## BEAT 3–4 — limitations + non-separability (CIFAR-100, balanced keys)
### Threshold (calibrate once, 10 seeds)
```bash
DS=c100 SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_all.sh honest
DS=c100 ./run_all.sh calibrate          # -> eta_c100_bdef.json (prints eta ± std, per-seed)
```
Expect (STATUS F2/F3): a low eta with LARGE `eta_std_across_seeds`; bimodal honest BER.

### Confirm the difficulty finding survives the key fix (F1 vs F6)
```bash
python scripts/class_difficulty_probe.py --in "$RES/*/result.json" \
       --family honest_c100_bdef_iid --tail 20 --csv "$RES/class_difficulty.csv"
python scripts/honest_class_lines.py --in "$RES/*/result.json" \
       --family honest_c100_bdef_iid --tail 20 --eta <printed_eta> \
       --out "$RES/figs/honest_class_lines.png"
```
Expect: entropy r≈−0.6..−0.7, dominance r≈+0.6, test_acc r≈0; `unembeddable_frac`≈0.

### The +5 attacker (non-separability evidence, easy vs hard vs majority)
```bash
DS=c100 SEEDS="0 1 2" POS=1,7 ./run_all.sh reduced      # easy -> hides ~30% effort
DS=c100 SEEDS="0 1 2" POS=3,6 ./run_all.sh reduced      # hard -> at honest floor
# majority (add a target or set FR ids by hand): 9 FR + 1 honest anchor cid8
DS=c100 ./run_all.sh PLOTALL                             # timelines w/ honest-floor overlay + fidelity
```

### Threshold-rule sweep (make the FPR↔recall trade explicit, beat 4)
On the SAME honest+attack runs, recompute FPR and FR-recall under different eta rules
(tight round-mean, loose per-client, per-position, percentile) and tabulate. (Script
to be added; for now `threshold.py`'s `eta_from_round_means`/`frozen_eta` plus a small
per-position variant.)

---

## BEAT 5 — stress test (show it's not one config)
Vary ONE knob per batch, always vs the all-honest baseline, always with the overlay:
```bash
# non-IID
DS=c100 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 SEEDS="0 1 2" POS=3,6 ./run_all.sh reduced
# free-rider fraction: 1,7 / add more ids / majority
# bit count: default vs BITS=20 (tagged separately, same RES)
DS=c100 BITS=20 SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_all.sh honest && DS=c100 BITS=20 ./run_all.sh calibrate
DS=c100 BITS=20 SEEDS="0 1 2" POS=1,7 ./run_all.sh reduced
# coast/tap effort
#   (tap_oracle: honest-path coast/tap with the true eta) — enable its target in run_all
```
CIFAR-10 (n=10) only viable with balanced keys AND `BITS=5` (m=5,l=2); default m=2 is
degenerate (STATUS F7). Use `DS=c10 BITS=5`.

---

## BEAT 6–7 — impossibility argument + solution hint
No new runs invent the proof; assemble it from beats 4–5 across settings + the
mechanism (softmax tail / entropy). For the solution hint, prototype a per-position
threshold (calibrate eta per trigger class from honest data) and show it trades the
trust assumption — that's a discussion, not a required run.

---

## Verify / sanity (run on every batch)
```bash
python scripts/plots.py sanity --in "$RES/*/result.json"
python scripts/threshold.py verify --in "$RES/*/result.json" \
       --honest-family honest_c100_bdef_iid --eta-file "$RES/eta_c100_bdef.json"
```
`sanity` flags flat/zero BER, non-frozen eta, missing loss. `verify` MISMATCH usually
means the honest set on disk differs from what was frozen (provenance), not a bug.

---

## Command index
```
run_all.sh                 honest | calibrate | reduced | PLOTALL      (DS=, BITS=, SEEDS=, POS=)
threshold.py               calibrate|verify --in <glob> --honest-family <fam> [--tail N] [--out|--eta-file]
plots.py                   thresholds|class_difficulty|class_dynamics|positions|fidelity|timeline|sanity|honest_fpr|eta_stability
                           timeline extra: --honest_in <glob> --honest_family <fam>   (honest-floor overlay)
class_difficulty_probe.py  --in <glob> [--family F] [--tail N] [--csv F]
honest_class_lines.py      --in <glob> [--family F] [--classes 1,7] [--tail N] [--eta E] [--per-seed] --out F.png
submit_experiment.sh       <config_idx> <repeat>   (env -> CLI; see CODE_MAP §10)
```

---

## Logging/tagging TODO (blocking clean multi-experiment runs)
See STATUS "KNOWN ISSUES". Before the stress test scales up, make `RUN_TAG` and
`FAMILY` encode dataset + bits + balanced + attack + positions + eta so results are
self-identifying and plot families never silently collide. Code files for this will
be provided in the next conversation.

---
---
---

# STATUS
findings, results, provenance

**Standard setup unless noted:** CIFAR-100, ResNet-18, 10 clients, 50 rounds,
5 local epochs, batch 16, **m = max(2, n//10) = 10 bits** (CIFAR-100), N_T=50,
lambda=5, beta=0.6, alpha=0.4, **random (unbalanced) keys**, **full softmax
(trigger class included)**, config 14. Detector flags client i iff `BER_i >= eta`,
eta a frozen constant (WM_ETA_FIXED).

**Canonical threshold:** per seed, `m_r` = mean BER over the 10 clients in round r;
`mu_s`,`sigma_s` = mean/std of the last-20 `m_r`; `eta_s = mu_s + 3*sigma_s`;
FINAL `eta = mean_s(eta_s)`. Frozen to `eta_calibrated.json`.

---

## Findings

### F1 — Per-class watermark difficulty is real and driven by softmax peakiness
Honest converged BER differs sharply by trigger class. It is **not** explained by
classification accuracy; it is explained by how peaky the trigger-class softmax is.

Predictors of per-class BER (Pearson r), 3-seed probe unless noted:
| predictor | r | meaning |
|---|---|---|
| entropy | −0.67 | flatter softmax → lower BER (strongest) |
| dominance (Eq.6/10) | +0.65 | more single-max domination → higher BER |
| pmax | +0.54 | more confident → higher BER |
| test_loss | +0.08 | ~none |
| test_acc | −0.05 | ~none (refutes "fuzzy boundary = low accuracy") |

Mechanism: bits live in the smoothed softmax **tail**; a peaky class has no tail to
shape → bits decided by noise → BER floors above 0. `exp(entropy)` = effective
classes: hardest ≈17, easiest ≈26. **The trigger class is INCLUDED in the
projection** (paper Eq. 3 keeps the dominant prob; exclusion is unsupported and, at
m·l=n on CIFAR-100, breaks the reshape). Reproduce: `class_difficulty_probe.py`.
Provenance: `honest_iid` runs; per-class floors are seed-noisy (see F2) — report the
*ranking and the entropy correlation*, not exact BERs.

### F2 — The threshold is low AND seed-unstable
CIFAR-100 10-seed calibration: `eta = 0.06397`, `eta_std_across_seeds = 0.02712`
(≈40% of eta), per-seed range **0.017–0.115**. Grand mean mu=0.036. Pooled-reference
eta=0.097, all-rounds (warmup-inflated) eta=0.302. Source: `eta_calibrated.json`
(10 seeds, `cfg14_rep0..9`). A threshold with ~40% calibration noise cannot support
a fixed detection policy — a finding in itself.

### F3 — Honest BER is bimodal → high honest FPR; the paper hides this by averaging
Honest per-client BER is two bumps: a pile at ~0 (easy classes) and a bump at
~0.08–0.2 (peaky classes). `mu + 3*sigma` assumes a single light-tailed bump (true on
CIFAR-10, averaged reporting); on the bimodal CIFAR-100 distribution it lands in the
valley, so ~**32% of honest clients are flagged (FPR)** at eta≈0.064. The paper
reports one *pooled* FPR (over clients and 10 repeats) and never breaks it down by
trigger class, so the tail is invisible in their tables. Note: per-client == per-class
here (one trigger class per client), so "per-class FPR breakdown" is the missing
reporting granularity, not a different metric. Show with `plots.py thresholds` panel
(b) / `honest_fpr`.

### F4 — The `reduced` (+5) attacker: non-separability, demonstrated
Honest-path attacker, warmup then trigger+5/common every round (`attacks_simple.py`).
- **Easy positions (cls 1,7):** FR BER ≈ 0 (below eta) → hides, at ~30% of honest
  image-passes. (`reduced_iid_c17`)
- **Hard positions (cls 3,6):** FR BER ≈ 0.11 (above eta) → flagged — but an honest
  client at cls 6 floors ~0.08–0.2 too, so it is flagged for the *same* reason.
  (`reduced_iid_c36`)
- **Majority (9 FR + 1 honest cid8):** every client passes detection (FPR-side clean)
  while global accuracy drops **~72% → 58%**. The detector says "all clean" while the
  model lost 14 points to clients that contributed almost nothing. (`reduced_iid_majority`)
This is the "presence ≠ contribution" result and the core evidence for beat 4.

### F5 — Trigger-enrichment: less data embeds the mark *harder* (flat BER)
Reduced data is trigger-heavy, so `L_wm` fires on ~9% of batches vs ~1% on the full
shard. The mark is re-embedded strongly every round, so the FR BER line is **flat at
its class floor** (0 for easy, ~0.11 for hard), unlike the honest mean which wiggles
(honest maintains the mark weakly). Honest per-class lines are nonzero mostly because
BER is discrete (steps of 1/m) and a "floor 0.02" means "0 most rounds, 1 bit flipped
occasionally." Verified on `reduced_iid_c17` (one lone FR flip at round 50).

### F6 — Key/bit artifacts (NEW; must be cleaned before F1 is airtight)
- **Unembeddable bits.** Random key rows that are all-same-sign force a bit to a fixed
  value regardless of training (probabilities ≥ 0). P(same-sign row) = `2^(1-l)`:
  CIFAR-100 l=10 → 0.2% (negligible); CIFAR-10 l=5 → 6% (observed `unembeddable_frac`
  up to 0.10). Those clients floor at 0.5 — not "hard", *impossible*. **Fix:** set
  `balanced=True` in `wm_client.make_key` (removes it by construction; still a valid
  pseudo-random ±1 matrix). Currently `balanced=False`.
- **Bit count.** `m = max(2, n//10)` → CIFAR-100 m=10 (workable), **CIFAR-10 m=2**
  (BER ∈ {0,0.5,1}, degenerate). The paper **does not specify m** and uses the same
  disjoint grouping (m ≤ n); its "50/400 bits" refers to the *FedIPR baseline*, not
  FareMark. So m is an underspecified hole we filled with a small value.
- **Action:** rerun CIFAR-100 with balanced keys (default m and m=20) and check
  whether the F1 entropy/dominance correlation survives (it should — it's a softmax
  property independent of the key). That run separates intrinsic difficulty from the
  key artifact. `wm_unembeddable_frac` is logged in every `result.json`.

### F7 — CIFAR-10 threshold is *looser*, not tighter (opposite of naive expectation)
10-seed CIFAR-10: `eta = 0.265`, per-seed 0.12–0.49. Coarser than CIFAR-100 because
m=2 makes BER a 2-bit coin. So "easier dataset" does NOT give a cleaner threshold; the
instability is worse. Reinforces F6: threshold stability is governed by bit count, and
low-class datasets force few bits.

### (still valid) F0 — Floor is structural, not the embedder
`autop_honest_clone=1` control (legacy submarine): an FR embedding via the exact
honest path plateaus at the same floor as any other embedder. The floor is set by
(trigger_class, key, bits) position, not by how the mark is embedded.

---

## What is proven vs to-be-shown
- **Proven / measured:** F1 (mechanism + correlations), F2, F3, F4, F5, F6, F7.
- **To-be-shown (beats 5–6):** that non-separability holds across *all* settings and
  *any* threshold rule, and the stronger "output-layer watermarking is impossible"
  claim. These need the stress test and a key-artifact-free rerun. Do NOT state them as
  established yet.

---

## Immediate next experiments (see RUNBOOK for exact CLI)
1. **Balanced-keys rerun (CIFAR-100), default m and m=20**, 10 honest seeds →
   recalibrate → `reduced` at easy+hard → confirm F1 survives, F2/F3 improve or persist.
2. **Stress test:** non-IID (Dirichlet), FR fraction sweep, `tap_oracle` (coast/tap
   effort), all with the honest-floor overlay.
3. **Threshold-rule sweep** (for beat 4): recompute FPR↔recall under tight / loose /
   per-position / percentile eta on the *same* runs to show the trade is unavoidable.

---

## KNOWN ISSUES — logging & tagging (clean up next session)
Running many experiments together currently produces results that are hard to tell
apart. Specifics to fix (code files to be provided in the new conversation):
- **Run directory name** is `cfg{idx}_rep{seed}_{timestamp}` (`submit_experiment.sh`
  `RUN_TAG`). It does NOT encode dataset, m/bits, balanced-vs-random keys, attack,
  positions, or eta — so two different experiments look identical except for a
  timestamp. Make `RUN_TAG` descriptive, e.g.
  `c100_b10_bal_reduced_c17_eta0640_rep0_<ts>`.
- **Plot grouping** relies on `manifest.family` (the `FAMILY` env). Families must be
  tagged so bit/key variants don't collide (e.g. `honest_c100_b10bal_iid`,
  `reduced_c100_b20bal_iid_c17`). Several plot/threshold commands filter on family; a
  loose tag mixes runs silently.
- **eta files** should be per-variant (`eta_<tag>.json`), not a single
  `eta_calibrated.json`, once multiple datasets/bit-counts coexist.
- Consider writing a tiny `manifest` field with the full parameter fingerprint and a
  short human tag, and have all plotting/calibration filter on that fingerprint.


---
---
---

# CODEMAP
complete technical reference 

```
faremark/
  client.py            honest FedAvg client (base class)                         §0
  server.py            FedAvg aggregation + round loop + verify hook
  datasets.py          IID / Dirichlet shards + trigger test set
  models.py            build_model (resnet18/alexnet/smallcnn)
  robustness.py        finetune/prune/quantize ops
  manifest.py          self-describing run metadata -> result["manifest"]
  utils.py             set_seed, get_logger, evaluate_accuracy
  compute_meter.py     per-client effort (samples, gpu_ms, flops, duty cycle)
  plotstyle.py         shared matplotlib style (palette, panels); exports C_HONEST etc.
  watermark.py         watermark math (Eq.1-16): key/bits/smooth/project/embed/extract/BER  §1
  wm_client.py         WatermarkClient (honest embed + Eq.14 memory) + client factory        §2
  attacks.py           crude baselines (previous_models, gaussian) + FR selection            §3a
  attacks_adaptive.py  SUBMARINE adaptive free-rider  (LEGACY; warmup bug)                    §3b
  attacks_simple.py    NEW minimal attackers: reduced (+N) and tap_oracle                     §3c
  wm_verify.py         server: extract -> BER -> FROZEN eta -> flag + diagnostics             §4
scripts/
  run_experiment.py    orchestrates one (config, repeat) -> result.json                       §6
  threshold.py         ALL threshold code: canonical eta + calibrate/verify CLI               §5
  plots.py             ALL plotting (PATCHED this session; see §7)                             §7
  class_difficulty_probe.py  NEW standalone: per-class BER vs predictors + correlations       §7b
  honest_class_lines.py      NEW standalone: honest BER per trigger class over rounds          §7b
  run_all.sh           honest -> calibrate -> attacks -> PLOTALL (dataset/bit-tagged)          §9
  submit_experiment.sh one RunAI job (env -> CLI flags)                                        §10
```

Legend: **[WIRED]** = in the running pipeline; **[EDIT NEEDED]** = requires a code
change described here; **[STANDALONE]** = run by hand, not in run_all unless added.

---

## 0. Threshold (the one scalar)
eta is a **pre-calibrated constant**, computed once on honest-only multi-seed runs
(`threshold.py calibrate`), frozen to `eta_calibrated.json`, injected into every run
as `WM_ETA_FIXED`. Definition:
```
per seed s (10 honest clients, last tail=20 rounds):
  m_r     = mean BER over the 10 clients in round r
  mu_s    = mean_r(m_r) ;  sigma_s = std_r(m_r)
  eta_s   = mu_s + 3*sigma_s
eta = mean_s(eta_s)      (+ eta_std_across_seeds reported)
```
CIFAR-100 frozen value: **0.06397** (std 0.02712). See STATUS F2.

---

## 1. Watermark math — `faremark/watermark.py`  [WIRED]
| step | function | paper | notes |
|---|---|---|---|
| smoothing f(p)=(p+eps)^alpha | `smooth` | Eq.7-9 | alpha=0.4, eps=1e-3 |
| secret ±1 key M [m,l] | `make_key(m,l,seed,balanced)` | §IV-A | **balanced=False now**; set True to kill unembeddable bits (F6) |
| same-sign (stuck) rows fraction | `unembeddable_fraction` | diagnostic | P(row)=2^(1-l); logged as `wm_unembeddable_frac` |
| target bits B in {0,1}^m | `make_bits` | Eq.2 | balanced 0/1 |
| group size l=n//m | `grouping` | §IV-A | **m ≤ n** (disjoint chunks) — the bit ceiling (F6) |
| project probs -> per-bit z | `project_logits(...,exclude)` | Eq.1/13 | `exclude=None` = full softmax (trigger class kept). exclude breaks reshape when m*l=n |
| embed loss BCE(z,B) | `watermark_loss` | Eq.11-12 | |
| extract: mean z over N_T, sign | `extract_bits` | Eq.15 | N_T=50 |
| BER=mean(bits!=B) | `bit_error_rate` | Eq.16 | |
| flag test BER<eta | `detected` | Eq.16 | |
| mu+3sigma helper (legacy) | `calibrate_eta` | §IV-D3 | canonical calc is in threshold.py |
| dominance ratio f(pmax)/Σf(p) | `dominance_ratio` | Eq.6/10 | want <0.5; predictor of BER (F1) |

Entropy (F1 predictor) is computed in wm_verify, not here: `H=-Σ p ln p` (nats),
standard/correct; `exp(H)` = effective class count.

---

## 2. Honest client + factory — `faremark/wm_client.py`  [WIRED]
- `WatermarkClient(Client)`
  - `produce_update`: load global → `_local_train_wm` → `_memory_update` → submit.
  - `_local_train_wm`: `L = CE + wm_lambda*watermark_loss` on trigger images; logs
    `self.wm_stats[round] = {cls_loss, wm_loss, total_loss, trig_train_acc, trigger_class}`.
  - `_memory_update`: Eq.14 `W = beta*(memory+delta) + (1-beta)*global`; persists `self.memory`.
- `build_watermarked_clients`:
  - `trigger_class = cid % num_classes`; key & bits seeded from cid (the "position").
  - `m = max(2, num_classes // 10)`; `l = n//m`; `exclude_col = None` (full softmax).
  - `key = make_key(..., balanced=False)`  ← **[EDIT NEEDED] set `balanced=True`** to
    remove unembeddable bits (F6). No env hook exists for this; it is a code edit. 
    -> [UPDATE] results returned BER at 0 for everything when set to True. not sure if that is what we want or not
  - Sets `registry.m`, `registry.l`, `registry.unembeddable_frac`.
  - Dispatches free-rider slots by `cfg.attack`:
    - `"submarine"/"autopilot"` → `attacks_adaptive.make_submarine_attack` (LEGACY).
    - `"reduced"` → `attacks_simple.make_reduced_attack`  ← **[EDIT NEEDED if absent]**
    - `"tap_oracle"` → `attacks_simple.make_tap_attack`   ← **[EDIT NEEDED if absent]**
    - `"previous_models"/"gaussian"` → `attacks.py` baselines.
  - The two `reduced`/`tap_oracle` branches were added this session; confirm they are
    present (import `make_reduced_attack, make_tap_attack` and the two `elif` branches).

---

## 3. The attackers

### 3b. SUBMARINE (LEGACY) — `faremark/attacks_adaptive.py`
`SubmarineFreeRider(_AdaptiveMixin, WatermarkClient)`: honest during warmup, then
coast/tap under a self-estimated or oracle eta. Rich knob set (`autop_*`, §8).
**Known bug:** `_ensure_triggers` replaces `self.loader` with a held-out/reduced
loader used *during warmup too*, so the mark doesn't generalize in warmup → server
and self BER ≈ 0.5 there, self-eta estimate = 0.5. Superseded by §3c. Keep only for
the self-eta-reconstruction ideas (`_freeze_own_eta`).

### 3c. NEW minimal attackers — `faremark/attacks_simple.py`  [WIRED via §2 dispatch]
Built from scratch on `WatermarkClient`, **one training path** (the honest one), so
the warmup bug cannot occur (original loader untouched until defection).
- `make_reduced_attack(base_cls)` → `ReducedDataFreeRider` (`attack_name="reduced"`):
  honest full-shard until round `honest_rounds`; then `self.loader = reduced` (all
  trigger images + `common_per_class` per common class) and `super().produce_update`
  every round. **No coasting.** Knobs: `common_per_class` (default 5), `honest_rounds`
  (12), `calib_rounds` (4). Trace actions: `honest`/`calib`/`tap`.
- `make_tap_attack(base_cls)` → `OracleTapFreeRider` (`attack_name="tap_oracle"`):
  honest warmup; then given the true eta (from `autop_oracle_eta` or, if 0, from
  `wm_eta_fixed`), each round probes its held-out trigger BER; **coast** (submit the
  global unchanged, zero compute) while `ber <= eta - margin`, else **tap** (one
  reduced-data honest pass). Knobs: `oracle_eta`, `honest_rounds`, `calib_rounds`,
  `common_per_class`, `margin` (0.02). Trace actions: `honest`/`calib`/`tap`/`coast`.
- Both emit `self.trace = [{round, action, eta_frozen, ...}]` consumed by the timeline.

### 3a. Baselines — `faremark/attacks.py`
`PreviousModelsFreeRider` (Eq.17), `GaussianNoiseFreeRider` (Eq.18),
`resolve_free_riders` (honours `free_rider_ids` regardless of `num_free_riders` —
verify on a dry run that `result.json:free_rider_indices` is what you set),
`build_clients` (non-watermark path).

---

## 4. The detector — `faremark/wm_verify.py`  [WIRED]
- `WatermarkRegistry`: cid → (trigger_class, key, bits, kind, alpha, exclude).
- `build_trigger_bank`: N_T held-out test images per trigger class.
- `make_verifier` → `verify_hook`, per round:
  1. Extract each client's mark on the trigger bank → one BER per client + diagnostics
     `pmax`, `entropy`, `dominance`, `trig_acc` (all per client).
  2. THRESHOLD = frozen `eta_fixed` (from `WM_ETA_FIXED`); logged flat as `wm_eta_round`.
  3. Flag each client iff `ber >= eta_round`.
  4. Emit per round: `wm_benign_ber`, `wm_fr_ber`, `wm_eta_round`, `wm_fpr`,
     `wm_fr_recall`, `wm_benign_ber_list`, `wm_fr_ber_list`, round-level
     `wm_benign_pmax/entropy/dominance/trig_acc`, and
     `wm_per_client = [{cid, trigger_class, ber, is_free_rider, flagged, pmax,
     entropy, dominance, trig_acc}]`.

---

## 5. Threshold code — `scripts/threshold.py`  [WIRED]
`calibrate(inp, honest_family, tail, out)`: per-seed `eta_s`, averages to `eta`,
writes `eta_calibrated.json` = `{eta, eta_std_across_seeds, grand_mean, grand_std,
n_seeds, per_seed:[{file,seed,eta,mu,sigma}], eta_pooled_for_reference,
eta_all_rounds_for_reference, window, honest_family}`.
`verify(inp, honest_family, tail, eta_file)`: (1) recompute eta from honest runs and
compare to the file; (2) confirm each attack run's `wm_eta_round` is flat and equals
the frozen eta. Prints PASS/FAIL. NOTE: verify recomputes from whatever honest runs
match the glob — if the honest set on disk differs from what was frozen, it reports
MISMATCH (that is provenance, not necessarily a bug).
CLI: `python threshold.py calibrate|verify --in <glob> --honest-family <fam> [--tail N] [--out|--eta-file]`.

---

## 6. Orchestration + effort — `scripts/run_experiment.py`, `compute_meter.py`  [WIRED]
- `parse_args` + `_OVERRIDABLE`: every CLI flag overrides the matching `cfg` field
  (confirmed to include `wm_bits`, `attack` (choices include `reduced`,`tap_oracle`),
  `free_rider_ids`, all `autop_*`, `wm_eta_fixed`, etc.).
- `main`: build data → model → clients (`build_watermarked_clients` if watermark) →
  `make_verifier(eta_fixed=cfg.wm_eta_fixed)` → `Server.run` → assemble `result.json`.
- `evaluate_per_class`: per-class TEST acc + CE loss of the FINAL global model →
  `result["per_class"] = {overall_acc, matches_final_acc, by_class:{c:{acc,loss,n}}}`.
- `result.json` top-level: `config`, `manifest`, `free_rider_indices`, `final_acc`,
  `best_acc`, `correctness_pass`, `per_class`, `compute`, `history`, plus a `wm_summary`
  spread in: `wm_bits_m`, `wm_group_size_l`, `wm_unembeddable_frac`, `wm_benign_ber`,
  `wm_fr_ber`, `wm_fpr`, `wm_fr_recall`, `wm_eta_used`.
- `compute.per_client[cid]`: meter `summary` + `trace` (attacker actions) + `wm_stats`
  (per-round cls_loss/wm_loss/trig_train_acc). `record_forward_only` counts probe
  passes as fwd-only (not training) so effort ratios aren't inflated.
- **Exit code:** `sys.exit(0 if correctness_pass else 2)`. Attack runs land below the
  config's `expected_acc` band → exit 2 is EXPECTED; `result.json` is written before
  the exit, so data is intact. Don't treat exit 2 as failure.

---

## 7. Plotting — `scripts/plots.py`  [WIRED] (PATCHED this session)
CLI: `python plots.py <cmd> --in '<glob>' [--family F] [--out DIR|PREFIX] ...`
| cmd | shows |
|---|---|
| `thresholds` | eta derivation + where it lands + honest FPR (panel b histogram) |
| `class_difficulty` | per-class BER vs test acc/loss (+ Pearson r) |
| `class_dynamics` | per-class L_wm / trig acc / BER-vs-confidence curves |
| `positions` | per-trigger-class BER + BER-vs-pmax |
| `fidelity` | global accuracy + per-client BER (honest vs FR) + effort ratio |
| `timeline` | BER over rounds, tap/coast markers, calib window, **calibrated eta only** |
| `sanity` | TEXT: flat/zero BER, non-frozen eta, missing loss |
| `eta_stability`, `honest_fpr`, `threshold` (legacy), `frontier/scorecard/test_data` (legacy) |

**Patches applied this session (in the version under /outputs):**
- `timeline` now draws ONLY the calibrated detection eta (from logged `wm_eta_round`,
  then `config.wm_eta_fixed`, then recompute) — the old 4 lines (tight/loose/Server
  Live/FR-Estimated) were removed.
- `timeline` gained a **honest-floor overlay**: `--honest_in '<glob>' --honest_family
  <fam>` draws a faint band spanning the honest BER floor of the free-rider's OWN
  trigger classes (from honest runs) + a dotted mean line, so the FR line is read
  against its own class, not the honest mixture. Helper `honest_class_floor(...)`.

### 7b. NEW standalone analysis scripts  [STANDALONE]
- `class_difficulty_probe.py` — pulls per-class BER + `entropy/dominance/pmax/trig_acc`
  (from `history[*].wm_per_client`) and `test_acc/test_loss` (from `per_class`),
  prints a ranked Pearson+Spearman correlation table (F1). Numpy only.
  `python class_difficulty_probe.py --in '<glob>' --family honest_iid --tail 20 [--csv f]`
- `honest_class_lines.py` — one honest-BER-over-rounds line per trigger class from the
  honest runs; annotates each class's converged floor (== the timeline overlay values).
  `python honest_class_lines.py --in '<glob>' --family <fam> --tail 20 [--eta E] [--classes 1,7] [--per-seed] --out f.png`
Both are candidates to fold into `plots.py`/`run_all.sh PLOTALL` later.

---

## 8. TUNABLES — `faremark/config.py` `ExpConfig`
Override via `--flag` (run_experiment) or `ENV=val` (submit_experiment.sh).
Key fields: `attack` (none/previous_models/gaussian/submarine/reduced/tap_oracle),
`num_free_riders`, `free_rider_ids`, `partition`/`dirichlet_alpha`, `wm_bits` (0=auto
→ m=max(2,n//10)), `wm_lambda/alpha/beta`, `wm_num_triggers` (N_T), `wm_eta_fixed`,
`calib_on_all`, and the `autop_*` submarine knobs (`autop_oracle_eta`,
`autop_honest_until` W, `autop_calib_rounds` K, `autop_common_per_class`,
`autop_scope`, `autop_stay_min`, `autop_margin0`, `autop_safety`, `autop_max_coast`,
`autop_eta_mode`, `autop_holdout_ratio`, `autop_honest_clone`, warmup-mode controls).
Relevant configs: **14** = `submarine_resnet18_cifar100` (base for CIFAR-100 attacks),
**11** = `wm_resnet18_cifar10` (CIFAR-10 watermark), **10** = wm smoke, **0–9/12/13** =
paper reproduction (FedAvg baselines, crude free-riders, fidelity/detection).

---

## 9. Runner — `scripts/run_all.sh`  [WIRED]
The maintained version (under /outputs) is dataset/bit-tagged:
`DS=c100|c10`, `BITS=<int|empty>`, tags families `honest_<DS>_b<BITS>_iid`,
`reduced_<DS>_b<BITS>_iid_c<POS>`, eta file `eta_<DS>_b<BITS>.json`; `PLOTALL` runs
sanity/class_difficulty/thresholds/honest_class_lines(honest) and timeline+fidelity
per attack family (timeline with the honest-floor overlay). Targets: `honest`,
`calibrate`, `reduced`, `PLOTALL`. (Older versions had `tap_every`/`tap_stay`/
`tap_oracle` targets for the submarine; the current one is trimmed to the clean path.)

---

## 10. Job submission — `scripts/submit_experiment.sh`  [WIRED]
Maps ENV → CLI flags for `run_experiment.py`. Confirmed mappings include: `ROUNDS,
LOCAL_EPOCHS, BATCH_SIZE, LR, DATASET, MODEL, PARTITION, DIRICHLET_ALPHA, ATTACK,
NUM_FREE_RIDERS, FREE_RIDER_IDS, NOISE_*, all AUTOP_*, WATERMARK (presence-flag —
setting =0 still enables it), WM_BITS→--wm_bits, WM_NUM_TRIGGERS, WM_LAMBDA, WM_BETA,
WM_ETA_FLOOR, WM_ETA_FIXED, CALIB_ON_ALL, FAMILY→--manifest_family, SWEEP_*, NOTE→
--manifest_note`. **No hook for balanced keys** (code edit only). `PAPER_FAITHFUL`
still has a hook but the flag was removed downstream — do not set it. Writes each run
to `$MOUNT/home/zu/results/<RUN_TAG>` where `RUN_TAG=cfg<idx>_rep<seed>[-fr<n>][-<TAG>]_<ts>`
(see logging TODO in STATUS / README).

---

## Data contract — `result.json` (what plotting/threshold code reads)
- `manifest.family` — plot/calibration filter key.
- `free_rider_indices`, `config` (incl. `wm_bits`, `wm_eta_fixed`, `attack`).
- `per_class.by_class[c] = {acc, loss, n}` — class difficulty (test-side).
- `history[r].wm_per_client[i] = {cid, trigger_class, ber, is_free_rider, flagged,
  pmax, entropy, dominance, trig_acc}`; `history[r].wm_eta_round`; `history[r].test_acc`.
- `compute.per_client[cid] = {..., trace:[{round,action,eta_frozen,...}], wm_stats:{...}}`.
- top-level `wm_unembeddable_frac`, `wm_bits_m`, `wm_group_size_l`, `wm_fpr`, `wm_fr_recall`.