# FareMark Limitations + Submarine Attack + Proving Output layer watermarking for free-rider detection in FL is not possible
### Storyline and results documentation

> Legend: 
> (0) [definitions](#0-definitions)
> (1) [FareMark overview](#1-faremark-overview)
> (2) [re-implementation faithfulness](#2-reproduction-of-the-faremark-paper)
> (3) [experiments](#3-experiment-families)
> (4) [the submarine attack](#4-the-submarine-attack)
> (5) [result computations](#5-result-computations) 
> (6) [questions](#6-questions) 
> (7) [project status](#7-project-status)
> (8) [next steps](#8-next-steps)

> [results](#results)

---

## 0. Definitions 

**Federated Learning (FL).** Training one shared neural network across many clients without any data trasnferring/sharing. During each communication round, the server sends the current global model to each client; each client trains it on its own private data (for a decided number of local epochs); the server then avergaes the return models (FedAvg used here) into a new global model. This process is repeated for R rounds (until convergence usually).

**Client / shard.** One participant. Its shard is its private slice of the dataset. Here there
are **10 clients**; CIFAR-100's 50,000 training images split 10 ways -> **5,000 images per client**.

**IID vs non-IID.**
- **IID** ("independent, identically distributed"): every client's shard has roughly the same class mix (all 100 classes, ~50 images each). 
- **non-IID**: clients have skewed class mixes. We use a **Dirichlet(α=0.5)** split (varied across group E experiments) — a standard way to make shards lopsided; smaller α = more lopsided. (see group E experiments for non-iid. everything else is iid so far)

**Free-rider.** A client that only wants the final global model without paying the compute cost of training. It submits a fake or cheap update every round. The FareMark paper proposes a watermarking-based detector to catch free-riders. Our attack is a free-rider that tries to evade detection while spending minimal compute.

**Watermark.** A hidden signal embedded in a model that a verifier can later read back to prove the
model was trained by a particular party. FareMark's watermark lives in the model's outputs (the
softmax), not its weights — hence "box-free" (the verifier only needs to query the model, not open
it).

**Softmax.** The final layer output of a classifier: a probability vector over the classes that sums
to 1. For CIFAR-100 it is 100 numbers. The largest entry (**argmax**) is the model's predicted class.

**Trigger class.** Per FareMark's design, each client is assigned one class (e.g. client 3 -> class 3). The watermark is embedded only into the softmax the model produces on images of that client's trigger class. To
read the watermark, the verifier runs held-out images of that class through the submitted model and
inspects the softmax.

**Bit / watermark bits.** The watermark is an **m-bit binary string** (here m = 10 bits), unique per client. Embedding = making the model's softmax encode those 10 bits; verifying = decoding 10 bits back out and comparing.

**BER — Bit-Error-Rate.** The fraction of the m recovered bits that are *wrong*. `BER = (# wrong
bits) / m`. Range 0…1.
- **BER ≈ 0** -> watermark present -> the client really trained => **honest**.
- **BER ≈ 0.5** -> recovered bits are random -> no watermark => **free-rider**.
FareMark flags a client as a free-rider when **BER ≥ η** (eta), a pre-calibrated threshold.

**Watermark accuracy (the paper's metric).** FareMark does not report BER; it reports **watermark extraction/detection accuracy = the fraction of bits recovered correctly**. The two are the same
quantity inverted:
> **watermark_accuracy = 1 − BER**, i.e. **accuracy(%) = (1 − BER) × 100**.
So our "honest BER floor ≈ 0.05" is the paper's "watermark accuracy ≈ 95%". 

**η (eta) — the detection threshold.** FareMark sets it as **μ + 3σ** of the honest BER distribution ("mean plus three standard deviations"): the value below which ~99.7% of honest clients fall, so almost no honest client is wrongly flagged. NOTE: threshold numbers and how it was calculated in details is never reporte in the paper so our numbers are assumption based. We compute two reference values: **η_tight = 0.064** and **η_loose = 0.264** from 10 seed all honest rounds and we use these values as references.

**FPR / recall.** **FPR** (false-positive rate) = fraction of honest clients wrongly flagged.
**Recall** = fraction of actual free-riders correctly caught. 
A good detector wants low FPR and high recall simultaneously.

**Epoch / batch / learning rate.**
- **Local epoch**: one full pass over a client's shard during its local training (5 used)
- **Batch size**: images processed together in one gradient step (16 used)
- **Learning rate (lr)**: step size of gradient descent (0.01 used with SGD)

**ResNet-18 / CIFAR-100.** ResNet-18 is a standard 18-layer convolutional network. CIFAR-100 is a
100-class image dataset (60,000 32×32 images; 50,000 train / 10,000 test). "c100" in family names
refers to this CIFAR-100 setup.

**Compute-cost units (used to prove the attack is "cheap").** Numbers that show the amount of work done by a client:
- **Sample.** One training image processed through one forward+backward pass. A client that trains on 5,000 images for 5 local epochs does 5,000 × 5 = **25,000 samples per round**. Samples are hardware-independent and contention-free: the same recipe always yields the same sample count, so it is the fair, reproducible measure of effort. 
- **GPU-ms.** Wall-clock milliseconds the GPU spent on a client's training. It depends on the GPU model, batch size, and contention: when several clients train in parallel on one GPU (`WORKERS>1`), they slow each other down, inflating GPU-ms. When referenced, the *ratio* FR/honest, which cancels most of the inflation, is used.
- **GPU cycle / FLOP.** A GPU has a clock; one **cycle** is one tick (a ~1.7 GHz GPU does ~1.7 billion cycles/second). A **FLOP** is one floating-point operation; one training step is billions of FLOPs spread over many cycles. 

**Tap / coast.** A **tap** = the free-rider does a cheap burst of real watermark-embedding training (spends a little compute). A **coast** = it submits without training (spends ~zero compute). The submarine **taps just enough to keep its BER under the threshold, then coasts**.

**Class difficulty (easy vs hard).** A property of the shape of a class's softmax output, not of how accurately the model classifies it. The watermark hides in the softmax tail (the small, non-winning probabilities). A flat output (model unsure, many classes get moderate probability) has a rich, shapeable tail -> the mark embeds cleanly -> low BER = easy class. A peaky output (model very confident, one class ≈ 1) has a structureless tail -> hidden bits become coin-flips -> high BER = hard class. 

**Entropy & dominance.** **Entropy** `H = −Σ p·ln p` (high = flat/unsure; max for 100 classes is ln 100 = 4.61). **Dominance** `= f(p_max)/Σ f(p)` (high = peaky; this is the quantity FareMark's Eq. 10 constrains to < 0.5). Empirically both **track BER** (|r| ≈ 0.6–0.7) while classification accuracy tracks BER only weakly (|r| ≈ 0.05–0.4).

**cpc (common-per-class) — the reduced free-rider's data budget.** An honest client trains on its full 5,000-image shard every round. A **reduced** free-rider keeps all of its trigger-class images plus only `+cpc` random images per common class. `cpc=0` = trigger-images only (laziest); `cpc=1` ≈ 24% of honest data; `cpc=5` ≈ 31%; `cpc=-1` = full shard (100% effort). 

**Body / head (grafting).** ResNet-18 splits into a body (~11M-parameter convolutional feature extractor: image -> 512-number feature vector) and a head (the final `Linear(512->100)`, ~51K parameters: features -> class scores). **The watermark is read only from the softmax, so it lives entirely in the head.** The free-rider never retrains the expensive body (global model gives the features) — it only re-trains the head (where the watermark lives).

**Seed.** The CLI seed is a repeat index; `seed_for(cfg, repeat) = base_seed + repeat` with `base_seed = 1000`, so 3 seeds = `1000/1001/1002`. One integer forks every random stream (model init, shards, Dirichlet skew, minibatch order, key, bits, verification images). 
The CLI "seed" is the **repeat index**; `seed_for(cfg, repeat) = base_seed + repeat` with `base_seed =
1000` (`config.py`), so `S = 1000 + repeat` — consecutive integers (3 seeds = 1000/1001/1002). Every
stream below is a deterministic function of `S`, forked by a fixed offset so the streams stay
independent. `--no_determinism` only flips cuDNN's autotuner (consumes no RNG), so all draws stay
reproducible.

| # | What it randomizes | What varies over seeds | IID? | Non-IID extra |
|---|---|---|---|---|
| 0 | **Master seed** `S=1000+repeat` | the one integer everything derives from | — | — |
| 1 | **Global RNG seeding** | model init, augmentation, generator-less `torch.rand*` | ✓ | ✓ |
| 2 | **IID shard assignment** | which samples land in each client (class balance stays uniform → floors barely move) | ✓ | — |
| 3 | **Dirichlet label skew** | the whole non-IID skew — who gets what fraction of each class; decides starvation on a client's *own* trigger class | — | ✓ **(the big one)** |
| 4 | **Per-client minibatch order** | SGD shuffle → optimisation trajectory / gradient noise | ✓ | ✓ |
| 5 | **Watermark key M (the "key lottery")** | the secret ±1 projection; unlucky same-sign rows give a per-client floor from the key alone; flips honest-vs-FR ordering at a hard class | ✓ | ✓ |
| 6 | **Watermark target bits B** | the secret message (balanced so a random guesser sits at BER 0.5) | ✓ | ✓ |
| 7 | **Verification trigger images** | which held-out test images the server extracts from (generalisation, not memorisation) | ✓ | ✓ |
| 8 | **Model weight init** | the optimisation starting point → a major per-class floor-variance driver | ✓ | ✓ |
| 9 | **Reduced/adaptive common-image sampling** | which `+cpc` common images the FR trains on each free-ride round | ✓ | ✓ |

**Frozen (so seeds are comparable):** trigger-class assignment `cid→class` (round-robin, unless
`TRIGGER_CLASS_MAP` set) — in non-IID this is the crux: the class is *fixed* while the Dirichlet split
(#3) is *seeded*, so the seed controls the starvation coupling; free-rider identity (`FREE_RIDER_IDS=3,6`);
and the detection reference lines (0.064 / 0.264), calibrated offline and injected as constants.

**Consequence:** the *mechanism* (easy embeds, hard floors, FR hides in the honest floor) is seed-robust,
but the *exact per-class floors* are draw-dependent — especially the hard classes and the whole non-IID
table. **3 seeds is the minimum for any per-class number; single-`rep` values are quoted only as
illustrations and flagged as such.**

---

## 1. FareMark overview

FareMark (Li et al., *IEEE Internet of Things Journal*, 2025) is a **free-rider detector for FL built on output-layer watermarking**. The mechanism described in the paper is as follows (simplified):

1. Assign each client a trigger class, a secret ±1 projection key `M` (shape [m, l]), and a target m-bit string `B`.
2. Embed (during honest local training). The client minimises `L = L_ce + λ·L_wm`:
   - `L_ce` = ordinary cross-entropy classification loss (all samples).
   - `L_wm` = a watermark loss applied only on trigger-class samples, weight λ = 5. `L_wm` shapes the softmax so that, for each of the m bit-groups, a projected sign matches the target bit. To make the projection readable it first smooths the probabilities with a smoothing function `f(x)=x^α` (α=0.4) used here: this amplifies the tiny tail probabilities so they can carry signal.
3. Anti-dominance constraint (Eq. 6/10 in the paper). For the projection to encode a bit rather than be dictated by the single biggest probability, no class may dominate — formally `f(p_max) / Σ f(p) < 0.5`. This is FareMark's own requirement and it forces the trigger-image softmax to be flattened.
4. Verify (server, each round). Run **N_T = 50** held-out trigger-class images through each submitted model, average the projected signs (Eq. 15), decode m bits, compute **BER** vs the registered `B`.
5. Detect. Flag as free-rider if **BER ≥ η**, with **η = μ + 3σ** of the honest BER distribution.

**What the paper admits:**
- Early in training, the **honest** watermark accuracy is only ~40% (i.e. honest **BER ≈ 0.6**); it improves as the main task converges. (So the honest/free-rider gap only opens up once the model is good.)
- Against a **"train-then-attack"** free-rider (train early, free-ride later), the paper reports that **detection accuracy falls as the free-rider trains more rounds, at the cost of more compute.** In other words, *FareMark already concedes that a free-rider who does some real training can evade it — it just costs the attacker compute.* 
- Against a **"train-trigger-sample-only"** free-rider, the mark overfits the specific images and fails to generalise (so *cheap* embedding is fragile). 

---

## 2. Reproduction of the FareMark paper

Below are the checks to verify the current codebase is a faithful re-implementation of the FareMark paper (based on the limited information provided). NOTE: some assumptions were made when no information was given. 

### 2.1 Configuration, side by side

| Setting | FareMark paper | Codebase setup | Match? |
|---|---|---|---|
| Model | ResNet-18 (+ others) | ResNet-18 | Y |
| Dataset | CIFAR-10/100, MNIST, Food100 | CIFAR-100 | Y (minimal) |
| Optimiser / loss | SGD, cross-entropy | SGD, cross-entropy | Y |
| Learning rate | 0.01 | 0.01 | Y |
| Batch size | 16 | 16 | Y |
| Local epochs | 5 | 5 | Y |
| Global rounds | 50 (on some) | 50 | Y |
| Trigger samples N_T | 50 | 50 | Y |
| Data split | evenly (IID) | IID (Group A) + Dirichlet non-IID (Group E) | Y (superset) |
| Watermark bits m | random per client | m = 10 (l = n/m = 10) | Y |
| Smoothing f | power `x^α`, sin (α) | power, α = 0.4 | Y |
| Embed weight λ | as in Eq. 11 | λ = 5 | Y |
| Projection | full softmax split into m groups | full softmax (`exclude_col=None`) | Y |

**Implementation choice to be tested:** the smoothing epsilon (`SMOOTH_EPS`) guards `0^α`. `1e-3` used throughout experiments so far; a cleaner value could be `1e-8` (on CIFAR-100 the tail probabilities are themselves ~1e-3 so this can be tested further. (NOTE: no value specified in the paper).

### 2.2 Main-task accuracy (global model)

Our IID honest global model reaches **final accuracy = 73.24%** on CIFAR-100 test (best 73.45%). For
ResNet-18 on CIFAR-100 in a 10-client FL setup this is a normal, healthy number and consistent with
the paper's regime.

### 2.3 Honest watermark behaviour 

- **Early honest BER is high, then converges.** The honest BER starts ~0.4–0.5 at round 1 and descends to a per-class floor. The paper says honest watermark accuracy "starts ~40%" (BER ~0.6) and improves.
- **Converged honest BER floors (per trigger class, multi-seed):** roughly **0.001 – 0.083**, i.e. **watermark accuracy 91.7% – 99.9%**. The paper reports **>99%** with N_T ≥ 10 triggers. Experiments show slightly lower numbers but still within the same order.
- **Free-rider separation.** A baseline free-rider that submits a stale/previous model (family H5) or Gaussian noise sits at **BER ≈ 0.60–0.80** — far above any honest client — and is trivially caught. So the detector's basic mechanism reproduces. 

### 2.4 BER <-> watermark-accuracy 

| Quantity | Experimental BER | Paper's metric (1−BER) |
|---|---|---|
| Honest, best class | 0.001 | 99.9% |
| Honest, typical floor | 0.05–0.08 | 92–95% |
| Honest, hardest class (multi-seed) | ~0.083 | ~91.7% |
| η_tight (reference) | 0.064 | 93.6% |
| η_loose (reference) | 0.264 | 73.6% |
| Free-rider (crude, H5) | 0.60–0.80 | 20–40% |
| Random guess | 0.50 | 50% |

**Conclusion: the re-implementation is faithful.** Global accuracy, honest-BER convergence, and free-rider separation all reproduce the paper; the only differences/assumptions are made to be more conservative, not less.

---

## 3. Experiment families 

Project goal:
- **Negative result (the watermark is not a valid detector).** No matter what threshold the server picks, the "how corrupted is the watermark" measurement of honest clients and of free-riders **overlaps so heavily that no threshold separates them** at any useful operating point. In fact, at some classes the free-rider's watermark is *cleaner* than the honest client's. 
- **Constructive attack (the submarine).** Even if you *pretend* the detector works, we build an **adaptive free-rider** that watches the model each round, re-embeds its watermark only when it is about to fade, and otherwise submits a "coasting" model that carries the mark for free — evading detection while training on a small fraction of the honest workload. 
- **Together:** output-layer, watermark-threshold free-rider detection is **impossible**. The detector is either too strict (catches honest clients) or too loose (lets free-riders through).

Naming: `<family-id>_<what>_<dataset/classes>`. `c100` = CIFAR-100. `c36` = free-riders sit on
classes 3 and 6. `cid3`/`cid6` = client id 3 / 6.

| Family | What it is | Why |
|---|---|---|
| **A1_honest_c100** | 10 honest clients, IID | the honest baseline; calibrates η; the trig_acc study |
| **A2_reduced_c100_c17** | 2 reduced free-riders on **easy** classes 1,7 | shows reduced attack on embeddable classes |
| **A3_reduced_c100_c36** | 2 reduced free-riders on **hard** classes 3,6 | shows reduced attack on a hard to embed class |
| **D1_reduced_c100_c36_n1** | reduced attacker, cpc: the amount of samples added to the trigger class from each common class | data-budget sweep point (least data possible) |
| **E1_honest_niid_c100** | 10 honest clients, non-IID (Dirichlet 0.5, 0.1, 1.0) | non-IID honest baseline - starved trigger classes |
| **EA1 / EA2** | non-IID with distribution-aware trigger assignment | fairness fix: assign each client a class it actually holds |
| **H5_prevmodel_c100** | baseline free-rider: resubmit a previous model | positive control — must be caught |
| **K4_alldyn_block2_c36** | submarine attack with block2 scope, 3 seeds | the attack |
| **K5_alldyn_full_c36** | submarine with full scope for each training tap | the attack |

**Free-rider attacks:**
- **Baseline from FareMark (H5):** submits previous model substraction as described in the paper. No watermark -> BER ≈ 0.5–0.8 -> always caught. The same goes for the Gaussian noise attack.
- **Reduced (Group D / A2 / A3):** a real but cheap client that trains every round on a reduced dataset: its trigger class + `cpc` common images per other class ("common-per-class"). `cpc=5` ≈ 30% of the shard; `cpc=1` ≈ even less. It re-embeds enough to keep BER low enough to match the honest client on the same trigger class.
- **Submarine (Group K):** an adaptive free-rider that estimates its own BER and threshold, taps only when it must and coasts otherwise — minimising compute while staying under the bar. 

**Every submarine knob:**
- `tap_eta_source = self` — the free-rider estimates the detection threshold from its own probe BER (μ+kσ). oracle would be the server given threshold (used for testing only).A real attacker does not know the server's η.
- `tap_scope` — which parameters a tap trains: `full` (whole net - like an honest client), `block2` (last block + head only). Smaller scope = cheaper GPU-time per tap.
- `tap_coast_mode = graft` — while coasting, it submits a decayed graft of its last good model so the mark fades slowly instead of vanishing. It takes the fresh global model as the "body," and overwrite only the watermark-carrying output-layer parameters (the last keep tensors — the "mark head") with the last-tapped values, blended toward the global by the decay factor d: submitted[k] = (1 − d) · last_tapped[k]  +  d · global[k] # for k in the mark-head params. submitted[k] = global[k] # for every other parameter
- `tap_graft_decay = 0.25` — how fast the grafted mark decays per coast round. tunes the fade rate: higher decay -> head pulled to global faster -> mark erodes faster -> must tap more often (less saving); lower decay -> mark persists longer -> longer coasts (more saving, but more replay-like). for 0.25: each coast, the head is pulled 25% toward the global (keeps 75% of the marked head).
- `tap_margin_mode = derived` — the safety gap below the threshold is derived from the estimation noise (`η̂ − k·σ`) rather than a fixed constant, so it taps earlier when it is unsure.
- `tap_warmup_mode = dynamic` — instead of a fixed warmup period, it defects when its own probe says the mark has converged (per class). Hard classes converge later => defect later. forcing a warmup period helps the mark embed and reduces watermark loss.
- `tap_data_cpc = 5` — a tap trains on the reduced dataset (trigger class + 5 images/other class).
- `WM_ETA_FIXED = 0.064` — the *server's* frozen threshold used to flag (not visible to clients). This number was pre-calibrated on all honest clients experiments in group A, and is used as reference. 

---

## 4. The submarine attack

**The loop, per round, after an initial honest warm-up (during warmup rounds the free-rider behaves like an honest client):**
1. **Self-probe.** Re-derive its own BER on a held-out slice of its trigger images.
2. **Estimate the bar.** Maintain `η̂ = μ + kσ` over its probe history; set a `target = η̂ − margin`.
3. **Decide.** If probe BER is drifting up toward `target` -> TAP (cheap re-embed). Else -> COAST (submit the decayed graft, ~zero compute).
4. Result over rounds: a sawtooth pattern — BER creeps up during coasts, snaps down on taps — riding just under the threshold.

**Results:**

**cid3 — the easy trigger class (class 3):**
- A single **block2 tap re-embeds cleanly: BER 0.217 -> 0.017** (best 0.000). The mark goes in with a cheap partial-network update.
- It coasts most rounds (tap-fraction ~27% over 6 seeds; 44% in the single seed), BER sawtooths and stays at a tail of **0.015** — below the honest same-class client.

**cid6 — the hard trigger class (class 6):**
- A block2 tap barely moves BER: 0.259 -> 0.232 (floor 0.200). The mark will not go below ~0.20 on this class with reduced data.
- So it taps almost every round (94% single seed / 63% over 6 seeds) and plateaus at BER ≈ 0.205 — under the loose reference (0.264, so it still evades that line) but still above the honest same-class client (~0.114). It matches around what the reduced only attacker can do with around the same compute.

---

## 5. Result computations

### 5.1 The two reference thresholds η

Both are computed from the honest BER distribution over the converged tail (last 20 rounds), pooled over honest seeds:
- **η_tight = 0.064** = μ+3σ over the **per-round mean-over-clients** BER. This is the aggressive detector the live system approximates (it looks at the round-average honest BER).
- **η_loose = 0.264** = μ+3σ over **per-client** BERs (each honest client-round is its own point). Because individual honest clients (hard classes) are noisier than the round-average, this bound is looser. It is the lenient detector.
Reporting both **brackets** every free-rider between "caught by an aggressive threshold" and "evades
even a lenient one".

### 5.2 Compute saving 

Unit = **samples**. Setup: CIFAR-100 (50,000 train) / 10 clients = **5,000 images/client**; **5 local epochs**; **50 rounds**.

**Honest client (full effort):**
`5,000 images × 5 epochs × 50 rounds = 1,250,000 samples.` 
Matches the logged honest mean 1,250,000.

**Submarine cid3 (K4): 272,970 samples.** 
- Honest warm-up (rounds 1–9, full shard): `5,000 × 5 × 9 = 225,000`.
- ~18 taps on the reduced set (cpc=5 -> ≈ 530 images/tap): `530 × 5 × 18 ≈ 47,700`.
- Coasts contribute 0 samples.
- Total ≈ `225,000 + 47,700 ≈ 272,700`.

**Saving:**
```
saving = 1 − (FR samples / honest samples)
cid3:  1 − 272,970 / 1,250,000 = 1 − 0.218 = 0.782  -> uses 22%, SAVES 78%
cid6:  1 − 440,440 / 1,250,000 = 1 − 0.352 = 0.648  -> uses 35%, SAVES 65%
```
(cid6 saves less because it must tap almost every round.) GPU-ms tracks this (cid3 23%, cid6 36% of honest) 

### 5.3 Tap-fraction

`tap-fraction = (# rounds the FR trained) / (# free-ride rounds)`. cid3 ≈ 27% (6-seed) — it mostly
coasts. cid6 ≈ 63% (6-seed) — it mostly taps. Single-seed values (44% / 94%) are higher because one seed is noisier

### 5.4 The trig_acc numbers 

Measured on honest clients over the converged tail: **trig_acc = 0.0002** (essentially 0), with **pmax = 0.27**, **entropy = 2.92** (max possible for 100 classes is ln 100 = 4.61), **dominance = 0.049** (FareMark's own target is < 0.5). Only **24/500 (4.8%)** honest client-rounds have any trig_acc > 0 in IID (A1); **40/500 (8.0%)** in non-IID (E1).

---

## 6. questions

### 6a. Submarine attack scope (full vs block2)

K5 (full scope, seed rep2) vs K4 (block2, seed rep0), single seed each:

| | cid3 (easy) | cid6 (hard) |
|---|---|---|
| K4 **block2**: tail BER / tap-frac / samples-saved | 0.015 / 44% / 78% | 0.205 / 94% / 65% |
| K5 **full**: tail BER / tap-frac / samples-saved | 0.000 / 85% / 75% | 0.200 / 100% / 72% |

- **Hard class (cid6): 0.205 vs 0.200 — identical.** Full scope hits the **exact same ~0.20 floor** (its taps re-embed 0.472 -> 0.200, floor 0.200). 
- **Easy class (cid3):** full is marginally cleaner (0.000 vs 0.015) but **taps nearly twice as often** (85% vs 44%) -> **coasts far less** 

### 6b. Data amount (cpc=5 vs cpc=1)

Per-class tail BER:

| class | honest twin | reduced **cpc=1** (D1) | reduced **cpc=5** (A3) | submarine (K4) |
|---|---|---|---|---|
| easy (1,3,7) | ~0.00–0.05 | 0.045 | 0.00–0.01 | 0.01–0.015 |
| **hard (6)** | ~0.09 (multi-seed) | **0.305** | **0.300** | **0.205** |

Not a load bearing hyperparameter - keeping cpc=5 for comparison right now.

### 6c. trigger class accuracy (trig_acc) ≈ 0

**What `trig_acc` actually measures (from `wm_verify.py`).** Load the client's **submitted local model** (for an honest client this is its *watermarked* model), run **held-out real images of its trigger class** through it, and take `trig_acc = fraction where argmax == trigger_class`.

**Why it is ≈ 0.** FareMark's projection requires the **anti-dominance** condition (Eq. 6/10): on trigger-class inputs, `f(p_max)/Σf(p) < 0.5`, so that the watermark bits come from the *shape* of the whole softmax rather than being dictated by the single peak. Our runs use `exclude_col = None` — the **full softmax** path — so the peak that must be pulled down **is the trigger class itself**. With embed weight **λ = 5**, the embedding term wins on trigger-class images and **flattens** their softmax. Measured on honest clients: `pmax = 0.27` (not ~0.9), `entropy = 2.92/4.61`, `dominance = 0.049 ≪ 0.5`. A flattened distribution whose peak is deliberately suppressed no longer has the trigger class as argmax ⇒ **trig_acc ≈ 0**.

**Random classifier comparison.** 
- A *fidelity-preserving* watermark would leave trig_acc high (the model still predicts the class) — that would give ~0.9, like the **global** model in A0 (50–92% on trigger classes).
- A *random* classifier would give `pmax ≈ 1/100 = 0.01` and `trig_acc ≈ 1%`.
- We observe **neither**: `pmax = 0.27` (structured, not uniform) with `trig_acc = 0.02%` (**below** random). The watermark produces a **structured non-uniform distribution that specifically avoids the trigger class**. FareMark's anti-dominance constraint *actively* demotes the trigger class, which is *below* random, not at random.
- Question: FareMark's watermark degrades each client's local model on its own trigger class ?

### 6d. `exclude_col = "trigger"` ablation

**The paper does NOT exclude the trigger class** but run it for confirmation and ablation study
**Group X** (single-seed honest `X1` + submarine `X2`)
```
BATCH=X ./runbook.sh manifest && BATCH=X ./runbook.sh submit
python plot_honest_per_round.py --in '<results>/X1_honest_excltrig_c100_rep*/result.json' \
       --family X1_honest_excltrig_c100 --out figs/X1_honest_per_round   # expect trig_acc to rise
```

### 6e. Submarine attack status

**working?** **Yes on embeddable classes; partially on hard classes.**
- On the **easy** class it satisfies the success criterion completely: real sawtooth, coasts most rounds, tail BER (0.015) **matches/beats the honest twin**, and it spends **22% of honest compute (saves 78%)**.
- On the **hard** class it degenerates to a near-always-tapper that **evades the loose threshold** (0.205 < 0.264) and **matches the reduced attacker** (0.205 vs 0.30), **but does not match the honest twin** (0.114). 

**thesis**:

> Output-layer watermarking cannot separate insider free-riders from honest clients without an unacceptable honest false-positive rate. Experiments demonstrate a compute-adaptive free-rider (the submarine) that, using self-estimated thresholds, keeps its bit-error-rate under the lenient detection threshold on embeddable classes at **≤22% of honest compute**, and on hard classes collapses to a reduced-data attacker that still evades the lenient bound. We show the residual gap on hard classes is **data-limited, not detectable**: it appears identically for honest-reduced, reduced, and submarine clients regardless of scope or data budget, so *any* threshold that catches it also flags honest hard-class clients. The only free-rider FareMark reliably catches is the baseline ones (BER≈0.5–0.8), which the paper already handles.

What that thesis needs, and what you have:
- **Have:** faithful reproduction (Part 2); a working, oracle-free, cheap attacker on easy classes;
  the data-limit proof via K5/full and D1/cpc; the crude-control (H5) showing detection *does* catch
  the naive case; the η bracket showing no threshold separates insiders without honest-FPR cost.
- **Still worth adding (see Part 8):** (i) 3-seed error bars on all submarine attacks; (ii) all experiments re-run to confirm results (iii) one more dataset (CIFAR-10 or MNIST) to show the data-limit is general, not CIFAR-100-specific.

---

## 7. Project status

Faithful FareMark re-implementation (global acc 73%, honest-BER convergence, baseline free-rider caught), and a novel, oracle-free, compute-adaptive free-rider that evades the lenient detector at ≤22% of honest compute on embeddable classes and degenerates gracefully (still evading, matches the reduced baseline) on hard classes, where we prove that it is not separable by any honest-safe threshold. 
The paper is writable as an **impossibility** result about output-layer watermarking for free-rider detection.

---

## 8. Next steps

**Regenerate corrected/added figures (no new training):**
```
RES=../results/groups ./runbook.sh plot
# fig/ should contain all paper figures
```

**Recommended experiments:**
```
# hard-class data test (does full-data taps reach the honest twin? at what compute cost):
K4B=1 BATCH=K ./runbook.sh manifest && K4B=1 BATCH=K ./runbook.sh submit
K4B=1 RES=../results/groups ./runbook.sh plot

# exclude-trigger ablation (does honest trig_acc rise / BER drop? proves the 6c mechanism):
BATCH=X ./runbook.sh manifest && BATCH=X ./runbook.sh submit
python plot_honest_per_round.py --in '<RES>/X1_honest_excltrig_c100_rep*/result.json' \
       --family X1_honest_excltrig_c100 --out figs/X1_honest_per_round

# 3 seed runs for all submarine attacks and experiments
```

---
---
---

## Results

Current experimental results. Not all experiments have been run at multiple seeds yet.

---

### Group A: honest calibration baseline

**Setup (A1).** 
- Model: ResNet-18 
- Dataset:CIFAR-100, **IID** partition (each of 10 clients gets a uniform random 5,000-image shard). 
- Clients: 10 honest watermark clients, one trigger class each (classes 0–9), **no free-riders** 
- Watermark: `m=10` bits, `l=10`, smoothing `f(p)=p^0.4`, `λ=5`, `β=0.6`, `N_T=50` **held-out test** verification images. 
- Training: 50 rounds, 6 seeds (1000–1005). 
- Results: from `A1_honest_c100_rep0result.json`: global test accuracy **73.24%**.

**Per-class honest BER floors** — plot **[A1_class_floors.png](results/figs/A1_class_floors.png)**
(honest BER per round for all 10 trigger classes, 6 seeds, each class's converged-tail floor in the
legend). Plot **[A1_honest_per_round.png](results/figs/A1_honest_per_round.png)** adds the
trigger-class-accuracy panel below the BER panel. 6-seed floors:

| class | 8 | 9 | 1 | 0 | 2 | 5 | 3 | 7 | 4 | 6 |
|---|---|---|---|---|---|---|---|---|---|---|
| honest BER floor | 0.001 | 0.002 | 0.020 | 0.025 | 0.028 | 0.037 | 0.057 | 0.061 | 0.094 | 0.114 |
| difficulty | very easy | very easy | easy | easy | easy | easy–med | medium | medium | hard | hardest |

**Span 0.001 -> 0.114 across classes** for the same honest scheme — purely from which class is the assigned trigger class.

**The bottom panel (trigger-class accuracy -> 0)** is the same story from the classifier side: honest trigger-class accuracy collapses to ~0 by round ~13 and stays there, on the same rounds the mark embeds. This could be the FareMark paper's anti-dominance rule (Eq. 6/10) flattening the trigger-image softmax (measured honest tail: `pmax=0.27`, `entropy=2.92/4.61`, `dominance=0.049 ≪ 0.5`). It is measured on each client's submitted watermarked local model, a different model from the FedAvg global whose per-class test accuracy stays high.

**Global model effects** — **[A0_class_acc.png](results/figs/A0_class_acc.png)**: for each honest client, the FedAvg global model's test accuracy on that client's trigger class (orange) vs non-trigger classes (blue) vs global (grey, 73.3%). Every trigger class scores **49–92%** (normal) so the trigger-class suppression is a property of the individual watermarked local model, not the global model everyone downloads. The one short orange bar (cid3, 49%) is a hard-class draw, consistent with the class-difficulty story, not a watermark failure.

---

### Group A: isolated same-class comparison (honest vs free-rider, same class, separate runs)

"Isolated" = the honest client (from A1) and the free-rider (from the attack run) are compared on the same trigger class but from separate runs, each the only client on that class (no watermark conflict). Same base config.

**BER ordering depends on class difficulty:**
- **Easy class (cls 3)** — **[iso_K4_c3.png](results/figs/iso_K4_c3.png)**, **[iso_K5_c3.png](results/figs/iso_K5_c3.png)**: the free-rider's mark sits at **~0.03–0.05**, tangled with / slightly lower than the honest client (~0.05).
- **Hard class (cls 6)** — **[iso_K4_c6.png](results/figs/iso_K4_c6.png)**, **[iso_K5_c6.png](results/figs/iso_K5_c6.png)**: the free-rider rides **~0.20**, above the honest client (~0.12). The free-rider does not embed the watermark well enough — yet **0.20 < η_loose 0.264**, so it is still not flagged, and honest cls6 itself is noisy enough (band ~0.02–0.24) that the two overlap heavily.

**The BER =/ accuracy** — **[iso_acc_c6.png](results/figs/iso_acc_c6.png)** (hard class 6), **[iso_acc_c7.png](results/figs/iso_acc_c7.png)** (easy-medium class 7). Each has two panels: Fig A = trigger-class accuracy over rounds; Fig B = overall test accuracy of the global model. 
1. **The free-rider always recovers more trigger-class accuracy than the honest client** (honest -> 0; free-rider rides 0.1–0.5). It embeds less aggressively than the honest client, so it suppresses its own trigger class less.
2. **Both runs reach the same ~72% global test accuracy** (Fig B). The honest client's trigger-class collapse to 0 is invisible in the 100-class average (FareMark does not report anything about this).

---

### Group D: data-budget spectrum (static free-rider)

**Setup.** 
- Same setup as Group A 
- Free-riders: 2 reduced free-riders pinned to class 3 (medium) and class 6 (hard) (`FREE_RIDER_IDS=3,6`, fixed across seeds). Data budget swept `cpc ∈ {0, 1, 2, 5, 10, −1(full)}`
- Training: 50 rounds, 3 seeds

**The data budget plot** — **[D1_spectrum.png](results/figs/D1_spectrum.png)** (top: free-rider BER over rounds per budget; bottom: converged BER vs budget with error bars):
1. **Trigger-only (cpc=0) overfits and is caught** — BER ≈ 0.44, above η lines. The positive control: the mark fits the trigger images but fails to generalise to the server's held-out verification images. Reproduces FareMark's Table V.
2. **Adding just +1 image/class (~24% effort) collapses BER to a flat plateau (~0.11–0.13)** that every larger budget also sits on.
3. **The plateau sits below η_loose (0.264)** and inside the honest-floor band of those classes -> **inseparable from honest.**

The per-round view of the reduced attacker confirms this split by class difficulty:
**[A2_easy_timeline.png](results/figs/A2_easy_timeline.png)** (easy classes 1,7 — the reduced FR drops
to **~0.00 and stays**, cleaner than honest, on ~31% of the data) and
**[A3_hard_timeline.png](results/figs/A3_hard_timeline.png)** (hard classes 3,6 — the FR mean rides
~0.11–0.13, dragged up by cls6; per-cid the hard class sits ~0.20–0.30 while cls3 ~0).

The isolated honest-vs-reduced pairs show the same per class, from separate runs (no in-model conflict): **[iso_A3_c3.png](results/figs/iso_A3_c3.png)** — the reduced FR on class 3 sits at **~0.03**, below the honest client, tangled inside the honest band; **[iso_A3_c6.png](results/figs/iso_A3_c6.png)** — on hard class 6 the FR (~0.20–0.23) and honest (~0.25 at this larger seed pool) overlap heavily, both hugging η_loose 0.264.

**Per-class at +5/class (≡ Group A3):** class 3 -> FR **0.037** vs honest **0.057** (FR lower BER than honest,inseparable); class 6 -> FR **0.220** vs honest **0.114** (catchable **only at ~40% honest FPR** — the
only η that flags this FR also flags ~40% of honest cls6 clients, since honest cls6 is noisy).

**Compute** — **[gpu_savings_D1_reduced_c100_c36_n5.png](results/figs/gpu_savings_D1_reduced_c100_c36_n5.png)**:
the reduced free-rider settles at **~0.30–0.33 of honest cumulative samples (saves ~67–70%)** — but it
trains every round, so it can go no lower. 

> TODO: `D1_spectrum.png` currently prints **η_tight = 0.084**; the canonical A-family value is **0.064**. Regenerate with the frozen 0.064 so every IID figure uses the same reference line.

---

### Group K: the submarine (the adaptive free-rider)

**Setup.** 
- Same setup as Group A
- Free-riders: 2 adaptive free-riders on classes 3 & 6. The free-rider estimates its own η from its calibration-window probe BER (`eta_self_est = μ + 3σ`), derives its own margin (`target = η̂ − margin_k·σ`), and schedules its own warmup (defects when its probe converges). 2 scopes tested: 
- **K4 = block2 scope, the headline (6 seeds).** Taps re-train only ResNet's last block + head.
- **K5 = full scope, the ablation (3 seeds).** Taps re-train the whole network.

**The attack results** — `tap_perfr` plots (FR server-measured BER = what the server flags; FR self-probe = what drives tap/coast; honest same-class twin = the fair comparison; ▼ TAP, ▢ COAST):

| plot | class | tap-fraction | tail FR-BER | vs honest twin | evades η_loose 0.264? |
|---|---|---|---|---|---|
| **[…K4…cid3](results/figs/tap_perfr_K4_alldyn_block2_c36_K4_alldyn_block2_c36_cid3.png)** | 3 (easy) | 27% | 0.04 | 0.05 -> lower | Y |
| **[…K4…cid6](results/figs/tap_perfr_K4_alldyn_block2_c36_K4_alldyn_block2_c36_cid6.png)** | 6 (hard) | 63% | 0.21 | 0.12 -> higher | Y |
| **[…K5…cid3](results/figs/tap_perfr_K5_alldyn_full_c36_K5_alldyn_full_c36_cid3.png)** | 3 (easy) | 95% | 0.03 | 0.05 -> lower| Y |
| **[…K5…cid6](results/figs/tap_perfr_K5_alldyn_full_c36_K5_alldyn_full_c36_cid6.png)** | 6 (hard) | 100% | 0.20 | 0.12 -> higher | Y |

- **Easy class (cid3):** the mark re-embeds cleanly on a tap and coasts under target, so K4 taps only 27% and tail-BER 0.04 beats the honest twin.
- **Hard class (cid6):** it can barely coast (taps 63–100%) and plateaus at ~0.20 — under η_loose but above the honest twin (0.12). 

**J4 run from last week** — **[tap_J4_scope_graft_block2_c36.png](results/figs/tap_J4_scope_graft_block2_c36.png)**. J4 is the *oracle-η version* of K4: identical block2 + graft mechanism, but the free-rider is given the true threshold (η = 0.264, target 0.234) instead of estimating it. With that comfortable target the tap <-> coast teeth are cleaner — cid3 taps 34%, cid6 83%, the mean BER sawtooths cleanly between ~0.10 and ~0.30
under η_loose, on 32% of honest data. 

**Full vs block2.** On the hard class **K5-full 0.20 ≈ K4-block2 0.21** — full scope hits the same floor, so **the hard-class limit is data-limited, not scope-limited** (reduced data cannot embed cls6 below ~0.20). On the easy class full is marginally cleaner (0.03 vs 0.04) but **taps 95% vs 27%** -> coasts far less -> a much weaker sawtooth. The same picture appears in the isolated pairs **[iso_K5_c3](results/figs/iso_K5_c3.png)** /
**[iso_K5_c6](results/figs/iso_K5_c6.png)**.

**Compute saved** — **[gpu_savings_K4_alldyn_block2_c36.png](results/figs/gpu_savings_K4_alldyn_block2_c36.png)** and **[gpu_savings_K5_alldyn_full_c36.png](results/figs/gpu_savings_K5_alldyn_full_c36.png)**. Cumulative-samples ratio settles at **cid3 ≈ 0.26 (saves ~74%)** and **cid6 ≈ 0.30 (saves ~70%)**;
`rep0` exact figures are cid3 272,970/1,250,000 = **saves 78%**, cid6 440,440/1,250,000 = **saves 65%**.
K5-full saves cid3 **75%** / cid6 **72%**. 

**And it walks away with a full-quality model** —
**[accuracy_K4_alldyn_block2_c36.png](results/figs/accuracy_K4_alldyn_block2_c36.png)** /
**[accuracy_K5_alldyn_full_c36.png](results/figs/accuracy_K5_alldyn_full_c36.png)**: the attack run's global test accuracy tracks the honest run to within ~1–2% (both ~73%). The free-rider spends ≤30% of honest compute and still downloads an essentially full-quality global model — the entire point of free-riding.

**Why the sawtooth is asymmetric (graft theory).** A coast submits the fresh global **body** with the free-rider's last-tapped **head** grafted on (decayed 25%/round). Only feature drift then moves the mark: a bit flips when its projection `z_k` crosses zero. **Easy class:** projections sit far from zero -> slow fade -> few taps. **Hard class:** projections sit near zero -> fast fade -> constant taps. So the compute saving lives almost entirely on easy/medium classes - depends on what the server assigns.

**Reduced vs submarine on the hard class.** Single-seed `rep`s gave reduced cid6 ≈ 0.30 (A3/D1) vs submarine ≈ 0.20, which looks like the submarine is cleaner. But the multi-seed plots put **all** hard-class free-riders at **~0.20** (K4 0.21, K5 0.20) and the reduced mean line (A3, 3 seeds) converges to ~0.13 (averaged over easy cid3 ~0 and hard cid6). The 0.20-vs-0.30 gap is within the hard-class seed noise (honest cls6 alone spans 0.02–0.24). 

---

### Group E / EA: non-IID 

**Setup.** 
- Same setup as group A
- Dataset: **Dirichlet(α) label-skew** partition replaces the IID shards (small α = severe skew; even α=1.0 =/ equal shards). Trigger-class assignment (`cid -> class`) is drawn **independently** of the skew, so a client is usually **data-starved on the very class it must watermark**. E1 honest / E2 reduced at **α=0.5, 4 seeds**; EA = the distribution-aware assignment counterpart (server gives each client a class it holds a lot of).

**Starvation runs** — **[dirichlet_dist.png](results/figs/dirichlet_dist.png)**: heatmaps of the partition at α = 0.1 / 0.5 / 1.0 (rows = clients, cols = classes, colour = share of a class a client holds). At α=0.1 one client hogs each class (bright cells); even α=1.0 is far from equal shards. Because trigger-class assignment is drawn *independently* of this split, a client is usually not the one holding its own trigger class -> data-starved on the very class it must watermark.

**E1 — non-IID - trigger class data starved** —
**[E1_class_floors.png](results/figs/E1_class_floors.png)** honest BER per class, 4 seeds; the per-round
BER + trig_acc plot **[E1_honest_per_round.png](results/figs/E1_honest_per_round.png)**, η_tight **0.161**, η_loose **0.576** — the non-IID value. 4-seed honest floors span **0.008 -> 0.341**: cls2 0.008, cls7 0.014, cls8 0.015, cls5 0.079, cls9 0.083, cls4 0.092, cls0 0.133, cls6 0.174, cls1 0.175, cls3 0.341 (hard).

**E2 — the reduced free-rider in non-iid - trigger class data starved** —
**[E2_niid_timeline.png](results/figs/E2_niid_timeline.png)** (η_tight **0.161**, η_loose **0.576**).
Honest floors at the FR's own classes: **cls3 ≈ 0.34, cls6 ≈ 0.17**. In the tail the global honest mean drops to ~0.06–0.10 (dragged down by easy classes) while the free-rider rides **~0.24** — i.e. it sits inside the honest-floor band of its own classes, above η_tight but far below η_loose 0.576. 

**The α sweep plots** — **[E3_a01_timeline.png](results/figs/E3_a01_timeline.png)** (α=0.1, extreme skew) and **[E3_a10_timeline.png](results/figs/E3_a10_timeline.png)** (α=1.0, near-IID), both 4 seeds, η_tight frozen 0.161:
- **α=0.1:** honest floors cls3 0.38 / cls6 0.34; honest mean ~0.30, FR mean ~0.34 — **total overlap**. η_loose = **1.064** (μ+3σ on the ultra-wide honest tail exceeds 1.0 = flag *nobody*); η_tight 0.161 = flag *everybody*.
- **α=1.0:** honest floors cls3 0.34 / cls6 **0.56**; the *global* honest mean is low (~0.10, most classes embed) but the FR's own classes stay starved, so the FR rides ~0.41 — **inside the 0.34–0.56 honest band of its own classes**, above η_tight, below η_loose 0.801. Starvation persists even near-IID (Dirichlet(1.0) =/ equal shards).

**Isolated non-IID pair and compute.** **[iso_E2_c6.png](results/figs/iso_E2_c6.png)** — honest (E1) and
reduced (E2) on class 6 sit on top of each other (~0.18) across all 50 rounds.
**[gpu_savings_E2_reduced_niid_c36.png](results/figs/gpu_savings_E2_reduced_niid_c36.png)** and
**[gpu_savings_EA2_reduced_niid_distrib_c36.png](results/figs/gpu_savings_EA2_reduced_niid_distrib_c36.png)**
show the non-IID reduced free-rider also settles at **~0.30 of honest samples (saves ~70%)**.

**EA — distribution-aware assignment.** Giving each client a class it holds a lot of lowers the honest floor (honest clients now embed cleanly), but the reduced/submarine free-rider — training on a concentrated trigger-heavy also embeds cleanly, so the two still coincide. (Per-round honest baseline: **[EA1_honest_per_round.png](results/figs/EA1_honest_per_round.png)**.) *Note:* **[iso_EA2_c6.png](results/figs/iso_EA2_c6.png)** TODO regenerate the plots for EA2.

---

## Related works - to read

- **Uchida et al. (2017)** — first DNN watermark, bits in the *weights* via a projected regulariser
  (white-box). Origin of the "project onto a pseudorandom key" template FareMark still uses; contrasting
  white-box (robust, needs weights) with box-free (convenient, fragile) motivates why the output-layer
  choice is the weak point.
- **Adi et al. (2018)** — first *black-box* watermark, read from outputs on trigger inputs (single-bit).
  The move toward output-only verification that our attack exploits.
- **BlackMarks (2019)** — first *multi-bit* black-box scheme; signature in the output-activation
  distribution. The multi-bit output-distribution reader FareMark inherits.
- **Universal BlackMarks (IEEE SPL 2023)** — FareMark's direct reader ancestor: power-function-on-softmax
  + projection onto a pseudorandom key (Eq. 8 + Eq. 10), before FL. Clearest evidence that the
  peaky-softmax failure is a property of the whole reader family — cite to generalise the negative result.
- **FedIPR (TPAMI 2022/23)** — brought the free-rider angle into FL (client-side secret marks that
  *identify* free-riders). The intellectual origin of the exact claim we refute, and a natural second
  target — our result applies to it too.
- **WAFFLE (SRDS 2021)** — a *server-side* FL watermark; it cannot police free-riders (a free-rider's
  submitted model is the global model, which already carries the server's mark). Explains why free-rider
  detection falls to client-side schemes like FareMark, closing off the "just use a server-side mark"
  escape.
- **FedSMW (2024) and kin** — reuse the same power-on-softmax + projection reader; more evidence the
  failure mode is shared across the family.
- **FRAD (IEEE IoT J 2024), RFFL, ST-DAGMM/DAGMM** — the non-watermark (contribution-evaluation /
  anomaly-detection) free-rider detectors FareMark compares against; the natural **baselines** for our
  evaluation. **ST-DAGMM is the intended baseline for the positive-control (crude free-rider) work.**

**The gap.** None of these evaluate an **adaptive, effort-minimising insider that holds a valid key and tunes its behaviour to sit under η.** Their free-rider is always the crude Gaussian / previous-models attacker, caught trivially. Our threat model and the submarine are exactly that missing evaluation — which is what makes the two-sided result (impossible-to-detect *and* cheaply-evadable) novel.

---
