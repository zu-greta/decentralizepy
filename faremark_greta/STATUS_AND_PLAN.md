# Status + experiment plan

## STATUS
implementation done for experiment set (see [experiment_plan](#experiment-plan---to-be-run)). group A experiments - sans AK are complete so far. 
- status: get Jade up to date with the progress so far from last week (threshold limitations, reduced FR, class difficulty) + go through experiment plan list for experiments to run or not or more suggestions

---
---

## DEFINITIONS

### Measurements definitions

**BER — bit-error-rate.** The watermark is `m` bits; BER = (number of bits read back wrong) ÷ m. It can only take the values 0, 1/m, 2/m, … 
*Technical:* the extraction reads m signs from the softmax and compares to the target message. 
*Plain:* "what fraction of the secret did the model get wrong." **0 = perfect mark, 0.5 = coin flip = no mark**, 1 = every bit flipped. 
Note 0.5, not 1, is "no watermark" — a random model gets half its bits right by luck.

**η (eta) — the detection threshold.** One number pre-calibrated on honest clients. 
The server flags a client as a free-rider if its BER ≥ η. Everything in the THRESHOLD sections is about how to choose it.

**FPR — false-positive rate.** Of the honest clients, the fraction wrongly flagged as free-riders. 
*Plain:* "how many innocent clients you accuse." Best near 0; the paper's headline claim is that it stays low.

**Recall (detection rate, TPR).** Fraction of the actual free-riders caught. 
*Plain:* "how many cheaters you catch." Best near 1. 
FPR and recall trade off: lowering η catches more free-riders but also flags more honest clients.

**Balanced accuracy.** `((1 − FPR) + recall) / 2`. One number combining both. 
*Plain:* the average of "how often you leave honest clients alone" and "how often you catch cheaters."
**0.5 = exactly as good as flipping a coin.** 1.0 = perfect.

**Balanced error.** `1 − balanced accuracy`. **0.5 = useless**, 0 = perfect. 
Used because "best balanced error" over all η is a single verdict on separability.

**best_threshold_balanced_error.** The lowest balanced error achievable by any η,
including the oracle rules that cheat by looking at the free-riders. 
*Plain:* "the best any threshold could possibly do."
If best_threshold_balanced_error = 0.5, no threshold can work — the two populations are inseparable.

**OVL — overlap coefficient.** Draw the histogram of honest BER and the histogram of free-rider BER on the same axis; 
OVL is the area they share (Σ over bins of the smaller of the two densities). 
*Technical:* Weitzman's overlapping coefficient, related to total variation distance. 
*Plain:* "how much the two bell-shapes sit on top of each other." **1.0 = the two distributions are identical** 
(nothing could ever tell them apart); 0 = they don't touch at all. 
High OVL means honest and free-rider BER look the same.

**Headroom (σ).** `(η − mean honest BER) / (per-client std of honest BER)`. 
*Plain:* "how many standard deviations of safety margin the threshold leaves above the honest average."
The paper's rule claims 3σ; we measure 0.1–0.6σ. A Shewhart 3σ limit is meant to false-alarm ~0.13% of the time; less headroom means far more false alarms.

**Degenerate (threshold).** A threshold with η < 1/m. Because BER is quantised to multiples of 1/m, 
any η in (0, 1/m) produces the identical detector ("flag if ≥1 bit wrong"), so the calibrated number is doing no real work.

**Entropy (of the softmax).** `H = −Σ p·ln(p)` over the class-probabilities of one image.
*Technical:* Shannon entropy in nats; `exp(H)` = "the model is effectively choosing between about this many classes." 
*Plain:* how spread-out / unsure the model's prediction is.
High entropy = flat, unsure, many classes plausible. 
Low entropy = peaky, confident, one class dominates. 
Use it because a flat (high-entropy) softmax has a rich tail to hide the watermark in, while a peaky (low-entropy) one does not — so entropy predicts class difficulty.

**Dominance / p_max.** `p_max` = the single largest class-probability for an image;
"dominance" = p_max relative to the rest of the tail (the paper's Eq. 10 quantity, `f(p_max)/Σf(p)`). 
*Plain:* how much the top guess hogs all the probability.
High dominance = peaky = hard to watermark. It is the mirror image of entropy.

**Pearson r.** A number from −1 to +1 measuring how tightly two quantities move together in a straight line. 
0 = unrelated; +1/−1 = perfect positive/negative line. Use it to ask "does class difficulty track entropy (yes, strong r) or classification accuracy (weak r)?"

**N_T (trigger sample count).** How many held-out images of the trigger class the server averages over when extracting the watermark. 
Larger N_T = less noisy extraction. The paper sweeps it (Table VII); I used 50–100.

**cpc (common-per-class, the `+N` of the reduced attacker).** How many images per non-trigger class (common class) the free-rider keeps. 
`cpc=5` means "all your trigger-class images + 5 random from every other class" ≈ 30% of an honest client's data. 
`cpc=-1` = a full honest shard (the free-rider that does 100% of the work but is still labelled a free-rider — used for comparison and sanity check).

**Effort / "data used %".** The free-rider's cumulative image-passes ÷ an honest client's. The number in the timeline inset. 
*Plain:* "what fraction of the work the cheater actually did."

---

### 1. Seed variation 

**Seed:** A single starting number that determines every "random" choice a run makes. 
Same seed -> identical run. Change the seed -> every random choice is re-rolled. 
Each experiment is run at several seeds (3 or 6) and average, so a result is not a fluke of one lucky draw. 
In code, `seed = base_seed + repeat`.

**What the seed re-rolls:**

| re-rolled by the seed | why it is random | how much it moves the result |
|---|---|---|
| which images each client gets | in real FL nobody controls who holds what | moderate |
| the order batches are shuffled during training | standard practice | small |
| the model's initial weights | networks start from random values | small–moderate |
| **the key matrix `M` (per client)** | keys must be secret and unique | large |
| **the target message `B` (per client)** | messages must be unpredictable | large |
| which N_T images go in the trigger bank | it is a sample of the class | small |

**Held constant across seeds:** the trigger class assignment (`trigger_class = cid % n` — client 6 always gets class 6), the number of clients, bits `m`, rounds, epochs, λ, β, α, the smoothing function, and the dataset itself.

**Where `M` and `B` come from.** Each client's key and message are generated by a random-number generator seeded with `seed + 1000·cid + 1`. 
The `1000·cid` part makes every client different from every other client within a run; the `seed` part makes the same client different across runs. 
Changing the seed from 0 to 1 hands every client a brand-new decoder matrix `M` and a brand-new message `B`. 

**large variance** There are two *kinds* of randomness:

- *Nuisance* randomness (data split, shuffle, init): measuring the **same thing** slightly imprecisely. 
  Averaging over seeds sharpens the estimate. This is the ordinary kind.
- *Task-changing* randomness (`M` and `B`): measuring a **different thing** each time. 
  "How hard is class 6 to watermark?" has no single answer — it depends which decoder matrix and which message you drew. 
  Averaging over seeds does not sharpen one number; it averages over a population of different questions.

Three concrete lotteries inside the `M`/`B` draw:

1. **Stuck bits.** A key row that comes out all-one-sign (all +1 or all −1) forces its bit
   to a fixed value forever, because smoothed probabilities are always ≥ 0. Probability of
   a stuck row is `2^(1−l)` where `l = n/m`. At `l=10` that is 0.2% (rare); at `l=5`,
   6.25%; at `l=2`, 50%. Whether a client draws one is pure luck of the seed.
2. **Bias strength.** Even without a fully stuck row, the row sum `s_k = Σ M_kj` sets how
   hard the bit is to flip. About a quarter of rows sum to 0 (easiest); the rest sit on a
   spectrum. Each client's difficulty is a mixture over whichever rows it drew.
3. **Message alignment.** Some target bits agree with their row's bias, some fight it. A
   different `B` re-rolls which bits are fighting.

**On top of all that, pure counting noise.** BER is "how many of `m` bits came back wrong,
divided by `m`". Even if every bit failed independently at the same rate `p`, BER would
jump around by `√(p(1−p)/m)` from one client to the next. With small `m` this is large:
on CIFAR-10 at `m=2` it is essentially the *entire* observed spread. This is why the
calibrated threshold η itself has ~40% seed-to-seed variation (finding F2) — η is derived
from these quantities and inherits every lottery.

---

### 2. Thresholds

**The setup.** After training, the server extracts the watermark from each client and computes its BER. then it compares that BER to a threshold η, flagging any client with BER ≥ η as a free-rider. The whole question is how to pick the single number η.
All rules below compute η from the *honest* BER values collected over the converged tail (last 20 rounds).

| rule | where it is from | what it computes | in plain terms | value from experiments |
|---|---|---|---|---|
| **μ + 3σ (the paper's "coded" rule)** | FareMark Section IV-D3 | mean of the per-round mean-BER, plus 3× its standard deviation | "the normal range is mean ± 3 std; flag anything above." Assumes a bell curve; ~0.13% of honest clients should trip it if the assumption holds | TODO |
| **pooled μ + 3σ** | same, computed differently | pool every (seed, round) mean into one sample, then μ + 3σ | same idea, but the spread now includes between-seed variation, so it lands higher (looser) | TODO |
| **loose (per-client μ + 3σ)** | same formula, client population | mean and std of the individual client BERs, then μ + 3σ | the only μ+3σ variant whose σ matches the population the test is applied to | TODO |
| **median + 3·MAD** | robust statistics (Hampel) | median instead of mean; 1.4826×median-absolute-deviation instead of σ | "normal range, but ignore outliers." Breaks (-> 0) when more than half the honest clients sit at BER 0 | TODO |
| **trimmed-10% μ + 3σ** | Tukey trimmed statistics | drop the top and bottom 10% of clients, then μ + 3σ | same as coded but throw away the extremes first | TODO |
| **adaptive σ-clip** | iterative sigma-clipping (astropy `sigma_clipped_stats`; also DP-SGD adaptive clipping) | repeatedly drop clients above μ+3σ and recompute until it stabilizes, then μ+3σ | "keep tightening the range until only the well-behaved clients define it." Adaptive clipping idea | TODO |
| **honest p95 / p99** | non-parametric empirical quantile | the 95th (or 99th) percentile of honest BER | "put the line where the worst 5% (or 1%) of honest clients sit." Fixes the false-positive rate directly, assumes no distribution shape | TODO |
| **equal-error-rate (EER)** | biometrics | the η where false-positive rate = false-negative rate | "the balance point where you wrongly flag as many honest as you miss free-riders." **Needs to see free-riders** | TODO |
| **Youden-optimal** | Youden's J statistic (1950) | the η minimizing (FPR + FNR)/2 — the single best possible threshold | "the best any threshold could ever do." **Needs free-riders**, used only to prove an upper bound | TODO |

---

### 3. Class difficulty 

**What "class difficulty" means.** Each client hides its watermark in the model's behaviour on its assigned trigger class.
From experiments, it can be seen that some classes are easier to hide a watermark in than others - a "hard" class is one where honest clients end up with high BER (the mark does not embed well) no matter how long they train.

**How it is measured.** For each trigger class:
1. Take the honest-only runs 
2. For each client, average its BER over the converged tail (last 20 rounds) and over all
   seeds. Since each client owns exactly one trigger class, this per-client number is the
   per-class number (class's floor)
3. Compare floors across classes. In the known-good CIFAR-100 10-client run they span
   **0.00 (classes 7,8,9) to 0.21 (class 6)** 

**Class difficulty — mechanism.** The watermark is read from the shape of the softmax tail (all the class probabilities except the dominant one):
- A **flat** softmax (the model is unsure — many classes get similar probability) has a
  rich, shapeable tail. The watermark loss can nudge those tail values to encode bits
  cheaply. -> low BER, easy class.
- A **peaky** softmax (the model is very confident — one class gets ~0.9, the rest near 0)
  has a flat, structureless tail of near-equal tiny numbers. There is nothing to shape, so
  the bits become coin flips. -> high BER, hard class.

Measure "peakiness" two ways: **entropy** (how spread-out the probabilities are; high =
flat) and **dominance / p_max** (how much the top class takes; high = peaky). In the
10-client regime these correlate with BER at |r| ≈ 0.6–0.7, while classification accuracy
correlates only weakly (|r| ≈ 0.05–0.4). That gap is the point: difficulty is about the
shape of the output distribution, not about whether the model classifies the class
correctly

**Plots** (`honest_class_lines`, `class_difficulty`):
- one BER-vs-round line per class; the tail floor is the number that matters.
- the four-panel `class_difficulty` figure sorts classes easy->hard, shows their accuracy
  in the same order (visibly scrambled = accuracy does not explain difficulty), and
  scatters BER against error and loss with Pearson r.

**What a "good result for the thesis" looks like:** floors that span a wide range and
correlate with entropy/dominance, not accuracy. That kills the obvious objection ("your
hard classes are just the classes the model is bad at") and shows the difficulty is baked
into the watermarking scheme, not the model quality.

---

### 4. FareMark -> more clients than classes problem

**Context.** Each client gets its own trigger class, and there are only 10 classes (CIFAR-10) or 100 (CIFAR-100). Once there are more clients than classes, clients must share trigger classes — two or more clients hide watermarks in the same class (forced sharing). 

**FareMark (table IX).** 
TODO: check the paper again - in table IX. describe the paper's "more clients than classes" experiments, how they do it, and what they show.

**Missing from the paper:** there is no statement of how the training set is partitioned across
clients — no images-per-client, no IID-vs-non-IID scheme, no mention of whether client
shards are disjoint or overlapping, and no learning rate, batch size, or optimizer for the
FL training. This matters enormously: 100 clients over CIFAR-100's 50,000 images is 500
images each if disjoint, which undertrains ResNet-18 (our R1 reached ~46% vs the paper's
75.31%). The paper's 75% is only reachable if clients see more data than a strict
disjoint split gives — e.g. IID sampling with replacement, larger overlapping shards, or
more rounds than stated. FareMark is not reproducible on the data-partition axis without guessing.

---


### 5. `m` — the number of watermark bits

**Watermark bits `m`.** The watermark is a string of `m` bits (0s and 1s). Each bit is read from one group of softmax outputs: the `n` class-probabilities are chopped into `m` disjoint groups of `l = n/m` each, and each group's projection sign gives one bit. 
So `m · l = n`, i.e. `m ≤ n` — you cannot have more bits than classes. 

**`m` importance.** BER is "wrong bits ÷ m", so:
- **small m -> coarse BER.** At m=2, BER can only be 0, 0.5, 1 — a single client-round is
  either perfect, half-wrong, or fully wrong. The "floor" numbers become lumpy and the
  threshold has almost nothing to grip.
- **large m -> each bit is weaker.** More bits means fewer softmax outputs per bit
  (`l = n/m` shrinks), so each bit is read from a shorter projection and is noisier. It also
  raises the chance of a stuck bit: a key row that is all-one-sign can never be embedded,
  and P(stuck row) = `2^(1−l)` grows fast as l shrinks (0.2% at l=10, 6.25% at l=5, 50% at
  l=2).

**Experiment setup m values:**

| dataset | n (classes) | **m used** | l = n/m | stuck-row rate | why this m |
|---|---|---|---|---|---|
| **CIFAR-100** | 100 | **10** | 10 | 0.2% (negligible) | the default `max(2, n//10)`. l=10 is long enough that bits are reliable and stuck rows are rare. Watermark-accuracy ceiling 99.9%, so the paper's 99.7% is reachable. |
| **CIFAR-10** | 10 | **1** | 10 | 0.2% | m=2 (the naive default) gives l=5 -> 6.25% stuck rows -> accuracy caps at 96.9%, below the paper's 99.72%. Only m=1 (l=10) can reach the paper's number — which is itself a finding: the paper's headline CIFAR-10 result forces a *single-bit* watermark, i.e. BER is one coin flip. |
| (exploratory) CIFAR-100 | 100 | 20 | 5 | 6.25% | used only to show the degradation: doubling the bits halves l, injects stuck bits, and raises the honest floor — demonstrating the `m·l=n` squeeze. |

**In short:** m=10 on CIFAR-100 is the best spot so far (reliable bits, negligible stuck rows,
paper-reachable accuracy) and is what all the Group A results use. m=1 on CIFAR-10 is forced
by the paper's own accuracy number. The tension between "more bits = more watermark capacity"
and "more bits = weaker, stuck-prone bits, bounded by m ≤ n" is a structural limitation of
output-layer watermarking.

---
---

## RESULTS

Batch **Group A** (CIFAR-100 / 10-client / m=10 config). 

Group A storyline: 
**[(1)](#r0--watermark-embedding-on-all-honest-clients-sanity-runs)** sanity runs to show that the watermark embeds correctly and honest clients converge to low BER (A1) -> 
**[(2)](#r3--threshold-calibration-across-seeds)** but different trigger classes converge to very different floors, and the threshold built from them has almost no safety margin (A1 thresholds) -> 
**[(3)](#r2--threshold-calibration)** and its own value swings 30% just by changing the random key (A1 eta-stability) -> 
**[(4)](#r4--reduced-free-rider-easy-classes)** so a free-rider doing a third of the work is invisible at easy classes (A2) and only catchable at hard classes by flagging honest clients too (A3).
**[(K)](#experiment-plan---to-be-run)** TODO: a free-rider with the same key and matrix as an honest client on the same trigger class -> prove that the watermark can be embedded and free-rider cannot be caught with a threshold

---

### R0 — watermark embedding on all honest clients (sanity runs)

**Setup.** A1: CIFAR-100, 10 clients (one per trigger class 0–9), ResNet-18, m=10 bits, 50 rounds, 5 local epochs, λ=5, all clients honest, 6 seeds.

**Plot: [A1_class_floors.png](results/groupA/figs/A1_class_floors.png)** 
— *Honest BER per trigger class over rounds.*
- **x** = communication round (1–50). **y** = bit-error-rate; **0 = mark embeds perfectly, 0.5 = coin flip = no mark.** BER only takes multiples of 1/m = 0.1.
- one coloured line per trigger class (10 classes); shaded band = ±1 std over 6 seeds.
- grey block on the right = the converged tail (last 20 rounds) used for calibration.
- the legend floor value is each class's mean BER over that tail.

**Information.** All ten classes start near 0.3–0.45 (random, untrained) and drop within ~8 rounds to their floors, then stay flat. 
Low, stable floors = the watermark is embedding. Health check for all honest code — final classification 73% (near the paper's 75%)

---

### R1 — class difficulty

**Same plot:** [A1_class_floors.png](results/groupA/figs/A1_class_floors.png)  

The floors are not all equal — they span 0.001 (class 8) to 0.114 (class 6), a **>100× range**:

| class | floor | | class | floor |
|---|---|---|---|---|
| 8 | 0.001 | | 5 | 0.037 |
| 9 | 0.002 | | 3 | 0.057 |
| 1 | 0.020 | | 7 | 0.061 |
| 0 | 0.025 | | 4 | 0.094 |
| 2 | 0.028 | | **6** | **0.114** |

**Good result:** a wide spread of floors that is stable across the tail and consistent across seeds. 
Class 6 is intrinsically ~100× harder to watermark than class 8, and no amount of extra training closes the gap — the lines are flat for 40 rounds. 

---

### R2 — threshold calibration

Threshold rules — `A1_honest_c100`
- seeds: **6**, calibration window: last **20** rounds
- honest client-rounds: **1200**, mean BER **0.0438**, per-client sd **0.0736**
- watermark bits m = **10**, so BER can only take values 0, 0.100, 0.200, …

**Plot: [A1_thresholds.png](results/groupA/figs/A1_thresholds.png)**
- **x** = round, **y** = BER (0.5 = coin flip). **Thick blue** = honest mean BER (what η
  is built from). **Dashed / dotted blue** = per-round p90 and worst client. **Pale blue
  band** = the spread from mean up to worst client — *the population the test is actually
  applied to.*
- each **coloured horizontal line** = one threshold rule; legend gives its η and honest FPR.
- **red dash-dot at 0.1** = 1/m. Any η below it is degenerate (calibration does nothing).
- grey block = calibration window (last 20 rounds). 

-> threshold table (every candidate threshold on one honest BER trace.):

| rule | eta | how it is computed | honest FPR | headroom | degenerate? |
|---|---|---|---|---|---|
| median + 3*MAD (robust location/scale) | 0.0000 | median instead of mean, 1.4826*MAD instead of sigma. Immune to outliers, but collapses to 0 when more than half the honest clients sit at BER=0. | 100.0% | -0.59σ | **yes** — below 1/m, so this is exactly 'flag if ≥1 bit wrong'; the value of eta does nothing |
| coded (paper, mean-over-clients then mu+3s over rounds, avg over seeds) | 0.0841 | for each seed: average BER over the N clients in each round -> one number per round; take mu+3*sigma of those; average across seeds. This is what the paper's text most plausibly means and what run_all.sh freezes. | 31.4% | +0.55σ | **yes** — below 1/m, so this is exactly 'flag if ≥1 bit wrong'; the value of eta does nothing |
| pooled (mu+3s over all seeds' round-means at once) | 0.1077 | same as above but pool every (seed, round) mean into one sample before mu+3*sigma. Looser, because between-seed spread is added to the sigma. | 9.9% | +0.87σ | no |
| trimmed-10% mu+3s | 0.1596 | drop the top and bottom 10% of client-rounds, then mu+3*sigma on the rest. | 9.9% | +1.57σ | no |
| honest p95 | 0.2000 | the 95th percentile of honest client-rounds. Fixes the false-positive rate at 5% by construction -- no distributional assumption at all. | 9.9% | +2.12σ | no |
| adaptive sigma-clip (kept 0.98) | 0.2242 | iteratively drop points above mu+3*sigma and recompute until stable, then mu+3*sigma on what survives. Excludes the hard-class tail from its own calibration. | 2.4% | +2.45σ | no |
| loose (mu+3s over PER-CLIENT BER) | 0.2644 | mu and sigma of individual client-round BERs -- no averaging over clients. This is the ONLY variant whose sigma matches the population the test is applied to. Roughly sqrt(N) larger than 'coded'. | 2.4% | +3.00σ | no |
| honest p99 | 0.3000 | the 99th percentile. Targets 1% FPR. | 2.4% | +3.48σ | no |

**Note:** The paper's rule ("coded", orange) lands at **η = 0.084 with FPR 31%** and **+0.55σ of headroom** - stricter threshold than the 3σ. 
To follow 3σ, implement the "loose" (pink, η = 0.264) threshold, which computes σ on individual clients instead of on the mean-over-clients. Summary from the table:

| rule | η | headroom | honest FPR |
|---|---|---|---|
| **coded (paper's rule)** | 0.084 | **+0.55σ** | **31%** |
| pooled | 0.108 | +0.87σ | 10% |
| honest p95 | 0.200 | +2.12σ | 10% |
| **loose (per-client μ+3σ)** | 0.264 | **+3.00σ** | 2% |

---

### R3 — threshold calibration across seeds

**Plot: [eta_stability_ber_A1_honest_c100.png](results/groupA/figs/A1_eta_stability/eta_stability_ber_A1_honest_c100.png)** — *per-seed honest BER curves + the η each seed produces.*
- **thick black** = mean honest BER over seeds; **grey band** = ±1 std across seeds.
- **faint blue lines** = individual seeds' BER traces.
- **green horizontal line** = the final η (averaged over seeds); **green band** = η ± its
  seed-to-seed std.

**Results:** η = **0.084 ± 0.025** — the standard deviation is **~30% of η's own value**, purely from re-drawing the random key `M` and message `B` each seed. The faint per-seed η lines are spread across roughly 0.06–0.14.

---

### R4 — Reduced free-rider (easy classes)

**Setup.** A2: same base, but clients at trigger classes **1 and 7** are reduced free-riders 
—> honest for 12 warmup rounds, then training on only their trigger class + 5 images per other class (**31% of an honest client's data**). 
3 seeds. η frozen at the calibrated 0.064.

**Plot: [A2_easy_timeline.png](results/groupA/figs/A2_easy_timeline.png)** — *BER vs round, honest vs free-rider, with η.*
- **thick blue** = honest mean BER; **thick orange** = free-rider mean BER; bands = ±1 std.
- **black dashed** = calibrated η = 0.064. **Pale blue dotted band** = honest floor at the free-rider's own classes (1, 7) for comparison
- yellow = warmup; green = calibration window; grey dashed = free-riding starts (round 12); orange ▽ = re-embed taps. 

**Results.** After defection the free-rider's BER drops to 0.00 and stays there, which is below both the honest curve and η — for all 40 post-defection rounds. 
It re-embeds its mark harder than honest clients because its shrunken data is concentrated on the trigger class.

**The numbers (`A2_easy_sep.json`):** per class 1, honest 0.020 vs **FR 0.000**; per class 7, honest 0.061 vs **FR 0.000**. 
Both classes: **best-threshold balanced error = 0.500** — no threshold.

**Conclusion.** A free-rider doing a third of the work has a stronger watermark than honest clients at the same position.

---

### R5 — Reduced free-rider (hard classes)

**Setup.** A3: identical to A2 but free-riders at the hard classes 3 and 6

**Plot: [A3_hard_timeline.png](results/groupA/figs/A3_hard_timeline.png)** — same layout as A2.

**Results.** After defection the free-rider rides at ~0.11–0.13, sitting on top of the honest-floor band for classes 3/6 (pale blue dotted) and oscillating around η. 

**The numbers (`A3_hard_sep.json`), read per class — the fair comparison:** 

| class | honest | FR | OVL | best balanced error | reading |
|---|---|---|---|---|---|
| **3** | 0.057 | 0.037 | 0.88 | **0.500** | FR *cleaner* than honest -> inseparable |
| **6** | 0.114 | 0.220 | 0.53 | 0.267 | FR separable — but at a cost (below) |

At class 3 the free-rider is cleaner than honest and no threshold can be drawn.
Class 6 is the one place the free-rider is catchable but with recall 0.867 only by flagging FPR 0.400 of honest clients at that same class. 
To catch the free-rider you falsely accuse 40% of honest clients sharing its position.

---

### R6 — same trigger class, different keys for free-rider, class 6

**Setup.** A4: a `reduced` free-rider (cid 0) pinned to trigger class **6**, sharing that
class with honest client 6. The free-rider draws its own random key `M` and message `B`
(the normal condition). 3 seeds, cpc=5 (31% data), η frozen at 0.064.

**Plot: [A4_sameclass_timeline.png](results/groupA/figs/A4_sameclass_timeline.png)** — BER vs round; layout as in R4/R5. Pale blue dotted = the honest floor at class 6 (0.114).

**Results.** After defection the free-rider's BER drops to **~0.067** and sits at or
just below the honest class-6 floor for all 40 rounds — the orange and blue lines are
interleaved, both hugging η. Per class 6 (`A4_sameclass_sep.json`): honest 0.114 vs FR
0.067 -> the free-rider is cleaner than honest, OVL 0.658, best balanced error 0.500.
The coded rule gets recall 0.333 at FPR 0.400 — to catch the free-rider, will falsely accuse 40% of honest clients at that same class. 

---

### TODO: R7 — same trigger class, same key for free-rider, class 6

**Intended setup.** AK was meant to give the free-rider (cid 0) honest client 6's **exact
key and message** (`WM_KEY_TWINS="0:6"`), so the only difference would be training effort.

**What actually ran (verified from the AK pod log).** `wm_key_twins` appears **nowhere** in
the pod log's config-override block, and the saved config shows `wm_key_twins: None`. Every
other flag (trigger_class_map 0:6, free_rider_ids 0, cpc 5, eta 0.064) IS echoed there, so
the twin flag simply did not reach the run — most likely the pod did not clone the
`ak-samekey` branch, or `WM_KEY_TWINS` was not exported into that submit. **So AK is a second
free-rider-at-class-6 run with its OWN key — a repeat of A4 at a different draw, not the
controlled same-key experiment.**

**What it still shows.** With its own key, the free-rider reads **FR 0.163** at class 6
(`AK_samekey_sep.json`) vs honest 0.114 — OVL 0.508, best balanced error 0.308. Compared to
A4's own-key draw (FR 0.067), this is a *third* independent class-6 key draw, and it lands
in a completely different place. That is not the isolation result, but it is more evidence
for the key lottery (next section).

**To get the real R7, rerun with the twin verified:**
```bash
# confirm the branch has the patch, then submit WITH the env var visible in the log
grep -n wm_key_twins scripts/run_experiment.py faremark/clients.py   # must be present
WM_KEY_TWINS="0:6" ... FAMILY="AK_sameclass_samekey_c6" ./submit_experiment.sh 14 <seed>
# then CHECK the pod log shows:  wm_key_twins  ->  0:6
```
Until the pod log echoes `wm_key_twins -> 0:6`, treat the same-key result as **not yet
collected**. ⚠️

---

### The key-lottery finding — reconciling A3, A4, AK 

You will notice the free-rider at class 6 gives a **different BER in every run**, even though
the class, the attack, and the 31% data budget are identical:

| run | free-rider's key | honest c6 | **FR c6** | best balanced error | separable? |
|---|---|---|---|---|---|
| **A3** (R5) | its own (random draw #1) | 0.114 | **0.220** | 0.267 | somewhat |
| **A4** (R6) | its own (random draw #2) | 0.114 | **0.067** | **0.500** | no — FR *cleaner* |
| **AK** (R7) | its own (random draw #3 — twin knob failed) | 0.114 | **0.163** | 0.308 | slightly |

**All three are the SAME experiment** — free-rider at class 6, own key, 31% data, 3 seeds
each — differing only by the random key draw. (AK was *meant* to hold the key fixed but the
twin did not apply, R7 above.) **The free-rider's class-6 BER swings from 0.067 to 0.220 — a
3× range — driven by nothing but which random key it drew.** Same position, same attack, same
effort. This is the answer to "is separability just position luck?": **it is worse than
position luck — it is KEY luck.**
Even at a single fixed position (class 6), whether the free-rider looks separable depends
entirely on the secret it happened to be assigned.

Two things follow, and both strengthen the thesis:

1. **Separability at hard classes is not a property of free-riding — it is a property of the
   draw.** In A3 the free-rider drew a key that embeds poorly (BER 0.220, looks catchable);
   in A4 it drew one that embeds well (0.067, looks honest). The *server cannot know which*,
   because it does not know the key quality in advance. A detector that only works when the
   attacker is unlucky is not a detector.

2. **The true effort effect (fixed key) is not yet measured** — AK was supposed to isolate
   it but the twin knob failed (R7). Once rerun correctly, the prediction is a small gap
   (honest embeds its mark fully, the 31%-data free-rider slightly less), which would be
   swamped by the 0.15-wide key swing above. **This is the single most valuable rerun** —
   it is the clean isolation and it is one verified env var away.

**How to present this.** The strongest, most defensible sentence is: *"Whether a free-rider
at a given position appears separable is dominated by the random key it draws (a 3× BER swing
at class 6 alone), not by whether it free-rode. The genuine effect of doing less work,
isolated by giving the free-rider the honest client's exact key, is expected to be small and
buried under this key variance — the controlled same-key run (AK) must be repeated to confirm,
as the twin key did not apply in the first attempt."* That converts your worry into your strongest anti-separability argument.

*(This also explains R5's class-6 result that looked separable last week: it was one key
draw. The correct claim is not "hard classes are always separable" but "separability at hard
classes is a coin-flip of the key, so no deployable threshold can rely on it." Update R5's
wording accordingly — the honest framing is the key-lottery one.)*

### Future groups (A same key, C smoothing, D +N spectrum, E non-IID, F capacity)  ⏳ TO RUN

Placeholders — see EXPERIMENT PLAN. Each will get its calibrated-threshold plot, its
BER-vs-round timeline with η, and its separability JSON, following the same template above.

---


## EXPERIMENT PLAN - to be run

### Group A — proven baselines
| label | setting | proves (notes ref) | status |
|---|---|---|---|
| A1 | honest, cifar100, 10cl, 6 seeds | class difficulty; threshold calibration | done |
| A2 | reduced +5, classes 1,7, 3 seeds | non-sep at EASY classes (FR cleaner than honest) | done |
| A3 | reduced +5, classes 3,6, 3 seeds | non-sep at HARD classes | done | 
| A4 | sameclass, FR on class 6, 3 seeds | FR vs honest, SAME trigger class, same training | done - need to replot |
| AK | sameclass, same key/message, FR on class 6, 3 seeds | FR vs honest, SAME trigger class, SAME key/message | TODO | 

### Group B — thresholds (notes: THRESHOLD regime)
All computed offline from A1 — **no new runs needed**. `plot_all_thresholds.py` + the
9-rule table in `detection.py` cover: μ+3σ, median+MAD, trimmed, adaptive-clip, p95/p99,
EER, Youden. Deliverable = the one-timeline figure + the headroom table.

### Group C — difficulty mechanism (notes: DIFFICULTY, smoothing)
| label | setting | proves |
|---|---|---|
| C1 | honest, sin smoothing (WM_ALPHA=1.5708), 3 seeds | does a different f() move the floors? |
| C2 | class_probe on A1 (offline) | entropy/dominance vs accuracy correlation |

### Group D — the +N free-riding spectrum (notes: FR spectrum)
| label | setting | proves |
|---|---|---|
| D1 | reduced, classes 3,6, N ∈ {-1,0,1,2,5,10,25,50}, 3 seeds | price of invisibility; N=-1 (full data, still FR) is the anchor |

### Group E — non-IID (notes: non-iid)
| label | setting | proves |
|---|---|---|
| E1 | honest, Dirichlet α=0.5, 3 seeds | does label skew widen honest BER? |
| E2 | reduced, classes 3,6, α=0.5, 3 seeds | non-sep under non-iid |
| (E requires the `n_trigger_samples` split before quoting — already logged in clients.py) |

### Group F — more clients than classes (notes: more clients)
| label | setting | proves |
|---|---|---|
| F1 | honest, 200 clients, MORE ROUNDS (100), 3 seeds | capacity — but needs enough rounds to train |
| F2 | reduced, 200cl, classes 6,7, 3 seeds | forced class-sharing overlap |

### Group G — detection policy (notes: DETECTION — currently untouched)
Design-only for Tuesday; implement in `wm_verify.py` next week:
consequence of crossing η, k-warnings-before-flag, detection window.

### Deferred / paper reproduction
- R0/R1/R2 reproduction: only after the probe confirms peaky-softmax embedding.
- `client_train` twins: the memorisation control, once embedding is confirmed.