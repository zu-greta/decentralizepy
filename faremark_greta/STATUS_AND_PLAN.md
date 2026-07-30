# Status + experiment plan

## STATUS
implementation done for experiment set (see [experiment_plan](#experiment-plan---to-be-run)). **Complete (3 seeds):** Group A (A1–A4), **AK** (R7), **Group D** (R8), **isolated same-class pairs** (R10), **operating-point money plot** (R11, insiders). **E3 non-IID** done but α-sweep buggy (R12). **C1 FAILED** (sin crash — R14, deprioritised). Compute + trimmed next batch in **R13**. Remaining to run: **E1/E2 + E3-fixed, F, H, I (1-seed sweep), V2** → `BATCH=EFHIV`. 
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
| **μ + 3σ (the paper's "coded" rule)** | FareMark Section IV-D3 | mean of the per-round mean-BER, plus 3× its standard deviation | "the normal range is mean ± 3 std; flag anything above." Assumes a bell curve; ~0.13% of honest clients should trip it if the assumption holds | η=**0.084**, FPR **31%**, headroom **+0.55σ** — **degenerate** (< 1/m=0.1). ✓ impl `detection.py:coded_eta` |
| **pooled μ + 3σ** | same, computed differently | pool every (seed, round) mean into one sample, then μ + 3σ | same idea, but the spread now includes between-seed variation, so it lands higher (looser) | η=**0.108**, FPR **10%**, **+0.87σ**. ✓ impl `plot_all_thresholds.py:pooled` (JSON field literally named `coded (mu+3s round-mean)`) |
| **loose (per-client μ + 3σ)** | same formula, client population | mean and std of the individual client BERs, then μ + 3σ | the only μ+3σ variant whose σ matches the population the test is applied to | η=**0.264**, FPR **2%**, **+3.00σ** — the only true-3σ variant, but so high it catches nothing. ✓ impl `detection.py:_mu_k_sigma(H)` |
| **median + 3·MAD** | robust statistics (Hampel) | median instead of mean; 1.4826×median-absolute-deviation instead of σ | "normal range, but ignore outliers." Breaks (-> 0) when more than half the honest clients sit at BER 0 | η=**0.000**, FPR **100%**, **−0.59σ** — collapses to 0 (>half honest at BER 0), so it flags everyone. ✓ impl `detection.py:_median_k_mad` |
| **trimmed-10% μ + 3σ** | Tukey trimmed statistics | drop the top and bottom 10% of clients, then μ + 3σ | same as coded but throw away the extremes first | η=**0.160**, FPR **10%**, **+1.57σ**. ✓ impl `detection.py:_trimmed_mu_sigma` |
| **adaptive σ-clip** | iterative sigma-clipping (astropy `sigma_clipped_stats`; also DP-SGD adaptive clipping) | repeatedly drop clients above μ+3σ and recompute until it stabilizes, then μ+3σ | "keep tightening the range until only the well-behaved clients define it." Adaptive clipping idea | η=**0.224**, FPR **2%**, **+2.45σ**. ✓ impl `detection.py:adaptive_clip_eta` (iterative, kept 98%) |
| **honest p95 / p99** | non-parametric empirical quantile | the 95th (or 99th) percentile of honest BER | "put the line where the worst 5% (or 1%) of honest clients sit." Fixes the false-positive rate directly, assumes no distribution shape | p95 η=**0.200** FPR **10%** (+2.12σ); p99 η=**0.300** FPR **2%** (+3.48σ). ✓ impl `detection.py:_percentile` |
| **equal-error-rate (EER)** | biometrics | the η where false-positive rate = false-negative rate | "the balance point where you wrongly flag as many honest as you miss free-riders." **Needs to see free-riders** | **oracle** (peeks at FRs), attack-dependent: η≈0.05 (A2/easy) to 0.15 (AK). ✓ impl `detection.py:eer_threshold` |
| **Youden-optimal** | Youden's J statistic (1950) | the η minimizing (FPR + FNR)/2 — the single best possible threshold | "the best any threshold could ever do." **Needs free-riders**, used only to prove an upper bound | **oracle** upper bound = 1 − `best_threshold_balanced_error`: 0.50 (A2/A4, inseparable) to 0.84 (AK global). ✓ impl `detection.py:best_threshold` |

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

**FareMark (Table IX).** The paper's capacity experiment is ResNet-18 / CIFAR-10 with **50 clients** and 10 classes — i.e. **5 clients per class** (forced sharing). Reported watermark accuracy stays high (~95.8%) and main-task accuracy ~88.4%, which the paper presents as evidence the scheme scales past one-client-per-class. **The crucial caveat (paper §V-F3 + our reading of the protocol):** Table IX uses **trigger-sample consistency** — the verification images are the *same images the client trained on* (`wm_trigger_mode="client_train"` in our code). That is **memorisation, not generalisation**: the paper itself notes (Table V) that a mark fitted to specific samples "cannot be generalised to other trigger-class samples". So Table IX's high capacity number is measured under the most favourable possible test (verify on training data). Our F3 reproduces this row *and* runs a held-out twin (`wm_trigger_mode="class"`, same class, different held-out images); `paper_check.py` then reports the **memorisation gap** = paper-mode watermark-acc − held-out watermark-acc. A large positive gap means the "capacity" is memorisation. This is why our own capacity experiments (Group F, 200 clients on CIFAR-100 with held-out banks) are the stricter test the paper avoids.

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
**[(K)](#experiment-plan---to-be-run)** and finally, even at a single fixed class with the free-rider handed the honest client's exact key and message (AK), the only remaining difference is training effort — and the free-rider still embeds a mark at least as clean as the honest client's, so no threshold catches it.

---

### R0 — watermark embedding on all honest clients (sanity runs)

**Setup.** A1: CIFAR-100, 10 clients (one per trigger class 0–9), ResNet-18, m=10 bits, 50 rounds, 5 local epochs, λ=5, all clients honest, 6 seeds.

**Plot: [A1_class_floors.png](results/groupD/figs/A1_class_floors.png)** 
— *Honest BER per trigger class over rounds.*
- **x** = communication round (1–50). **y** = bit-error-rate; **0 = mark embeds perfectly, 0.5 = coin flip = no mark.** BER only takes multiples of 1/m = 0.1.
- one coloured line per trigger class (10 classes); shaded band = ±1 std over 6 seeds.
- grey block on the right = the converged tail (last 20 rounds) used for calibration.
- the legend floor value is each class's mean BER over that tail.

**Information.** All ten classes start near 0.3–0.45 (random, untrained) and drop within ~8 rounds to their floors, then stay flat. 
Low, stable floors = the watermark is embedding. Health check for all honest code — final classification 73% (near the paper's 75%)

---

### R1 — class difficulty

**Same plot:** [A1_class_floors.png](results/groupD/figs/A1_class_floors.png)  

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

**Plot: [A1_thresholds.png](results/groupD/figs/A1_thresholds.png)**
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

**Plot: [eta_stability_ber_A1_honest_c100.png](results/groupD/figs/A1_eta_stability/eta_stability_ber_A1_honest_c100.png)** — *per-seed honest BER curves + the η each seed produces.*
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

**Plot: [A2_easy_timeline.png](results/groupD/figs/A2_easy_timeline.png)** — *BER vs round, honest vs free-rider, with η.*
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

**Plot: [A3_hard_timeline.png](results/groupD/figs/A3_hard_timeline.png)** — same layout as A2.

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

**Plot: [A4_sameclass_timeline.png](results/groupD/figs/A4_sameclass_timeline.png)** — BER vs round; layout as in R4/R5. Pale blue dotted = the honest floor at class 6 (0.114).

**Results.** After defection the free-rider's BER drops to **~0.067** and sits at or
just below the honest class-6 floor for all 40 rounds — the orange and blue lines are
interleaved, both hugging η. Per class 6 (`A4_sameclass_sep.json`): honest 0.114 vs FR
0.067 -> the free-rider is cleaner than honest, OVL 0.658, best balanced error 0.500.
The coded rule gets recall 0.333 at FPR 0.400 — to catch the free-rider, will falsely accuse 40% of honest clients at that same class. 

---

### R7 — same trigger class, SAME key for free-rider, class 6  

**Setup.** AK gives the free-rider (cid 0) honest client 6's **exact key and message** (`WM_KEY_TWINS="0:6"`) on top of the same trigger class (`TRIGGER_CLASS_MAP="0:6"`),
so cid 0 and cid 6 are identical in every watermark parameter and the only difference is that cid 0 free-rides (trains on 31% of the data).

**Plot: [AK_samekey_timeline.png](results/groupD/figs/AK_samekey_timeline.png)** — *BER vs round,
honest vs free-rider at the SAME class/key, aggregated over 3 seeds.*
- **thick blue** = honest **mean** BER over *all* clients (the global floor); **thick orange** =
  free-rider mean BER; bands = ±1 std over 3 seeds.
- **light-blue dashed** = the honest twin *at the free-rider's class (6)* — the fair comparator.
- **black dashed** = η tight 0.064 (frozen, used); **blue dashed** = η loose 0.264 (pooled μ+3σ).
- yellow = forced-honest warmup; green = calibration window [8,11]; grey dashed = free-riding
  starts (~round 12); orange ▽ = re-embed taps. Inset: 31 % of honest data, avg over 3 seeds.

**Result (3 seeds, converged tail, read from the figure).** After defection the free-rider and
its honest twin at class 6 oscillate together in the same band (~0.13–0.30) and their ±1 std
ribbons overlap the whole way — they are not separable. The FR mean settles a touch *lower*
(≈0.14–0.17, ending ~0.135) than the honest twin (≈0.18–0.20), i.e. the 31 %-data free-rider is
if anything cleaner, never noisier. Meanwhile the **global** honest mean (~0.06) sits right on
η tight — so at the hard class 6 even the honest client rides above η tight.
"Same class, same secret, 31 % effort -> the free-rider's BER is indistinguishable from (slightly below) its honest twin's; no η separates them."

### R8 — +N price-of-invisibility spectrum  

**Setup.** `D1_reduced_c100_c36_n{-1,0,1,2,5,10,25,50}`, free-riders pinned to classes 3 and
6, cpc = N images/other-class, η frozen 0.064. `N = 0` = trigger-images-only; `N = -1` = a
full honest shard (still labelled a free-rider — the effort anchor).

**Plot: [D1_spectrum.png](results/groupD/figs/D1_spectrum.png)** — three panels: (1) FR BER over
rounds, one line per budget; (2) converged FR BER vs budget with honest floor + η; (3) actual
samples/round spent. Aggregated over 3 seeds.

**Converged free-rider BER vs data budget — pooled over classes 3 & 6 (3-seed mean, panel 2):**

| N (cpc) | effort (samples vs honest) | converged FR BER (pooled) | reading |
|---|---|---|---|
| triggers only (0) | 23 % | **≈ 0.44** (±0.18) | overfits → fails held-out verification → **caught** |
| +1 / class | 24 % | ≈ 0.13 | plateau reached already |
| +2 / class | 26 % | ≈ 0.11 | — |
| +5 / class | 31 % | ≈ 0.13 | — |
| +10 / class | 38 % | ≈ 0.12 | — |
| +25 / class | 61 % | ≈ 0.17 | mild upswing (within error) |
| +50 / class | 96 % | ≈ 0.17 | (2 seeds) |
| full shard (−1) | 100 % | ≈ 0.14 | doing 100 % of the work ≠ lower BER |

pooled honest floor (fig) **0.028**; η tight **0.064**; per-class A1 floors c3 **0.057** / c6 **0.114**.

**Headline (3 seeds): the price of invisibility is one +1/class increment.** Trigger-**only**
(N=0) overfits and blows up to **≈0.44**. But the first sliver of common data (+1/class,
**24 % effort**) already collapses BER to its plateau ≈0.11–0.13, and every larger budget —
up to and including a full honest shard — stays on that same plateau. So extra effort buys the
free-rider nothing: not lower BER, not higher. A free-rider that does ~a quarter of an honest
client's work looks exactly like one that does all of it. 

| trigger class | honest BER (6 seeds) | FR BER (3 seeds) | OVL | best balanced error | reading |
|---|---|---|---|---|---|
| **3 (easy)** | 0.057 | **0.037** | 0.88 | **0.500** | FR *cleaner* than honest -> **inseparable** |
| **6 (hard)** | 0.114 | **0.220** | 0.53 | 0.267 | FR noisier -> catchable only at FPR 0.40 |

### R9 — Isolated same-class pairs 

Honest client and free-rider on class 6 from seperate runs to compare the BER

- **[iso_c1.png](results/groupD/figs/iso_c1.png)** — class 1 (easy): honest A1 cid1 vs FR
  A2_reduced_c100_c17 cid1.
- **[iso_c7.png](results/groupD/figs/iso_c7.png)** — class 7 (easy): honest A1 cid7 vs FR A2 cid7.
- **[iso_c3.png](results/groupD/figs/iso_c3.png)** — class 3 (mid): honest A1 cid3 vs FR
  A3_reduced_c100_c36 cid3.
- **[iso_c6.png](results/groupD/figs/iso_c6.png)** — class 6 (hard): honest A1 cid6 vs FR A3 cid6.

*Axes:* BER vs round; **blue** = honest (6 seeds ±std), **orange** = free-rider (3 seeds ±std);
black dashed η tight 0.064, blue dashed η loose 0.264; yellow = warmup, green = calibration
window, grey dashed = free-riding starts (~round 12).

**Result — the free-rider embeds at least as cleanly as honest at every class except the hard
one, where it is noisier only by the key draw:**

| class | honest (converged) | free-rider (converged) | verdict |
|---|---|---|---|
| 1 (easy) | ~0.02 | **0.00** (flat for 40 rounds) | FR *cleaner*; inseparable |
| 7 (easy) | ~0.05–0.08 | **0.00** (flat) | FR *cleaner*; inseparable |
| 3 (mid) | ~0.03–0.08 | ~0.033 | tangled, FR ≤ honest; inseparable |
| 6 (hard) | ~0.10–0.15 | ~0.20–0.235 | FR *above* honest — but see key-lottery |

### R12 — non-IID TODO: BUG (RERUN this and re-analyze)

**Setup.** Dirichlet label skew; reduced free-riders on classes 3 & 6, cpc=5 (30 % data), η
recalibrated for non-IID (frozen 0.161, up from IID 0.064). **a01 = α 0.1** (most skewed),
**a03 = α 0.3**, **a10 = α 1.0** (least skewed).

- **[E3_a01_timeline.png](results/groupD/figs/E3_a01_timeline.png)** (α 0.1)
- **[E3_a03_timeline.png](results/groupD/figs/E3_a03_timeline.png)** (α 0.3)
- **[E3_a10_timeline.png](results/groupD/figs/E3_a10_timeline.png)** (α 1.0)

| α (skew) | η tight (frozen) | η loose (pooled) | honest floor c3 / c6 | FR mean (converged) | flagged? |
|---|---|---|---|---|---|
| 0.1 (a01) | 0.161 | 0.127 | 0.07 / 0.20 | ~0.11–0.13 | **no** (FR < η) |
| 0.3 (a03) | 0.161 | 0.127 | 0.07 / 0.20 | ~0.11–0.13 | **no** |
| 1.0 (a10) | 0.161 | 0.141 | 0.06 / 0.23 | ~0.11–0.14 | **no** |

**Finding (robust across all three).** Non-IID widens the honest floor and pushes η up to
0.161 (2.5× the IID 0.064). The 30 %-data free-rider then rides at ~0.11–0.13, below η tight
the whole post-defection period -> never flagged, sitting inside the honest-floor band. Skew does
not help detection; it hurts it (a wider floor = a taller ceiling to hide under). Confirms the
Group-E hypothesis: **label skew erases separability at least as thoroughly as IID.**

### Remaining groups  ⏳ NOT YET RUN — placeholders

Each will follow the same template (calibrated-threshold plot + BER-vs-round timeline with the
two η lines + separability JSON), and each subsection below is a stub to fill when its seeds land.

- **C1 — smoothing function (sin vs power).** *Proves:* does a different `f()` move the honest
  class floors? *Result:* **❌ FAILED — sin-smoothing crash, see R14.** Disabled in `run_now.sh`
  and excluded from `BATCH` until the `wm_f=sin` branch is fixed. Not blocking.
- **E — non-IID (Dirichlet α).** E1 honest floor under skew; E2 reduced-FR under skew; E3 α
  severity sweep. *Result:* **◑ PARTIAL — E3 reduced timelines done (R12), but the α sweep is
  buggy** (a01≡a03) and **E1/E2 (α=0.5) not yet run.** Fixed to sweep {0.1, 1.0}; delete the stale
  E3 dirs and re-run with E1/E2. See R12.
- **F — more clients than classes (capacity).** F1 honest 200-client/100-round floor; F2 forced
  class-sharing overlap; F3 Table IX repro (`client_train`) **+ held-out twin → memorisation
  gap**. *Expected:* forced sharing tangles marks; F3's gap shows the paper's "capacity" is
  largely memorisation. *Fill:* `F1_thresholds`, `F2_capacity_*`, `grade → Table IX`. *Result:*
  _TODO (probe-gated: needs the embedding probe to pass first)._
- **H — paper baselines (crude attacks the scheme SHOULD catch).** H1 fidelity (CIFAR-10 matches
  Table I/II); H3 previous-models FR and H4 Gaussian FR both caught (BER ≫ η); H5 previous-models
  on c100 = the money-plot positive control. *Expected:* all crude attacks light up near recall
  1.0 — the necessary contrast that makes "insiders are invisible" meaningful. *Fill:*
  `H_sep_*.json`, `grade → c10`. *Result:* _TODO._
- **I / J — adaptive-tap attacker.** One knob at a time (I: when/margin/data/scope/eta/coast) and
  several at once (J1–J4). *Proves:* an attacker that tracks η and coasts under it stays
  invisible at a fraction of compute, even self-estimating η. *Fill:* `tap_*` timelines,
  `tap_dynamics` frontier, `operating_point`. *Result:* _TODO (BATCH=IJ)._
- **V — verify-mode × N_T (Table VII/V memorisation).** V1 client_train vs held-out gap at
  N_T ∈ {1,10,50}; V2 FR trained on few trigger samples overfits → caught. *Expected:* small-N_T
  marks memorise and don't generalise (mirrors R8's trigger-only c3 = 0.60). *Fill:*
  `V2_sep_tn*.json`, `grade → memorisation gap`. *Result:* _TODO (BATCH=V)._
- **operating_point (money plot) & tap_dynamics.** `operating_point` is **BUILT and run — see
  R11**: with one deployable η, insider recall ≤ 0.17 at any usable FPR and 0.00 under the
  per-class oracle for the same-class families; nothing reaches the 0.9 target. Remaining: add the
  crude **H5** previous-models baseline so the positive control (~1.0) appears on the same axis,
  and `tap_dynamics` once I/J land. *Result:* _R11 done (insiders); H5 control + tap_dynamics TODO._

---


## EXPERIMENT PLAN - to be run

### Group A — proven baselines
| label | setting | proves (notes ref) | status |
|---|---|---|---|
| A1 | honest, cifar100, 10cl, 6 seeds | class difficulty; threshold calibration | done |
| A2 | reduced +5, classes 1,7, 3 seeds | non-sep at EASY classes (FR cleaner than honest) | done |
| A3 | reduced +5, classes 3,6, 3 seeds | non-sep at HARD classes | done | 
| A4 | sameclass, FR on class 6 (**1 free-rider**, cid 0; 10 clients total), 3 seeds | FR vs honest, SAME trigger class, same training | **done** — replot with `plot_sameclass_pair.py` (cid0 FR vs cid6 honest on class 6) |
| AK | sameclass, same key/message, FR on class 6, 3 seeds | FR vs honest, SAME trigger class, SAME key/message | **done (3 seeds)** — twin key applied; FR indistinguishable from honest twin (R7 addendum) |

### Group B — thresholds 
All computed offline from A1. TODO verify the implementation of thresholds - for now just use tight threshold 0.064 (from the 10 seed honest runs) and the loose threshold 0.264 (from the 10 seed honest runs client means).

### Group C — difficulty mechanism 
| label | setting | proves |
|---|---|---|
| C1 | honest, sin smoothing (WM_ALPHA=1.5708), 3 seeds | does a different f() move the floors? |
| C2 | class_probe on A1 (**offline** — reads `wm_benign_entropy/pmax/dominance` already in A1 result.json) | entropy/dominance vs accuracy correlation |

### Group D — the +N free-riding spectrum 
| label | setting | proves |
|---|---|---|
| D1 | reduced, classes 3,6, N ∈ {-1,0,1,2,5,10,25,50}, 3 seeds | price of invisibility; N=-1 (full data, still FR) is the anchor |

### Group E — non-IID 
| label | setting | proves |
|---|---|---|
| E1 | honest, Dirichlet α=0.5, 3 seeds | does label skew widen honest BER? |
| E2 | reduced, classes 3,6, α=0.5, 3 seeds | non-sep under non-iid |
| E3 | honest **+** reduced at α ∈ {0.1, 0.3, 1.0}, 3 seeds each | severity sweep: does worsening label skew widen the honest floor and further erase separability? (each α gets its own honest run so η recalibrates per-α offline) |
| (E requires the `n_trigger_samples` split before quoting — already logged in clients.py) |

### Group F — more clients than classes 
| label | setting | proves |
|---|---|---|
| F1 | honest, 200 clients, MORE ROUNDS (100), 3 seeds | capacity — but needs enough rounds to train |
| F2 | reduced, 200cl, classes 6,7, 3 seeds | forced class-sharing overlap |
| F3 | **Table IX** repro: cifar10, 50cl, `client_train` mode **+** held-out twin, 3 seeds | prove settings match the paper AND expose the memorisation gap (`paper_check.py`). **PROBE-GATED** — build with `PAPER_OK=1` only after the embedding probe passes |

### Group H — paper baselines 

| label | setting | proves |
|---|---|---|
| H1 | honest, **cifar10**, watermarked, 3 seeds | fidelity: watermark-accuracy and main-task accuracy match Table I/II |
| H2 | honest, **cifar100** (= **A1**, not rerun) | fidelity on the base config; cite `A1_honest_c100` |
| H3 | **previous-models** free-rider, cifar10, 3 seeds | the crude attack IS caught (FR BER ~0.5 ≫ η) — the detector works as the paper claims |
| H4 | **gaussian-noise** free-rider, cifar10, 3 seeds | second crude attack IS caught — baseline sanity |
| (Table IX capacity baseline lives in **F3**, `PAPER_OK=1`.) |

### Group G — detection policy (notes: currently untouched)
implement in `wm_verify.py` next week:
consequence of crossing η, k-warnings-before-flag, detection window.

### Deferred / paper reproduction
- R0/R1/R2 reproduction: only after the probe confirms peaky-softmax embedding.
- `client_train` twins: the memorisation control, once embedding is confirmed.
---
---

## N1. General setting / threat model 

**System.** A federated-learning system with a server and *N* clients training a shared
classifier over *n* classes for *R* rounds (FedAvg). Ownership/effort is protected by an
**output-layer, box-free, multi-bit watermark** (based on FareMark): each client *i* is
assigned a private key matrix `M_i` and message `B_i` of `m` bits, and embeds its mark by adding
a regulariser `λ·L_wm` to its local loss. The mark is **read only from the model's outputs**: the
verifier pushes `N_T` held-out trigger-class images through the model, applies a smoothing
function `f` (a power/sin map, Eq. 8) to the softmax, projects onto `M_i` (Eq. 10), and thresholds
the sign to recover `m` bits. Detection is a single scalar test: flag client *i* as a
free-rider iff its bit-error-rate `BER_i ≥ η`, with `η` calibrated on honest clients.

**Attacks (free-rider).** A client that wants the aggregated global model without paying the
full training cost, while keeping `BER < η` so it is never flagged. It acts like an honest client, holds a valid `(M_i, B_i)`, sees the global model each round, and can (a) train on a reduced shard, (b) train intermittently like a submarine ("tap"), (c) submit the unchanged global between taps ("coast"), and (d) estimate `η` from its own honest-phase BER. 
The whole strategy is to look exactly like a slightly-below-average honest client.

**Our claim:** The watermark cannot function as an effort/free-rider detector if, for every threshold `η` a server could deploy (calibrated on honest data alone), the honest-BER and free-rider-BER distributions
overlap enough that no `η` separates them at a useful operating point. We show this holds because:
1. **BER does not measure effort.** A reduced free-rider concentrates its shrunken data on the
   trigger class and embeds at least as cleanly as an honest client (A2/A4: FR BER ≤ honest).
2. **Apparent separability is a key/class lottery the server cannot observe** (A3/A4/AK: a ~3× BER
   swing at one fixed class from the key draw alone).
3. **The scheme's own threshold has no headroom** (+0.55σ, not 3σ) and its BER axis is quantised
   to `1/m`, so the calibrated `η` is degenerate and false-alarms 10–31 % of honest clients.
4. **An adaptive attacker closes any residual gap** by tracking `η` and coasting under it
   (Group I/J): even a self-estimated `η` with a small margin keeps `BER < η` at a fraction of the
   honest compute.

**Non-goal.** We do not claim the *mark* cannot be embedded (it can, robustly — that is Group H's
baseline). We claim that there exists no threshold `η` that can separate honest from free-rider clients when embedding watermarks in the output layer and that there exists an adaptive free-rider strategy that can exploit this to avoid detection using minimal effort.

## N2. Related work — output-layer watermarking lineage 

- **Uchida et al. (2017)** — the first DNN watermark: embeds bits in *weights* via a regulariser
  projected onto a secret matrix (white-box). Establishes the spread-spectrum "project onto a
  pseudorandom key" template that everything below reuses. [link](https://arxiv.org/abs/1701.04082)
- **Adi et al. (2018), "Turning your weakness into a strength"** — backdoor watermark: ownership
  proved by the model's *outputs* on trigger inputs (black-box), but single-bit/behavioural. [link](https://arxiv.org/abs/1802.04633)
- **BlackMarks (Chen, Rouhani, Koushanfar, 2019, arXiv:1904.00344)** — the first *multi-bit*
  black-box scheme: encodes the owner signature in the **distribution of output activations**,
  clustering classes to bits. [link](https://arxiv.org/abs/1904.00344)
- **Universal BlackMarks (IEEE SPL, 2023)** — the **direct ancestor of FareMark's reader**: it
  <cite index="26-1">applies a power function to the softmax output to map it from an impulse-like to a smooth distribution, then extracts watermark bits by projecting the output onto a pseudorandom key vector</cite>. FareMark's Eq. 8 (power smoothing) + Eq. 10 (projection) is this construction, moved into FL. [link](https://ieeexplore.ieee.org/document/10025674)
- **FedIPR (Li, Fan, Gu, Li, Yang, TPAMI 2022)** — the FL free-rider angle: client-side secret
  watermarks (feature + backdoor) that, unlike server-side schemes, can *identify free-riders*.
  This is the "watermark ⇒ free-rider detection" claim FareMark inherits and extends. [link](https://arxiv.org/pdf/2109.13236)
- **WAFFLE (Tekgul et al., SRDS 2021)** — server-side FL watermark; the survey literature notes
  server-side marks *cannot* police free-riders because the free-rider's model is just the global
  model. This is why the detection burden falls on *client-side* schemes like FareMark/FedIPR. [link](https://arxiv.org/abs/2008.07298)
- **FedSMW (2024)** and related FL schemes reuse the same power-function-on-softmax + projection
  reader, so the peaky-softmax failure we document is a property of the **reader**, shared across
  the family — not a FareMark-specific bug. [link](https://ieeexplore.ieee.org/document/10827540)


to read:
WAFFLE: good paper but the goal is not free-rider detection. Easy breakable i guess ? I think FedTracker is maybe more relevant
FedIPR: they consider the same free-riders as in FareMark :eyes: In my personal note i see "Depending on Greta's plots, she can have a look on this" :joy: so yes i recomand this one ! (positive point that i liked: Theorem 1, the range of watermark bit-length)
I like this survey: "When Federated Learning meets Watermarking: A  Comprehensive Overview of Techniques for  Intellectual Property Protection" (from 2023) if you want a general overview
the rest of the papers you mention are for centralized setting right ? (The first one is: Adversarial frontier stitching for remote neural network watermarking :rooster:) So maybe not the priority for next week, i'll say ?

**The gap our work fills.** None of these evaluate an *adaptive, effort-minimising insider* that
holds a valid key and tunes its behaviour to sit under `η`; the free-rider they test is the crude
Gaussian/previous-models attacker (caught trivially). Our threat model (N1) and the adaptive-tap
attacker (N4) are exactly that missing evaluation. 

## N3. Two reference thresholds on every timeline 

From now on **two** horizontal lines are drawn on every BER-vs-round timeline automatically
(`plots.py timeline`), so no plot shows a single cherry-picked η:
- **η tight = 0.064** — the frozen aggressive line the server actually used (`WM_ETA_FIXED`; below
  `1/m`, degenerate). Black dashed. Overridable with `--eta_tight`.
- **η loose = 0.264** — the loosest *sane* deployable rule = pooled μ+3σ over honest client-means.
  Blue dashed. If honest runs are passed via `--honest_in`, it is recomputed from them; else the
  `ETA_LOOSE_DEFAULT = 0.264` constant is used. Overridable with `--eta_loose`.

## N4. The adaptive-tap attacker — one knob at a time 

Built as `make_adaptive_tap_attack` in `clients.py`. Enabled with `attack=adaptive_tap`. Warmup/calibration
schedule is identical to the reduced attacker (honest for `W` rounds, calib window `[W-K, W-1]`,
free-ride from `W`), so it is directly comparable. sweep one knob at a time (Group I) or several (Group J):

| knob (config / env) | question it answers | values |
|---|---|---|
| `tap_eta_source` / `TAP_ETA_SOURCE` (+`tap_eta_k`) | **threshold estimation** — does the attack still work when the FR must *guess* η from its own honest-phase BER instead of being handed it? | `oracle` \| `self` |
| `tap_margin` / `TAP_MARGIN` | how far under η to aim (safety vs cost) | 0.0, 0.02, 0.05, 0.10 |
| `tap_when` / `TAP_WHEN` (+`tap_period`) | **when to tap** — react to a rising BER, tap on a fixed clock, or tap every round | `threshold` \| `every_k` \| `always` |
| `tap_max_coast` / `TAP_MAX_COAST` | force a tap after this many coasts (stealth-vs-safety cap) | 1, 2, 4, 999 |
| `tap_data_cpc` / `TAP_DATA_CPC` | **how much data per tap** | -1 (full), 0 (trigger-only), 1, 5, 25 |
| `tap_scope` / `TAP_SCOPE` | **how much of the model a tap trains** (cheaper backward) | `full` \| `block2` \| `block` \| `head` |
| `tap_coast_mode` / `TAP_COAST_MODE` | **how you free-ride between taps** | `resend` (global, zero compute) \| `decay` (resend own last tapped weights) |
| `tap_probe_holdout` / `TAP_PROBE_HOLDOUT` | held-out trigger images for the FR's self-BER probe | 64 (default) |

**Two OBSERVABLES you asked to see** are *measured, not set*. Each round the attacker records
`ber_before` (probe BER of the aggregated model, before it acts) and `ber_after` (after a tap) in
`self.trace`, so from any run you can extract:
- **fade time** — while coasting, how many rounds the mark takes to climb from post-tap back up to
  the target `η − margin` (how long a single tap "lasts"); and
- **recovery time** — after a tap, how many rounds until `ber_after` dips back under target.
These fall straight out of the `coast`/`tap` trace rows; a small extractor can be added to
`plot_all_thresholds.py`/`plots.py` once the runs land (no new file needed).

**Why this is safe for the running jobs.** `make_reduced_attack` and `make_tap_attack` are
**byte-identical** to the uploaded code; the only additions are the new factory and a new
`elif attack == "adaptive_tap"` branch that nothing reaches unless explicitly selected. All
`tap_*` config fields default to inert values and are ignored by every other attack. Pods clone
the repo at start, so the jobs already running are unaffected regardless; new families only appear
in the *next* manifest.

## N5. Every setting used
Base config for all CIFAR-100 experiments = **`config_idx 14`** (`submarine_resnet18_cifar100`),
overridden per run by env vars.

| setting | value | why this value |
|---|---|---|
| model / dataset | ResNet-18 / CIFAR-100 | the paper's CIFAR-100 row; the "known-good" config that reproduces embedding (73% acc) |
| num_clients | 10 | one client per used trigger class (0–9) -> no forced sharing in the base; sharing is studied deliberately in F |
| rounds / local_epochs | 50 / 5 | paper's §V-B settings |
| lr / batch / momentum / wd | 0.01 / 16 / 0.9 / 5e-4 | the config defaults that give the reproduced accuracy; held constant so results are comparable |
| watermark / λ / β / α / f | on / 5.0 / 0.6 / 0.4 / power | paper Eq. 11/14/8 defaults |
| m (bits) | auto = `max(2, n//10)` = **10** | l = n/m = 10 → reliable bits, negligible (0.2%) stuck rows, paper-reachable wm-accuracy |
| wm_num_triggers `N_T` | 50 | paper's fidelity setting; V sweeps it (1/10/50) |
| wm_trigger_mode | `class` (held-out shared) | the strict generalisation test; V compares against `client_train` (paper Table IX) and `client` |
| wm_balanced_keys | False (paper-faithful ±1) | keeps the structural stuck-bit artifact the paper has; `True` only for the F6/F7 demo |
| **attack knobs** | | |
| ATTACK | none / reduced / adaptive_tap | none = calibration; reduced = the thesis attacker (running); adaptive_tap = next-batch knobbed attacker |
| FREE_RIDER_IDS | e.g. `3,6` or `0` | pins exact cids; **overrides** `num_free_riders` (so A4/AK = 1 FR despite the config's default 2) |
| TRIGGER_CLASS_MAP | `0:6` | pins a FR onto an honest client's class (same-class control) |
| WM_KEY_TWINS | `0:6` (AK only) | FR takes the honest client's key+msg -> effort-only isolation |
| AUTOP_COMMON_PER_CLASS (cpc) | 5 | trigger + 5/other-class ≈ 31% data; the "+N" is swept in D |
| AUTOP_HONEST_UNTIL `W` / AUTOP_CALIB_ROUNDS `K` | 12 / 4 | forced-honest warmup then a 4-round calibration window before defection; matches the timeline shading |
| WM_ETA_FIXED | 0.064 (IID base) | frozen tight η (see N3); the offline sweep is the real verdict |

---
---
---

# RUN & PLOT REFERENCE 

**thresholds (all from the same A1 honest tail, last 20 of 50 rounds, m=10):**

| name | value | what is averaged before μ+3σ | why that value |
|---|---|---|---|
| tight / online-frozen | 0.064 | per seed: mean-over-clients each round → μ+3σ per seed → **average the etas** | σ is within-seed round-mean spread (narrow) |
| loose / per-client | 0.264 | **no averaging** — μ+3σ over individual client-round BERs | σ is the *real* per-client spread (~√10× bigger) |

## R3. Master experiment table

**Legend:** ▶ = running now (batch `./run_now.sh ACDEFH` → `WORKERS=6 PODS=2 ./submit_pool.sh`).
⏳ = to run (batch `BATCH=IJVF PAPER_OK=1 ./runbook.sh manifest` → `./runbook.sh submit`).
All families land in `results/<family>_rep<seed>_<ts>/result.json`. Plots are the runbook `plot`
phase (`RES=~/local/results ./runbook.sh plot`) unless noted.

| # | st | proves | family(ies) | build/run | plots produced |
|---|----|--------|-------------|-----------|----------------|
| A1 | ▶ | class-difficulty floors; threshold calibration; η seed-instability | `A1_honest_c100` (×6) | `./run_now.sh A` | `A1_thresholds(.png/.md)`, `A1_class_floors`, `A1_class_probe`, `A1_eta_stability` |
| A2 | ▶ | non-sep at EASY classes (FR cleaner than honest) | `A2_reduced_c100_c17` (×3) | `./run_now.sh A` | `A2_easy_timeline` (2-η), `A2_easy_sep.json`, `iso_c1`, `iso_c7` |
| A3 | ▶ | non-sep at HARD classes; catch only by flagging honest | `A3_reduced_c100_c36` (×3) | `./run_now.sh A` | `A3_hard_timeline`, `A3_hard_sep.json`, `iso_c3`, `iso_c6` |
| A4 | ▶ | same trigger class, own key (confounded by sharing) | `A4_sameclass_c100_c6` (×3) | `./run_now.sh A` | `A4_sameclass_timeline`, `A4_sameclass_sep.json` |
| AK | ▶ | same class + SAME key (effort-only isolation) | `AK_sameclass_samekey_c6` (×3) | `./run_now.sh A` | `AK_samekey_timeline`, `AK_samekey_sep.json` (verify `wm_key_twins -> 0:6`) |
| C1 | ▶ | does a different smoothing f() (sin) move the floors? | `C1_honest_sin_c100` (×3) | `./run_now.sh C` | `C1_class_floors`, `C1_class_probe` |
| D1 | ▶ | price-of-invisibility curve (N images/common class) | `D1_reduced_c100_c36_n{-1,0,1,2,5,10,25,50}` (×3) | `./run_now.sh D` | `D1_spectrum` sweep plot |
| E1 | ▶ | non-IID honest floor (label skew widens BER) | `E1_honest_niid_c100` (×3) | `./run_now.sh E` | `E1_thresholds`, `E1_class_floors` |
| E2 | ▶ | non-sep under non-IID (hard classes) | `E2_reduced_niid_c36` (×3) | `./run_now.sh E` | `E2_niid_timeline`, `E2_niid_sep.json` |
| E3 | ▶ | severity sweep: does more skew erase separability further? | `E3_{honest,reduced}_niid_*_{a01,a03,a10}` (×3) | `./run_now.sh E` | per-α timelines/floors |
| F1 | ▶ | capacity: 200 clients (forced sharing), honest | `F1_honest_nc200` (×3) | `./run_now.sh F` | `F1_thresholds` |
| F2 | ▶ | non-sep under >clients-than-classes | `F2_reduced_nc200_c67` (×3) | `./run_now.sh F` | `F2_capacity_timeline`, `F2_sep.json` |
| H1 | ▶ | fidelity: all-honest CIFAR-10 matches paper Table I/II | `H1_honest_c10` (×3) | `./run_now.sh H` | `grade` → paper_check c10 |
| H3 | ▶ | crude previous-models FR IS caught (baseline sanity) | `H3_prevmodel_c10` (×3) | `./run_now.sh H` | `H_sep_H3_prevmodel_c10.json` |
| H4 | ▶ | crude Gaussian FR IS caught (baseline sanity) | `H4_gaussian_c10` (×3) | `./run_now.sh H` | `H_sep_H4_gaussian_c10.json` |
| I_* | ⏳ | adaptive-tap, ONE knob at a time (when/margin/data/scope/eta/coast/maxcoast) | `I_<knob>_<val>_c36` (×3) | `BATCH=I ./runbook.sh manifest submit` | `tap_I_*` timelines (2-η) |
| J_* | ⏳ | adaptive-tap, several knobs at once (4 hypotheses) | `J{1..4}_*_c36` (×3) | `BATCH=J …` | `tap_J*` timelines |
| V1 | ⏳ | verify-mode × N_T → memorisation gap (Table VII) | `V1_verify_{client_train,client,class}_nt{1,10,50}_c100` (×3) | `BATCH=V …` | `grade` → memorisation gap |
| V2 | ⏳ | Table V: FR trained on few trigger samples → overfits, caught | `V2_tableV_attack_c36_tn{1,5,10,25,50,m1}` (×3) | `BATCH=V …` | `V2_sep_tn*.json` |
| F3 | ⏳ | Table IX capacity repro (client_train) + held-out twin | `F3_tableIX_c10_nc50(_heldout)` (×3) | `PAPER_OK=1 BATCH=F …` | `grade` → Table IX + gap |

**Still worth adding for a strong paper (not yet built):** a **consolidated operating-point figure**
(recall at a fixed 5% FPR across A2/A3/A4/AK/D/V2 on one axis — the "no threshold works" money plot),
and the **tap fade/recovery-time** extraction (the `ber_before/after` trace fields exist; a small
reader turns them into "rounds a tap lasts / rounds to recover"). Both are offline from existing JSONs.

## to run / plot reference
- **Plotting:** after the batch, `RES=~/local/results ./runbook.sh calibrate plot grade`. Timelines
  auto-draw both η lines; isolation pairs (`iso_c*`) come from A1 (honest) vs A3/A2 (FR) — separate
  runs, no same-class conflict.
---

# R5. Two new analysis plots (built into plots.py)

Both are offline (read existing `result.json`), added as `plots.py` subcommands, and wired into
`runbook.sh plot`. Also: the loose reference η is now **0.264** (`ETA_LOOSE_DEFAULT`, per-client
μ+3σ, ~2% FPR) so timelines bracket 0.064 (aggressive) .. 0.264 (lenient).

**1. `operating_point` — the "no threshold works" money plot.** Calibrates ONE deployable η on the
pooled honest BER for each FPR budget {1%, 5%, 10%}, then reads each attack's recall off that same
line (a real server has one η and cannot know a client's class in advance). Emits a grouped-bar PNG
+ a `.md` table with both the global (deployable) recall and a per-class **oracle** recall (η
calibrated on the FR's own class — an upper bound the server can't actually use). Crude baselines
(`H5_prevmodel_c100`) are the positive control: they light up near recall 1.0 while the reduced/
adaptive insiders sit near 0 at any usable FPR. Because BER is quantised to 1/m, exact 5% FPR is
often unattainable; the bar labels show the **actual** FPR each η achieves.
```
python scripts/plots.py operating_point --in 'results/*/result.json' \
  --honest_family A1_honest_c100 --tail 20 \
  --families A2_reduced_c100_c17 A3_reduced_c100_c36 A4_sameclass_c100_c6 \
             AK_sameclass_samekey_c6 D1_reduced_c100_c36_n5 V2_tableV_attack_c36_tn5 \
             H5_prevmodel_c100 --out figs/operating_point
```

**2. `tap_dynamics` — fade & recovery from the adaptive-tap trace.** Reads
`compute.per_client[fr].trace` (`ber_before/after`, `target`, `action`). ONE `--family` → a trace
plot (see a tap dip and the coast climb, with tap fraction / a-tap's-lifetime / drop-per-tap
annotated). MANY `--families` → the **stealth frontier**: tap fraction (compute actually spent) vs
rounds-a-tap-lasts (persistence) — lower-left-and-higher = a cheaper, longer-lived free-rider.
Metrics: `tap_fraction`, `rounds_between_taps` (fade time), `ber_drop_per_tap` (recovery),
`fade_per_coast`, `stayed_below_target`.
```
python scripts/plots.py tap_dynamics --in 'results/*/result.json' --family J1_cheapest_c36 --out figs/tap_J1
python scripts/plots.py tap_dynamics --in 'results/*/result.json' --out figs/tap_frontier \
  --families I_when_threshold_c36 I_margin_m005_c36 I_data_n0_c36 I_scope_head_c36 J1_cheapest_c36 ...
```

**Master-table additions:**

| # | st | proves | family(ies) | build/run | plots |
|---|----|--------|-------------|-----------|-------|
| OP | plot | one deployable η, recall≈0 for insiders at any usable FPR (the headline) | (uses A2/A3/A4/AK/D/V2 + H5) | `plots.py operating_point` | `operating_point(.png/.md)` |
| TD | plot | adaptive FR holds the mark for a fraction of the compute; stealth frontier | (uses I_*/J_*) | `plots.py tap_dynamics` | `tap_dyn_*`, `tap_frontier(.png/.md)` |
| H5 | ⏳ | crude previous-models FR on **c100** IS caught (same-dataset money-plot control) | `H5_prevmodel_c100` (×3) | `./run_now.sh H` | feeds `operating_point` |

The earlier "still worth adding" note is now **done**: both the consolidated operating-point figure
and the tap fade/recovery extraction are built and tested.
