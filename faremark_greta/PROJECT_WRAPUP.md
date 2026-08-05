# Output-Layer Watermarking Cannot Detect Free-Riders in Federated Learning 

> For the FareMark family of *output-layer (box-free) watermarks*, there is **no detection threshold η that separates honest clients from free-riders**, and an **adaptive free-rider ("the submarine") can stay under any deployable η while doing only a small fraction of the honest training work**.


**Reading the status tags used throughout:**
- ✅ **Done (3+ seeds, table-ready)** — a finalized result that can go in the paper.
- ◑ **Done (1 seed, shape only)** — the effect is demonstrated but must be repeated at ≥3 seeds before it is quoted with numbers.
- ⏳ **To run** — planned, not yet executed.
- ⚠️ **Flag** — a caveat, conflict, or known instability a reader must know about.

---

## Table of contents

1. [Introduction — what the project is and why it matters](#1-introduction)
2. [The FareMark paper — what we build on and what we attack](#2-the-faremark-paper)
3. [Threshold analysis + class difficulty](#3-threshold-analysis--class-difficulty)
4. [Reduced-data free-riding](#4-reduced-data-free-riding)
5. [The submarine — adaptive reduced-data free-rider (main attacker, in progress)](#5-the-submarine)
6. [Non-IID setting](#6-non-iid-setting)
7. [Related work](#7-related-work)
8. [Master list of what is left to run](#8-what-is-left-to-run)
9. [Canonical configuration reference](#9-canonical-configuration-reference)

---

<a name="1-introduction"></a>
## 1. Introduction — what the project is and why it matters

### 1.1 The five ideas you need before anything else

**(a) A neural network classifier.** A *classifier* is a program that looks at an input (here, a small 32×32-pixel colour image) and outputs a guess about which of several categories it belongs to. A modern classifier is a *neural network*: a long stack of mathematical layers full of tunable numbers called **parameters** (or "weights"). **Training** means showing the network many labelled example images, measuring how wrong its guesses are (a number called the **loss**), and nudging every parameter a tiny bit in the direction that reduces the loss. One nudge per small batch of images is one **step**; one full pass over all the training images is one **epoch**. The network we use, **ResNet-18**, has about **11 million** parameters. Its final layer produces one score per class; those scores are squashed by a function called **softmax** into a list of probabilities that sum to 1 (e.g. "80% cat, 5% dog, …"). Everything the watermark does happens on this final probability list, the **softmax output**.

**(b) The datasets.** We train on **CIFAR-100**: 60,000 tiny photos (50,000 for training, 10,000 for testing) sorted into **100 categories** ("classes") such as *apple, bicycle, dolphin, …*. A closely related smaller dataset, **CIFAR-10**, has the same images grouped into just **10 classes**. The number of classes, written **n**, matters a great deal here (Section 3).

**(c) Federated learning (FL).** Normally you gather all training data in one place and train one model. In FL the data cannot be centralized (privacy, regulation, bandwidth). Instead there are **N separate clients** (think: N phones or hospitals), each holding its own slice of data. Training runs in **rounds**. In each round: the central **server** sends its current shared model (the **global model**) to every client; each client trains it a little on its own private data and sends back its updated copy (the **local model**); the server **averages** all the returned copies into a new global model. This averaging rule is called **FedAvg**. After many rounds the global model is as good as if all the data had been pooled — but no raw data ever left a client.

**(d) The free-rider problem.** In FL, a dishonest client can try to get the finished, valuable global model **without doing its share of the work**. It pretends to participate — sends *something* back every round — but that something is cheap: random noise, a lightly-edited copy of last round's model, or a model trained on far less data than everyone else. This cheater is a **free-rider**. Crucially, a free-rider does **not** try to break the model (that would be a different attack); it just wants the good model for free. The server would like a **detector** that flags free-riders so they can be removed or denied the model.

**(e) Watermarking as a proposed detector.** A **watermark** is a hidden signal embedded into a model that proves who made it — like an invisible signature. The **FareMark** paper (the work we build on, Section 2) proposes using a watermark as a *free-rider detector*: every honest client is required to embed its own private watermark into the model during training. The reasoning is: *embedding a watermark requires real training, so only clients that actually trained can produce a detectable watermark; a free-rider that skipped the work will have no watermark and gets caught.* The server reads each client's submitted model, tries to extract that client's watermark, and if the watermark is too corrupted, flags the client as a free-rider.

### 1.2 What this project claims

**This project's entire purpose is to show that idea (e) does not work** for *output-layer* watermarks (defined in Section 2.2). We make a **two-sided argument**:

- **Negative result (the watermark is not a valid detector).** No matter what threshold the server picks, the "how corrupted is the watermark" measurement of honest clients and of free-riders **overlaps so heavily that no threshold separates them** at any useful operating point. In fact, at some classes the free-rider's watermark is *cleaner* than the honest client's. (Groups A, D, E — all done at 3+ seeds.)

- **Constructive attack (the submarine).** Even if you *pretend* the detector works, we build an **adaptive free-rider** that watches the model each round, re-embeds its watermark only when it is about to fade, and otherwise submits a "coasting" model that carries the mark for free — evading detection while training on a small fraction of the honest workload. (Groups I, J, NOW — in progress; the core proof-of-concept exists at 1–3 seeds.)

**Together:** output-layer, watermark-threshold free-rider detection is **both theoretically unsound and practically evadable**.

### 1.3 Key measurement definitions (used everywhere below)

- **BER — Bit-Error-Rate.** A watermark here is a string of **m bits** (0s and 1s). To check it, the server re-extracts those m bits from the model and counts how many came out wrong. **BER = (wrong bits) / m.** It ranges from 0 to 1. **BER = 0 means a perfect watermark. BER = 0.5 means a coin-flip — i.e. no watermark at all** (a random model gets half its bits right by luck). BER = 1 means every bit is flipped. So the meaningful "no mark" value is **0.5, not 1.**
- **η (eta) — the detection threshold.** A single number the server picks in advance (FareMark paper claims that this threshold is pre-calibrated on honest clients). Rule: **flag client *i* as a free-rider if its BER ≥ η.** The whole thesis is about whether *any* good η exists.
- **FPR — False-Positive Rate.** The fraction of **honest** clients wrongly flagged as free-riders. The server wants this near 0 (you cannot go around accusing honest participants).
- **Recall (TPR — True-Positive Rate).** The fraction of **free-riders** correctly caught. The server wants this near 1. FPR and recall trade off against each other as you move η.
- **Best balanced-error.** The lowest possible error any threshold can achieve, averaging the two mistake types, *even if you let the threshold cheat by peeking at the free-riders*. **Best balanced-error = 0.5 means the two groups are perfectly inseparable — no threshold beats a coin flip.** This is the single most important number in the negative result.
- **OVL — Overlap coefficient.** How much the honest-BER histogram and the free-rider-BER histogram physically overlap. **OVL = 1.0 means the two distributions are identical** (nothing can separate them); OVL = 0 means they are completely disjoint (a perfect detector exists).
- **Seed.** Every experiment uses a random-number **seed** — one integer that determines all the "random" choices (which images go to which client, the initial weights, the secret watermark keys, etc.). Re-running with a *different* seed is like re-running the whole experiment from scratch under new luck. **A result is only trustworthy if it holds across several seeds.** Our standard is **≥ 3 seeds; ideally 10.** Seeds are consecutive integers `1000, 1001, 1002, …`.

### 1.4 The experiment groups at a glance

| Group | What it proves | Status |
|---|---|---|
| **A** | Class-difficulty floors; threshold calibration; threshold instability; free-rider is often *cleaner* than honest | ✅ 3 seeds (A1 honest = 6 seeds) |
| **D** | "Price of invisibility": how little data a free-rider needs to become undetectable | ✅ 3 seeds (largest budget = 2 seeds) |
| **E** | Non-IID (realistic uneven data) makes detection **worse**, not better | ✅ 3 seeds |
| **I** | First adaptive-tap attacker; the effort dial works; naïve "coasting" fails | ◑ 1 seed (shape only) |
| **J** | The **graft-coast** mechanism ships and works; J2 is a confirmed submarine | ◑ 1 seed (shape only) |
| **NOW** | First 3-seed submarine (J2 confirmed at 3 seeds; J5 crashed) | ◑ partial — see Section 5 |
| **H, V, F, C** | Positive controls & robustness | ⏳ to run |

All finalized experiments use **CIFAR-100** at present. **A CIFAR-10 replication is requested for the paper** (Section 8) because CIFAR-10 is the more common benchmark and is the dataset FareMark reports its headline numbers on.

---

<a name="2-the-faremark-paper"></a>
## 2. The FareMark paper — what we build on and what we attack

### 2.1 What FareMark is

**FareMark** (Li et al., *IEEE Internet of Things Journal*, Vol. 12 No. 18, 2025 — "FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning Model") proposes exactly the detector described in Section 1.1(e). Its pitch: the server assigns each client a unique **trigger class** (one of the n categories, e.g. client 1 → "dog", client 2 → "cat"). During training every honest client embeds a private multi-bit watermark that is readable **only from the model's softmax outputs on images of its trigger class**. The server can then, each round, push a handful of trigger-class images through each client's submitted model, read out the watermark, and check it. Honest clients (who trained) produce clean watermarks; free-riders (who did not) cannot, and are flagged. FareMark reports near-perfect watermark recovery, minimal loss of accuracy, and good free-rider detection, and shows robustness to fine-tuning, pruning, quantization, and differential privacy.

### 2.2 The three key design choices FareMark makes (and why they matter to us)

1. **Output-layer / "box-free" reading.** The watermark is read **only** from the model's output probabilities — never from its internal weights. (In watermarking jargon, "box-free" / "black-box" = you only need to query the model and look at outputs; "white-box" = you need to open the model and inspect weights.) This is convenient for verification but, as we show, it is the source of the failure: the signal lives in a fragile corner of the softmax output.

2. **A per-client secret key + message.** Each client *i* holds a private random **key matrix `M_i`** (entries +1 or −1) and a private **message `B_i`** (a string of m bits). Both are secret and unique. This is how many clients can share one model without their watermarks colliding.

3. **A "smoothing" trick to make embedding possible.** A well-trained classifier is *over-confident*: on a trigger-class image it puts almost all probability on one class and near-zero on the other 99. That leaves no "room" in the output to hide a multi-bit signal. FareMark therefore passes each probability `p` through a **smoothing function `f`** — the paper's main choice is a **power function `f(p) = p^α` with 0 < α < 1** (they use **α = 0.4**) — which *inflates the tiny tail probabilities* so the hidden bits have somewhere to live. (The paper also floats a sine function `f(p) = sin(αp)`, Eq. 9; we show in Section 3.5 it barely smooths at all.)

### 2.3 How the watermark works, step by step (the reader family)

For client *i* with trigger class *c*, on a trigger-class image:
1. **Group.** Take the n-dimensional softmax output `P` and cut it into **m groups** of length **l = n / m** each. (CIFAR-100: n = 100, m = 10, l = 10.)
2. **Smooth.** Apply `f(p) = p^0.4` to each probability, inflating the tail.
3. **Project.** For each group *k*, multiply the smoothed probabilities by the secret key row and sum: `z_k = Σ_j f(p_j) · M_{k,j}`. (Paper Eq. 13.)
4. **Threshold to a bit.** `bit_k = 1 if z_k ≥ 0, else 0`. Averaged over several trigger images. (Paper Eq. 15.)
5. **Embed** by adding a penalty `λ · L_wm` to the client's training loss (`λ = 5`), which pushes the softmax tail into the shape that reproduces the client's secret message `B_i`. (Paper Eq. 11–12.)
6. **Detect** by re-extracting and computing **BER**; flag if BER ≥ η. (Paper Eq. 16.)

FareMark also adds a **memory-enhanced local update** (paper Eq. 14): because FedAvg *averaging* would otherwise wash a single client's watermark out, each client blends in its own past trajectory with a factor **β = 0.6** to keep its mark alive through aggregation.

### 2.4 The specific FareMark limitations this project tests and attacks

We attack **three** weaknesses that FareMark's own evaluation leaves unexamined:

- **(i) The threshold η.** FareMark says to set η at "μ + 3σ" of the honest error rate (mean plus three standard deviations). We show (Section 3) that on a 100-class problem this recipe lands in a **degenerate** region (η < 1/m), that the *only* non-degenerate variants either flag a large fraction of honest clients or catch no free-riders, and that η is **unstable across seeds** (it wobbles ~30% from the secret-key lottery alone). There is **no η with both low FPR and useful recall.**

- **(ii) The non-IID (realistic, uneven-data) setting.** FareMark, like most FL watermarking papers, evaluates mostly on **IID** data (every client holds a uniform random slice, so all classes are equally represented everywhere). Real FL is **non-IID**: clients have skewed, uneven class distributions. A natural defence of FareMark would be "sure, but in messy real FL the honest and free-rider clients would separate." We show (Section 6) the opposite: **non-IID makes detection strictly worse**, because clients are usually *data-starved on their own trigger class* and thus already embed badly — the free-rider hides in that honest failure.

- **(iii) The "crude" free-rider attackers.** FareMark only tests **weak** free-riders: ones that submit Gaussian noise or a re-extrapolated copy of previous global models (paper Eqs. 17–18), plus a "train-then-attack" and a "few-trigger-sample" client (paper Tables IV–V). These are all *trivially* caught precisely because they don't embed a real watermark. FareMark **never tests a free-rider that legitimately holds its own key and adaptively does just enough real work to keep its watermark alive under η.** That is exactly our **submarine** (Section 5) — the missing evaluation.

### 2.5 What we do NOT dispute

We do **not** claim the watermark can't be embedded — it can, robustly. On IID CIFAR-100 the honest scheme reaches ~73% test accuracy and embeds cleanly on most classes. We reproduce FareMark's embedding faithfully; our positive-control experiments (Section 8, Group H) confirm that a *crude* free-rider that skips training really is caught. Our claim is narrower and sharper: **(a) no deployable η separates honest from free-rider, and (b) an adaptive free-rider exploits this to evade at minimal effort.**

---

<a name="3-threshold-analysis--class-difficulty"></a>
## 3. Threshold analysis + class difficulty

**Group A. Status: ✅ done, 3 seeds (A1 honest calibration = 6 seeds).**

This is the foundation of the negative result. It establishes two things: (1) different trigger classes are **intrinsically** easy or hard to watermark — creating a per-class **BER floor** the honest client cannot beat — and (2) no threshold η calibrated on honest clients does the job.

### 3.1 Context — why "class difficulty" exists

The watermark hides in the **tail** of the softmax (the small, non-winning probabilities). Some classes naturally produce a **flat** softmax (the model is unsure, many classes get moderate probability) — a flat output has a rich, shapeable tail, so the mark embeds cleanly → **low BER (easy class)**. Other classes produce a **peaky** softmax (the model is very confident, one class ≈ 1, the rest ≈ 0) — a peaky output has a structureless tail, so the hidden bits become coin-flips → **high BER (hard class)**. Whether a class is easy or hard is a property of the **shape of the output distribution**, *not* of how accurately the model classifies that class. This distinction is what defeats the natural objection "your hard classes are just badly-classified classes."

We quantify output shape with two numbers, both computable from the softmax alone:
- **Entropy** `H = −Σ p·ln p` — high = flat/unsure.
- **Dominance** `= f(p_max) / Σ f(p)` — high = peaky (this is exactly the quantity FareMark's Eq. 10 constrains).

Empirically, entropy and dominance **track BER** (correlation |r| ≈ 0.6–0.7), while **classification accuracy tracks BER only weakly** (|r| ≈ 0.05–0.4). That gap is the point: difficulty is about output *shape*, not model quality.

### 3.2 Experimental setup (A1 — the honest calibration run)

- **Model / data:** ResNet-18 on CIFAR-100, IID partition (each of the N = 10 clients gets a uniform random 5,000-image shard).
- **Clients:** 10 honest watermark clients, one per trigger class (classes 0–9). No free-riders in A1 — this run *calibrates* what "honest" looks like.
- **Watermark:** m = 10 bits, l = 10, smoothing `f(p) = p^0.4`, λ = 5, β = 0.6, N_T = 50 verification images (held-out **test** images — a strict generalisation test, not the images the client trained on).
- **Rounds:** 50. **Seeds:** 6 (1000–1005).
- **Verified from the uploaded `A1_honest_c100_rep0result.json`:** final test accuracy 73.24%, mean honest BER ≈ 0.075–0.079, m = 10, l = 10, unembeddable fraction = 0.0.

### 3.3 Results — the per-class BER floors

The plot **[A1_class_floors.png](results/groups/figs/A1_class_floors.png)** shows honest BER over 50 rounds for all 10 trigger classes, aggregated over 6 seeds, with each class's **converged-tail floor** (mean BER over the last 20 rounds) printed in the legend. These floors are the ceiling a free-rider only has to reach to be indistinguishable from honest.

**Per-class honest BER floors (6-seed, from `A1_class_floors.png`):**

| Trigger class | Honest BER floor | Difficulty |
|---|---|---|
| cls 8 | 0.001 | very easy |
| cls 9 | 0.002 | very easy |
| cls 1 | 0.020 | easy |
| cls 0 | 0.025 | easy |
| cls 2 | 0.028 | easy |
| cls 5 | 0.037 | easy–medium |
| cls 3 | 0.057 | medium |
| cls 7 | 0.061 | medium |
| cls 4 | 0.094 | hard |
| cls 6 | **0.114** | **hardest** |

**Span: 0.001 → 0.114 across classes** — a **>100× range** in how well the *same honest scheme* embeds, purely from which class you were assigned.

### 3.4 Results — the threshold sweep (why every η fails)

We computed every reasonable recipe for η from the honest tail (6 seeds). Each row is a different rule for turning "honest BER statistics" into a single threshold; for each we report the resulting honest FPR (how many honest clients it wrongly flags), the **headroom** (how many honest standard deviations of safety margin it leaves — a proper industrial "3σ" control chart wants ≥ 3), and whether it is **degenerate** (η < 1/m = 0.10, so it does no real work).

| Rule for η | η value | Honest FPR | Headroom (σ) | Degenerate? |
|---|---|---|---|---|
| median + 3·MAD | 0.000 | 100% | −0.59σ | yes (collapses) |
| **coded (FareMark's own μ+3σ over round-means)** | **0.084** | **31%** | +0.55σ | **yes (< 0.10)** |
| pooled μ+3σ | 0.108 | 10% | +0.87σ | no |
| trimmed-10% μ+3σ | 0.160 | 10% | +1.57σ | no |
| honest p95 | 0.200 | 10% | +2.12σ | no |
| adaptive σ-clip | 0.224 | 2% | +2.45σ | no |
| **loose (per-client μ+3σ)** | **0.264** | **2%** | **+3.00σ** | no |
| honest p99 | 0.300 | 2% | +3.48σ | no |

**How to read this table (the core of the negative result):**
- **FareMark's own recommended rule ("coded," μ+3σ) lands at η ≈ 0.084, which is below 1/m = 0.10 → it is degenerate.** The carefully-calibrated threshold reduces to the trivial "flag if any single bit is wrong," and even then it flags **31% of honest clients**. Its headroom is +0.55σ, not the +3σ the recipe claims — the "3σ" math breaks on quantised, floor-limited BER.
- **The only rule that achieves a genuine +3σ of safety is "loose" at η = 0.264.** But 0.264 is *so high* that it sits **above every free-rider we build** — it catches essentially nothing (Sections 4–5).
- **There is no middle η with both low FPR and useful recall.** Every threshold is either degenerate/high-FPR, or safe-but-catches-nothing.

**The two frozen reference lines used on every timeline plot in this project** are therefore:
- **η_tight = 0.064** — the degenerate end (μ+3σ over round-mean honest BER, from a 10-seed calibration). Any "recall" reported at this line is an artifact of flagging honest clients.
- **η_loose = 0.264** — the only genuine 3σ operating line (per-client μ+3σ). This is the line that *actually matters*: if a free-rider's BER stays under 0.264, it evades a properly-calibrated detector.

The threshold also **wobbles ~30% from seed to seed** purely from the secret-key/message lottery — which is why it must be frozen from a multi-seed calibration rather than recomputed live (a live threshold judging the very round it was calibrated on would be circular).

### 3.5 BER ≠ accuracy — the "cleaner free-rider" paradox

Plots **[iso_acc_c6.png](results/figs/iso_acc_c6.png)** and **[iso_acc_c7.png](results/figs/iso_acc_c7.png)** make a counter-intuitive but central point. On an isolated easy class (cls 6 shown here as class "6"/"7" trigger panels), the **free-rider has the *lower* BER (cleaner watermark) AND the *higher* trigger-class classification accuracy**, while the **honest client has the *higher* BER AND ~0% trigger-class accuracy** — yet both runs reach the *same* ~72% overall test accuracy (right panel of each plot). This is not a contradiction:
- BER measures **sign-alignment of the smoothed softmax tail with a secret key**, not whether the top guess is correct. The mark lives in the tail *shape*.
- FareMark's scheme **requires suppressing the trigger class's own probability below 0.5** to embed at all (its Eqs. 4–6, 10). So a watermarked model is *designed* to be un-confident on its own trigger class → **low trigger-class accuracy is expected, not a bug.**
- The honest client (≈50 trigger images diluted in a 5,000-image shard, fighting 9 other clients through averaging) **over-suppresses** its trigger class to accuracy 0 while its tail stays unstable → BER floor ~0.11. The free-rider (a small, concentrated shard, re-embedding aggressively every round) fits *both* objectives → cleaner mark **and** some recovered trigger accuracy.

⚠️ **A cost FareMark never reports:** the honest client's trigger-class accuracy **collapses to 0**. FareMark only ever publishes *overall* test accuracy (its Table I), which barely moves because sacrificing ~10 of 100 classes is invisible in a 100-class average. The right panels of `iso_acc_c6/c7.png` show this directly.

### 3.6 Same-class and same-key controls (isolating the cause)

To rule out "maybe the free-rider just got an easier class," Group A includes tight controls:
- **iso plots (isolated, separate runs):** at easy classes 1 and 7 the free-rider's mark drops to **0.00 and stays there** (cleaner than honest). At hard class 6 the ordering is a **key lottery** — one key draw puts the FR above honest (`iso_c6`), a different draw puts it below (`iso_c6_A4_cleaner`). The server cannot see the key, so it cannot know which case it is in. [iso_c6_A4_cleaner.png](results/groups/figs/iso_c6.png) and [iso_c1.png](results/groups/figs/iso_c1.png) show the two different difficulty classes.

### 3.7 Reference code / theory for this section

- **BER, η, detection:** `watermark.py` — `extract_bits` (average-then-sign, Eq. 15), `bit_error_rate` (Eq. 16), `detected(ber, η) = ber < η`. Documented degeneracies: a perfect mark gives μ = σ = 0 → η = 0 → everyone flagged; any η < 1/m is the same detector.
- **Key lottery / stuck bits:** because probabilities and `f(p)` are ≥ 0, a key row that comes out **all-same-sign** forces its bit to a constant regardless of input — structurally unembeddable. `P(same-sign row) = 2^(1−l)`: at l = 10 (our CIFAR-100 case) this is 0.2% (negligible); at l = 5 it is 6.25%; at l = 2 it is 50%. This is why the number of bits m must be chosen carefully (Section 3.9).
- **Threshold sweep:** `detection.py` computes all rules offline from the honest tail.

### 3.8 Why m = 10 (and why CIFAR-10 is different — the `m·l = n` squeeze)

The bits and the group length are locked together by **m · l = n**. More bits (m↑) means shorter projections (l↓), which means **noisier bits and more stuck rows**. Fewer bits means coarser BER (with m = 2, BER can only be 0, 0.5, or 1). For CIFAR-100 (n = 100), **m = 10, l = 10** is the sweet spot: reliable bits, only 0.2% stuck rows.

### 3.9 What is needed to finish Section 3

1. ⏳ **Group H (H5): the positive-control bar.** Run the *crude* previous-models free-rider on CIFAR-100 at 3 seeds to show it *is* caught at recall ≈ 1.0. This completes the operating-point figure by giving the "detector works on lazy cheaters" contrast bar. **Not yet run.**
2. ⏳ **CIFAR-10 replication** of the class-floor + threshold sweep (3 seeds min, ideally 10), including the m = 1 forced-single-bit finding.
3. ⚠️ Re-run any single-seed A-group plot that is being quoted; only 6-seed floors are table-ready.

---

<a name="4-reduced-data-free-riding"></a>
## 4. Reduced-data free-riding

**Group D. Status: ✅ done, 3 seeds (largest data budget = 2 seeds).**

Section 3 showed no threshold separates honest from free-rider *in principle*. Group D shows the **static** free-rider that exploits this: the **reduced-data free-rider**, which trains honestly but on a drastically shrunken dataset, concentrated on its trigger class. The question Group D answers precisely: **how little real work does a free-rider need to become undetectable?**. It exploits the fact that training just the trigger sample class and a few other images is enough to keep the watermark alive, while skipping the rest of the honest workload. By training less images, the free-rider takes less steps per round (keeping the same batch size but with less data), and thus also seeing more trigger samples eahc step, it can keep the watermark alive while doing a fraction of the honest work.

### 4.1 Context and the attacker's idea

An honest client trains on its full 5,000-image shard, spread across all 100 classes, every round. The reduced free-rider instead keeps **all of its trigger-class images plus only `+N` random images per other class** — a tiny, trigger-heavy diet — and trains on that through the normal honest embedding path. The parameter **`cpc` ("common-per-class")** is that `+N`:
- `cpc = 0`: trigger-class images only (the laziest — the positive control - proven also by FareMark to overfit).
- `cpc = 1`: all trigger images + 1 image per other class ≈ **24% of an honest client's data**.
- `cpc = 5`: all trigger images + 5 per other class ≈ **31% of honest data**.
- `cpc = -1`: full honest shard (100% effort — the "effort anchor," still labelled a free-rider for comparison).

Because the reduced free-rider still runs the *real* embedding loss, it produces a *real* watermark — it just does it cheaply and with a trigger-class focus.

### 4.2 Experimental setup

- Same substrate as Group A (ResNet-18, CIFAR-100, N = 10, IID, m = 10, N_T = 50, β = 0.6, λ = 5).
- **Two free-riders**, pinned to **class 3 (medium, honest floor ≈ 0.057)** and **class 6 (hardest, honest floor ≈ 0.114)** — the ID set `FREE_RIDER_IDS = 3,6`, fixed across seeds.
- **The `+N` spectrum swept:** cpc ∈ {−1 (full), 0, 1, 2, 5, 10, 25, 50}. 
!!! TODO: reduce the +25 and +50 points and just run 0 to full shard because the plateau is already clear at +10.
- **Rounds:** 50. **Seeds:** 3 (cpc = 50 has 2). Reference lines η_tight = 0.064, η_loose = 0.264.

### 4.3 Results — the "price of invisibility" curve

The plot **[D1_spectrum.png](results/groups/figs/D1_spectrum.png)** has two panels: (top) free-rider BER over rounds for each data budget; (bottom) the converged BER versus data budget with error bars, against the η_tight and η_loose lines.

**What the curve shows:**
1. **Trigger-only (cpc = 0) overfits and is caught** — BER ≈ 0.44, *above both* η lines. This is the **positive control**: a lazy free-rider that trains only on its trigger images overfits to those specific images, the mark fails to generalise to the server's held-out verification images, and it is detected. This reproduces FareMark's own Table V finding.
2. **Adding just +1 image per class (~24% effort) collapses BER to a flat plateau (~0.11–0.13)** that *every larger budget also sits on.* In other words, there is **no gradual "you get caught less as you work more" gradient** — there is a cliff. Below the cliff (trigger-only) you are caught; one step past it (cpc ≥ 1) you are already at the honest floor and stay there no matter how much more you add.
3. **That plateau sits below η_loose (0.264)** and lands right in the band of the honest client's own floor on those classes → **inseparable from honest.**

**The exact +5/class per-class numbers (`D1_sep_n5.json`, ≡ Group A3):**

| Class | Free-rider BER | Honest BER | Separable? |
|---|---|---|---|
| **class 3** (medium) | **0.037** | 0.057 | No — FR *cleaner* |
| **class 6** (hardest) | 0.220 | 0.114 | Catchable **only at 40% honest FPR** |

So even at the *hard* class, the only way to flag the free-rider is to simultaneously flag 40% of honest clients — which no operator can do.

### 4.4 Reference code / theory

- **Reduced attacker:** `make_reduced_attack` (`attack=reduced`) in `clients.py`. It is honest through a warmup, then every round trains the reduced `TensorDataset` (`trigger_train + cpc images × each common class`) via the honest embed path.
- **Shared reduced-loader + self-probe machinery:** `_SimpleFRMixin._prepare(cpc, holdout)` scans the shard once, splits trigger vs common images, and builds the reduced dataset.
- **Compute meter:** records **image-passes** = `images × epochs`. Confirmed from the uploaded D1 JSON: an honest round = 25,000 passes; a cpc = 1 reduced round = 765 passes (≈ 3% of an honest round), giving the 24% overall effort ratio.

### 4.5 Conclusion so far (Section 4)

- A free-rider needs only **~24% of honest effort** (cpc = 1) to reach the honest BER floor and become **statistically inseparable** from honest clients.
- **BER does not measure effort.** Past a tiny cliff, more work does not lower BER — it plateaus at the honest floor. The only detectable free-rider is the *trigger-only* one that does almost nothing (the positive control).
- This is the **static** version of the impossibility: even a non-adaptive cheater is safe at ~a quarter of the workload.

### 4.6 What is needed to finish Section 4

1. ⏳ **CIFAR-10 replication** of the +N spectrum (3 seeds, ideally 10) — requested for the paper.
2. ⏳ **V2 (paper Table V complement):** the few-trigger-sample free-rider positive control at 3 seeds, to pair with the cpc = 0 result. **Not yet run.**
3. ✅ The IID CIFAR-100 spectrum is complete and table-ready (with the +50 point at 2 seeds — a third seed would tidy it).
4. ⏳ TODO replot the spectrum with up to +10 instead of +25 and +50, because the plateau is already clear at +10. This is a cosmetic change to make the figure more readable. 

---

<a name="5-the-submarine"></a>
## 5. The submarine — adaptive reduced-data free-rider (the main attacker this paper presents)

**Groups I, J, NOW. Status: ◑ proof-of-concept exists (J2 at 1 seed; NOW = J2 at 3 seeds confirmed, J5 crashed). The self-sufficient version is not yet finished — see Section 5.7.**

⚠️ **This is the paper's headline attacker and it is still under construction. Everything in this section that is not explicitly marked ✅/3-seed should be treated as provisional, and the items in Section 5.7 must be run and added before the section is final.**

### 5.1 The idea

Section 4's reduced free-rider re-trains **every round** — cheap per round, but still constant work. The **submarine** goes further: it re-embeds its watermark **only when the mark is about to fade below the threshold**, and the rest of the time it **coasts** — submitting a model that carries the mark essentially for free. Its life cycle:

> **Reduced-train until the mark is embedded and you have estimated the threshold η → then coast, submitting a mark-carrying model each round for free → tap (do one cheap reduced-training step) only when your own probe says the mark is fading toward η → back to coasting.**

A "tap" is a single cheap re-embedding step; a "coast" is a free round with zero training. The name "submarine" captures the behaviour: it runs silent (coasting) and only surfaces briefly (taps) when it must.

### 5.2 The turning knobs (what is dynamic / adaptable)

The submarine is defined by a set of **tap knobs**. These are exactly the levers the attacker can turn:

| Knob | Values (default) | What it controls | Effect of turning it up |
|---|---|---|---|
| **`tap_coast_mode`** | resend / decay / **graft** | how a free "coast" round's model is built | — (graft is the winner; see 5.3) |
| **`tap_scope`** | **head** (2 params) / block (8) / block2 (20) / full | how much of the network a tap re-trains | deeper/cleaner re-embed but costlier per tap and the mark fades faster → more taps |
| **`tap_data_cpc`** | 0 / 1 / 2 / **5** / 25 / −1 | images per common class in a tap's reduced set | more data per tap → cleaner embed, higher cost; cpc = 0 overfits → caught |
| **`tap_margin`** | **0.03** / 0.05 | safety gap below η: `target = η − margin` | raises the target lower → taps **more** |
| **`tap_max_coast`** | **12** / 20 | force a tap after this many coasts | longer coasts allowed → fewer forced taps |
| **`tap_probe_holdout`** | **16** / 48 | held-out trigger images the FR uses to self-estimate its own BER | bigger = less noisy self-estimate → fewer wasted taps |
| **`tap_when`** | **threshold** / always / every_k | the tap decision rule | threshold = adaptive; always = the reduced attack |
| **`tap_eta_source`** | **oracle** / self | where the FR gets its η estimate | oracle = *given* the true η (a crutch); self = estimates it itself (the goal) |

**What is dynamic vs fixed right now:** the tap/coast decision is genuinely dynamic (it responds to a live self-probe each round, per free-rider). The **η it aims under is currently *handed* to the attacker (`oracle`)** — the single biggest remaining crutch (Section 5.7). The warmup schedule and safety margin are currently fixed constants that *should* be derived dynamically.

### 5.3 The coast mechanism — why "graft" is the key invention

The whole attack hinges on **how a coast round submits a mark-carrying model without training.** Three modes were tested:

- **`resend`** — coast by re-submitting the fresh global model. **Fails:** the mark washes out in a *single* FedAvg averaging step, so the free-rider's self-probe reads ~0.5 the very next round and it is forced to tap every round anyway. No saving. (This was the Group-I failure.)
- **`decay`** — coast by re-submitting the free-rider's *own* last-tapped weights, verbatim. **The mark never fades**, so it evades the BER test — **but every coast submission is byte-for-byte identical**, a trivially detectable **replay/staleness** tell. It defeats the BER detector only by handing the server a different, easy tell.
- **`graft`** — **the invention.** Coast by submitting **{ the fresh global model's body (feature extractor) + the free-rider's frozen last-tapped mark head (final layer) }**. The body tracks the global every round (so there is **no replay tell** — the submission looks live and current), while the tiny frozen final layer keeps the watermark alive. Because only the body drifts (the head is frozen at mark-carrying values), the mark **fades *gradually*** over several coast rounds — a smooth sawtooth — rather than all at once. **Graft is the only mode that both evades the BER test *and* keeps the submission live.**

**Why graft is cheap (the ML background):** ResNet-18 splits into a **body** (the ~11M-parameter convolutional feature extractor that turns an image into a 512-number "feature vector") and a **head** (the final `Linear(512 → 100)` layer, ~51K parameters, that turns features into class scores). **The watermark is read only from the softmax, so it lives entirely in the head.** The free-rider never has to retrain the expensive body — the aggregated global model hands it good features for free every round; it only occasionally re-tunes the cheap head to keep its watermark alive. A `head`-scope tap is ~170 training steps on ~51K parameters (about **11% of an honest round**); a coast is literally **zero**.

### 5.4 The gradual-fade theory (why easy classes coast and hard classes don't)

A watermark bit is `bit_k = sign( Σ_j f(p_j) · M_{k,j} )`, computed from `p = softmax(W_head · z + b_head)`. Grafting **freezes the head** `(W_head, b_head)` at mark-carrying values and lets **only the features `z`** drift (the body follows the fresh global). So the only thing moving the watermark on a coast is feature drift:
- At a tap, features gave projections `z_k` with the correct signs → BER ≈ 0.
- On later coasts the body drifts, so a bit **flips only when its projection crosses zero** — after drift ≈ `|z_k| / (drift rate)`.
- **Easy class:** projections sit *far* from zero (large `|z_k|`) → many rounds of drift needed to flip a bit → **slow fade, few taps.**
- **Hard class:** projections sit *near* zero (small `|z_k|`) → tiny drift flips bits → **fast fade, frequent taps.**

That margin *is* the sawtooth period, and it is why the submarine's compute saving is **asymmetric**: it lives almost entirely on easy/medium classes.

### 5.5 Results so far

**Group I (◑ 1 seed) — the effort dial and the coasting failure.** The data dial works exactly as predicted: cpc = 0 → caught (BER 0.60); cpc = 1 → evades η_loose at ~38% effort; cpc = 5 → evades at ~43% effort. But under the naïve `resend` coast, the mark fades in one round, so "threshold" tapping **degenerates to always-tap** — no saving. This motivated graft.

**Group J (◑ 1 seed, seed 1000, 40 rounds) — graft ships.** The J-suite (J0 gate → J4 scope) confirms graft is a real, gradually-fading coast. The winner is **J2 = graft coast + head scope + threshold decision + cpc = 5 + margin 0.03**. J4 (`block2` scope) produces the *cleanest* sawtooth (plot **`tap_J4_scope_graft_block2_c36.png`**, uploaded — note the crisp orange sawtooth where each tap drops BER to ~0 and it ramps back up) but taps ~59% of rounds at ~2× tap cost, so `head` is the stealthier, cheaper scope.

**Group NOW (◑ partial) — the first 3-seed submarine.** NOW ran J2 (confirmed config) and J5 (a re-tuned variant) at 3 seeds each (1000/1001/1002, 40 rounds).

**J2 at 3 seeds (from the `tap_perfr_J2` aggregation — the headline submarine result):**

| Free-rider | Class | Difficulty | Tap fraction (attack phase) | Tail server-BER | vs η_loose 0.264 |
|---|---|---|---|---|---|
| **cid3** | class 3 | easy–medium | **10%** | 0.13 | **evades** |
| **cid6** | class 6 | hardest | **43%** | 0.22 | **evades** |

**Reading it:** both free-riders **evade** at the server on all 3 seeds. The **compute win is asymmetric**: cid3 (easy) genuinely coasts — it taps only ~10% of rounds — while cid6 (hard) taps ~43% (near the reduced-attack cost) and saves little. The reason is a **self-probe blind spot**: cid6's small 16-image self-probe *over-reads* its own BER (~0.30) while the server actually measures ~0.22, so the free-rider taps far more than it needs to. The mark itself is *not* caught — cid6's server BER 0.22 is exactly the plain reduced-attack value for class 6, so **on the hard class the submarine is no better and no worse than the static reduced attack.**

### 5.6 Compute breakdown (the "free-riding is cheap" number — J2, seed 1000, verified from the trace)

| Quantity | Honest client | Free-rider (mean) | cid3 (easy) | cid6 (hard) |
|---|---|---|---|---|
| Sample-passes, whole run | 1,000,000 | 305,600 | 285,660 | 325,540 |
| Effort ratio vs honest | 100% | **30.6%** | 28.6% | 32.6% |
| — of which **warmup** (11 forced-honest rounds) | — | 275,000 | 275,000 | 275,000 |
| **Attack-phase** passes (post-defection) | 725,000 | 30,600 | 10,660 | 50,540 |
| **Attack-phase effort vs honest** | 100% | **~4%** | **~1.5%** | **~7%** |
| Taps / freeride rounds | — | — | 4/29 | 19/29 |

⚠️ **The headline "31% effort" is a warmup artifact — quote the attack-phase number.** About 90% of the free-rider's total compute is the 11 forced-honest warmup rounds it does *before* the attack even starts (needed to embed the mark once). The **marginal** cost of the attack itself is **~4% of honest effort** (as low as **~1.5% on the easy class**). Always lead with the attack-phase number: it is smaller, and it is the honest one. The pipeline already logs per-round samples (`compute.per_client[cid].per_round`), so the **attack-phase-only effort inset is straightforward to add** and is a missing piece that makes the win legible.

### 5.7 What is hardcoded / still missing (must be done to claim a *complete* attack)

In priority order — this is the real to-do list for the main contribution:

1. ⏳⚠️ **Oracle η → self-estimated η (the biggest crutch).** Right now the free-rider is *handed* the true η = 0.264 (`tap_eta_source = oracle`). A real server never publishes η. The self-estimation path exists in code (`_resolve_eta`: μ + k·σ over the FR's own calibration-window probe BERs) but is **untested at scale.** **Do:** run a `tap_eta_source = self` family at 3 seeds; show the FR's self-estimate lands near 0.264 from its own ~4-round calibration window and still evades. **Until this passes, every evasion claim carries an asterisk.** This is the single most important experiment left in the whole project.
2. ⏳ **Close the self-probe/server gap on hard classes.** cid6's 16-image self-probe over-reads (0.30 vs server 0.22), forcing needless taps. The fix is a *better probe* (probe on ≥ the server's 50 images, or smooth/average the probe) — **not** a bigger safety margin (J5 tried that and made it worse). This is what would let the hard class actually coast. Constraint: the trigger class is data-starved, so you cannot hold out too many images without breaking embedding.
3. ⏳⚠️ **Resolve the R33–40 tail instability.** In the seed-1000 traces, around rounds 33–40 the test accuracy and honest BER spike violently (in the crashed J5 run, accuracy crashed 69% → 24% at R33 and honest BER spiked to 0.44). This looks like a **late aggregation / memory-update (Eq. 14) instability, not the attack**, but it sits *in the converged tail we read the verdict from.* It must be confirmed seed-specific and fixed or excluded before any submarine number is tabled.
4. ⏳ **Finish NOW at 3 seeds for both families.** ⚠️ **J5 crashed** at round 39 with no `result.json` (likely a pod eviction during finalization). **There is currently no J5 compute/detection summary — only a partial log.** J5's specific knob combo (margin 0.05, max_coast 20, holdout 48) is **not** an improvement — it made cid6 tap *more* — so **do not re-run J5 as-is.** Re-run instead: the J2 config (the winner) at 3 seeds to completion, plus the self-η variant from item 1.
5. ⏳ **Add the attack-phase-only effort metric + per-cid tap plots.** Report effort over post-defection rounds only, and always plot **per-free-rider** (the old single-family plot averaged the two free-riders and made the attack look like "all coast" when cid6 was actually always-tapping). The `tap_perfr` command does this.
6. ⏳ **(Optional) cheaper taps.** Graft holds the mark, so sweep `tap_data_cpc ∈ {1, 2}` under graft/head — if cpc = 1 still re-embeds, each tap roughly halves again (to ~2.7% of an honest round).

### 5.8 The verdict on the submarine (state this plainly)

A genuine adaptive submarine **exists**: on easy/medium classes (cid3) it truly coasts on a gradually-fading graft mark, its submissions track the global (no replay tell), and it evades η_loose at **~1.5% attack-phase compute**. That is categorically more than "the reduced attack with a schedule."

**But — and this is important — the submarine does not threaten the impossibility thesis; it confirms it.** To keep every coast peak under a *usable* η, the mark must be held near the **honest floor of its class** — which means the free-rider is doing *enough real work to be indistinguishable from an honest client there*, and any η low enough to catch its peaks also flags honest clients on the same hard class. On hard classes the submarine collapses to the (already-undetectable) reduced attack. **The submarine makes free-riding *cheaper* on easy classes; it does not make free-riding *detectable* anywhere.** Output-layer, BER-thresholded free-rider detection still has no deployable operating point.

### 5.9 Reference code / theory for the submarine

- **The attacker class:** `AdaptiveTapFreeRider` (`clients.py`), inheriting `(_SimpleFRMixin, WatermarkClient)` so that a tap uses the genuine honest embed path (the mark it writes is real).
- **Per-round decision (`produce_update`):** probe the *coast candidate* (not the raw global), then `tap = (probe is None) or (probe > target)`; `target = η − margin`.
- **Graft candidate (`_coast_candidate`):** clone the fresh global, then overwrite only the head parameter names (`fc.weight`, `fc.bias`) with the free-rider's last-tapped head.
- **Memory subtlety:** the memory buffer (Eq. 14) only advances on taps, never on coasts — so a coasting free-rider carries its last-*tapped* trajectory.
- ⚠️ **Naming:** the fully-dynamic `SubmarineFreeRider` (dynamic warmup, own-η estimation) is **commented out** in the code; NOW ran the simpler `AdaptiveTapFreeRider`. Do not claim dynamic-warmup results until that class is revived.

---

<a name="6-non-iid-setting"></a>
## 6. Non-IID setting

**Group E. Status: ✅ done, 3 seeds; α ∈ {0.1, 0.5, 1.0}, CIFAR-100, 10 clients.**

### 6.1 Context — what "non-IID" means and why it is the key rebuttal

Everything so far used **IID** data: every client got a uniform random slice, so all 100 classes appear about equally in every client's shard. Real federated learning is almost never like that — real clients have **skewed, uneven** class distributions (one hospital sees mostly one disease; one phone mostly one kind of photo). This is **non-IID** data. A natural defence of FareMark is: *"Your IID non-separability is a lab artifact. In messy, realistic, heterogeneous FL, honest and free-rider clients would separate."* **Group E tests that defence directly — and refutes it.**

**How we create non-IID data (`dirichlet_partition`, Hsu et al. 2019):** for each class, we draw a proportion vector from a **Dirichlet distribution** with concentration parameter **α**, and hand out that class to the 10 clients in those proportions. **Small α = severe skew** (one client hogs each class); **large α ≈ uniform** (but even α = 1.0 is *not* equal shards — it still has high variance). We sweep **α ∈ {0.1 (extreme skew), 0.5 (moderate), 1.0 (near-IID)}**.

**The crucial coupling (this is the whole mechanism):** a client's **trigger-class assignment is drawn *independently* of the data split.** So a client is usually *not* the one that happened to receive the bulk of its own trigger class → it is **data-starved on the very class it must watermark.** A client that holds only a handful of its trigger-class images cannot embed a clean mark — *even when honest.* That raises the honest BER floor, and the free-rider hides underneath it.
!!! TODO: add a figure showing the Dirichlet distribution for α = 0.1, 0.5, 1.0.
!!! TODO: also test different distribution - assume server knows the data distribution -> assign the trigger classes such that the clients are assigned the classes they have more data on. This is a more realistic scenario, but it is also a more favorable one for the server. The free-rider will be even harder to hide in this case.

### 6.2 Experimental setup

- Same substrate (ResNet-18, CIFAR-100, N = 10, m = 10, N_T = 50), but the **non-IID Dirichlet partition** replaces the IID shards.
- **E1:** honest-only calibration at α = 0.5 (the non-IID analogue of A1).
- **E2:** reduced free-rider (classes 3 & 6) vs honest at α = 0.5.
- **E3:** the α severity sweep — honest and reduced at α = 0.1 and α = 1.0. (α = 0.3 was dropped as ≡ α = 0.1.)
- **Rounds:** 50. **Seeds:** 3. Frozen reference line on E2/E3 timelines: `WM_ETA_FIXED = 0.161` (a coded/pooled η for the non-IID tail); η_tight 0.064 and η_loose lines vary per α (see below).

### 6.3 Results

**E1 — non-IID raises and spreads the honest floor** (plot **[E1_class_floors.png](results/groups/figs_2/E1_class_floors.png)**, uploaded).
Honest per-class BER floors at α = 0.5 span **0.007 → 0.255** (vs IID's 0.001 → 0.114 — the skew roughly *doubles* the worst floors): cls2 0.007, cls7 0.018, cls8 0.020 (still embed well) rising to cls6 0.167, cls0 0.168, cls1 0.187, **cls3 0.255 (hardest)**. **Some honest clients simply cannot watermark their assigned trigger class under skew** — and that honest failure is the free-rider's hiding place.

**E1 thresholds — the skew forces a lose-lose choice.** Recalibrated on the non-IID honest tail: coded 0.150 (FPR **24%**), pooled 0.182 (24%), trimmed-10% 0.318 (9%), loose 0.550 (4%). The skew lifts the tight/coded rule to 0.150 — now *above* 1/m = 0.10, so **no longer degenerate** — **but its honest FPR is 24%.** The only way to make the tight rule non-degenerate under skew is to accept **flagging a quarter of honest clients.**

**E2 — the reduced free-rider hides in the honest floor** (plot **[E2_niid_timeline.png](results/groups/figs_2/E2_niid_timeline.png)**, uploaded).
Honest floors at the free-rider's own classes: **cls3 0.26, cls6 0.17.** In the converged tail, the **global** honest mean drops to ~0.07–0.10 (dragged down by easy classes) while the free-rider rides ~0.18–0.20 — i.e. **the free-rider sits *inside* the honest-floor band (0.17–0.26) for its own trigger classes.** The frozen 0.161 line does "catch" the free-rider, but only by *also* flagging every honest client on cls3 and cls6 — exactly the 24% FPR.

**E2 separability — the decisive threshold-independent numbers (`E2_niid_sep.json`, 3 seeds):**

| View | Honest BER | FR BER | OVL | Best balanced-error | Reading |
|---|---|---|---|---|---|
| **class 3** (FR's own) | 0.255 | **0.222** | 0.667 | **0.500** | FR *cleaner*; **inseparable** |
| **class 6** (FR's own) | 0.167 | **0.143** | 0.783 | **0.500** | FR *cleaner*; **inseparable** |
| GLOBAL (server pools all) | 0.109 | 0.183 | 0.690 | 0.388 | looks weakly separable — **the illusion** |

**This table is the payoff of the whole non-IID section.** At each of the free-rider's *own* trigger classes, the **best possible threshold scores balanced-error 0.500 — a coin flip — and the free-rider's BER is actually *lower* than honest.** The only place any separation appears is the **GLOBAL** row (balanced-error 0.388), and that is precisely the **pooling illusion**: the server, comparing one free-rider against *all* honest clients (most on easy classes with BER ≈ 0), sees a gap **that does not exist at the class where detection actually has to happen.** A server can only act on the global view, so it *believes* it can separate — the per-class rows prove it cannot.

**E3 — the α sweep confirms the mechanism across severities** (plots **[E3_a01_timeline.png](results/groups/figs_2/E3_a01_timeline.png)** and **[E3_a10_timeline.png](results/groups/figs_2/E3_a10_timeline.png)**, uploaded).
- **α = 0.1 (extreme skew):** η_loose rises to **0.521**; honest floors cls3 0.31 / cls6 0.35. Honest mean ~0.25–0.28, FR mean ~0.31–0.33 — **total overlap**, both riding the honest-floor line. Skew is so severe *nobody* embeds well; the free-rider vanishes into the honest cloud.
- **α = 1.0 (near-IID):** η_loose 0.330; honest floors cls3 0.27 / cls6 0.45. The **global** honest mean is low (~0.05–0.08, most classes embed) **but the FR's own classes are still starved** (Dirichlet(1.0) ≠ equal shards), so their floors are *higher* than the IID floors. The FR rides ~0.32: above η_tight, below η_loose, inside the 0.27–0.45 honest band of its own classes. **This held over 3 seeds — it is the starvation mechanism, not a bad-luck draw.**

### 6.4 Reference code / theory

- **Partition:** `datasets.py::dirichlet_partition` (Hsu et al. 2019 label-skew). For each class, `props ~ Dirichlet(α · ones(K))` over K = 10 clients; hand client k the fraction `props[k]` of that class.
- **The starvation coupling:** trigger-class assignment (`cid → class`) is deterministic and drawn *independently* of the seeded Dirichlet split — so the seed controls whether a client is starved on its own trigger class. This is why **3 seeds is the minimum** for any non-IID table: the *mechanism* is seed-robust, but the *exact floors* are draw-dependent.
- **Separability metrics:** OVL (overlap coefficient = shared histogram area = Weitzman's coefficient = 1 − total-variation distance) and best-balanced-error (lowest error any threshold achieves, even an oracle one).

### 6.5 Conclusion so far (Section 6)

**Non-IID does not rescue the detector — it erodes it further.** Label skew raises the honest floor on each client's own trigger class (because that client rarely holds much of it), so the reduced free-rider hides among honest clients who *also* cannot embed on hard classes. Per-class, honest and free-rider BER **coincide across all three α values** (best balanced-error = 0.500 at both free-rider classes); the only η that flags the free-rider also flags ~24% of honest clients. **This kills the "real heterogeneous FL would separate them" rebuttal — heterogeneous FL is strictly *worse*.**

### 6.6 What is needed to finish Section 6

1. ✅ The 3-seed α ∈ {0.1, 0.5, 1.0} CIFAR-100 story is complete and table-ready (E2 per-class separability table is done).
2. ⏳ **CIFAR-10 non-IID replication** (3 seeds, ideally 10) — requested for the paper, to show the mechanism is not CIFAR-100-specific.
3. ⚠️ **Consider a 10-seed E1 calibration** — non-IID floors are the most seed-sensitive quantity in the project (the Dirichlet draw is "the big one" among random streams), so the threshold table would be more defensible at more seeds.
4. ⏳ **Add a figure showing the Dirichlet distribution for α = 0.1, 0.5, 1.0** (to illustrate the skew visually).
5. ⏳ **Test a more realistic trigger-class assignment** — assign trigger classes to clients based on their data distribution (the server knows the data distribution). This is a more favorable scenario for the server, but it is also more realistic. The free-rider will be even harder to hide in this case, so it is worth testing.

---

<a name="7-related-work"></a>
## 7. Related work — where this sits, and how each piece helps

The scheme we attack is the endpoint of a lineage of watermarking methods. Understanding the lineage is what lets us claim our failure is a property of the **reader family**, not a one-off FareMark bug.

- **Uchida et al. (2017), "Embedding watermarks into deep neural networks."** The first DNN watermark: bits embedded in the *weights* via a projected regulariser (white-box — you must open the model to read it). It established the "project onto a pseudorandom key" template that FareMark still uses. *How it helps us:* it is the origin of the projection idea; contrasting white-box (robust but needs weights) vs FareMark's box-free (convenient but fragile) motivates why the output-layer choice is the weak point.
- **Adi et al. (2018), "Turning your weakness into a strength: watermarking by backdooring."** The first *black-box* watermark, read from *outputs* on special trigger inputs (single-bit). *How it helps us:* it is the black-box ancestor; it shows the field's move toward output-only verification, which is exactly the move that our attack exploits.
- **BlackMarks (2019).** The first *multi-bit* black-box scheme; the signature lives in the output-activation distribution. *How it helps us:* it introduces the multi-bit output-distribution reader that FareMark inherits.
- **Universal BlackMarks (IEEE SPL 2023) — FareMark's direct reader ancestor.** Power-function-on-softmax + projection onto a pseudorandom key — precisely FareMark's Eq. 8 + Eq. 10, before they were moved into FL. *How it helps us:* this is the clearest evidence that the **peaky-softmax failure we document is a property of the whole reader family**, not FareMark specifically. Cite it to generalise the negative result.
- **FedIPR (TPAMI 2022/2023).** Brought the free-rider angle into FL: client-side secret marks that can *identify* free-riders. This is the "watermark ⇒ free-rider detection" claim FareMark inherits, and FareMark's own experiments compare against it. *How it helps us:* it is the intellectual origin of the exact claim we refute, and a natural second target — our negative result applies to it too.
- **WAFFLE (SRDS 2021).** A *server-side* FL watermark. It cannot police free-riders (a free-rider's submitted model is just the global model, which already carries the server's mark), which is *why* free-rider detection falls to client-side schemes like FareMark. *How it helps us:* it explains why the client-side design (the one we attack) is the only game in town for this task — closing off the "just use a server-side mark" escape.
- **FedSMW (2024) and kin.** Reuse the same power-on-softmax + projection reader. *How it helps us:* more evidence that our documented failure mode is shared across the reader family — strengthening the generality claim.
- **FRAD (IEEE IoT J 2024), RFFL, ST-DAGMM/DAGMM (the non-watermark detectors FareMark compares against).** These are *contribution-evaluation* and *anomaly-detection* free-rider detectors (they look at gradient similarity or fit an autoencoder to benign updates). FareMark positions itself as better than these. *How they help us:* they are the natural **baselines** for our own evaluation — if we show the watermark detector fails where these succeed (or that these also fail on the submarine), we sharpen the contribution. **ST-DAGMM is already the intended baseline for our positive-control work (Group H).**

**The gap we fill.** *None* of these methods evaluate an **adaptive, effort-minimising insider that holds a valid key and tunes its behaviour to sit under η.** Their free-rider is always the crude Gaussian / previous-models attacker, caught trivially. Our threat model (Section 2.4) and the submarine (Section 5) are exactly that missing evaluation — which is what makes the two-sided result (impossible-to-detect *and* cheaply-evadable) novel.

---

<a name="8-what-is-left-to-run"></a>
## 8. Master list of what is left to run (consolidated)

### 8.2 Experiments to run (priority order)

| # | Experiment | Why | Seeds | Status |
|---|---|---|---|---|
| 1 | ⏳ **Self-η submarine** (`tap_eta_source = self`, J2 config) | Removes the last crutch (oracle η) → self-sufficient attack. **The single most important run left.** | 3 (→10) | not run |
| 2 | ⏳ **H5 positive control** (crude previous-models FR, CIFAR-100) | Completes the operating-point figure (the "detector catches lazy cheaters" bar) | 3 | not run |
| 3 | ⏳ **Re-run J2 submarine to completion** at 3 seeds; **do NOT re-run J5 as-is** | J5 crashed (no result.json); its knobs are not an improvement | 3 | partial |
| 4 | ⏳ **Fix/confirm the R33–40 tail instability** (memory-update artifact) | It sits in the tail we read the verdict from | across seeds | open |
| 5 | ⏳ **CIFAR-10 replication** of Groups A, D, E (incl. the m = 1 forced-single-bit finding) | The common benchmark + FareMark's headline dataset; sharpens the 1/m argument | 3 (→10) | not run |
| 6 | ⏳ **V2** (few-trigger-sample FR, paper Table V complement) | Positive control pairing with cpc = 0 | 3 | not run |
| 7 | ⏳ **F1/F2/F3** (capacity, 200 clients, Table IX repro + held-out twin) | Tests FareMark's capacity claim under a strict generalisation test | 2–3 | not run |
| 8 | ⏳ **C1** (sin vs power smoothing ablation) | Low priority; sin is just a weak smoother | 3 | not run |
| 9 | ⏳ **Attack-phase-only effort metric + per-cid tap plots** | Makes the compute win legible and honest | (re-plot) | code exists |
| 10 | ⏳ **10-seed E1 non-IID calibration** (optional) | Non-IID floors are the most seed-sensitive quantity | 10 | optional |
!!! TODO: add all the TODO added in the above sections that arent in this table here too

### 8.3 Standing rules for anything that goes in the paper

- **≥ 3 seeds minimum, 10 ideal.** Single-seed results are shape-only and must not be tabled. (The A1 single-seed-vs-6-seed floor discrepancy in Section 3.3 is the cautionary example.)
- **Trust uploaded results over the source notes** where they conflict; several note-values are stale (flagged inline throughout).
- **Report per-class / per-free-rider, never pooled**, for anything about separability or tap fraction — the "global pooling illusion" (Section 6.3) and the "cid3-only tap-fraction" bug (Section 5.7) both come from pooling.
- **Lead with attack-phase effort** (~4%, ~1.5% on easy classes), and note the warmup-dominated ~31% headline as secondary.

---

<a name="9-canonical-configuration-reference"></a>
## 9. Canonical configuration reference

The exact settings, verified against the uploaded `A1_honest_c100_rep0result.json` and `D1_reduced_c100_c36_n1_rep0result.json`.

| Setting | Value | Note |
|---|---|---|
| Model / dataset | ResNet-18 (32×32) / CIFAR-100 | reproduces ~73% test accuracy |
| Clients N | 10 | one client per used trigger class |
| Rounds | 50 (Groups A/D/E); 40 (Groups J/NOW) | — |
| local_epochs | **5** ⚠️ | **paper uses 2** — flag before any FareMark comparison |
| lr / batch / momentum / weight-decay | 0.01 / 16 / 0.9 / 5e-4, SGD | frozen for comparability |
| Aggregation | sample-weighted FedAvg (= mean under equal IID) | — |
| Watermark on? / λ / β / α / f | yes / 5.0 / 0.6 / 0.4 / power `p^α` | paper Eqs. 11/14/8 |
| Bits m / group length l | **10 / 10** | 1/m = 0.10; any η < 0.10 is degenerate |
| N_T (verification images) | 50 | held-out **test** images (generalisation, not memorisation) |
| Partition | IID (Groups A/D); Dirichlet non-IID (Group E) | α ∈ {0.1, 0.5, 1.0} for E |
| Free-rider IDs | 3, 6 | pins the medium + hardest classes; fixed across seeds |
| Warmup / calibration window | 12 / 4 | honest rounds 1–7, calibrate 8–11, defect at 12 |
| Reference thresholds (frozen) | η_tight = 0.064 (degenerate) / η_loose = 0.264 (the real operating line) | from 10-seed honest calibration |
| Submarine defaults | graft coast, head scope, cpc = 5, threshold decision, margin 0.03, oracle η ⚠️ | oracle η is the crutch to remove |
| Seeds | 1000, 1001, 1002, … (`seed = 1000 + repeat`) | — |

**What each seed re-randomizes** (why ≥ 3 seeds is required): the client data shards, the minibatch order, the model initialization, **the secret watermark key `M` and message `B` per client (the "key lottery" — the largest driver of BER swings)**, the verification images, and — **in non-IID only** — the entire Dirichlet skew pattern (the single biggest mover of non-IID floors). Held constant across seeds: trigger-class assignment, N, m, rounds, epochs, λ, β, α, the frozen thresholds, and the free-rider set.

---