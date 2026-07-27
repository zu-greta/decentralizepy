# Status and the full experiment plan

## STATUS
TODO: summarize current implementation status, what experiments were implemented and what were run. 

## DEFINITIONS

### 1. Seed variation 

**Seed:** A single starting number that determines every "random" choice a run makes. 
Same seed → identical run. Change the seed → every random choice is re-rolled. 
We run each experiment at several seeds (3 or 6) and average, so a
result is not a fluke of one lucky draw. In code, `seed = base_seed + repeat`.

**What the seed re-rolls, and how much each one matters:**

| re-rolled by the seed | why it is random | how much it moves the result |
|---|---|---|
| which images each client gets | in real FL nobody controls who holds what | moderate |
| the order batches are shuffled during training | standard practice | small |
| the model's initial weights | networks start from random values | small–moderate |
| **the key matrix `M` (per client)** | keys must be secret and unique | **large** |
| **the target message `B` (per client)** | messages must be unpredictable | **large** |
| which N_T images go in the trigger bank | it is a sample of the class | small |

**What is NOT re-rolled (held constant across seeds):** the trigger class assignment
(`trigger_class = cid % n` — client 6 always gets class 6), the number of clients, bits
`m`, rounds, epochs, λ, β, α, the smoothing function, and the dataset itself.

**Where `M` and `B` come from, concretely.** Each client's key and message are generated
by a random-number generator seeded with `seed + 1000·cid + 1`. The `1000·cid` part makes
every client different from every other client *within* a run; the `seed` part makes the
same client different *across* runs. So changing the seed from 0 to 1 hands **every client
a brand-new decoder matrix `M` and a brand-new message `B`**. This formula is our
implementation detail, not from the paper; the paper only says `M` is a per-client secret
pseudorandom matrix (§IV-A) and `B` is the watermark to embed.

**Why the variance is unusually large — the key point.** There are two *kinds* of
randomness here and they are not equal:

- *Nuisance* randomness (data split, shuffle, init): you are measuring the **same thing**
  slightly imprecisely. Averaging over seeds sharpens the estimate. This is the ordinary
  kind.
- *Task-changing* randomness (`M` and `B`): you are measuring a **different thing** each
  time. "How hard is class 6 to watermark?" has no single answer — it depends which
  decoder matrix and which message you drew. Averaging over seeds does not sharpen one
  number; it averages over a *population of different questions*.

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

**ELI5.** Imagine each kid (client) is given a secret decoder ring (`M`) and a secret word
to spell (`B`) using colored beads. Some kids get an easy ring where every bead position
can be either color. Some kids get a broken ring where one position is glued to red no
matter what — they can never spell their word right there. When you change the seed, you
hand out a completely new set of rings and words to everyone. So "how well can the kids
spell?" changes every time, not because the kids got better or worse, but because they got
different rings. That is why the numbers bounce around so much even though nothing about
the actual kids changed.

---

### 2. Thresholds — what each rule is, where it comes from, what it does

**The setup.** After training, the server measures each client's BER and must decide: is
this client honest (BER low, it embedded its mark) or a free-rider (BER high, no mark)?
It flags anyone with **BER ≥ η**. The whole question is how to pick the single number η.
All rules below compute η from the *honest* BER values collected over the converged tail
(last 20 rounds); the last two also peek at free-riders and are therefore not deployable.

| rule | where it is from | what it computes | in plain terms |
|---|---|---|---|
| **μ + 3σ (the paper's "coded" rule)** | FareMark §IV-D3; underneath, the Shewhart 3-sigma control limit from 1920s quality control | mean of the per-round mean-BER, plus 3× its standard deviation | "the normal range is mean ± 3 std; flag anything above." Assumes a bell curve; ~0.13% of honest clients should trip it *if* the assumption holds |
| **pooled μ + 3σ** | same, computed differently | pool every (seed, round) mean into one sample, then μ + 3σ | same idea, but the spread now includes between-seed variation, so it lands higher (looser) |
| **loose (per-client μ + 3σ)** | same formula, correct population | mean and std of the *individual* client BERs, then μ + 3σ | the only μ+3σ variant whose σ matches the population the test is applied to. Delivers the true 3σ margin |
| **median + 3·MAD** | robust statistics (Hampel) | median instead of mean; 1.4826×median-absolute-deviation instead of σ | "normal range, but ignore outliers." Breaks (→ 0) when more than half the honest clients sit at BER 0 |
| **trimmed-10% μ + 3σ** | Tukey trimmed statistics | drop the top and bottom 10% of clients, then μ + 3σ | same as coded but throw away the extremes first |
| **adaptive σ-clip** | iterative sigma-clipping (astropy `sigma_clipped_stats`; also DP-SGD adaptive clipping) | repeatedly drop clients above μ+3σ and recompute until it stabilizes, then μ+3σ | "keep tightening the range until only the well-behaved clients define it." This is the "adaptive clipping in warmup rounds" idea from the meeting notes |
| **honest p95 / p99** | non-parametric empirical quantile | the 95th (or 99th) percentile of honest BER | "put the line where the worst 5% (or 1%) of honest clients sit." Fixes the false-positive rate directly, assumes no distribution shape |
| **equal-error-rate (EER)** | biometrics | the η where false-positive rate = false-negative rate | "the balance point where you wrongly flag as many honest as you miss free-riders." **Needs to see free-riders — not deployable** |
| **Youden-optimal** | Youden's J statistic (1950) | the η minimizing (FPR + FNR)/2 — the single best possible threshold | "the best any threshold could ever do." **Needs free-riders — not deployable**, used only to prove an upper bound |

**Why we test all of them.** Rules 1–7 are things a real server could deploy (they use
honest data only). EER and Youden *cheat* by looking at the free-riders — we include them
to bound what is even possible. The argument becomes: *"even the cheating rule, which no
real server could run, only reaches balanced accuracy X."* If that X is ≈ 0.5, no
deployable rule can do better, because none of them can beat the cheating one.

**The two things the regime reveals** (findings F8, F9):
- The paper's rule computes σ over a **mean across clients** (spread σ/√N) but applies the
  threshold to **individual clients** (spread σ). So "3σ of margin" becomes ≈ 3/√N σ —
  measured at 0.13σ–0.51σ across our runs, never the promised 3σ, giving 7–53% false
  positives instead of 0.13%.
- BER only takes values 0, 1/m, 2/m, … . When the watermark works well, η drops below
  1/m, and "flag if BER ≥ η" becomes exactly "flag if ≥ 1 bit wrong" — the numeric value
  of η stops mattering entirely.

**ELI5.** You are a teacher deciding who cheated on a spelling test by how many letters
they got wrong. μ+3σ says "the normal number of mistakes is the average plus a bit; anyone
worse than that, I accuse." The problem: you measured "the average class's mistakes"
(which barely varies, because it is an average of 30 kids) but then accuse *individual*
kids (who vary a lot). So your line is way too strict and you end up accusing a third of
honest kids. The other rules are different ways to draw the line — use the middle kid
instead of the average (median), throw out the best and worst few first (trimmed), or just
say "the worst 5% get accused" (p95). We try every possible line and show that no line
separates the honest kids from the cheaters, because their mistake-counts overlap too much.

---

### 3. Class difficulty — what it is and how it is measured

**What "class difficulty" means.** Each client hides its watermark in the model's behavior
on **one class of images** (its trigger class). It turns out some classes are much easier
to hide a watermark in than others, and this has nothing to do with how hard the client
works — it is a property of the *class itself*. A "hard" class is one where honest clients
end up with high BER (the mark does not embed well) no matter how long they train.

**How it is measured.** For each trigger class:
1. Take the honest-only runs (nobody cheating).
2. For each client, average its BER over the converged tail (last 20 rounds) and over all
   seeds. Since each client owns exactly one trigger class, this per-client number *is* the
   per-class number. We call it the class's **floor**.
3. Compare floors across classes. In the known-good CIFAR-100 10-client run they span
   **0.00 (classes 7,8,9) to 0.21 (class 6)** — a class can be effectively impossible to
   watermark while its neighbor is trivial.

**What actually makes a class hard — the mechanism.** The watermark is read from the
*shape of the softmax tail* (all the class probabilities except the dominant one). Two
regimes:
- A **flat** softmax (the model is unsure — many classes get similar probability) has a
  rich, shapeable tail. The watermark loss can nudge those tail values to encode bits
  cheaply. → low BER, easy class.
- A **peaky** softmax (the model is very confident — one class gets ~0.9, the rest near 0)
  has a flat, structureless tail of near-equal tiny numbers. There is nothing to shape, so
  the bits become coin flips. → high BER, hard class.

We measure "peakiness" two ways: **entropy** (how spread-out the probabilities are; high =
flat) and **dominance / p_max** (how much the top class takes; high = peaky). In the
10-client regime these correlate with BER at |r| ≈ 0.6–0.7, while classification *accuracy*
correlates only weakly (|r| ≈ 0.05–0.4). That gap is the point: difficulty is about the
**shape of the output distribution**, not about whether the model classifies the class
correctly. (Caveat from finding F1: the correlation with the key draw means difficulty is
partly a key-class interaction, not purely intrinsic — balanced keys remove much of it.)

**How the plot shows it** (`honest_class_lines`, `class_difficulty`):
- one BER-vs-round line per class; the tail floor is the number that matters.
- the four-panel `class_difficulty` figure sorts classes easy→hard, shows their accuracy
  in the same order (visibly scrambled = accuracy does not explain difficulty), and
  scatters BER against error and loss with Pearson r.

**What a "good result for the thesis" looks like:** floors that span a wide range and
correlate with entropy/dominance, not accuracy. That kills the obvious objection ("your
hard classes are just the classes the model is bad at") and shows the difficulty is baked
into the watermarking scheme, not the model quality.

**ELI5.** You have to hide a secret note in a picture. If the picture is a messy scribble
(flat softmax), there are lots of little places to tuck the note — easy. If the picture is
almost entirely one giant red circle with a few faint specks (peaky softmax), there is
nowhere to hide the note except the specks, and they are too faint to arrange reliably —
hard. Some image-classes make the model draw scribbles, some make it draw the giant circle.
The "hard" classes are the giant-circle ones, and no amount of trying lets you hide the
note well there. Importantly, this is not about whether the model *recognizes* the picture
correctly — it is about whether the picture leaves you any room to hide things.

---

### 4. How the paper handles "more clients than classes"

**Why it is a special case.** Each client gets its own trigger *class*, and there are only
10 classes (CIFAR-10) or 100 (CIFAR-100). Once you have more clients than classes,
**clients must share trigger classes** — two or more clients hide watermarks in the same
class. This is the crux of the "more clients than classes" experiments: does forced sharing
create overlap that a threshold cannot resolve?

**What the paper actually does (from Tables I, VII, and §V).** Going through the paper
carefully:
- The paper's main CIFAR-10 / MNIST fidelity runs use **10 clients** (Table I). With 10
  classes, that is exactly one client per class — no sharing.
- The **CIFAR-100 row uses 100 clients** (Table I: "Clients 100"), again one client per
  class — the maximum before sharing begins.
- So in its headline tables the paper **never actually oversubscribes** — it stops at
  clients = classes. The "capacity" question (more clients than classes) is one the paper
  does not directly stress-test in its fidelity numbers.

**The settings the paper does specify** (§V-B): N_T (trigger sample count) varies by
experiment — 100 for the Table I/II fidelity rows, and Table VII sweeps N_T ∈ {1, 10, 50,
100, 150, 200, 300, 400}; ResNet-18 and AlexNet; 50 communication rounds; 5 local epochs
per round; watermark bit-length N (for the FedIPR feature-baseline) set to 100.

**What the paper does NOT specify — the reproducibility gap.** After going through the
whole paper: there is **no statement of how the training set is partitioned across
clients** — no images-per-client, no IID-vs-non-IID scheme, no mention of whether client
shards are disjoint or overlapping, and no learning rate, batch size, or optimizer for the
FL training. This matters enormously: 100 clients over CIFAR-100's 50,000 images is 500
images each if disjoint, which undertrains ResNet-18 (our R1 reached ~46% vs the paper's
75.31%). The paper's 75% is only reachable if clients see *more* data than a strict
disjoint split gives — e.g. IID sampling with replacement, larger overlapping shards, or
more rounds than stated. **We could not find these details because they are not in the
paper.** This is itself a reportable finding: FareMark is not reproducible on the
data-partition axis without guessing.

**How our experiments differ and why.** Our Group F pushes *past* the paper into genuine
oversubscription — 200 clients on CIFAR-100 (2 per class, forced sharing) — precisely
because that is the regime the paper avoids and where our non-separability argument is
strongest: two clients on the same class, one honest and one free-riding, produce
overlapping BER that no threshold can split. We also run it with **more rounds (100)** than
the paper's 50, specifically to rule out "you just undertrained" as an objection — if the
model is well-trained and the free-rider is *still* inseparable, the result is clean. And
we use held-out trigger images (`class` mode) rather than the client's own training images,
which is the stricter generalization test; the paper's Table IX-style capacity numbers lean
on trigger-sample consistency (verifying on the same images used in training), which is
memorization rather than true embedding.

**ELI5.** Imagine assigning each kid one shelf in a classroom to hide their treasure, and
the teacher checks the right shelf to find each kid's treasure. If there are 10 shelves and
10 kids, everyone gets their own shelf — easy to tell whose treasure is whose. But if there
are 20 kids and only 10 shelves, two kids must share a shelf, and now their treasures are
mixed together on the same shelf — the teacher cannot tell which kid hid what, or whether a
lazy kid hid anything at all. The paper's experiments quietly stop at "one kid per shelf"
and never test the crowded-shelf case. Our experiments deliberately crowd the shelves,
because that is exactly where the teacher's checking method falls apart — which is the whole
point we are trying to prove.

---

## RESULTS
TODO: each plot + setting + results



## EXPERIMENT PLAN - to be run
TODO: summarize the experiments to be run based on last meeting + each experiment goal and whta it proves, implementation status. 


Every experiment maps to a meeting-note item. All use the **known-good CIFAR-100
10-client base** unless the point *is* to vary clients/dataset.

### Group A — the proven baseline, more seeds (safe, always runnable)
Solidifies class difficulty + reduced non-separability on the config that already works.

| label | setting | proves (notes ref) |
|---|---|---|
| A1 | honest, cifar100, 10cl, 6 seeds | class difficulty; threshold calibration |
| A2 | reduced +5, classes 1,7, 3 seeds | non-sep at EASY classes (FR cleaner than honest) |
| A3 | reduced +5, classes 3,6, 3 seeds | non-sep at HARD classes |
| A4 | sameclass, FR on class 6, 3 seeds | FR vs honest, SAME trigger class, same training |

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