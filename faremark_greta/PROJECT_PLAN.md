# Project Plan — Limits of Output-Layer Watermarking for Free-Rider Detection in Federated Learning

Status: 
[x] Reproduction of FareMark paper codebase 
[~] Repoduction of FareMark paper results
[] Define limits of FareMark paper design and detection scheme
[] Implement new attacks to demonstrate limits
[] Run experiments to demonstrate limits
[] Define and implement new detection scheme to overcome limits
[] Run experiments to demonstrate new detection scheme
[] Write up results

---

## 1. Goal 

Show that **output-layer watermarking/fingerprinting cannot reliably detect
free-riders under identifiable conditions** — i.e. detection is not just hard
but *impossible* under certain conditions.

### Things to consider

1. **Threshold fragility** — the paper's detection threshold is `η = μ + 3σ` 
2. **Forgeability** — a free-rider with the key + a few trigger samples can re-embed the mark
3. **Collusion** — multiple free-riders can combine their updates to evade detection
4. **Non-IID false positives** — skewed data → some honest clients can't embed → their BER rises → they get misflagged
5. **Train-then-attack** - on random rounds, a free-rider can train on the global model and then attack, which may allow it to evade detection
6. **Trigger-sample only + pick and choose** - a free-rider can train trigger sample class only + a select few from the rest of the dataset, which may allow it to evade detection

---

## 2. Definitions

> **Note:** BER (bit-error-rate) is not defined as such in the paper but rather
> as the left-hand side of Eq. 16. The paper never names it, and it reports
> only the complement `1 − BER` as "watermark detection rate" / `Acc_wm`. 
> Using BER for consistent vocabulary. 
> Per-client bit budget m = 4 on CIFAR-10 is a derivation under our embeddability 
> assumptions; the paper leaves m as an unspecified free parameter. 

**Bit.** One yes/no piece of a client's secret watermark message. The paper
treats the message length **m as a free parameter**: it states only that
`n ≥ m`, that the group size is `l = n / m`, and that the first `m·l` softmax
outputs carry the watermark (section IV-A, Eq. 1). It never reports the m it used for
its own detection experiments. (The `N = 100` in Table I and `N_T = 50/100`
elsewhere are *not* this m — the former is the FedIPR baseline's feature-bit
length, the latter the trigger-*sample* count.)

**Bit budget m = (n−1)/2 — derived from the paper.** In `config.py`,
the default sets `m = (num_classes − 1)//2`, giving **m = 4 on CIFAR-10** (10
classes) and **m ≈ 49 on CIFAR-100** (100 classes). This follows from 2
implementation choices made drifting apart from the paper:
- **Excluding the trigger class** from the projection (its softmax peak would
  freeze one bit at small l), leaving `n − 1` usable classes.
- Requiring **sign-balanced ±1 key rows**, which need group size `l ≥ 2` to be
  embeddable, so at most `(n−1)//2` groups fit.
The paper's worked example instead uses a mixed-sign row `M = [1,−1,1]` over a
group of 3 on the full softmax, so it is not bound to 4 either. Treat the
small-m regime as an unexamined gap we are entitled to probe, since the
paper never pins m down — and present the "few classes → few bits → overlap"
argument as our analysis, not a reproduced FareMark finding.

**Watermark (B^i).** Each client `i` has its own m-bit string `B^i` plus a
secret ±1 key `M`, embedded into the model's softmax on that client's *trigger class*.

**Bit-error-rate (BER) — Eq. 16.** The paper thresholds, in Eq. 16, the 
fraction of recovered bits that disagree with the registered bits:

```
(1/m) · Σ_k | b̂_k − b_k |  < η          (paper Eq. 16, left-hand side)
```

We call this left-hand side the **bit-error-rate (BER)**. The paper never uses
that term; the only place it names the quantity at all is the sentence after
Eq. 16 ("typical error rate for legitimate clients ... η = μ + 3σ"). Everywhere
else it reports the **complement**:

| our term | paper's term | where |
|---|---|---|
| BER = (1/m)Σ\|b̂−b\| | (unnamed) Eq. 16 error rate | Eq. 16 |
| 1 − BER ("watermark accuracy") | "watermark detection rate" / `Acc_wm` | Fig. 8, Tables II, VII |
| "BER ≥ η → free-rider" | Eq. 16 detection rule | Eq. 16 |
| η = μ+3σ of benign BER | same | text after Eq. 16 |

- Honest client (trained the watermark in) → bits come back correct → **BER ≈ 0**
  (paper: detection rate ≈ 100%).
- Free-rider (fabricated its update) → each bit is a coin-flip vs the secret
  target → on average half wrong → **BER ≈ 0.5** (paper: rate < ~40%, Fig. 8).

So BER is the single scalar that says "is the watermark present in this model?"
0 = present (contributor), 0.5 = absent (free-rider). In a writeup, introduce it
once as "we refer to the Eq. 16 error rate as the BER," then use it freely.

**Watermark accuracy.** `1 − BER`, as a percentage — this *is* the paper's
"watermark detection rate" / `Acc_wm` (a recovered watermark ≈ 100%).

**Which metric, and why FareMark uses BER (not trigger accuracy).** Two metric
families exist in DNN watermarking, and it is easy to attach the wrong one:

| Watermark family | Payload | Recovered from | Metric |
|---|---|---|---|
| Feature-based (white-box): Uchida, FedIPR-feature | literal bit-string | model **weights** | **BER** |
| Backdoor-based (black-box): Adi, WAFFLE, FedIPR-backdoor | (input → assigned-label) pairs | model **predictions** on trigger images | **trigger / task accuracy** |
| **FareMark (box-free, output-space)** | literal bit-string | model **softmax outputs** on trigger-class inputs | **BER** |

So the common rule of thumb — "BER for a bit-string payload, trigger accuracy
for prediction behavior on trigger images" — is right as a *general* taxonomy,
but two things must be stated correctly for FareMark:
- FareMark's BER is on a bit-string extracted from **outputs, not weights**
  (Eq. 1/13/15 act on the probability vector p̂). The "from the weights" phrasing
  applies only to feature-based schemes; FareMark is box-free precisely to avoid
  weight access.
- The "trigger prediction error rate" branch describes **backdoor-based**
  watermarking, which FareMark defines itself *against*. FareMark still
  classifies trigger-class images to their **true** label; the mark lives in the
  *shape of the tail probabilities*, not in the predicted class. There is no
  trigger-prediction error rate in FareMark's verification — measuring one would
  just be measuring main-task accuracy.

**Terminology collision to avoid.** FareMark *does* report a "task accuracy,"
but that is the **fidelity** metric (main-task classification over the whole
test set, Table I) — a separate axis from watermark BER. Never let "trigger
accuracy" (a backdoor metric absent here) blur into either FareMark's BER or its
fidelity accuracy.

**Threshold η.** Decision boundary on BER. A client is flagged as a free-rider
when `BER ≥ η`. The paper sets `η = μ + 3σ` of the benign BER distribution.
Our verifier adds two guards (sliding window + cap at 0.25) for the high-
free-rider regime the paper does not stress.

**Confusion-matrix metrics** (over the clients in a round):
- **Detection accuracy** = (honest kept + free-riders caught) / all clients.
  This is the single percentage the paper reports as `Acc_fr`.
- **Recall (TPR)** = free-riders caught / all free-riders.
- **FPR** = honest flagged / all honest.

**Separability margin.** `min(free-rider BER) − max(honest BER)`. Positive =
separable; ≤ 0 = overlap = impossible.

---

## 3. How the paper evaluates (all via the Eq. 16 error rate = our BER)

The Eq. 16 error rate — which we call BER, and which the paper reports as its
complement `1 − BER` ("watermark detection rate" / `Acc_wm`) — is the
measurement behind almost every table:

| Paper section | What is measured (in the paper's words) | = in our BER terms | Pass condition |
|---|---|---|---|
| **Fidelity** (Tables II, VII) | watermark detection rate `Acc_wm` after embedding | `1 − BER` | `Acc_wm ≈ 100%` while task accuracy stays near baseline |
| **Free-rider detection** (Tables III–V, Fig. 8) | `Acc_fr`, FPR via the Eq. 16 threshold | benign BER ≈ 0, free-rider BER ≈ 0.5, high `Acc_fr` | clean separation by η |
| **Robustness** (Figs 9–10, Table VI) | watermark recovery rate after fine-tune / prune / quantize / DP | `1 − BER` vs attack strength | recovery stays high until the attack also destroys task accuracy |

In other words: the paper never evaluates "detection" as a standalone quantity —
it evaluates the Eq. 16 error rate (our BER) and dresses it up as detection
rate, `Acc_fr`, and robustness curves. That is exactly why this project attacks
BER separability: collapse the gap between the benign and free-rider error-rate
distributions and every one of these claims fails at once.

---

## 4. Baseline status — finalize before drifting

Four required capabilities and their evidence:

| Capability | Evidence | Remaining |
|---|---|---|
| Federated learning (FedAvg) | ResNet-18/CIFAR-10 93.2%, MNIST 99.5% | done |
| Free-rider attacks | previous-model + Gaussian degrade accuracy on the Fig-7 trend | done |
| Watermark embedding | ~2-pt fidelity cost, benign BER ≈ 0 | done |
| Detection | 0.94–1.00 across attacks/datasets, FPR ≈ 0 | **run robustness** |

To declare the baseline finished:
1. **Run robustness** (`run_robustness.py`) — the only un-run piece (Figs 9–10 / Table VI).
2. **10 repeats** on the core fidelity + detection configs (paper averages 10; we mostly have 1).
3. Keep the provenance note for the two documented deviations (trigger-class
   exclusion at l=2, sign-balanced keys) and the two assumed hyperparameters
   (momentum 0.9, weight-decay 5e-4).

After that the code is a validated baseline and any change is "our contribution,"
not a reproduction question.

---

## 5. Weaknesses to demonstrate (thesis pillars)

1. **Bit-count ceiling (our analysis, not a paper finding).** The paper leaves
   m open; under our embeddability assumptions m = (n−1)/2, so few classes → few
   bits → a random free-rider's BER lands under η by chance (~⅓ of the time at
   m = 4) → the distributions overlap. Information-theoretic, and an
   unexamined gap in the paper.
2. **Forgeability.** A free-rider with the key + a few trigger samples re-embeds
   the mark; its BER falls with effort. Detection silently assumes the attacker
   won't try.
3. **Threshold fragility.** `μ+3σ` degenerates exactly under heavy free-riding,
   when honest BER spikes — the regime it is meant for.
4. **Non-IID false positives.** Skewed data → some honest clients can't embed →
   their BER rises → they get misflagged.

---

## 6. Knobs to tweak

| Knob | Flag | Effect to probe |
|---|---|---|
| dataset / #classes | `--dataset cifar10\|cifar100` | bit-count ceiling |
| free-rider effort | `--blend`, `--n_trigger_samples` | forgeability |
| free-rider fraction | `--num_free_riders 2..8` | threshold under stress |
| data skew | `--partition dirichlet --dirichlet_alpha` | non-IID false positives |
| attack timing | `--attack random_round --honest_prob` | sporadic-honesty evasion |
| watermark strength | `--wm_lambda`, `--wm_beta` | fidelity vs robustness trade |
| smoothing | `--wm_bits`, `wm_alpha` | group size l, embeddability |

---

## 7. Experiments to run now (exact CLI + expected results)

All commands assume you are in the repo root (or pod working copy), with
`--data_root` and `--output_dir` pointing at your NFS paths. Each writes one
`result.json` into its `--output_dir`; point `plot_results.py` at the parent
folder afterward. On the cluster, submit each as a separate fire-and-forget job
(`submit_experiment.sh`, `WAIT=0`) using the same overrides as env vars.

### 7.0 Finalize baseline — robustness (do first)

```
python scripts/run_robustness.py --config_idx 11 --repeat 0 \
    --data_root $DATA --output_dir $OUT/robustness
```
**Expect:** baseline task ≈ 91%, watermark ≈ 100%. Fine-tune: watermark accuracy
decays as epochs rise (2→20) while task stays near baseline (Fig. 9). Prune:
watermark + task tolerant to ~50%, both collapse past ~60% (Fig. 10). Quantize:
8-bit ≈ unchanged, 2-bit degrades. Writes `robustness.json`.

### 7.1 Forgeability sweep — the headline run (CIFAR-10, mixed attack)

```
# vary disguise blend
for B in 0.3 0.5 0.7; do
  python scripts/run_experiment.py --config_idx 12 --attack mixed \
      --blend $B --n_trigger_samples 8 \
      --data_root $DATA --output_dir $OUT/mixed_blend_$B
done
# vary attacker trigger samples
for NS in 2 8 32; do
  python scripts/run_experiment.py --config_idx 12 --attack mixed \
      --blend 0.5 --n_trigger_samples $NS \
      --data_root $DATA --output_dir $OUT/mixed_ns_$NS
done
```
**Expect:** as blend↑ (more of the attacker's own lightly-trained weights) and
n_trigger_samples↑, free-rider BER **drops toward the honest cluster**. On
CIFAR-10's 4 bits, expect fr-BER to dip under 0.25 at high effort → recall falls
below 1.0 and detection drops below 1.0. This is the overlap that supports the
forgeability claim. If detection stays perfect, raise blend toward 0.8.

### 7.2 Threshold fragility — free-rider fraction sweep (CIFAR-10)

```
for N in 2 4 6 8; do
  python scripts/run_experiment.py --config_idx 12 --attack previous_models \
      --num_free_riders $N \
      --data_root $DATA --output_dir $OUT/fr_$N
done
```
**Expect:** detection ≈ 1.0 at N=2,4,6; a dip at N=8 (≈0.94, recall ≈0.9) —
matching the paper's "≈6% degradation at 80%." Watch `wm_eta_round`: if the
model briefly collapses, η behaviour at N=8 is the fragility evidence.

### 7.3 Non-IID false positives — Dirichlet sweep (all honest, watermarked)

```
for A in 0.1 0.5 1 100; do
  python scripts/run_experiment.py --config_idx 12 --num_free_riders 0 \
      --partition dirichlet --dirichlet_alpha $A \
      --data_root $DATA --output_dir $OUT/noniid_$A
done
```
**Expect:** at α=100 ≈ IID → benign BER ≈ 0, FPR ≈ 0. As α↓ to 0.5 then 0.1,
some honest clients lack their trigger class → benign BER rises → **FPR climbs
above 0** even with no free-riders present. The α at which FPR departs from 0 is
the non-IID failure point.

### 7.4 Bit-count ceiling — CIFAR-10 vs CIFAR-100 (mixed attack, same effort)

```
python scripts/run_experiment.py --config_idx 12 --attack mixed \
    --blend 0.5 --n_trigger_samples 8 --dataset cifar10 \
    --data_root $DATA --output_dir $OUT/bits_cifar10
python scripts/run_experiment.py --config_idx 12 --attack mixed \
    --blend 0.5 --n_trigger_samples 8 --dataset cifar100 \
    --data_root $DATA --output_dir $OUT/bits_cifar100
```
**Expect:** CIFAR-100 (≈49 bits) stays cleanly separable (fr-BER ≈ 0.5,
detection 1.0); CIFAR-10 (4 bits) shows the same attacker getting much closer to
(or under) the threshold. The contrast is the bit-count argument.

### 7.5 Make the figures

```
python scripts/plot_results.py --in $OUT/mixed_blend_0.3 $OUT/mixed_blend_0.5 $OUT/mixed_blend_0.7 --out figs/forgeability_blend
python scripts/plot_results.py --in $OUT/fr_2 $OUT/fr_4 $OUT/fr_6 $OUT/fr_8     --out figs/threshold
python scripts/plot_results.py --in $OUT/noniid_0.1 $OUT/noniid_0.5 $OUT/noniid_1 $OUT/noniid_100 --out figs/noniid
python scripts/plot_results.py --in $OUT/bits_cifar10 $OUT/bits_cifar100        --out figs/bitcount
```
Each produces `separability.png` (the thesis figure), a `sweep.png`, and per-run
BER trajectories. Read the printed `margin=±…` line: negative = overlap = a
demonstrated impossibility regime.

### 7.6 Repeats (credibility)

Re-run configs 1, 11, 12 with `--repeat 0 … 9` and aggregate:
```
python scripts/aggregate_results.py $OUT
```
**Expect:** tighter means and small std; the single-run jitter (esp. CIFAR-10
detection at N=8) shrinks toward the paper's values.

---

## 8. Next steps / two-month timeline

- **Weeks 1–2:** robustness + 10-repeats → lock the baseline.
- **Weeks 3–5:** run 7.1–7.4 → one separability figure each.
- **Weeks 6–7:** push the mixed/forgery attack until honest & free-rider BER
  overlap on CIFAR-10 (raise blend, give the attacker the key + more samples).
- **Week 8:** write up the conditions for impossibility; draft the limitations paper.

**Decide early:** the **threat model** (does the attacker hold the key? the
trigger data?) — every forgeability claim depends on it. And whether the
argument is empirical (detection fails in these regimes) or also **formal** (a
short lemma: with m bits, a random model's `P(BER < η)` is a Binomial tail; on
m=4, `P(BER ≤ 0.25) ≈ 0.31`, so overlap is unavoidable). The formal lemma is
tractable and strengthens the paper.

**Out of scope** for a limitations paper: ShuffleNet/GoogLeNet, Food-101, and
the DAGMM/FedIPR baselines — skip once the baseline is locked.