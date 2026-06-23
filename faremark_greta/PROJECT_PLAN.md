# Project Plan — Limits of Output-Layer Watermarking for Free-Rider Detection

Status: reproduction of FareMark essentially complete; pivoting to an attack /
limitations study. This document is the working reference for the new goal,
the definitions it rests on, the weaknesses to demonstrate, and the exact
experiments to run next.

---

## 1. Goal (reframed)

Show that **box-free, output-layer watermarking cannot reliably detect
free-riders under identifiable conditions** — i.e. detection is not just hard
but *impossible* in regimes that occur in normal federated learning.

The entire argument reduces to one question, asked over and over in different
settings:

> Are the honest-client and free-rider **bit-error-rate (BER)** distributions
> separable by a single threshold η?
>
> - **Disjoint** → a band of η gives perfect detection (the scheme works).
> - **Overlapping** → no η separates them → detection is impossible *here*.

Every experiment below produces a *separability figure* answering this for one
regime. The collection of regimes where the gap closes **is** the paper.

---

## 2. Definitions

**Bit.** One yes/no piece of a client's secret watermark message. The scheme
can pack only about `(num_classes − 1) / 2` bits into the output layer, so the
message length is **m = 4 bits on CIFAR-10** and **m ≈ 49 on CIFAR-100**.

**Watermark (B^i).** Each client `i` has its own m-bit string `B^i` plus a
secret ±1 key `M`, embedded into the model's softmax on that client's *trigger
class*.

**Bit-error-rate (BER).** The fraction of the m recovered bits that disagree
with the registered target bits:

```
BER = (number of wrong bits) / m        ∈ [0, 1]
```

- Honest client (trained the watermark in) → bits come back correct → **BER ≈ 0**.
- Free-rider (fabricated its update) → each bit is a coin-flip vs the secret
  target → on average half wrong → **BER ≈ 0.5**.

So BER is the single scalar that says "is the watermark present in this model?"
0 = present (contributor), 0.5 = absent (free-rider).

**Watermark accuracy.** `1 − BER`, expressed as a percentage. This is how the
paper reports fidelity and robustness (a recovered watermark = ~100%).

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

## 3. How FareMark uses BER to evaluate (three roles)

BER (usually reported as `1 − BER` = "watermark accuracy") is the measurement
behind almost every table in the paper:

| Paper section | What they measure with BER | Pass condition |
|---|---|---|
| **Fidelity** (Tables II, VII) | Watermark accuracy `1−BER` after embedding — does the mark survive training + FedAvg aggregation? | `1−BER ≈ 100%` while task accuracy stays near baseline |
| **Free-rider detection** (Tables III–V, Fig. 8) | Each client's BER → threshold η → `Acc_fr`, recall, FPR | honest BER≈0, free-rider BER≈0.5, high `Acc_fr` |
| **Robustness** (Figs 9–10, Table VI) | Watermark accuracy `1−BER` after a removal attack (fine-tune / prune / quantize / DP) vs attack strength | watermark accuracy stays high until the attack also destroys task accuracy |

In other words: the paper never evaluates "detection" directly — it evaluates
**BER**, and detection/fidelity/robustness are all thresholds or trends on BER.
That is exactly why the project attacks BER separability: break the BER gap and
you break every claim at once.

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

1. **Bit-count ceiling.** Few classes → few bits → free-rider matches by luck
   (~⅓ of the time on 4 bits) → distributions overlap. Information-theoretic.
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
