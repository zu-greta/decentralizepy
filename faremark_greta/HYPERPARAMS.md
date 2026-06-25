# Hyperparameter & Toggle Reference

Every movable part in the code, what it maps to, how to set it, and its effect.
Set anything via CLI on `run_experiment.py` (`--flag`) or via the matching env
var on `submit_experiment.sh`. Defaults live in `faremark/config.py` (`ExpConfig`).

## Mode toggles (decide which algorithm you are running)

| Flag / env | Default | What it is | Effect / when to use |
|---|---|---|---|
| `--paper_faithful` / `PAPER_FAITHFUL` | off | Strips our 3 deviations at once | Runs the **bare paper algorithm**: random (not sign-balanced) keys, **no** trigger-class exclusion (full softmax), and a **cumulative uncapped** μ+3σ threshold. Use with CIFAR-100 so the full-softmax projection is embeddable. This is the mode for "is the weakness real or my artifact?" |
| `--calib_on_all` / `CALIB_ON_ALL` | off | Calibrate η over **every** client, not just benign | Demonstrates the **threshold-poisoning / circularity** weakness: free-rider BER ≈ 0.5 inflates μ+3σ. Off = the paper's assumed trusted benign pool. |

When `paper_faithful=off` (our robust mode), three guards are active: trigger-class
exclusion, sliding-window η (last 15 rounds), and η capped at 0.25. Disclose these
in any writeup; they are why our detector behaves better than the bare paper.

## Federated-learning knobs

| Flag / env | Default | What it is | Effect |
|---|---|---|---|
| `--config_idx` (positional on submit) | — | Selects an `ExpConfig` (0–13) | Picks model/dataset/attack preset. 12 = CIFAR-10 detection, 13 = paper-faithful CIFAR-100. |
| `--repeat` (positional) | 0 | Seed selector (`base_seed+repeat`) | Run 0–9 for the paper's 10-repeat averaging. |
| `--rounds` / `ROUNDS` | 50 | Communication rounds | More rounds → better convergence and watermark embedding; longer runtime. |
| `--local_epochs` / `LOCAL_EPOCHS` | 5 | Local SGD epochs per round | More local work per round; affects embedding strength and convergence. |
| `--batch_size` / `BATCH_SIZE` | 16 | Local batch size | Paper uses 16. Larger = faster, slightly different optimization. |
| `--lr` | 0.01 | Learning rate | Paper value. |
| `--model` / `MODEL` | per config | resnet18 / alexnet / smallcnn | Architecture. SmallCNN only for fast smoke tests. |
| `--dataset` / `DATASET` | per config | mnist / cifar10 / cifar100 | **Class count = bit budget.** cifar10→~4 bits (overlap-prone), cifar100→~49 bits (clean). The bit-count lever. |
| `--num_clients` (config only) | 10 | FL clients | Oversubscription study: >#classes forces shared trigger classes. |

## Data-distribution knobs (non-IID)

| Flag / env | Default | What it is | Effect |
|---|---|---|---|
| `--partition` / `PARTITION` | iid | `iid` or `dirichlet` | Switches to label-skewed non-IID split. |
| `--dirichlet_alpha` / `DIRICHLET_ALPHA` | 0.5 | Skew strength | Small = severe skew (α=0.1: clients see few classes → honest BER↑, FPR↑); large (α≥100) ≈ IID. The non-IID lever. |

## Free-rider / attack knobs

| Flag / env | Default | What it is | Effect |
|---|---|---|---|
| `--attack` / `ATTACK` | per config | none / previous_models / gaussian / train_then_attack / trigger_only / random_round / mixed | Which fabrication the free-rider uses. `mixed` = the forgery adversary. |
| `--num_free_riders` / `NUM_FREE_RIDERS` | per config | How many of N clients cheat | Dilution / threshold-stress lever; high fractions can collapse the model. |
| `--noise_sigma` / `NOISE_SIGMA` | 0.1 | Gaussian-attack noise std | Bigger = more degradation, easier to detect. |
| `--blend` / `BLEND` | 0.5 | mixed: weight on attacker's own lightly-trained weights | Higher = more genuine signal → free-rider BER drops toward honest (forgeability). |
| `--n_trigger_samples` / `N_TRIGGER_SAMPLES` | 8 | trigger_only/mixed: # trigger samples the attacker fits | More → better forged mark → lower free-rider BER. |
| `--honest_prob` / `HONEST_PROB` | 0.5 | random_round: per-round prob of training honestly | Sporadic-honesty evasion. |
| `--attack_round` / `ATTACK_ROUND` | 50 | train_then_attack: round it defects | Earlier defect = easier to detect (mark didn't persist). |

## Watermark knobs

| Flag / env | Default | What it is | Effect |
|---|---|---|---|
| `--watermark` / `WATERMARK` | per config | Turn embedding on | Off = plain FedAvg (no detection). |
| `--wm_bits` / `WM_BITS` | 0 (auto) | m, message length | 0 → auto. **Non-paper-faithful:** auto = (classes−1)//2 (max bits; balanced keys make them all embeddable). **Paper-faithful:** auto = classes//10 → group size l≈10 (a *faithful* default: random keys are almost surely mixed-sign, so no same-sign floor). Set `WM_BITS=49` in paper-faithful mode to opt into the **max-payload** stress case (l=2, ~half the bits unembeddable) — that is the bit-count pillar, not the generic paper. |
| `--wm_lambda` / `WM_LAMBDA` | 5.0 | Weight of L_wm in total loss (Eq. 11) | Higher = stronger embedding, more fidelity cost. |
| `--wm_beta` / `WM_BETA` | 0.6 | Memory coefficient (Eq. 14), **per client** | 0 = plain FedAvg (mark washes out); higher = mark survives aggregation but convergence slows. Tuned heuristically. |
| `wm_alpha` (config) | 0.4 | Smoothing f() exponent (Eq. 8) | Smaller = flatter softmax tail, more room to shape bits; too small hurts accuracy. |
| `wm_f` (config) | power | Smoothing kind (power / sin) | Eq. 7–9 alternatives. |
| `wm_label_smoothing` (config) | 0.1 | Label smoothing | Keeps the softmax tail movable so bits can be shaped. |
| `--wm_num_triggers` / `WM_NUM_TRIGGERS` | 50 | N_T verification triggers (Eq. 15) | More = more reliable extraction (paper: ≥10 → >99%). |
| `wm_eta` (config) | 0.25 | Detection threshold floor / our cap (Eq. 16) | In our mode also the η cap; in paper mode only the floor. |
| `wm_verify_every` (config) | 1 | Verify every k rounds | Cost control. |

## Robustness driver (run_robustness.py)

Sweeps are hard-coded inside the script (fine-tune epochs 2/5/10/20, prune
0.2–0.8, quantize 8/4/2-bit). Change them by editing the loops in
`scripts/run_robustness.py`. Launch with `SCRIPT=scripts/run_robustness.py`.

## Where to change things for a fully paper-exact run

`--paper_faithful` already flips all three deviations. If you want them
individually: trigger-class exclusion lives in `wm_client.build_watermarked_clients`
(`exclude_col`), key balance in `watermark.make_key` (`balanced=`), and the
threshold window/cap in `wm_verify.make_verifier` (`paper_faithful` branch). The
detector's per-round metrics are written to `result.json["history"]`; the
converged summary (last-10-round mean) is the top-level `wm_*` fields.

---

## Measurement metrics — plain-language glossary

Every metric the detector reports, what it means, how it is computed, its range,
and how to read it. Per-round values live in `result.json["history"]`; the
top-level `wm_*` fields are the converged summary (mean over the last
`wm_detect_window = 10` rounds).

### Accuracy metrics

**`test_acc` / `final_acc` / `best_acc` — global model accuracy (%).**
Top-1 accuracy of the *aggregated* global model on the held-out test set, each
round. `final_acc` is the last round, `best_acc` the max over rounds. Range
0–100. This is **fidelity**: it tells you the watermark/attack isn't destroying
the model. It says nothing about detection.

### The core quantity

**`BER` — bit-error-rate (per client).** The fraction of a client's `m`
watermark bits that the server extracts *wrong*: `BER = (1/m) Σ |b̂ₖ − bₖ|`
(Eq. 16, left side). Range 0–1.
- `BER ≈ 0` → the client embedded its watermark → it really trained → **honest**.
- `BER ≈ 0.5` → bits are random w.r.t. the secret target → no watermark → **free-rider**.
- Granularity is `1/m`: CIFAR-10 (m=4) moves in steps of 0.25; CIFAR-100 (m=49)
  in steps of ~0.02. Coarse steps at low `m` are themselves a limitation signal.
"BER" is our name for the Eq. 16 error rate; the paper reports its complement,
"watermark accuracy" `Acc_wm = 1 − BER`.

**`wm_benign_ber` — mean BER over honest clients.** Average of the per-client
BER across all non-free-rider slots that round. Low is healthy. In paper-faithful
max-bit mode it floors above 0 because ~half the random key rows are
unembeddable (same-sign rows at group size l=2) — that floor is a property of
the algorithm at high payload, not a bug.

**`wm_fr_ber` — mean BER over free-rider clients.** Same average, restricted to
the planted free-riders. `null` when there are none. Expected ~0.5; values like
0.44 mean the free-rider partially fakes the mark.

### Detection-quality metrics (the ones that matter for the claim)

**`wm_eta_round` / `wm_eta_used` — the detection threshold η.** A client is
flagged free-rider when its `BER ≥ η`. Computed as `μ + 3σ` of the benign BER
distribution (Eq. 16). `wm_eta_round` is η that round; `wm_eta_used` is the
converged value. Two modes: paper-faithful = cumulative over *all* rounds (no
window, no cap) → inflates because early untrained rounds sit at BER~0.5; our
mode = sliding 15-round window, capped at 0.25 → tracks the converged honest
cluster. **If η sits far above both BER clusters, the detector flags nobody.**

**`wm_fpr` — false-positive rate.** Fraction of *honest* clients wrongly flagged
as free-riders: `benign_flagged / n_benign`. Range 0–1. **This is the non-IID
headline metric**: it rises with data skew even when zero free-riders exist.
0 = no honest client misclassified; 1 = every honest contributor wrongly accused.

**`wm_fr_recall` — free-rider recall (detection rate).** Fraction of *actual*
free-riders that were caught: `fr_flagged / n_fr`. Range 0–1; `null` if none.
**This is the true measure of whether detection works.** recall 1.0 = every
free-rider caught; recall 0.05 = they ride free. Read this, not detect_acc.

**`wm_detect_acc` — detection accuracy.** Overall fraction classified correctly:
`(true negatives + true positives) / total clients`. Range 0–1. **Misleading
when free-riders are rare**: with 8 honest + 2 free-riders, flagging nobody
still scores 0.8 because the 8 honest are "correct." Always cross-check against
recall; detect_acc flatters the scheme.

### Embeddability diagnostic (explains an honest-BER floor)

**`wm_unembeddable_frac` (with `wm_bits_m`, `wm_group_size_l`).** Fraction of
secret-key rows that are *same-sign* (`[+1,+1]` or `[−1,−1]`). Because `f(p) ≥ 0`,
such a row forces its bit to a fixed sign for every input → that bit is
**structurally unembeddable** and sits at ~50% error regardless of training.
Range 0–1, computed once at setup. With balanced keys (non-paper-faithful) it is
0. With random keys it grows as group size shrinks: `P(same-sign) = 2^(1−l)`, so
`l=2 → 0.50`, `l=3 → 0.25`, `l≥6 → negligible`. **Use it to attribute a floor:**
an honest BER stuck near `0.5 × wm_unembeddable_frac` is this artifact, not data
skew or attack — distinguishing "the scheme can't carry this payload" from "the
data distribution broke detection." `wm_bits_m` and `wm_group_size_l` record the
`m` and `l` actually used (which now differ between modes), so every run
self-documents its bit budget. A warning is logged at setup when the fraction
exceeds 10%.

A scheme "works" only if the honest and free-rider BER distributions are
**disjoint** *and* η sits in the gap between them. Then FPR→0 and recall→1
simultaneously. It fails two ways: **overlap** (honest BER rises into the
free-rider band — non-IID; no η can separate them, FPR and recall trade off) or
**mis-set threshold** (distributions are separable but η is parked outside the
gap — paper-faithful cumulative η; recall collapses while detect_acc still looks
fine). The **separability figure** visualizes both: it sweeps η and prints the
margin (gap between the two clusters). Positive margin = a good η exists;
negative margin = genuine impossibility.