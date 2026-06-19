# FareMark — Code ↔ Paper Map

A precise, module-by-module mapping between this reproduction and **Li et al.,
"FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning
Model," IEEE IoT-J 2025**. Every equation, table, and figure in the paper is
listed with the code that implements or produces it.

The codebase is a single-process simulation of centralized FedAvg (the paper's
setting): clients train sequentially on one GPU, the server aggregates by
parameter averaging. This is mathematically identical to the paper's protocol;
only the network/parallelism is collapsed.

---

## 1. Paper structure → code

| Paper section | What it defines | Code |
|---|---|---|
| §III-A/B | FL + watermarking setup | `server.py`, `client.py`, `datasets.py`, `models.py` |
| §IV-A (Eq. 1–10) | Watermark representation + smoothing | `watermark.py` |
| §IV-B (Eq. 11–13) | Client-side embedding loss | `watermark.py`, `wm_client.py` |
| §IV-C (Eq. 14) | Memory-enhanced update | `wm_client.py` |
| §IV-D (Eq. 15–16) | Free-rider detection | `wm_verify.py` |
| §IV-E | IPR / ownership verification | `wm_verify.py` (same extractor) |
| §V-A2 (Eq. 17–18) | Free-rider attack models | `attacks.py` |
| §V-D3/4 | Adaptive free-riders | `attacks.py` (train-then-attack, trigger-only) |
| §V-E | Robustness (fine-tune/prune/quantize/DP) | `robustness.py`, `scripts/run_robustness.py` |

---

## 2. Module-by-module

### `faremark/models.py` — networks (§V-A)
`build_model(name, num_classes, in_channels)`. Implemented: **ResNet-18**
(CIFAR stem: 3×3 stride-1 conv, no max-pool), **AlexNet** (small-image), and
`SmallCNN` (fast smoke tests). **Still to add for full reproduction:**
**ShuffleNet** and **GoogLeNet** (paper §V-A lists all four).

### `faremark/datasets.py` — data + IID split (§V-A1)
`build_data(...)`: **MNIST, CIFAR-10, CIFAR-100**, training set split **evenly
(IID)** across clients, full test set for evaluation, with `test_dataset` exposed
for trigger sampling. **Still to add:** **Food100** (not in torchvision; likely a
100-class Food-101 subset — confirm before use).

### `faremark/client.py` — honest local training (§III)
`Client.produce_update(global_state, prev_global_state, round_idx)` — the single
seam later stages override. Honest clients load the global model, run local SGD
on their own shard, return weights + sample count.

### `faremark/server.py` — FedAvg + round loop (§III, Eq. agg)
`Aggregator.aggregate` = sample-weighted mean `W_g = (1/N) Σ W_i`. `Server.run`
distributes → collects → aggregates → evaluates each round; keeps `W_{t-1}`
(needed by Eq. 17); `verify_hook` is the Stage 3/4 extraction/detection plug-in.

### `faremark/watermark.py` — the watermark (§IV-A/B/D)
| Function | Paper |
|---|---|
| `smooth(p, kind, alpha)` | f(): `x^α` (Eq. 7–8), `sin(αx)` (Eq. 9) |
| `make_key(m, l, seed)` | secret ±1 matrix M; **sign-balanced rows** (see note) |
| `make_bits(m, seed)` | watermark message B^i (balanced) |
| `grouping(n, m)` | `l = n // m` group size |
| `project_logits(probs, key, …, exclude)` | `z_k = Σ_j f(p_j)·M_{k,j}` (Eq. 1/13) |
| `watermark_loss(...)` | `L_wm` = BCE driving sign(z_k)→b_k (Eq. 11–12) |
| `extract_bits(...)` | average z over N_T triggers, then sign (Eq. 15) |
| `bit_error_rate`, `detected` | `BER = (1/m)Σ|b̂−b|`, `BER < η` (Eq. 16) |
| `calibrate_eta(...)` | `η = μ + 3σ` of benign BER (Eq. 16) |
| `dominance_ratio(...)` | `f(p_max)/Σf < 0.5` diagnostic (Eq. 4–6/10) |

**Grouping matches the paper exactly:** Eq. 15 reads the `(l·(k−1)+j)`-th output,
i.e. consecutive blocks of size `l` over the first `m·l` softmax classes.

**Two documented refinements** (faithful to intent; required at small `l`):
1. `exclude` the trigger class — the paper's Eq. 4–6/10 keeps `p_max ≤ 0.5`
   inside a group; at `l=2` with argmax = trigger that is impossible without
   breaking classification, so the trigger group's bit freezes. Excluding the
   trigger enforces the same anti-dominance intent by construction. Pass
   `exclude=None` for the paper-exact full-softmax read (use `l ≥ 3`).
2. sign-balanced key rows — because `f(p) ≥ 0`, a same-sign row forces a fixed
   bit; the paper's example `M=[1,−1,1]` is mixed-sign, which is automatic for
   random rows when `l` is large but must be enforced at `l=2`.

### `faremark/wm_client.py` — embedding + memory (§IV-B/C)
`WatermarkClient.produce_update`: trains `L = L_cl + λ·L_wm`, applying `L_wm`
**only to trigger-class samples** (§V-A1's normal/trigger data split), with label
smoothing to keep the tail movable; then the **memory-enhanced update** (Eq. 14)
`W_new = β·(memory + Δ) + (1−β)·W_global` (Δ = this round's local step). `β=0`
recovers plain FedAvg. `build_watermarked_clients` assigns each client a distinct
trigger class + secret key + message and registers them.

### `faremark/wm_verify.py` — detection (§IV-D, Eq. 15–16)
`WatermarkRegistry` (per-client trigger/key/bits), `build_trigger_bank` (N_T
trigger images per class from the test set), `make_verifier` → the server's
`verify_hook`: extract each submitted model's watermark, compute BER, flag a
free-rider when `BER ≥ η`. Reports benign BER, free-rider BER, detection
accuracy (`Acc_fr`), watermark accuracy (`Acc_wm`), and FPR.

### `faremark/attacks.py` — free-riders (§V-A2, §V-D3/4)
| Class | Paper |
|---|---|
| `PreviousModelsFreeRider` | Eq. 17, `W_free = 2W_t − W_{t-1}` |
| `GaussianNoiseFreeRider` | Eq. 18, `W_free = W_t + N(0,σ²)` |
| `make_train_then_attack` | §V-D3, Table IV (trains early, defects at round R) |
| `make_trigger_only` | §V-D4, Table V (trains on few trigger samples → overfits) |
Norm-buffer guard `_is_norm_buffer`: never extrapolate/perturb BatchNorm running
stats (would push variance negative → NaN). Only matters for ResNet/GoogLeNet.

### `faremark/robustness.py` — watermark removal (§V-E)
`finetune` (λ=0 retrain → Fig. 9), `prune_model` (global L1 → Fig. 10),
`quantize` (precision reduction → §V-E), `dp_noise` (clip + Gaussian, Table VI;
for a faithful Table VI train under Opacus and reuse the verifier).

### scripts
- `run_experiment.py` — one (config, repeat): Stage 1/2/3 train + evaluate +
  watermark metrics → `result.json`.
- `run_robustness.py` — Stage 4: train watermarked model, sweep
  fine-tune/prune/quantize → `robustness.json` (Figs. 9–10, Table VI).
- `aggregate_results.py` — mean ± std grouped by (config, attack, #free-riders);
  `--fig7` prints the accuracy-vs-free-rider trend.

---

## 3. Equation checklist (every equation in §IV)

| Eq. | Meaning | Code |
|---|---|---|
| 1 | `z_k = Σ_j p̂_j M_{k,j}` | `project_logits` (with f) |
| 2 | `b_k = δ(z_k)` (≥0→1) | `extract_bits` sign |
| 3–6 | worked example + `p_max≤0.5` constraint | `dominance_ratio` |
| 7 | `f(x)=x^α, α<0` | `smooth(kind="power", alpha<0)` |
| 8 | `f(x)=x^α, 0<α<1` | `smooth` default |
| 9 | `f(x)=sin(αx)` | `smooth(kind="sin")` |
| 10 | `f(max)/Σf < 0.5` | `dominance_ratio` |
| 11 | `L = L_cl + λ L_wm` | `WatermarkClient._local_train_wm` |
| 12 | `L_wm` = BCE | `watermark_loss` |
| 13 | `z̃_k = Σ_j f(p̂_j) A_{k,j}` | `project_logits` |
| 14 | memory-enhanced update | `WatermarkClient._memory_update` |
| 15 | extract = avg over N_T then sign | `extract_bits` |
| 16 | `BER<η`, `η=μ+3σ` | `bit_error_rate`/`detected`/`calibrate_eta` |
| 17 | previous-models free-rider | `PreviousModelsFreeRider` |
| 18 | Gaussian-noise free-rider | `GaussianNoiseFreeRider` |

---

## 4. Tables & figures → how to produce them

| Paper | Experiment | Driver / config |
|---|---|---|
| Table I | Fidelity (all watermarked, 0 FR; vs FedAvg/FedIPR) | cfg 11, `run_experiment.py` |
| Fig. 7 | Acc vs #free-riders (4 panels) | `submit_fig7.sh` on cfg 8/9 (+ AlexNet/MNIST) |
| Table II | Watermark detection accuracy, N_T=100 | cfg 11, read `wm_benign_ber` → `Acc_wm` |
| Fig. 8 | WM detection rate, benign vs FR over rounds | cfg 12, per-round `wm_benign_ber`/`wm_fr_ber` |
| Table III | Single + multi FR detection (20–80%) | cfg 12 with `num_free_riders` swept |
| Table IV | Train-then-attack FR | `make_train_then_attack`, `attack_round=50` |
| Table V | Trigger-sample-only FR | `make_trigger_only`, vary `n_trigger_samples` |
| Table VI | Robustness vs differential privacy | `run_robustness.py` / Opacus-trained client |
| Table VII | WM accuracy vs N_T | cfg 11, vary `wm_num_triggers` |
| Fig. 9 | Robustness vs fine-tuning | `run_robustness.py` finetune sweep |
| Fig. 10 | Robustness vs pruning | `run_robustness.py` prune sweep |

Baselines for the comparison columns (**not yet implemented**): **FedAvg** (set
`watermark=False` — done), **FedIPR** (feature-based N-bit + backdoor-based),
and **ST-/ATD-DAGMM** (anomaly-detection free-rider baseline, threshold 13).
These reproduce the "vs others" columns of Tables I–III; FareMark's own numbers
do not need them.

---

## 5. Extending the framework — where to add things

- **New model** → add a builder in `models.py` and a branch in `build_model`.
  (ShuffleNet/GoogLeNet go here.)
- **New dataset** → add a loader + normalization in `datasets.py` and return a
  `DataBundle`. (Food100 goes here.)
- **New free-rider behavior** → subclass `Client` (or `WatermarkClient` for
  watermark-aware ones) overriding `produce_update`; register in `ATTACKS`.
- **New watermark scheme** → add a smoothing/projection variant in `watermark.py`;
  `WatermarkClient` and the verifier consume it through the same functions.
- **New detection rule** → it all flows through `Server.verify_hook`; swap the
  function returned by `make_verifier`.
- **New robustness attack** → add a function in `robustness.py` and a sweep line
  in `run_robustness.py`.
- **Non-IID data, client sampling, secure aggregation** → these touch
  `datasets.py` (partition) and `server.py` (round loop); the watermark/detection
  code is independent of them.
