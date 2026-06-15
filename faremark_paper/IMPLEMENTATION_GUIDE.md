# FareMark — Implementation Breakdown & Results Guide

---

## Part 1 — How the Federated Learning System Works

### The Big Picture

Federated learning (FL) is a setup where multiple clients (think: hospitals, phones, IoT devices) each hold their own private data and train a shared model together without ever sharing their raw data. The central server only ever sees model weights, never the underlying data.

The standard algorithm is **FedAvg**:

```
For each training round:
  1. Server sends global model to all clients
  2. Each client trains locally on their own data
  3. Each client sends updated weights back to server
  4. Server averages all weights → new global model
  5. Repeat
```

The problem FareMark solves: a **free-rider** client skips step 2 entirely (no real training, no real data) but still receives the valuable global model at step 1. The server cannot tell the difference just by looking at the uploaded weights — the free-rider can fake them cheaply.

---

### How This Implementation Maps to the Code

#### The Server (`faremark/server.py`)

The server has three jobs:

**1. Trigger assignment (Stage I, run once before training)**
```python
trigger_map = self.server.assign_triggers(client_ids)
# Result: {0: 'dog', 1: 'cat', 2: 'car', ...}
# Each client gets one class that belongs exclusively to them
```
Each client is assigned a unique class from the dataset (e.g. client 0 "owns" the dog class). This is the key insight that avoids watermark conflicts: every client only touches their own slice of the output space.

**2. FedAvg aggregation**
```python
def aggregate(self, local_state_dicts):
    avg = sum(all_weights) / num_clients
    self.global_model.load_state_dict(avg)
```
This is the standard FedAvg from McMahan et al. (2017). Every parameter is averaged across all clients equally — no weighting by dataset size in this implementation, matching the paper.

**3. Free-rider detection**
```python
error, b_hat, is_free_rider = server.verify_watermark(
    model=submitted_model,
    client_id=cid,
    trigger_loader=trigger_loader,
    n_triggers=100,
)
```
After each client submits their model, the server queries it with trigger-class images and tries to extract the watermark. If the extracted watermark doesn't match the registered one (within threshold η), the client is flagged as a free-rider.

---

#### The Client (`faremark/client.py`)

Each client runs local SGD for `local_epochs` epochs before sending back.

**Normal training (benign clients):**

The key innovation is the **two-loss training loop**: common-class samples use only cross-entropy, trigger-class samples use cross-entropy *plus* watermark loss.

```python
# For images of the trigger class:
loss = cross_entropy(logits, labels)          # L_cl — classify correctly
     + lambda * watermark_loss(logits, key)   # L_wm — embed watermark
```

This forces the model to do two things simultaneously:
- Classify "dog" images correctly (normal training)
- Embed a specific watermark signal into the output distribution for "dog" images

The two objectives don't conflict because each client only modifies their own trigger class.

**The memory-enhanced update (Eq. 14):**

Standard SGD after FedAvg aggregation tends to "forget" the watermark — averaging with other clients' models washes it out. The fix:

```python
# Instead of: W_new = W_local + lr * gradient   (standard SGD)
# Do:
W_new = beta * (W_local + lr * gradient) + (1 - beta) * W_global
#         ↑                                    ↑
#   your local update              pulls back toward global model
```

With β = 0.9, 90% of the update comes from local gradient descent and 10% is "anchored" to the global model from the previous round. This acts as a regulariser that helps preserve the global model's knowledge while still allowing the watermark to embed.

**Free-rider clients:**

A free-rider does no training at all. Two strategies from the paper:

```python
# Strategy 1: "Previous Models" (Eq. 17)
W_fake = W_current - W_previous
# Produces near-zero weights — looks like a gradient update but isn't

# Strategy 2: "Gaussian Noise" (Eq. 18)
W_fake = W_global + small_gaussian_noise
# Perturbs the global model slightly — harder to detect naively
```

---

### One Full Training Round, Step by Step

```
Round k:
  ┌─ SERVER ──────────────────────────────────────────┐
  │  1. Broadcast W_global to all clients             │
  └───────────────────────────────────────────────────┘
         │
         ▼ (all clients receive W_global)
  ┌─ EACH BENIGN CLIENT ──────────────────────────────┐
  │  2. Load W_global as starting point               │
  │  3. For each local epoch:                         │
  │     a. On common-class batches:                   │
  │        loss = CrossEntropy(logits, labels)        │
  │        backward() → memory-enhanced step          │
  │     b. On trigger-class batches:                  │
  │        loss = CrossEntropy + λ * WM_loss          │
  │        backward() → memory-enhanced step          │
  │  4. Send W_local back to server                   │
  └───────────────────────────────────────────────────┘
  ┌─ FREE-RIDER CLIENT ───────────────────────────────┐
  │  2. Build fake W_fake (no real training)          │
  │  3. Send W_fake back to server                    │
  └───────────────────────────────────────────────────┘
         │
         ▼ (server collects all W_i)
  ┌─ SERVER ──────────────────────────────────────────┐
  │  5. Verify watermark in each submitted model      │
  │     → Flag clients without valid watermark        │
  │  6. FedAvg: W_global = mean(all W_i)             │
  │  7. Distribute new W_global                       │
  └───────────────────────────────────────────────────┘
```

---

## Part 2 — How the Watermarking Algorithm Works

### The Core Idea

Traditional backdoor watermarks work by training the model to output a specific wrong answer when it sees a special trigger image ("whenever you see this pixel pattern, say 'class 7'"). This is detectable and erasable.

FareMark uses a fundamentally different approach: the watermark is not in the model's *predictions* but in the *shape of its output probability distribution* across all classes. There is no trigger image — any image from the trigger class works.

---

### Step 1: The Softmax Output Is the Carrier

For a 10-class classifier, the model outputs a probability vector for every input:

```
Input: [dog image]
Output: [0.01, 0.02, 0.78, 0.03, 0.01, 0.04, 0.02, 0.05, 0.03, 0.01]
         airplane  car   dog  cat  deer  frog  horse  ship  truck  bird
```

The watermark is encoded in the *fine-grained shape* of this vector, not just which class wins. Even a correctly classified dog image contains a hidden watermark in the relative magnitudes of the non-dominant probabilities.

---

### Step 2: The Smoothing Function f(x)

The raw softmax output is too "spiky" — almost all probability mass sits on the correct class, making the rest nearly zero and numerically unstable for encoding. The smoothing function flattens this:

```python
# frac_power (default): f(x) = x^0.5  (square root — compresses peaks)
# Before: [0.01, 0.02, 0.78, 0.03, ...]
# After:  [0.10, 0.14, 0.88, 0.17, ...]  ← much more uniform
```

Three options are implemented (Eq. 7–9):
- `neg_power`:  f(x) = x^α,  α < 0  — inverts and amplifies small values
- `frac_power`: f(x) = x^α,  0 < α < 1  — mild compression (default, α=0.5)
- `sin`:        f(x) = sin(αx)  — periodic, nonlinear mapping

---

### Step 3: Projection Onto a Secret Key

Each client has a private pseudorandom matrix **M** (shape: `wm_bits × group_size`), values ±1. The smoothed probabilities are projected onto M to produce a scalar per watermark bit:

```
z_k = sum_j  f(p_j^k) * M_{k,j}       (Equation 1 / 13)

Where:
  p_j^k = j-th probability in the k-th group
  M_{k,j} = random ±1 key entry
  z_k = scalar "vote" for the k-th watermark bit
```

Concretely for 4-bit watermark, 10 classes (group_size = 10//4 = 2):
```
Group 1: probabilities [p0, p1]  →  z1 = f(p0)*M[0,0] + f(p1)*M[0,1]
Group 2: probabilities [p2, p3]  →  z2 = f(p2)*M[1,0] + f(p3)*M[1,1]
Group 3: probabilities [p4, p5]  →  z3 = ...
Group 4: probabilities [p6, p7]  →  z4 = ...
```

Then threshold:
```
b_hat_k = 1  if z_k >= 0
b_hat_k = 0  if z_k < 0
```

This gives 4 bits extracted from one forward pass. No special trigger image needed.

---

### Step 4: The Watermark Loss (Training)

To *embed* the watermark, we add a regularisation term that pushes z_k toward the desired bit value (Eq. 12):

```python
L_wm = sum_k [
    b_k * log(sigmoid(z_k))           # if target bit = 1, push z_k > 0
  + (1 - b_k) * log(1 - sigmoid(z_k)) # if target bit = 0, push z_k < 0
]
# This is binary cross-entropy on z_k with target b_k
```

So for each batch of trigger-class images, gradient descent simultaneously:
1. Minimises classification loss (correct label wins)
2. Shapes the output distribution so that z_k has the correct sign for each bit

---

### Step 5: Extraction and Detection (Inference)

To verify a client's watermark, the server collects N_T = 100 trigger-class images and:

```python
# 1. Get logits for all trigger samples
logits = model(trigger_images)  # shape: (100, num_classes)

# 2. Average z_k across all samples (Eq. 15)
z_mean = mean_over_samples( projection(logits) )  # shape: (wm_bits,)

# 3. Threshold
b_hat = (z_mean >= 0).float()

# 4. Compare with registered watermark
error = mean( |b_hat - b_registered| )

# 5. Detect
is_free_rider = (error > eta)
```

A free-rider never trained on the trigger class with the watermark loss, so the model's output distribution for trigger-class images has random z_k signs → about 50% bit accuracy → high error → flagged.

---

### Why Free-Riders Can't Fake It

A free-rider knows their assigned trigger class (public info) and could try to query the global model with trigger images and report back whatever watermark they observe. The paper addresses this: the secret key M is never revealed to the client. The server uses M to verify; the client can only submit weights, not claim a watermark value. If a free-rider tries to train on just a few trigger samples (Table V), the watermark overfits and fails to generalise to the server's held-out trigger set.

---

## Part 3 — Reading Your `results.json`

Here is your smoke test result annotated line by line:

```json
{
  "rounds": [1, 2],
```
Training was evaluated at rounds 1 and 2 (you ran `global_rounds=2`, `eval_every=1`).

---

### Main Task Accuracy

```json
  "main_acc": [0.0994, 0.1],
```

**What it means:** Top-1 classification accuracy on the CIFAR-10 test set after each round.

**Is this normal?** Yes — completely expected. CIFAR-10 has 10 classes, so a random model scores ~10% = 0.10. After only 2 rounds with 1 local epoch each, the model has barely trained at all. The paper runs 100 rounds with 2–5 local epochs to reach ~90%+ accuracy. Think of this as the model in the first 2 seconds of a 100-minute training session.

---

### Watermark Accuracy — Benign Clients

```json
  "wm_acc_benign": [0.625, 0.5],
```

**What it means:** Fraction of watermark bits correctly extracted from benign clients' models, averaged across the 2 benign clients. (You had 3 clients, 1 free-rider, so 2 benign.)

**Range:** 0.0 to 1.0. Higher = better watermark embedding.

**Is this normal?** Yes. With only 4-bit watermarks (`wm_bits=4`), the expected random baseline is 0.5 (coin flip). Your values of 0.625 and 0.5 are hovering around chance — the watermark has not converged yet. The paper reports near-100% after ~30 rounds. This is the same situation as main_acc: training just started.

**Why did it drop from 0.625 to 0.5?** With only 2 rounds and random initialisation, this is just noise. Bit accuracy oscillates early in training before the loss properly converges.

---

### Watermark Accuracy — Free-Rider

```json
  "wm_acc_freerider": [0.75, 0.5],
```

**What it means:** Fraction of watermark bits "accidentally" present in the free-rider's submitted model.

**What you want:** This should stay low (~0.5) throughout training — the free-rider never actually embeds the watermark, so extraction should return ~random bits. Your value of 0.5 at round 2 is exactly what's expected.

**Why is round 1 = 0.75?** In round 1, the free-rider strategy is "previous_models" — but on the first round there is no previous model, so the code submits the global model as-is. The global model at round 0 is randomly initialised and happens to produce z_k values that match 75% of the bits by chance. This is an artefact of the very first round only.

---

### Free-Rider Detection Accuracy

```json
  "fr_detection_acc": [0.0, 1.0],
```

**What it means:** Fraction of actual free-riders correctly identified by the server. Here there is 1 free-rider, so it's either 0.0 (not caught) or 1.0 (caught).

**Round 1 = 0.0:** The free-rider submitted the global model unchanged (first-round fallback), which still contains whatever random watermark signal exists at initialisation. The threshold η is not yet calibrated (fewer than 5 benign samples observed), so the fallback η = 0.30 is used. At round 1, the benign clients' watermarks are also weak (~62.5% accuracy = ~37.5% error), so the threshold is wide enough that the free-rider slips through.

**Round 2 = 1.0:** The free-rider now submits W_current − W_previous (near-zero weights). This gives ~50% bit accuracy (~50% error), which exceeds the threshold η. The server correctly flags the client as a free-rider. **This is the result you want to see.**

---

### False Positive Rate (FPR)

```json
  "fpr": [0.0, 0.0],
```

**What it means:** Fraction of *benign* clients incorrectly flagged as free-riders. 0.0 throughout means no benign client was ever falsely accused.

**This is the most important metric in practice.** Falsely accusing a legitimate client who contributed real data is costly. The paper reports FPR values close to 0 across all settings. Your FPR = 0.0 is correct.

---

### Config Block

```json
  "config": {
    "num_clients": 3,          ← 3 clients total
    "num_free_riders": 1,      ← 1 is a free-rider (client 2)
    "free_rider_type": "previous_models",
    "global_rounds": 2,        ← only 2 rounds (smoke test)
    "local_epochs": 1,         ← 1 local epoch per round (paper uses 2)
    "batch_size": 32,          ← paper uses 16
    "wm_bits": 4,              ← paper uses 8
    "beta": 0.9,               ← memory-enhanced blend
    "smooth_fn": "frac_power", ← f(x) = x^0.5
    "alpha_smooth": 0.5,
    "n_triggers": 100,         ← trigger samples for verification
    "eta": null,               ← threshold auto-estimated
    ...
  }
```

The smoke test uses reduced settings (fewer bits, fewer rounds, larger batch) to run fast. For paper-quality results, use the preset configs in `run_experiments.py`.

---

### Summary: What Does a "Good" Result Look Like?

After ~50–100 rounds with the full paper settings, you should see:

| Metric | Expected (paper) | Your smoke test |
|--------|-----------------|-----------------|
| `main_acc` | ~0.85–0.91 | ~0.10 (random init, expected) |
| `wm_acc_benign` | ~0.98–1.00 | ~0.50–0.62 (too early) |
| `wm_acc_freerider` | ~0.40–0.50 | ~0.50–0.75 (noise) |
| `fr_detection_acc` | ~0.95–1.00 | 1.0 at round 2 (lucky early) |
| `fpr` | ~0.00–0.05 | 0.0 ✓ |

---

## Part 4 — Mac M3 (Apple Silicon): CPU vs MPS

### What is MPS?

Your M3 chip has a built-in GPU called the **Apple Neural Engine / GPU**, accessible in PyTorch via the **Metal Performance Shaders (MPS)** backend, introduced in PyTorch 1.12. It is not CUDA (Nvidia), but it is GPU acceleration.

### Should you use it?

**Yes, absolutely — it will make a meaningful difference.** For this workload:

| Device | CIFAR-10 / ResNet-18 / 100 rounds | Notes |
|--------|----------------------------------|-------|
| M3 CPU (all cores) | ~3–5 hours | Uses only CPU cores |
| M3 MPS (GPU) | ~30–60 min | 4–6× faster typically |

The smoke test ran fine on CPU because it's only 2 rounds. For `table1` or any 50–100 round experiment, MPS will save you hours.

### How to enable it

**Step 1: Check your PyTorch version supports MPS**
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```
If it prints `True`, you're ready. If `False`:
```bash
pip install --upgrade torch torchvision
```
MPS requires PyTorch ≥ 1.12 and macOS ≥ 12.3. With miniforge3 on M3 you almost certainly have this.

**Step 2: Run with MPS**
```bash
python run_experiments.py --exp smoke --device mps
```

That's it. The code already handles this — `torch.device("mps")` is passed through everywhere.

**Step 3: Verify it's actually using the GPU**

Run Activity Monitor → GPU History (or Window → GPU History) while an experiment is running. You should see GPU usage spike.

### Known MPS gotchas to watch out for

1. **Some operations fall back to CPU silently.** PyTorch will warn you if an op isn't supported on MPS yet. This is harmless but means that op runs on CPU anyway.

2. **float64 not supported on MPS.** If you see errors like `"MPS does not support float64"`, add this to your training loop:
   ```python
   # In faremark/train.py, after device = torch.device(...)
   if str(self.device) == "mps":
       torch.set_default_dtype(torch.float32)
   ```

3. **`num_workers > 0` in DataLoader can cause issues on MPS.** If you get weird multiprocessing errors, set `num_workers=0` in the config.

4. **Memory is shared between CPU and GPU on Apple Silicon.** You don't need to worry about running out of GPU memory separately — it's unified with your system RAM.

### Bottom line

| Recommendation | Reason |
|---|---|
| Use `--device mps` for any experiment with ≥ 20 rounds | 4–6× faster, free, already supported |
| Keep `--device cpu` for the 2-round smoke test | Difference negligible for 2 rounds |
| Don't bother seeking cloud GPU yet | M3 MPS is sufficient for all single-dataset experiments in the paper. Only CIFAR-100 with 100 clients may take a few hours even on MPS. |

The paper used 2× RTX 3080 (each ~30 TFLOPS FP32). An M3's GPU is roughly ~7 TFLOPS — slower but perfectly usable for replicating all experiments given enough patience or overnight runs.