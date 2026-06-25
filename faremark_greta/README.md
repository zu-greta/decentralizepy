# FareMark reproduction

Re-implementation of **FareMark: Model-Watermark-Driven Free-Rider Detection in
Federated Learning** (Li et al., IEEE IoT-J 2025).

The repo reproduces the paper **in stages, with a correctness gate after each**,
so the foundation is always verified before the next layer is added.

| Stage | Content | Status | Gate |
|------:|---------|--------|------|
| 1 | Honest FedAvg (no free-riders, no watermark) | **done** | Table I FedAvg accuracies |
| 2 | Free-rider attacks (Eq. 17 prev-models, Eq. 18 Gaussian) | **done** | Fig. 7 accuracy-drop trend |
| 3 | Watermarking scheme (Eq. 1–16) + memory update (Eq. 14) | **done** | benign BER≈0; Tables II, VII |
| 4 | Free-rider detection + robustness | **done** | Tables III–VII, Figs. 8–10 |
| 5 | Limitations / attack study (non-IID, forgeability, bit-count, threshold) | **in progress** | separability figures |

> **Stages 1–4 are complete and verified.** Stage 1 is plain federated learning,
> Stage 2 adds the free-rider *attacks*, Stage 3 adds the *watermark* (honest
> clients embed it, benign BER ≈ 0; free-riders cannot, BER ≈ 0.5), and Stage 4
> turns that separation into the full detection + robustness sweeps. **Stage 5**
> is the current direction: the reproduction is now the validated baseline for an
> attack/limitations paper that probes the regimes FareMark never tested —
> non-IID data, a forging free-rider, low bit budgets, and threshold fragility.
> See `PROJECT_PLAN.md` for that plan and `HYPERPARAMS.md` for every tunable knob.

*Note*: FareMark is centralized FedAvg simulated on one GPU, plus a custom per-client
loss, a memory-enhanced update, and server-side verification. 

---

## Layout

```
decentralizepy/
├── ...                           # upstream decentralizepy (not used yet)
└── faremark_greta/
    ├── faremark/                 # the library
    │   ├── config.py             # experiment registry + expected-accuracy gates
    │   ├── datasets.py           # MNIST / CIFAR-10 / CIFAR-100, IID partition
    │   ├── models.py             # ResNet-18 (small-image), AlexNet, SmallCNN
    │   ├── client.py             # honest Client + the produce_update() seam
    │   ├── attacks.py            # Stage 2 free-rider clients + factory
    │   ├── watermark.py          # Stage 3 core: smoothing f, projection, embed/extract, BER
    │   ├── wm_client.py          # Stage 3 WatermarkClient (embed L_wm) + memory update (Eq.14)
    │   ├── wm_verify.py          # Stage 3 registry + server-side extraction/detection
    |   ├── robustness.py         # Stage 4 robustness tests: fine-tune, prune, DP (Opacus)
    │   ├── server.py             # FedAvg aggregator + round loop + verify_hook
    │   └── utils.py              # seeding, accuracy, logging
    ├── scripts/
    │   ├── run_experiment.py     # entry point: one (config, repeat) -> result.json
    |   ├── run_robustness.py     # Stage 4 robustness sweeps (fine-tune, prune, quantize)
    │   ├── plot_results.py       # result.json -> BER trajectory, sweep, separability figures
    │   └── aggregate_results.py  # mean ± std over repeats (paper table format)
    ├── infra/
    │   ├── Dockerfile 
    │   ├── build.sh
    │   ├── requirements.txt
    │   ├── submit_experiment.sh  # one RunAI job (env-overridable: PARTITION, ATTACK, PAPER_FAITHFUL, …)
    │   ├── submit_sweep.sh       # config × repeats grid
    │   └── submit_fig7.sh        # free-rider-count sweep (Fig. 7)
    ├── FareMark.md               # paper deep-dive
    ├── DOCUMENTATION.md          # code walkthrough
    ├── PROJECT_PLAN.md           # limitations/attack-paper plan (Stage 5)
    ├── HYPERPARAMS.md            # every tunable knob: flag, env var, effect
    ├── GRETA.md                  # project plan + notes
    └── README.md
```

---

## Code implementation

The whole simulator is one FedAvg loop with one extension point. Each file is
small and single-purpose; links are relative to this README.

### [`faremark/server.py`](faremark/server.py) — orchestration & aggregation
The control center. `Server.run(rounds)` is the FedAvg loop: each round it hands
every client the current global model, collects their weights, aggregates, and
evaluates the global model on the held-out test set.

- `Aggregator.aggregate(updates)` — **FedAvg**: a sample-weighted average of
  client weights. With equal IID shards this is exactly the paper's simple mean
  `W_g = (1/N) Σ Wᵢ`. Done tensor-by-tensor; integer buffers (e.g. BatchNorm's
  `num_batches_tracked`) are copied, not averaged.
- The server keeps the **previous** global model too (`prev_global_state`),
  because Stage-2 free-riders need `W_t` *and* `W_{t-1}`.
- `verify_hook(server, round, updates)` — a no-op placeholder where Stage 3/4
  will plug watermark extraction + detection (Eq. 15–16). It exists now so
  adding it later won't disturb the loop.

### [`faremark/client.py`](faremark/client.py) — honest local training
`Client.produce_update(global_state, prev_global_state, round_idx)` is **the one
method later stages override**. The honest client loads the global model, runs a
few epochs of local SGD on *its own shard only*, and returns the weights plus its
sample count. The data isolation (a client can only see its own loader) is what
makes this federated learning rather than centralized training.

### [`faremark/attacks.py`](faremark/attacks.py) — free-riders
Free-riders subclass `Client` and override only `produce_update` — no real
training happens, but they still report a normal sample count so they blend into
the average. The two paper attacks (paper §V-A2):
- `PreviousModelsFreeRider` (**Eq. 17**): `W_free = 2·W_t − W_{t-1}` —
  extrapolate from the last two globals to mimic continued progress.
- `GaussianNoiseFreeRider` (**Eq. 18**): `W_free = W_t + N(0, σ²)`, optional
  per-round decay `σ_t = σ₀ · t^(-γ)`.

Plus the **adaptive / watermark-aware** free-riders (factories that wrap a
`WatermarkClient` so the attacker can train a little before defecting). These
drive the limitations study:
- `train_then_attack` (**Table IV**): trains honestly until `attack_round`, then
  free-rides. Detectable unless it trained long enough for the mark to persist.
- `trigger_only` (**Table V**): trains on only `n_trigger_samples` trigger images
  → the mark overfits and fails on the held-out trigger bank.
- `random_round`: trains honestly each round with prob `honest_prob`, free-rides
  otherwise (sporadic-honesty evasion).
- `mixed`: the **forgery adversary** — a minimal trigger-only embed blended into a
  mostly-replayed global (`blend·own + (1−blend)·extrapolated`). The cheapest way
  to push free-rider BER toward the honest cluster.

`build_clients` / `build_watermarked_clients` return a mix of honest + free-rider
clients and the chosen indices (deterministic given the seed).

### [`faremark/watermark.py`](faremark/watermark.py) — Stage 3 core math
The box-free, output-space watermark (paper §IV-A). A client's m-bit message is
read from the model's **softmax output** on its trigger class:
- `smooth(p, …)` — the function f() (**Eq. 7–9**); default `f(p)=p^α`, `α<1`,
  which amplifies the tail probabilities so the argmax can't dominate the read.
- `make_key(m, l, …)` — the secret ±1 projection matrix M. Rows are
  **sign-balanced** (equal +1/−1): since `f(p) ≥ 0`, an all-same-sign row would
  force a fixed bit regardless of input, so balanced rows are required for a bit
  to be embeddable at all. (This was the key debugging insight.)
- `project_logits(probs, key, …, exclude)` — `z_k = Σ_j f(p_j)·M_{k,j}`
  (**Eq. 13**); the trigger class is excluded so its dominant probability can't
  freeze a bit. `extract_bits` averages z over N_T trigger samples then takes the
  sign (**Eq. 15**); `watermark_loss` is the BCE embedding term (**Eq. 11–12**);
  `bit_error_rate` / `detected` implement the detection threshold (**Eq. 16**).

### [`faremark/wm_client.py`](faremark/wm_client.py) — Stage 3 watermark client
`WatermarkClient(Client)` overrides `produce_update` to (1) train with
`L = L_cl + λ·L_wm` where the watermark term is applied to trigger-class samples
only (**Eq. 11–12**, plus label smoothing to keep the tail movable), and (2)
apply the **memory-enhanced update** (**Eq. 14**):
`W_new = β·(memory + Δ) + (1−β)·W_global`, where `Δ` is this round's local step.
Without it, FedAvg averaging would wash the watermark out; `β=0` recovers plain
local training. `build_watermarked_clients(...)` assigns each client a unique
trigger class + secret key + message and registers them.

### [`faremark/wm_verify.py`](faremark/wm_verify.py) — Stage 3 server verification
`WatermarkRegistry` stores each client's (trigger class, key, bits).
`build_trigger_bank` samples N_T trigger images per class from the test set.
`make_verifier(...)` returns the server's `verify_hook`: each round it extracts
every client's watermark from the submitted model, computes BER, and flags a
free-rider when `BER ≥ η`. It reports benign BER, free-rider BER, detection
accuracy and false-positive rate (**Eq. 15–16**; the mechanism behind Tables II–V).

Two **mode toggles** let you run either our robust detector or the bare paper
algorithm (both default off, so existing runs are unchanged):
- `paper_faithful` — strips our three deviations at once: random (not
  sign-balanced) keys, **no** trigger-class exclusion (full softmax), and a
  **cumulative uncapped** `μ+3σ` threshold (no sliding window, no 0.25 cap). Use
  with a high-class dataset (CIFAR-100) so the full-softmax projection is
  embeddable. This answers "is a weakness real or an artifact of my changes?"
- `calib_on_all` — calibrate `η` over **every** client instead of the assumed
  honest pool, exposing the paper's circularity: free-rider BER ≈ 0.5 poisons
  `μ+3σ`. Off = the paper's trusted-pool assumption.

### [`faremark/models.py`](faremark/models.py) — model zoo
`ResNet18`, `AlexNetSmall`, `SmallCNN` (the last for fast smoke tests). Both real
models are adapted for small (28×28 / 32×32) inputs. The ResNet stem is the
standard "CIFAR ResNet" fix (3×3 stride-1 conv, drop the max-pool); without it
the feature maps collapse and accuracy stalls. Add ShuffleNet/GoogLeNet here via
`build_model`. Maps to paper §V-A.

### [`faremark/datasets.py`](faremark/datasets.py) — data & partitioning
Loads MNIST / CIFAR-10 / CIFAR-100 and splits the training set across clients.
Two partition modes (`partition=` argument): **IID** (`iid_partition`, the
paper's even split) and **non-IID** (`dirichlet_partition`, a label-skewed split
we added for the limitations study). See **Dataset & data partitioning** below
for the full breakdown of what is split, what is held out, and how IID and
non-IID differ.

### [`faremark/config.py`](faremark/config.py) — experiment registry
Maps `config_idx → ExpConfig` (model, dataset, #clients, rounds, lr, …, and
Stage-2 `attack`/`num_free_riders` fields). `expected_acc` is the **Stage-1
correctness band**, anchored to the FedAvg column of Table I. `repeat` selects a
seed (the paper averages over 10 repeats).

### [`faremark/utils.py`](faremark/utils.py) — helpers
Reproducible seeding (so `(config, repeat)` is deterministic), top-1 accuracy,
and a logger that writes to both stdout and `run.log`.

### [`scripts/run_experiment.py`](scripts/run_experiment.py) — entry point
Parses `--config_idx/--repeat` (+ overrides), builds data/model/clients/server,
runs the loop, writes `result.json`, and prints a `PASS/FAIL` verdict against the
expected band (non-zero exit on failure, so sweeps/CI can detect regressions).

### [`scripts/aggregate_results.py`](scripts/aggregate_results.py) — summaries
Walks a results directory and rolls per-repeat `result.json` files into
**mean ± std** per config — the format the paper's tables use.

### [`infra/`](infra/) — cluster (RunAI)
`build.sh` builds/pushes the image; `submit_experiment.sh` runs one job (clones
the repo into the pod, runs the script, captures everything to `pod.log` on the
PVC, keeps failed pods for inspection); `submit_sweep.sh` and `submit_fig7.sh`
launch grids.

---

## Dataset & data partitioning

This section explains exactly what data exists, how it is divided among clients,
and what is used for training versus verification versus testing. All of it lives
in [`faremark/datasets.py`](faremark/datasets.py).

### The three datasets

| Dataset | Classes (`n`) | Train images | Test images | Image | Default clients |
|---|---|---|---|---|---|
| MNIST | 10 | 60,000 | 10,000 | 1×28×28 grayscale | 10 |
| CIFAR-10 | 10 | 50,000 | 10,000 | 3×32×32 colour | 10 |
| CIFAR-100 | 100 | 50,000 | 10,000 | 3×32×32 colour | 10 (or 100) |

The class count `n` matters beyond accuracy: it sets the **watermark bit budget**
(`m ≈ (n−1)/2` under our embeddability assumptions), so CIFAR-10 carries ~4 bits
and CIFAR-100 ~49. That is why detectability differs by dataset even for the same
attacker (see the bit-count experiments).

### What gets split, and what does not

Only the **training set** is partitioned across clients — each client trains on
its own shard and nothing else (this data isolation is what makes the simulation
*federated* rather than centralized). Two things are **never** partitioned and
are shared/global:

- **Test set** — the full held-out test split is kept whole and used only by the
  server to evaluate the *global* model's accuracy each round (`Server._evaluate`).
  No client trains on it.
- **Trigger bank** — for watermark verification the server samples up to `N_T`
  images **per trigger class from the test set** (`build_trigger_bank`, default
  `wm_num_triggers=50`). These are the held-out trigger samples the server queries
  to extract each client's watermark. Using *test* images here is deliberate: it
  means verification checks whether the watermark **generalizes** to images the
  client never trained on — which is what makes the trigger-only forgery fail.

So the data has three non-overlapping roles: client shards (training), the trigger
bank (verification, drawn from test), and the test set (global accuracy).

### Transforms

Standard per-dataset normalization; CIFAR train shards additionally get random
crop (32, padding 4) + horizontal flip for augmentation. Test/trigger images get
normalization only — no augmentation — so verification and accuracy are measured
on clean images.

### IID partition (the paper's setting)

`iid_partition(num_samples, num_clients, seed)`: shuffle **all** training indices
and cut them into `num_clients` near-equal shards. Every client therefore sees an
i.i.d. sample of the full distribution — roughly equal counts of every class.
Consequences that the watermark relies on:

- Every client's **trigger class is well represented in its own shard**, so it has
  plenty of trigger-class images to embed its watermark on → honest BER → ~0.
- Class balance is uniform across clients, so the FedAvg average is the simple
  mean the paper assumes, and the benign BER distribution is tight → the
  `η = μ+3σ` threshold sits low and separates cleanly.

This is the **only** regime the paper tests ("the training dataset was divided
evenly among the clients").

### Non-IID partition (our addition for the limitations study)

`dirichlet_partition(labels, num_clients, alpha, seed)` implements the standard
label-skew benchmark (Hsu et al. 2019). For **each class**, it draws a
`Dirichlet(α)` vector over clients and hands out that class's images in those
proportions. The single knob `α` controls severity:

| `α` | Effect on each client's data | Effect on the watermark |
|---|---|---|
| ≥ 100 | ≈ IID (near-uniform class mix) | honest BER ≈ 0, behaves like the paper |
| 1.0 | mild skew | some clients short on their trigger class → honest BER rises |
| 0.5 | strong skew | benign BER and false positives climb |
| 0.1 | severe — each client sees only a few classes | a client may hold **almost none of its own trigger class** → it cannot embed → its honest BER → ~0.5 |

The failure mechanism is direct: a client can only embed its watermark on
*trigger-class* images, but under heavy skew its shard may contain very few (or
zero) of them. It then looks identical to a free-rider to the verifier — honest
BER rises into the free-rider band, false positives climb, and `η` can no longer
separate the two populations. This happens **with no free-riders present at all**,
which is why non-IID is the cleanest limitation: the scheme misclassifies genuine
contributors purely because of how their data is distributed.

### How to select each mode

```bash
# IID (default) — paper setting
./submit_experiment.sh 12 0

# non-IID, severe skew
PARTITION=dirichlet DIRICHLET_ALPHA=0.1 ./submit_experiment.sh 12 0
```

or on the CLI: `--partition dirichlet --dirichlet_alpha 0.1`. Both modes share
the same call site, so every experiment can be run either way without code
changes. The split is deterministic given the seed, so a `(config, repeat,
partition, α)` combination is reproducible.

---
### Configs

```bash
python scripts/run_experiment.py --list_configs
```

| idx | name | model / dataset | clients | notes |
|----:|------|-----------------|--------:|-------|
| 0 | smoke_mnist_smallcnn | SmallCNN / MNIST | 5 | fast pipeline check (<1 min) |
| 1 | resnet18_cifar10 | ResNet-18 / CIFAR-10 | 10 | **Table I baseline (~91.85%)** |
| 2 | resnet18_mnist | ResNet-18 / MNIST | 10 | ~98.7% |
| 3 | resnet18_cifar100 | ResNet-18 / CIFAR-100 | 100 | ~76.5% |
| 4 | alexnet_cifar10 | AlexNet / CIFAR-10 | 10 | ~86.4% |
| 5 | alexnet_mnist | AlexNet / MNIST | 10 | ~91.5% |
| 6 | alexnet_cifar100 | AlexNet / CIFAR-100 | 10 | ~68.5% |
| 7 | fr_smoke_mnist | SmallCNN / MNIST | 10 | Stage-2 fast trend |
| 8 | fr_prev_resnet18_cifar10 | ResNet-18 / CIFAR-10 | 10 | previous-models attack |
| 9 | fr_gauss_resnet18_cifar10 | ResNet-18 / CIFAR-10 | 10 | gaussian attack |
| 10 | wm_smoke_mnist | SmallCNN / MNIST | 10 | Stage-3 fast embed+extract |
| 11 | wm_resnet18_cifar10 | ResNet-18 / CIFAR-10 | 10 | **fidelity vs baseline (Table I "Ours")** |
| 12 | wm_fr_resnet18_cifar10 | ResNet-18 / CIFAR-10 | 10 | watermark + free-riders → detection |
| 13 | paper_faithful_resnet18_cifar100 | ResNet-18 / CIFAR-100 | 10 | **bare paper algorithm** (random keys, full softmax, uncapped μ+3σ); many bits, no bit-count artifact |

### Locally

```bash
pip install torch torchvision numpy
python scripts/run_experiment.py --config_idx 0 --repeat 0 --device cpu \
  --output_dir ./out --data_root ./data
```

### On the cluster (RunAI)

```bash
cd infra
./build.sh                    # build + push image (once / on dependency changes)
./submit_experiment.sh 0 0    # smoke test:  config_idx=0 repeat=0 - expected >95% on MNIST, PASS
./submit_experiment.sh 1 0    # Table I baseline: ResNet-18 / CIFAR-10 - expected ~88-94%, PASS (paper: 91.85%)
# note: switch to BATCH_SIZE=64 for faster runs where an exact match isn't needed
./submit_experiment.sh 0 0    # deterministic smoke test:  config_idx=0 repeat=0 - expected >95% on MNIST, PASS again
./submit_experiment.sh 2 0    # ResNet-18 / MNIST - expected ~98.7%, PASS
python scripts/aggregate_results.py /mnt/nfs/home/zu/results  # aggregate after a couple of repeats to see mean ± std (paper's table format). gives the mean ± std for each config_idx over the repeats.
```

Each run writes to a timestamped dir on the PVC:
- `result.json` — config, per-round accuracy, final/best accuracy, PASS/FAIL
- `run.log` — the Python logger output
- `pod.log` — the full pod trace (clone, GPU, errors) for debugging

### Expected outputs

**Stage 1 — smoke (`config_idx 0`).** Finishes in under a minute; accuracy climbs
monotonically to ~97–98% and `correctness_pass: true`:

```
round 1/5  test_acc=90.48%   ...   round 5/5  test_acc=97.51%
CORRECTNESS CHECK: PASS (final 97.51% vs expected 95.0-100.0%)
```

**Stage 1 — Table I baseline (`config_idx 1`).** ~5 min/round on one GPU (batch 16),
~4 h total. Accuracy rises smoothly and plateaus near the paper's 91.85%
(observed run: 11% → 80% by round 5 → ~92% by round 30). `result.json`:

```json
{
  "config": { "name": "resnet18_cifar10", "rounds": 50, ... },
  "final_acc": 92.0, "best_acc": 92.4,
  "expected_acc": [88.0, 94.0], "correctness_pass": true,
  "history": [ { "round": 1, "test_acc": 11.03 }, ... ]
}
```

> Tip: for a faithful Table I match keep the defaults; for faster runs where an
> exact match isn't needed, `BATCH_SIZE=64 ./submit_experiment.sh 1 0`.

**Repeats → mean ± std** (the paper's table format):

```bash
./submit_sweep.sh "1" "0 1 2 3 4 5 6 7 8 9"          # 10 repeats
python scripts/aggregate_results.py /mnt/nfs/home/zu/results
# resnet18_cifar10   10   91.9 +/- 0.4   10/10
```

### Stage 2 — free-rider trend (Fig. 7)

```bash
# locally for quick check assignments + both attacks + the trend
python test_stage2.py # -> assignment OK; previous_models runs; trend: 0 FR=100% 2 FR=51.8% 4 FR=51.8%
```

```bash
# on the cluster for the full trend (Fig. 7)
cd infra
./submit_fig7.sh 7 0                # fast MNIST trend, counts 0 2 4 6 8 (minutes)
./submit_fig7.sh 8 0 0 2 4 6 8      # ResNet-18 / CIFAR-10, previous-models 
./submit_fig7.sh 9 0 0 2 4 6 8      # ResNet-18 / CIFAR-10, gaussian 
python ../scripts/aggregate_results.py /mnt/nfs/home/zu/results --fig7
```

> **Sweeps submit all jobs at once.** `submit_sweep.sh` / `submit_fig7.sh` call
> `submit_experiment.sh` in non-blocking mode (`WAIT=0`), so every job is queued
> immediately and the cluster runs them as GPUs free up — you are not waiting for
> one to finish before the next is submitted. A single `./submit_experiment.sh N M`
> still blocks and waits (and auto-cleans up) as before. Because each job's name and
> results dir carry a timestamp and a `-frN` tag, parallel jobs never collide.

Each job writes its own `result.json`; `aggregate_results.py` walks the results
root and groups by `(config, attack, #free-riders)` — so `--fig7` prints the
accuracy-vs-free-rider trend, while plain repeats collapse into mean ± std.

Override any single run via env vars:
```bash
NUM_FREE_RIDERS=4 ATTACK=gaussian ./submit_experiment.sh 9 0
```

**Expected:** main-task accuracy falls as the free-rider count rises (the Fig. 7
trend). `result.json` records `free_rider_indices` so you can see which clients
cheated. The Stage-2 gate is the *trend*, not a fixed accuracy band — there is no
detection yet (that needs the watermarks from Stage 3).

### Stage 3 — watermarking (embed, extract, detect)

First, the synthetic end-to-end test (no cluster, no dataset download — runs in
seconds and exercises the real `WatermarkClient`, server verifier and registry):

```bash
python test_stage3.py
```

Then on the cluster:

```bash
cd infra
./submit_experiment.sh 10 0     # fast watermark smoke (MNIST): embed + extract 
./submit_experiment.sh 11 0     # FIDELITY: ResNet-18/CIFAR-10, all honest + watermarked 
./submit_experiment.sh 12 0     # DETECTION: watermark + free-riders

WM_NUM_TRIGGERS=100 ./submit_experiment.sh 11 0     # Table II (watermark accuracy)
WM_NUM_TRIGGERS=50  ./submit_experiment.sh 12 0     # Table III (detection)
```

What each piece maps to and the **expected output**:
- **Embedding / extraction** (Eq. 11–15): honest clients reach mean benign
  `wm_benign_ber ≈ 0` — the watermark is recovered. (Tables II / VII; vary N_T
  via `wm_num_triggers`.)
- **Fidelity** (idx 11, Table I "Ours"): final accuracy within ~2 points of the
  Stage-1 baseline (your 93.22%). The watermark shouldn't cost much accuracy.
- **Detection** (idx 12, Tables III–V): free-riders show `wm_fr_ber ≈ 0.5` and
  are flagged (`wm_fr_recall`), benign clients are kept (`wm_fpr ≈ 0`).

The Stage-3 watermark metrics are written into `result.json`:

```json
{
  "watermark": true,
  "wm_benign_ber": 0.0,    "wm_fr_ber": 0.5,
  "wm_detect_acc": 1.0,    "wm_fpr": 0.0,  "wm_fr_recall": 1.0,
  "final_acc": 92.1, "correctness_pass": true, ...
}
```

Validated end-to-end on synthetic FL (6 clients, 2 free-riders) before any
cluster run: benign BER 0.00, free-rider BER 0.50, detection accuracy 100%,
FPR 0%. Two implementation notes that cost real debugging time and are baked into
the code: the trigger class is **excluded** from the projection (its ~1.0
probability would otherwise freeze a bit), and key rows are **sign-balanced**
(because `f(p) ≥ 0`, a same-sign row makes a bit unembeddable). Key knobs:
`wm_lambda` (embedding strength), `wm_beta` (memory vs global), `wm_alpha`
(smoothing), `wm_eta` (detection threshold) — overridable via `--wm_lambda` /
`--wm_beta` or the config.

---

### Stage 4 — detection + robustness

```bash
./submit_fig7.sh 12 0 2 4 6 8
python scripts/run_robustness.py --config_idx 11 --repeat 0 --output_dir $RESULTS/robust --data_root $DATA
```

Detection over time (Fig. 8) from the cfg-12 run, plot per-round `wm_benign_ber` vs `wm_fr_ber` from `history`. Expect benign to fall toward 0 within ~20-30 rounds while free-rider BER stays >~0.4. the 2 curves should visibly seperate and stay apart.
Detection vs free-rider rate (Table III). Sweep `--num_free_riders` 2/4/6/8 (20–80%) in cfg-12; (reuse `submit_fig7.sh 12 0 2 4 6 8`). for each, confirm `wm_detect_acc` stays hiigh and `wm_fpr` low even as the free-rider fraction rises to 80%. Headline detection table.
Adapive free-riders (Tables IV-V). Run with `attack=train_then_attack` (`attack_round=50`) and `attack=trigger_ony`, sweep `n_trigger_samples` (the N_T used for the attacker's fake watermark). Confirm the attack fails: `wm_fr_ber` should rise to ~0.5, and `wm_detect_acc` should stay high. The point to verify: train-then-attack is still flagged if it defected without enough honest training, and trigger-only fails verification because it overfits a few triggers and doesn't generalize to the held-out trigger bank.
Robustness (Figs. 9–10, Table VI). Run `scripts/run_robustness.py --config_idx 11 --repeat 0` (the watermarked ResNet-18/CIFAR-10 config) and sweep the fine-tuning epochs, pruning ratio, or DP noise multiplier. Confirm the watermark holds up: `wm_benign_ber` should stay low and `wm_detect_acc` high even as the model is fine-tuned, pruned, or noised. check the expected shapes: fine-tuning drives task accuracy back toward baseline while watermark accuracy decays (Fig. 9); pruning tolerates ~50% with the watermark intact, then both collapse past ~60% (Fig. 10); quantization/DP show graceful watermark degradation.

```bash
./reproduce_paper.sh fidelity     # Table I + II   (watermark, 0 FR, 10 repeats)
./reproduce_paper.sh fig7         # Fig. 7         (4 panels, both attacks)

./reproduce_paper.sh detection    # Table III+Fig8 (FR sweep, both attacks)
./reproduce_paper.sh robustness   # prints the run_robustness.py commands
./reproduce_paper.sh all          # everything
```

---

### Stage 5 — limitations / attack study

The reproduction above is the validated baseline. Stage 5 probes the regimes the
paper never tested. Each experiment emits a graph via `plot_results.py` (per-run
BER trajectory + global accuracy, a swept-variable summary, and the thesis
**separability** figure). Full plan in `PROJECT_PLAN.md`; every knob in
`HYPERPARAMS.md`.

```bash
# non-IID false positives (all honest, watermarked) — the cleanest limitation
for A in 100 1 0.5 0.1; do
  PARTITION=dirichlet DIRICHLET_ALPHA=$A NUM_FREE_RIDERS=0 TAG=noniida${A/./} \
    ./submit_experiment.sh 12 0
done

# forgeability — the mixed (forging) free-rider on CIFAR-10
for B in 0.3 0.5 0.7; do ATTACK=mixed BLEND=$B TAG=mixb${B/./} ./submit_experiment.sh 12 0; done

# bare paper algorithm + threshold poisoning (CIFAR-100, many bits)
PAPER_FAITHFUL=1 TAG=pfiid ./submit_experiment.sh 13 0
PAPER_FAITHFUL=1 CALIB_ON_ALL=1 TAG=pfpoison ./submit_experiment.sh 13 0

# figures (on the jumphost, where the PVC is mounted)
python scripts/plot_results.py --in $RESULTS/cfg12_rep0-fr0-noniida* --out figs/noniid
python scripts/plot_results.py --in $RESULTS/cfg12_rep0-mixb*        --out figs/forgeability
```

The four limitation pillars and what to expect: **non-IID** (FPR climbs 0→1.0 as
α shrinks, with zero free-riders), **forgeability** (mixed-attack free-rider BER
falls toward the honest cluster as blend/`n_trigger_samples` rise), **bit-count**
(CIFAR-10's ~4 bits overlap where CIFAR-100's ~49 separate), and **threshold
fragility** (heavy free-riding or `calib_on_all` poisons `μ+3σ`). Read each
figure's printed `margin = ±…`: negative means the honest and free-rider
distributions overlap and no `η` separates them — the impossibility result.

---

## The seam for later stages

`Client.produce_update(global_state, prev_global_state, round_idx)` is the single
method Stages 2 and 3 override:
- **free-rider** (Stage 2): ignore the data, fabricate weights from the two prior
  globals;
- **watermark client** (Stage 3): split trigger/common classes, add the `L_wm`
  regularizer, and apply the memory-enhanced update (Eq. 14).

`Server.verify_hook(server, round, updates)` is where Stage 3/4 plug watermark
extraction + detection (Eq. 15–16). Both exist as no-ops today, so adding the
stages won't require refactoring the loop.

---

## Reproducing the paper — settings, remaining work, and the experiment matrix

**End goal:** reproduce every table and figure in Li et al. (IoT-J 2025). This
section is the checklist to get there.

### Settings per experiment (the paper is inconsistent — use these)

The general settings prose says "local epoch 2, global epoch 100", but every
specific experiment overrides it. Use the per-experiment numbers:

| Experiment | Rounds × local epochs | Notes |
|---|---|---|
| Fidelity (Table I) | **50 × 5** | paper §V-B: "50 communication rounds … five epochs" |
| Watermark detection (Table II) | **50 × 5** | N_T = 100 triggers |
| Detection over time (Fig. 8) | ~60 rounds, 1 round = 10 epochs | benign →>98% by ~round 30 |
| Single/multi FR detection (Table III) | ≥ 60 rounds | N_T = 50; FR rate swept 20–80% |
| Fine-tune robustness (Fig. 9) | λ=0, validate every 10 epochs | — |

Common to all: SGD, lr 0.01, batch 16, IID even split, **10 repeats averaged**,
one distinct trigger class per client (so num_clients ≤ num_classes for the main
tables). Momentum/weight-decay are **not** specified by the paper; we use 0.9 /
5e-4 (state this as an assumption). So our `rounds=50, local_epochs=5` defaults
are correct for Tables I–II; do **not** switch to 2×100.

### Still to implement

1. **Models:** add **ShuffleNet** and **GoogLeNet** to `models.py` (have ResNet-18,
   AlexNet).
2. **Dataset:** add **Food100** to `datasets.py` (confirm it's a Food-101 subset).
3. **Baselines for the comparison columns:** **FedIPR** (feature-based N-bit +
   backdoor-based) and **ST-/ATD-DAGMM** (free-rider anomaly detector). FareMark's
   own numbers don't need these; they fill the "vs others" columns of Tables I–III.
4. **Valid/test split** in `datasets.py` if you tune λ/β/η (avoid tuning on test).
5. **Opacus** integration for a faithful Table VI (DP during training).

### The experiment matrix (what to run for each result)

| Paper result | Command(s) |
|---|---|
| Table I (fidelity) | `submit_sweep.sh "11" "0..9"` × {ResNet-18, AlexNet} × {MNIST, CIFAR-10/100} |
| Fig. 7 (acc vs #FR) | `submit_fig7.sh 8 0 0 2 4 6 8` (+ cfg 9; + AlexNet/MNIST variants) |
| Table II (Acc_wm) | cfg 11 runs; read `wm_benign_ber` → Acc_wm = 1−BER |
| Fig. 8 (rate over rounds) | cfg 12; plot per-round `wm_benign_ber` vs `wm_fr_ber` |
| Table III (FR detection) | cfg 12 with `--num_free_riders` 2/4/6/8 (20–80%) |
| Table IV (train-then-attack) | `attack=train_then_attack`, `attack_round=50` |
| Table V (trigger-only) | `attack=trigger_only`, sweep `n_trigger_samples` |
| Table VI (DP) | `run_robustness.py` (or Opacus-trained client) |
| Table VII (Acc_wm vs N_T) | cfg 11, sweep `wm_num_triggers` ∈ {10,50,100,…} |
| Figs. 9–10 (finetune/prune) | `run_robustness.py --config_idx 11` |

Each model×dataset×repeat is one `result.json`; `aggregate_results.py` turns the
tree into the paper's mean ± std tables. Budget: the full matrix is large
(4 models × 4 datasets × 10 repeats for Table I alone), so prioritize
ResNet-18/CIFAR-10 and AlexNet/MNIST first — those cover Fig. 7 and Fig. 8/9/10.

See `DOCUMENTATION.md` for the full code↔paper map and the "where to add things"
guide for extending the framework.