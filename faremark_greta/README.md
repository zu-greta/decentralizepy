# FareMark reproduction

Re-implementation of **FareMark: Model-Watermark-Driven Free-Rider Detection in
Federated Learning** (Li et al., IEEE IoT-J 2025).

The repo reproduces the paper **in stages, with a correctness gate after each**,
so the foundation is always verified before the next layer is added.

| Stage | Content | Status | Gate |
|------:|---------|--------|------|
| 1 | Honest FedAvg (no free-riders, no watermark) | **done** | Table I FedAvg accuracies |
| 2 | Free-rider attacks (Eq. 17 prev-models, Eq. 18 Gaussian) | **done** | Fig. 7 accuracy-drop trend |
| 3 | Watermarking scheme (Eq. 1–16) + memory update (Eq. 14) | planned | Tables II, VII |
| 4 | Free-rider detection + robustness | planned | Tables III–VI, Figs. 9–10 |

> **Stages 1–2 contain no watermarking.** Stage 1 is plain federated learning;
> Stage 2 adds the free-rider *attacks* on top of it. Watermarking is Stage 3.

*Note*: FareMark is centralized FedAvg simulated on one GPU, plus a custom per-client
loss, a memory-enhanced update, and server-side verification. 

---

## Layout

```
decentralizepy/
├── ...                         # upstream decentralizepy (not used yet)
└── faremark_greta/
    ├── faremark/               # the library
    │   ├── config.py           # experiment registry + expected-accuracy gates
    │   ├── datasets.py         # MNIST / CIFAR-10 / CIFAR-100, IID partition
    │   ├── models.py           # ResNet-18 (small-image), AlexNet, SmallCNN
    │   ├── client.py           # honest Client + the produce_update() seam
    │   ├── attacks.py          # Stage 2 free-rider clients + factory
    │   ├── server.py           # FedAvg aggregator + round loop + verify_hook
    │   └── utils.py            # seeding, accuracy, logging
    ├── scripts/
    │   ├── run_experiment.py   # entry point: one (config, repeat) -> result.json
    │   └── aggregate_results.py# mean ± std over repeats (paper table format)
    ├── infra/
    │   ├── Dockerfile build.sh requirements.txt
    │   ├── submit_experiment.sh# one RunAI job
    │   ├── submit_sweep.sh      # config × repeats grid
    │   └── submit_fig7.sh       # free-rider-count sweep (Fig. 7)
    ├── FareMark.md             # paper deep-dive
    ├── GRETA.md                # project plan + notes
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

### [`faremark/attacks.py`](faremark/attacks.py) — Stage 2 free-riders
Free-riders subclass `Client` and override only `produce_update` — no training
happens, but they still report a normal sample count so they blend into the
average. Maps to paper §V-A2:
- `PreviousModelsFreeRider` (**Eq. 17**): `W_free = 2·W_t − W_{t-1}` —
  extrapolate from the last two globals to mimic continued progress.
- `GaussianNoiseFreeRider` (**Eq. 18**): `W_free = W_t + N(0, σ²)`, optional
  per-round decay `σ_t = σ₀ · t^(-γ)`.
- `build_clients(cfg, ...)` — factory that returns a mix of honest + free-rider
  clients and the chosen free-rider indices (deterministic given the seed).

### [`faremark/models.py`](faremark/models.py) — model zoo
`ResNet18`, `AlexNetSmall`, `SmallCNN` (the last for fast smoke tests). Both real
models are adapted for small (28×28 / 32×32) inputs. The ResNet stem is the
standard "CIFAR ResNet" fix (3×3 stride-1 conv, drop the max-pool); without it
the feature maps collapse and accuracy stalls. Add ShuffleNet/GoogLeNet here via
`build_model`. Maps to paper §V-A.

### [`faremark/datasets.py`](faremark/datasets.py) — data & IID partition
Loads MNIST / CIFAR-10 / CIFAR-100 and splits the training set **evenly across
clients** (IID), matching the paper. `iid_partition` shuffles indices and gives
each client a near-equal shard; a `partition` argument is left in place so a
non-IID (Dirichlet) split can be added later without changing call sites.

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

## Running experiments

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
python ../scripts/aggregate_results.py /mnt/nfs/home/zu/results
```

Override any single run via env vars:
```bash
NUM_FREE_RIDERS=4 ATTACK=gaussian ./submit_experiment.sh 9 0
```

**Expected:** main-task accuracy falls as the free-rider count rises (the Fig. 7
trend). `result.json` records `free_rider_indices` so you can see which clients
cheated. The Stage-2 gate is the *trend*, not a fixed accuracy band — there is no
detection yet (that needs the watermarks from Stage 3).

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

## Operational notes (lessons from cluster bring-up)

- The submit command passes paths as **env vars** (`-e`) and keeps the pod
  command on single lines — RunAI mangles backslash line-continuations and
  re-parsed quoting, which silently emptied variables in early versions.
- Failed pods are **kept** for inspection; clean up with
  `runai delete job <name> -p sacs-zu`.
- `PKG_SUBDIR` in `submit_experiment.sh` must point at the canonical package dir
  (`faremark_greta`); push code there before launching jobs.