# FareMark reproduction

Re-implementation of **FareMark: Model-Watermark-Driven Free-Rider Detection in
Federated Learning** (Li et al., IEEE IoT-J 2025).

This repo reproduces the paper in stages, with a correctness gate after each so
we always know the foundation is sound before building on it.

| Stage | Content | Status | Gate |
|------:|---------|--------|------|
| 1 | Honest FedAvg (no free-riders, no watermark) | **done** | Table I FedAvg accuracies |
| 2 | Free-rider attacks (Eq. 17 prev-models, Eq. 18 Gaussian) | **done** | Fig. 7 trend |
| 3 | Watermarking (Eq. 1–16) + memory update (Eq. 14) | planned | Tables II, VII |
| 4 | Detection + robustness (Tables III–VI, Figs. 9–10) | planned | matching tables |

## Why a custom simulator (not decentralizepy)

FareMark is centralized FedAvg simulated on one GPU, plus a custom per-client
loss, a memory-enhanced update, and server-side verification. decentralizepy is
built for *real distributed* learning over topologies (ZMQ, multi-process, graph
configs) — its strengths aren't exercised here, and threading custom losses
through it adds risk during reproduction. We keep the core logic modular so it
could be ported into decentralizepy later if we need genuinely distributed runs.

## Layout

```
faremark_paper/
  faremark/            # the library
    config.py          # experiment registry (config_idx -> config), expected-acc gates
    datasets.py        # MNIST / CIFAR-10 / CIFAR-100, IID partition
    models.py          # ResNet-18 (small-image), AlexNet, SmallCNN
    client.py          # Client.produce_update(...)  <- the seam stages 2/3 override
    attacks.py         # Stage 2: free-rider clients (previous_models, gaussian) + factory
    server.py          # FedAvg Aggregator + round loop + verify_hook (no-op now)
    utils.py           # seeding, accuracy, logging
  scripts/
    run_experiment.py  # entry point: one (config, repeat) -> result.json + PASS/FAIL
    aggregate_results.py  # mean +/- std over repeats (the paper's table format)
  infra/
    Dockerfile build.sh requirements.txt
    submit_experiment.sh  # one RunAI job
    submit_sweep.sh       # config x repeats grid (10-repeat averaging)
    submit_fig7.sh        # free-rider-count sweep (Stage 2 / Fig. 7 trend)
```

## Run locally

```bash
pip install torch torchvision numpy
# fast smoke test (downloads MNIST):
python scripts/run_experiment.py --config_idx 0 --repeat 0 --device cpu \
  --output_dir ./out --data_root ./data
python scripts/run_experiment.py --list_configs
```

## Run on the cluster (RunAI)

```bash
cd infra
./build.sh                       # build + push the image (run once / on dep changes)
./submit_experiment.sh 0 0       # smoke test:  config_idx=0 repeat=0
./submit_experiment.sh 1 0       # Table I: ResNet-18 / CIFAR-10
./submit_sweep.sh "1" "0 1 2 3 4 5 6 7 8 9"   # 10 repeats for averaging
```

Each run writes `result.json` (config, per-round accuracy, final acc, PASS/FAIL)
and `stdout.log` to the results dir on the PVC. `aggregate_results.py` rolls the
repeats up into mean +/- std.

## Stage 2: free-rider attacks

Configs 7–9 add free-riders. Two attacks are implemented (both subclass `Client`,
overriding only `produce_update`):
- `previous_models` (Eq. 17): `W_free = 2*W_t - W_{t-1}` (delta-weights extrapolation).
- `gaussian` (Eq. 18): `W_free = W_t + N(0, sigma^2)`, optional per-round decay.

Run a Fig. 7 trend (accuracy vs free-rider count):

```bash
cd infra
./submit_fig7.sh 7 0                # fast MNIST smoke, counts 0 2 4 6 8
./submit_fig7.sh 8 0 0 2 4 6 8      # ResNet-18/CIFAR-10, previous_models
python ../scripts/aggregate_results.py /mnt/nfs/home/zu/results
```

You can also override on any single run via env vars:
`NUM_FREE_RIDERS=4 ATTACK=gaussian ./submit_experiment.sh 9 0`.
The Stage-2 gate is the trend (accuracy falls as free-riders rise), not a fixed band.

## The seam for later stages

`Client.produce_update(global_state, prev_global_state, round_idx)` is the only
method stages 2 and 3 override:
- **free-rider**: ignore the data, fabricate weights from the two prior globals;
- **watermark client**: split trigger/common classes, add `L_wm`, apply the
  memory-enhanced update.

`Server.verify_hook(server, round, updates)` is where stage 3/4 plug watermark
extraction + detection. Both exist (as no-ops) today so adding stages won't
require refactoring the loop.