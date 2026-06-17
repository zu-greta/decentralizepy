# FareMark reproduction

Re-implementation of **FareMark: Model-Watermark-Driven Free-Rider Detection in
Federated Learning** (Li et al., IEEE IoT-J 2025).

This repo reproduces the paper in stages, with a correctness gate after each so
we always know the foundation is sound before building on it.

| Stage | Content | Status | Gate |
|------:|---------|--------|------|
| 1 | Honest FedAvg (no free-riders, no watermark) | **done** | Table I FedAvg accuracies |
| 2 | Free-rider attacks (Eq. 17 prev-models, Eq. 18 Gaussian) | planned | Fig. 7 trend |
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
faremark_greta/
  faremark/            # the library
    config.py          # experiment registry (config_idx -> config), expected-acc gates
    datasets.py        # MNIST / CIFAR-10 / CIFAR-100, IID partition
    models.py          # ResNet-18 (small-image), AlexNet, SmallCNN
    client.py          # Client.produce_update(...)  <- the seam stages 2/3 override
    server.py          # FedAvg Aggregator + round loop + verify_hook (no-op now)
    utils.py           # seeding, accuracy, logging
  scripts/
    run_experiment.py  # entry point: one (config, repeat) -> result.json + PASS/FAIL
    aggregate_results.py  # mean +/- std over repeats (the paper's table format)
  infra/
    Dockerfile build.sh requirements.txt
    submit_experiment.sh  # one RunAI job
    submit_sweep.sh       # config x repeats grid (10-repeat averaging)
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

## The seam for later stages

`Client.produce_update(global_state, prev_global_state, round_idx)` is the only
method stages 2 and 3 override:
- **free-rider**: ignore the data, fabricate weights from the two prior globals;
- **watermark client**: split trigger/common classes, add `L_wm`, apply the
  memory-enhanced update.

`Server.verify_hook(server, round, updates)` is where stage 3/4 plug watermark
extraction + detection. Both exist (as no-ops) today so adding stages won't
require refactoring the loop.
