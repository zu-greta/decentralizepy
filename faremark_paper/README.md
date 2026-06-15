# FareMark — Re-implementation

Unofficial PyTorch re-implementation of:

> **FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning Model**
> Li Li, Xinpeng Zhang, Hanzhou Wu, Guorui Feng, Weiming Zhang
> *IEEE Internet of Things Journal*, vol. 12, no. 18, pp. 38965–38977, Sep. 2025
> DOI: [10.1109/JIOT.2025.3587745](https://doi.org/10.1109/JIOT.2025.3587745)

---

## What this implements

| Paper component | File |
|---|---|
| Watermark representation & loss (Eq. 1–13) | `faremark/watermark.py` |
| FL client + memory-enhanced update (Eq. 14) | `faremark/client.py` |
| FL server + FedAvg + detection (Eq. 15–16) | `faremark/server.py` |
| AlexNet / ResNet-18 / ShuffleNet / GoogLeNet | `faremark/models.py` |
| MNIST / CIFAR-10 / CIFAR-100 / Food-101 | `faremark/datasets.py` |
| All hyperparameters from Section V-A | `faremark/config.py` |
| Full FL training loop | `faremark/train.py` |
| Fine-tuning / pruning / DP robustness | `faremark/evaluate.py` |
| Experiment runner (all tables & figures) | `run_experiments.py` |

---

## Hardware requirements

The paper uses **2× NVIDIA RTX 3080 GPUs**.

- A single GPU (≥8 GB VRAM) is sufficient for CIFAR-10/MNIST experiments.
- CIFAR-100 with 100 clients is memory-intensive; use `--device cpu` if needed.
- All experiments also run on CPU (much slower — see time estimates below).

Approximate wall-clock times for 100 global rounds on a single RTX 3080:

| Dataset | Model | Clients | Approx. time |
|---|---|---|---|
| CIFAR-10 | ResNet-18 | 10 | ~45 min |
| MNIST | AlexNet | 10 | ~20 min |
| CIFAR-100 | ResNet-18 | 100 | ~4 hrs |

---

## Setup

### 1. Clone / download this repository

```bash
git clone <your-repo-url>
cd faremark
```

### 2. Create a Python environment

Using conda (recommended — matches the paper's Python 3.7 / PyTorch 2.1):

```bash
conda create -n faremark python=3.9    # 3.9 is the lowest that works well with torch 2.1
conda activate faremark
```

Or with venv:

```bash
python3 -m venv faremark_env
source faremark_env/bin/activate       # Windows: faremark_env\Scripts\activate
```

### 3. Install PyTorch

Go to [pytorch.org](https://pytorch.org/get-started/locally/) and select your CUDA version.
For CUDA 12.1 (most common on modern GPUs):

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

For CPU only:

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install remaining dependencies

```bash
pip install numpy opacus
```

`opacus` is only needed for the differential privacy experiment (Table VI).
All other experiments work without it.

### 5. Verify installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected output (GPU): `2.1.0  True`
Expected output (CPU): `2.1.0  False`

---

## Quick start — smoke test

Runs 2 training rounds with 3 clients (1 free-rider) on CIFAR-10.
Completes in ~1–2 minutes on any hardware.

```bash
python run_experiments.py --exp smoke # --device mps
```

You should see output like:
```
Using device: cuda
Loading dataset: cifar10
Building model: resnet18
Free-rider clients: [2]
...
  Round    1/2 | Acc=0.112 | WM_benign=0.531 | FR_det=nan | FPR=0.000 | Elapsed=38s
  Round    2/2 | Acc=0.134 | WM_benign=0.548 | FR_det=1.000 | FPR=0.000 | Elapsed=71s
Smoke test passed ✓
```

---

## Running experiments

All experiments write JSON results to `./results/` by default.

```bash
# Single experiment
python run_experiments.py --exp table1
python run_experiments.py --exp table2
python run_experiments.py --exp fig7
python run_experiments.py --exp fig8
python run_experiments.py --exp table3
python run_experiments.py --exp table6     # requires opacus
python run_experiments.py --exp table7
python run_experiments.py --exp table8
python run_experiments.py --exp fig9
python run_experiments.py --exp fig10

# All experiments (long — several hours total)
python run_experiments.py --exp all

# Custom output directory
python run_experiments.py --exp table1 --output_dir ./my_results

# Force CPU even if GPU available
python run_experiments.py --exp smoke --device cpu
```

---

## Running a custom configuration

```python
from faremark_mod import FareMarkConfig, FareMarkTrainer

cfg = FareMarkConfig(
    model_name="resnet18",      # 'alexnet', 'resnet18', 'shufflenet', 'googlenet'
    dataset_name="cifar10",     # 'mnist', 'cifar10', 'cifar100', 'food100'
    num_clients=10,
    num_free_riders=2,
    free_rider_type="previous_models",  # or 'gaussian_noise'
    global_rounds=100,
    local_epochs=2,
    batch_size=16,
    lr=0.01,
    wm_bits=8,
    lam=1.0,
    beta=0.9,
    smooth_fn="frac_power",     # 'neg_power', 'frac_power', or 'sin'
    alpha_smooth=0.5,
    n_triggers=100,
    device="cuda",
    output_dir="./results",
    exp_name="my_experiment",
)

trainer = FareMarkTrainer(cfg)
results = trainer.run()

print(f"Final accuracy:        {results['main_acc'][-1]:.4f}")
print(f"Watermark acc (benign): {results['wm_acc_benign'][-1]:.4f}")
print(f"FR detection rate:     {results['fr_detection_acc'][-1]:.4f}")
print(f"False positive rate:   {results['fpr'][-1]:.4f}")
```

---

## Hyperparameters from the paper (Section V-A)

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | 0.01 | SGD with momentum 0.9 |
| Batch size | 16 | Per-client local batch size |
| Local epochs | 2 | Except fidelity tests: 5 |
| Global rounds | 100 | 50 for fidelity/detection tables |
| lambda (λ) | 1.0 | Classification / watermark balance |
| beta (β) | 0.9 | Memory-enhanced blend factor |
| N_T (triggers) | 100 | For watermark extraction |
| Detection threshold η | μ + 3σ | Estimated from benign clients |
| Watermark bits m | 8 | Per client |
| Smoothing function | x^α, α=0.5 | `frac_power` |

---

## Results files

Each experiment produces a `results.json` under `./results/<exp_name>/`:

```json
{
  "rounds": [10, 20, ...],
  "main_acc": [0.71, 0.81, ...],
  "wm_acc_benign": [0.91, 0.99, ...],
  "wm_acc_freerider": [0.48, 0.51, ...],
  "fr_detection_acc": [1.0, 1.0, ...],
  "fpr": [0.0, 0.0, ...],
  "config": { ... }
}
```

Top-level JSON files (e.g. `table1_fidelity.json`) summarise final values
across all configurations for easy comparison with the paper's tables.

---

## Project structure

```
faremark/
├── faremark/
│   ├── __init__.py       — package exports
│   ├── watermark.py      — watermark representation, loss, extraction
│   ├── client.py         — FL client with memory-enhanced update
│   ├── server.py         — FL server: aggregation & detection
│   ├── models.py         — AlexNet / ResNet-18 / ShuffleNet / GoogLeNet
│   ├── datasets.py       — MNIST / CIFAR-10 / CIFAR-100 / Food-101
│   ├── config.py         — all hyperparameters & paper preset configs
│   ├── train.py          — main FL training orchestrator
│   └── evaluate.py       — fine-tuning / pruning / DP robustness
├── run_experiments.py    — CLI runner for all paper experiments
├── requirements.txt
└── README.md
```

---

## Implementation notes & known deviations from the paper

1. **GoogleNet auxiliary loss**: torchvision's GoogLeNet returns a
   `GoogLeNetOutputs` namedtuple in training mode. The trainer handles
   this automatically.

2. **Food-101**: The paper refers to "Food100" (100 classes). We use
   torchvision's `Food101` dataset with all 101 classes and a label map.

3. **FedIPR baseline**: Table I and Table II reference FedIPR as a
   comparison. This re-implementation does **not** include FedIPR;
   only FareMark is implemented. You can compare our numbers against
   the paper's Table I/II values directly.

4. **Threshold η**: The paper sets η = μ + 3σ from observed benign
   client errors. Early rounds use a fallback of η = 0.30 until ≥5
   benign samples are observed.

5. **IID data split**: The paper uses an even IID split across clients.
   Non-IID experiments are not described in the paper and not implemented.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{li2025faremark,
  title   = {{FareMark}: Model-Watermark-Driven Free-Rider Detection in
             Federated Learning Model},
  author  = {Li, Li and Zhang, Xinpeng and Wu, Hanzhou and
             Feng, Guorui and Zhang, Weiming},
  journal = {IEEE Internet of Things Journal},
  volume  = {12},
  number  = {18},
  pages   = {38965--38977},
  year    = {2025},
  doi     = {10.1109/JIOT.2025.3587745}
}
```