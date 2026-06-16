# FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning

This repository implements **FareMark**, a watermark‑based framework for detecting free‑riders in Federated Learning (FL) as proposed in:

> *FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning Model*  
> Li Li, Xinpeng Zhang, Hanzhou Wu, Guorui Feng, and Weiming Zhang

The implementation is built with PyTorch and supports:
- Multiple clients embedding unique watermarks in a shared FL model.
- Box‑free watermark representation using softmax outputs.
- Memory‑enhanced gradient updates to mitigate aggregation conflicts.
- Detection of various free‑rider strategies.
- Robustness tests (fine‑tuning, pruning, quantization, differential privacy).

## Repository Structure
faremark/
├── README.md
├── requirements.txt
├── Dockerfile
├── submit_job.sh
├── configs/
│ ├── exp_fidelity.yaml
│ ├── exp_detection.yaml
│ ├── exp_free_rider.yaml
│ ├── exp_robustness.yaml
│ └── exp_ablation.yaml
├── faremark/
│ ├── init.py
│ ├── models.py
│ ├── datasets.py
│ ├── watermark.py
│ ├── client.py
│ ├── server.py
│ ├── simulator.py
│ ├── free_rider.py
│ ├── robustness.py
│ ├── metrics.py
│ └── utils.py
└── scripts/
├── run_experiment.py
├── run_validation.py
└── run_simple_test.py


## Installation

```bash
git clone <this-repo>
cd faremark
pip install -r requirements.txt
```

## Quick start
Run a simple test with 10 clients, 1 free‑rider, 10 rounds on CIFAR‑10:

```bash
python scripts/run_simple_test.py
```

## Running an Experiment
Use a configuration file (YAML) to set parameters:
```bash
python scripts/run_experiment.py --config configs/exp_fidelity.yaml
```
Results (logs, metrics, plots) are written to `logs/exp_name/`.

## Reproducing Paper Tables
Execute all experiments from the paper:
```bash
python scripts/run_validation.py
```
This runs the configurations corresponding to Tables I–IX.
Note: Some experiments are computationally heavy; use a GPU cluster.

## Using the RunAI Cluster
1. Build the Docker image:
```bash
docker build -t faremark .
```
2. Edit `submit_job.sh` with your RunAI project and image details (e.g., python scripts/run_validation.py).
3. Submit the job:
```bash
sbatch submit_job.sh
```

## Customization
New models: add to faremark/models.py and update get_model().

New datasets: add to faremark/datasets.py and update get_dataset().

New free‑rider strategies: implement in faremark/free_rider.py.

Watermark parameters: adjust watermark_lambda, memory_mu, trigger_class, alpha in config.