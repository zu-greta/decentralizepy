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
| `--wm_bits` / `WM_BITS` | 0 (auto) | m, message length | 0 → auto = (classes−1)//2. Lower m on a fixed dataset = forces larger group size l but fewer bits (worse separation). |
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
