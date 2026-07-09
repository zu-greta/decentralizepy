> TODO: update the documentation files




# EXPERIMENTS — the registry

One place that says **what each experiment family is, why it exists, what to
sweep, and how to read the result.** If you add an experiment, add a row here.
Every run also stamps a machine-readable copy of its family/sweep/interpretation
into its own `result.json` under `"manifest"` (see `faremark/manifest.py`), so a
run is never opaque even in isolation.

## How a run describes itself

`run_experiment.py` writes into `result.json`:

- `manifest.family`  — which family below (e.g. `A7_submarine`).
- `manifest.sweep_var` / `manifest.sweep_level` — the one knob this run varies
  and its value (e.g. `mem_blend_global` = `0.3`).
- `manifest.note` — a one-line hypothesis you pass via `NOTE=...`.
- `manifest.interpretation` — per-metric "which direction is good" key.
- `compute.summary` — attacker-vs-honest effort (`effort_ratio_gpu`,
  `effort_ratio_samples`, duty cycle).

You set family/sweep from the submit script:

```bash
ATTACK=submarine CALIB_ON_ALL=1 MEM_BLEND_GLOBAL=0.3 \
  FAMILY=A7_submarine SWEEP_VAR=mem_blend_global NOTE="cheap evasion, realistic eta" \
  TAG=a7-mbg03 WAIT=0 ./submit_experiment.sh 14 0
```

`plot_adaptive.py` groups runs by `family[sweep_var=level]` and averages over
seeds (repeats), so consistent `FAMILY`/`SWEEP_VAR` tags are what make the plots
assemble themselves.

## Naming grammar (results directory)

```
cfg{IDX}_rep{SEED}{-frN}{-TAG}_{timestamp}/result.json
```

`TAG` is free-form but keep it aligned with the family, e.g. `a7-mbg03`,
`a8-warmup5`, `a5-blend05`. The authoritative descriptor is the manifest inside
the file; the directory tag is just for eyeballing `ls`.

## Threshold options (applies to every watermark+free-rider family)

The paper-faithful detector is `PAPER_FAITHFUL=1` (cumulative, uncapped μ+3σ).
The attacker-vs-threshold relationship is a **server-side** flag:

| Option | Flag | Meaning |
|---|---|---|
| 1 | `CALIB_ON_ALL=0` (default) | η is calibrated on a trusted benign pool that **excludes** the attacker. Idealized; the attacker must **guess** η. |
| 2 | `CALIB_ON_ALL=1` | η is μ+3σ over **all** clients incl. the attacker, computed each round during training. Realistic; the attacker **poisons/inflates** η. |

The attacker's *own* η-guess is a separate knob, `SUB_ETA_MODE` (`adaptive`
mirrors the server's μ+3σ over its submitted-BER history; `fixed` assumes
`SUB_ETA_FIXED`). Run **both** options for the adaptive families.

---

## Families

Legend for "read": ✅ = defender wins, ❌ = attack wins.

### Reproduction / fidelity (Part 1 — done)
| Family | config idx | What | Read |
|---|---|---|---|
| `baseline` | 1–6 | FedAvg Table I accuracy bands | final_acc in band |
| `fidelity` | 11, 13 | all-honest watermark; accuracy cost of embedding | final_acc ≈ baseline, honest BER→0 |

### Static free-riders (the paper's own attacks — the strawmen)
| Family | idx | What | Read |
|---|---|---|---|
| `previous_models` | 8, 12, 13 | Lin et al. delta-weights fabricator; never embeds | ✅ caught: fr_ber≈0.5, recall→1 |
| `gaussian` | 9 | Fraboni et al. stochastic perturbation; never embeds | ✅ caught: fr_ber≈0.5 |

These exist to show the detector catches the *trivial* case. They are **not**
the contribution — they never even try to embed.

### Key-holding evasion (our attacks)
| Family | idx | Sweep (`SWEEP_VAR`) | Hypothesis / what it shows |
|---|---|---|---|
| `A2_train_then_attack` | 12/13 + `ATTACK=train_then_attack` | `attack_round` | Embed then defect; how long the mark persists after training stops (Table IV). |
| `A5_trigger_only` | + `ATTACK=trigger_only` | `n_trigger_samples` | Overfitting the mark on few trigger samples → self-BER low but **test** BER high (generalization gap, Table V). |
| `A5_mixed` | + `ATTACK=mixed` | `blend`, `full_trigger_class`, `n_common_samples` | No-key-ish forgery blending a lightly-trained embed with the global. |
| **`A7_submarine`** | **14** | `sub_warmup`, `sub_coast_mode`, `sub_max_burst_batches`, `mem_blend_global` | **Warmup a mark, then coast; tap only when BER drifts up. PRELIMINARY: caught (blend coast → mark decays) or poisons (replay coast → stale). Coast modes: transplant/blend/replay/noise/global.** |
| **`A8_memory_exploit`** | **15** | `warmup_rounds` | **Train `warmup_rounds`, then replay the frozen mark. PRELIMINARY: evades η but POISONS the model (acc 72→55, honest BER→0.5). Cheapest but not stealthy.** |
| **`F_reembed`** | **16** | `reembed_scope`, `reembed_steps` | **NEW / not yet run. Fresh global backbone + cheap head-only trigger fine-tune → fresh + marked + cheap. The theoretically-motivated attack; the sweep decides if it reaches the "cheap + evades + healthy" corner.** |

**Read results by `wm_fr_ber` (below η = evades) and `final_acc` (~72 = healthy,
low = poisoned), NOT `wm_fr_recall`** — the paper-faithful η is cumulative and
swings 0.35–0.88, so recall is a noisy, threshold-dependent readout. The claim is
the pair: `effort_ratio ≪ 1` **with** `fr_ber < η` **and** `acc` healthy.

### Detector-fragility (context, not attacks)
| Family | flag | What |
|---|---|---|
| `noniid` | `PARTITION=dirichlet DIRICHLET_ALPHA=…` | Non-IID inflates benign-BER variance → η self-inflates → recall collapses even against static free-riders. A property of the threshold, distinct from the effort attacks. |

---

## Recommended matrix (CURRENT)

The single command that runs every attack's weak-point sweep (submarine warmup /
samples / coast type, memory warmup, and the reembed frontier) is:

```bash
./scripts/run_full_sweep.sh          # ~23 runs, 1 seed, WAIT=0, priority-ordered
RES=/mnt/nfs/home/zu/results ./scripts/make_sweep_figs.sh   # all figures
```

See **STATUS.md §4–7** for the preliminary results and what each sweep answers.
The headline figure is the **weak-point map** (`figs/weakpoint_all.png`): fr_ber
vs effort, colored by accuracy, with the η line — the target is the low-effort
corner that is below η AND still green (healthy). As of now that corner is empty
for the three coast attacks; `reembed` is the candidate to fill it.

Older per-family commands are in RUNSHEET_ADAPTIVE.md, but prefer the sweep above.