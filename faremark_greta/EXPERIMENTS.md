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
| **`A7_submarine`** | **14** | `mem_blend_global`, `sub_margin`, `calib_on_all`, `sub_eta_mode` | **Closed-loop, keeps BER just under its η-estimate; taps minimal bursts only when needed. Tracks the global to stay fresh (robust to staleness checks). The headline "cheap evasion" result.** |
| **`A8_memory_exploit`** | **15** | `warmup_rounds`, `mem_blend_global` | **Train once, replay the frozen marked memory forever. BER≈0 at ~1 round of compute — the cheapest break, but naive-detectable by staleness.** |

For A7/A8 the point is the **compute** block: `effort_ratio_gpu ≪ 1` **with**
`wm_fr_recall → 0` ⇒ the scheme is defeated cheaply (❌ for the defender).

### Detector-fragility (context, not attacks)
| Family | flag | What |
|---|---|---|
| `noniid` | `PARTITION=dirichlet DIRICHLET_ALPHA=…` | Non-IID inflates benign-BER variance → η self-inflates → recall collapses even against static free-riders. A property of the threshold, distinct from the effort attacks. |

---

## Recommended matrix for the adaptive claim

Run each of A7 and A8 at **≥3 seeds** (`repeat` 0,1,2), under **both** threshold
options, plus the static `previous_models` baseline as the "expensive-to-catch-
never" anchor:

```bash
# A7 submarine — option 1 (attacker guesses eta) and option 2 (eta poisoned).
# Note: bursts are trigger-enriched and the submarine warms up a generalizing
# mark first (SUB_WARMUP rounds) — a naive tap over the general shard does NOT
# embed on CIFAR-100 (trigger class ~1% of shard). Run >=50 rounds; 10 is a
# transient where paper-faithful eta is still inflated by untrained early rounds.
for CAL in 0 1; do for R in 0 1 2; do
  ROUNDS=50 ATTACK=submarine PAPER_FAITHFUL=1 CALIB_ON_ALL=$CAL SUB_WARMUP=3 \
    FAMILY=A7_submarine SWEEP_VAR=calib_on_all \
    TAG=a7-cal$CAL WAIT=0 ./submit_experiment.sh 14 $R
done; done

# A8 memory-exploit — warmup sweep (1 = pure exploit, 5 = "momentum")
for W in 1 3 5; do for R in 0 1 2; do
  ATTACK=memory_exploit PAPER_FAITHFUL=1 WARMUP_ROUNDS=$W \
    FAMILY=A8_memory_exploit SWEEP_VAR=warmup_rounds \
    TAG=a8-w$W WAIT=0 ./submit_experiment.sh 15 $R
done; done
```

Then:

```bash
python scripts/plot_adaptive.py effort --in "$RES/*a7-*" "$RES/*a8-*" "$RES/*prev*" \
       --out figs/effort_plane --effort gpu --metric wm_fr_recall
python scripts/plot_adaptive.py squeezing --in "$RES/*a7-cal1*" --out figs/a7_squeeze
python scripts/plot_adaptive.py sweep --in "$RES/*a8-*" --sweep_var warmup_rounds \
       --metric wm_fr_recall --out figs/a8_warmup
python scripts/plot_adaptive.py duty --in "$RES/*a7-cal1*rep0*" --out figs/a7_duty
```

The **effort plane** (`figs/effort_plane`) is the money figure: static
free-riders sit at high effort / high recall or zero effort / high recall
(caught), while A7/A8 sit at **low effort / ~zero recall** — outside the region
the scheme can defend.