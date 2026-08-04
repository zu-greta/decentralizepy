# FareMark — reproduction + limitations study

Re-implementation and limitations analysis of **FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning** (Li et al., IEEE IoT-J 12(18), 2025).
Centralized FedAvg simulated on one GPU, with a per-client output-layer watermark loss, a memory-enhanced update (Eq. 14), and server-side verification (Eq. 15–16).

---

## Goal

Show experimentally that under the paper's own setup and under extensions (non-IID, adaptive free-riders, more clients than classes), no threshold separates honest clients from free-riders - ie. output layer watermarking in federated learning for free-rider detection is impossible in general

---

## Layout - TODO update the layout with the current code status

```
faremark/
  clients.py           every client class:
                       PART 1  honest FedAvg Client
                       PART 2  WatermarkClient (Eq.11-12 + Eq.14) + factory
                       PART 3a crude baselines Eq.17/18 + FR selection + build_clients
                       PART 3b reduced/tap attackers [+ DISABLED submarine]
  watermark.py         Eq.1-16 math -- leaf, used by clients.py AND wm_verify.py
                       *** PATCHED: sin smoothing, SMOOTH_EPS, make_key default ***
  wm_verify.py         server: extract -> BER -> frozen eta -> flag + diagnostics
  server.py            FedAvg aggregation + round loop
  runlog.py            all run.log formatting
  compute_meter.py     per-client effort accounting (gpu_ms / samples / flops)
  config.py datasets.py models.py manifest.py utils.py plotstyle.py
  fast_data.py         [OPTIONAL, NOT WIRED] GPU-resident loader, see PERFORMANCE
  robustness.py        [UNWIRED] finetune/prune/quantize -- paper V-E, not run
scripts/
  run_experiment.py    one (config, repeat) -> result.json
  resultio.py          result.json data contract (load/select/BER extraction)
  detection.py         calibrate | verify | separability
  plots.py             all plotting (subcommands, see REFERENCE 3a-F)
  plot_all_thresholds.py  *** NEW: every threshold rule on one timeline + .md table ***
  paper_check.py       grade runs vs the paper's published rows
infra/
  run_now.sh           *** NEW: THE script. Builds jobs.tsv for the current batch ***
  plot_now.sh          *** NEW: makes exactly the four figure groups, nothing else ***
  submit_pool.sh       *** NEW: submits exactly PODS worker jobs that drain jobs.tsv ***
  submit_experiment.sh *** PATCHED: deterministic RUN_TAG + DRYRUN manifest mode ***
  run_everything.sh    the full matrix in legs (still works; superseded by run_now.sh)
  run_all.sh           one leg: honest -> calibrate -> attacks -> separability -> PLOTALL
  paper_check.sh       submit/grade the paper reproduction rows
```

---

## Standard setup

CIFAR-100, ResNet-18, 10 clients, 50 rounds, 5 local epochs, batch 16, N_T=50, λ=5,
β=0.6, α=0.4
Trigger class = `cid % n`

| dataset | m | l | stuck | ceiling | paper reports | 
|---|---|---|---|---|---|---|
| CIFAR-100 | 10 (code default) | 10 | 0.20% | 99.90% | 99.71 | 

Threshold: `η = mean over seeds of (μ_s + 3σ_s)` over per-round mean-over-clients honest BER, last 20 rounds; frozen and injected as `WM_ETA_FIXED`.
CIFAR-100 / 10 clients: **η = 0.063** (per-seed 0.017–0.115, std ≈ 40%).

---

## Quickstart - TODO finalise the quickstart commands

```bash
# 1. on the cluster login node, in infra/
./run_now.sh                    # builds jobs.tsv, submits NOTHING
wc -l jobs.tsv                  # expect 20
unset DRYRUN
RUNAI_EXTRA="--node-pools a100-80" PODS=2 WORKERS=6 ./submit_pool.sh
runai list jobs                 # must show exactly 2 -- then walk away

# 2. when done
scp -r <cluster>:$MOUNT/home/zu/results ~/local/results
RES=~/local/results ./plot_now.sh
```

---
---
---

# STORYLINE

## Plan

### 1. Reproduce the FareMark method — done
Eq. 1–16 implemented in `watermark.py`, embedding in `clients.py`, verification in `wm_verify.py`. Each implementation step was checked against the paper and tested against the paper's results.

### 2. Reproding FareMark results — in progress
Tested baselines against the paper's result tables. missing final tables for abaltions and extra studies

### 3. Limitations — done
Figure out paper's weak points, what to exploit: threshold (non-seperability), class difficulty, non-iid, oversubscription, adapted free-riders

### 4. Prove limitations + build an attack - in progress
Thresholds limitation shown through reduced attacker. Building submarine attacker

### 5. Stress test — todo
Stress tests each setting for the above limitations

### 6. Argue output-layer watermarking is impossible — todo
...

### 7. Hint at a solution — todo
...


---

# CODEMAP - TODO complete the codemap once all code is done, wired and tested

Legend: **[WIRED]** in the pipeline · **[NEW]** added this cycle · **[PATCHED]** changed
this cycle · **[UNWIRED]** exists but not called.

## 1. Watermark math — `faremark/watermark.py` [WIRED] [PATCHED]

| step | function | paper | notes |
|---|---|---|---|
| smoothing f(p) | `smooth(p, kind, alpha, eps)` | Eq. 7–9 | **[PATCHED]** `eps` → module const `SMOOTH_EPS` (env-switchable, default legacy 1e-3); `sin` branch validates α ∈ (0, π/2] and rejects gain < 1.10 |
| how much f() actually smooths | `smoothing_gain(kind, alpha)` | — | **[NEW]** 1.0 = f does nothing. Check any (kind, α) before spending GPU |
| secret ±1 key M [m,l] | `make_key(m, l, seed, balanced)` | §IV-A | **[PATCHED]** default now `False`, matching `config.wm_balanced_keys` and the paper |
| stuck-row fraction | `unembeddable_fraction` | diagnostic | `P = 2^(1−l)`; logged as `wm_unembeddable_frac`. Ceiling = `1 − 0.5·P` |
| target bits B | `make_bits` | Eq. 2 | balanced 0/1, so a random model sits at BER 0.5 |
| group size l = n//m | `grouping` | §IV-A | `m ≤ n` — the bit ceiling (F6) |
| project → per-bit z | `project_logits(..., exclude)` | Eq. 1/13 | `exclude=None` = full softmax, **paper-faithful** |
| embed loss | `watermark_loss` | Eq. 11–12 | BCE(z, B) |
| extract | `extract_bits` | Eq. 15 | mean z over N_T, then sign |
| BER | `bit_error_rate` | Eq. 16 | |
| flag test | `detected(ber, eta)` | Eq. 16 | `ber < eta`. Docstring now spells out the η=0 and η<1/m degeneracies (F9) |
| dominance ratio | `dominance_ratio` | Eq. 6/10 | want < 0.5; the diagnostic that exposed the sin bug |

## 2. Honest client + factory — `faremark/clients.py` [WIRED]
- `WatermarkClient.produce_update` → `_local_train_wm` (`L = CE + λ·wm_loss` on trigger
  images) → `_memory_update` (Eq. 14 `W = β(memory+Δ) + (1−β)·global`).
- `build_watermarked_clients`: `trigger_class = cid % n`; `m = cfg.wm_bits or max(2, n//10)`;
  `l = n//m`; `exclude_col = None`; `key = make_key(..., balanced=cfg.wm_balanced_keys)`
  seeded `seed + 1000·cid + 1`. Dispatches free-rider slots by `cfg.attack`.
- **Open edit:** if you adopt `fast_data.py`, the reduced attacker must call
  `self.loader.subset(idx)` instead of building a fresh CPU `DataLoader`.

## 3. Attackers — `faremark/clients.py` PART 3
`previous_models` / `gaussian` (Eq. 17/18) · `reduced` (+N, the main one) ·
`tap_oracle` (coast/tap adaptive) · `submarine` **[DISABLED — warmup bug]**.

## 4. Detector — `faremark/wm_verify.py` [WIRED]
`WatermarkRegistry` (cid → trigger_class, key, bits, kind, alpha, exclude);
`build_trigger_bank` (`class` mode, held-out, shared per class),
`build_trigger_bank_per_client` (`client`, disjoint held-out slices),
`build_trigger_bank_from_train` (`client_train`, the client's own training images —
paper §V-F3). **All three are used only inside `verify_hook`, never in training**, so
one training run can be extracted three ways (saves 18 jobs on the capacity legs).
Per round emits `wm_benign_ber{,_p90,_max}`, `wm_fr_ber`, `wm_fpr`, `wm_fr_recall`,
`wm_eta_round`, `wm_eta_source`, `wm_flagged_cids`, and `wm_per_client[]` with
`{cid, trigger_class, ber, is_free_rider, flagged, pmax, entropy, dominance, trig_acc}`.
**Missing:** `n_trigger_samples` per client per round — required for F11.

## 5. Analysis — `scripts/detection.py` [WIRED]
`calibrate` (freeze η) · `verify` (confirm attack runs used it) · `separability`
(the rule-independent tables: 9 rules + OVL + best-threshold balanced error).
**Open edit:** emit `rule=degenerate` instead of `fpr=1.0, recall=1.0` when the honest
support is a point mass.

## 6. Orchestration — `scripts/run_experiment.py` [WIRED]
Every CLI flag overrides the matching `cfg` field. Writes `result.json`.
**Exit code 2 = accuracy outside `expected_acc`** — EXPECTED for attack runs;
`result.json` is written before the exit. `submit_pool.sh` treats 0 and 2 as success.

## 7. Plotting — `scripts/plots.py` [WIRED] + `plot_all_thresholds.py` [NEW]
Subcommands: `thresholds`, `class_difficulty`, `class_probe`, `class_dynamics`,
`positions`, `fidelity`, `timeline`, `honest_lines`, `separability`, `sweep`,
`honest_fpr`, `sanity`, and legacy `threshold`/`frontier`/`scorecard`/`test_data`.
`plot_all_thresholds.py` **[NEW]** draws *every* honest-only rule on one BER-vs-round
timeline plus a red `1/m` line, and emits a `.md` table giving each rule's η, how it was
computed in prose, its honest FPR, its **headroom in σ**, and whether it is degenerate.
This is the figure `plots.py thresholds` could not make.

**Open plot fixes:** the green "USED eta" line is described in `thresholds`' title but
never drawn; `timeline` prefers `config.wm_eta_fixed` over the eta file, so figures show
the provisional 0.050; `honest_lines` bands extend below 0; the 200-client legend covers
60% of the canvas.

## 8. Runners — `infra/`
| script | role |
|---|---|
| `run_now.sh` **[NEW]** | builds `jobs.tsv` for the current batch (R1–R8). Submits nothing |
| `submit_pool.sh` **[NEW]** | submits exactly `PODS` worker jobs; each drains its shard with `WORKERS` concurrent runs. Resume-safe |
| `plot_now.sh` **[NEW]** | local; makes exactly the four figure groups |
| `submit_experiment.sh` **[PATCHED]** | deterministic `RUN_TAG` (no timestamp — required for resume) + `DRYRUN=1` manifest mode |
| `run_all.sh`, `run_everything.sh`, `paper_check.sh` | unchanged; still work for ad-hoc legs |

## Data contract — `result.json`
Authoritative in `scripts/resultio.py`. Top level: `schema_version`, `config`,
`manifest{family, sweep_var, sweep_level}`, `summary{}`, `env{git_commit, torch, host}`,
`free_rider_indices`, `final_acc`, `best_acc`, `correctness_pass`, `per_class`,
`compute`, `history[]`. Quick look:

```bash
python scripts/resultio.py digest   --in 'results/*/result.json'
python scripts/resultio.py contract --in results/<run>/result.json
```