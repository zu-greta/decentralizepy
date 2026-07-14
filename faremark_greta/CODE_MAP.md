# CODE_MAP — technical reference

Line numbers refer to the **cleaned** files (the autopilot-only refactor). Core
files (client, server, watermark, wm_verify, compute_meter, thresholds, datasets,
utils, manifest, plotstyle) are unchanged from the original upload.

Package layout (import root `faremark/`, scripts in `scripts/`):

```
faremark/
  client.py           honest FedAvg client (base class)
  server.py           FedAvg aggregation + round loop + verify hook
  datasets.py         IID / Dirichlet shards + trigger test set
  watermark.py        the watermark math (Eq. 1-16): key/bits/embed/extract/BER/eta
  wm_client.py        WatermarkClient (honest embed + Eq.14 memory) + client factory
  attacks.py          crude paper baselines (previous_models, gaussian) + FR selection
  attacks_adaptive.py AUTOPILOT adaptive free-rider (the attack under study)
  wm_verify.py        server: extract each client's mark -> BER -> calibrate eta -> flag
  compute_meter.py    per-client effort (samples, gpu_ms, flops, duty cycle)
  thresholds.py       offline eta variants for plots (frozen/converged/windowed/...)
  manifest.py         self-describing run metadata block for result.json
  models.py           build_model (resnet18/alexnet/smallcnn)   [not reviewed here]
  robustness.py       finetune/prune/quantize ops (Figs 9-10)   [not reviewed here]
  utils.py            set_seed, get_logger, evaluate_accuracy
  plotstyle.py        shared matplotlib style (colour-blind palette, stacked panels)
scripts/
  run_experiment.py   orchestrates one (config, repeat); writes result.json
  run_tests.py? -> run_tests.sh / submit_experiment.sh   cluster submit + the 3-test sweep
  plot_tests.py       the 3-test figures (FPR + per-FR/per-honest BER + effort)
  aggregate_results.py   mean+/-std tables over seeds (reproduction)
  submit_sweep.sh     generic configs x repeats launcher (reproduction)
  run_robustness.py   robustness driver (Figs 9-10)             [optional]
```

---

## 1. The signal: watermark math — `faremark/watermark.py` (214 lines)

The ONE quantity everything keys off: **BER** (bit-error-rate) of a client's m-bit
mark, extracted from its trigger-class softmax.

| Symbol / step | Function | Line | Paper |
|---|---|---|---|
| smoothing f(p)=p^alpha | `smooth` | 48 | Eq. 7-9 |
| secret +/-1 key M [m,l] | `make_key` | 67 | §IV-A |
| same-sign (stuck) key rows | `unembeddable_fraction` | 91 | floor diagnostic |
| target bits B in {0,1}^m | `make_bits` | 106 | Eq. 2 |
| group size l = n//m | `grouping` | 120 | §IV-A |
| project probs -> per-bit z | `project_logits` | 132 | Eq. 1/13 |
| embed loss BCE(z, B) | `watermark_loss` | 157 | Eq. 11-12 |
| extract: mean over N_T, sign | `extract_bits` | 170 | Eq. 15 |
| BER = mean(bits != B) | `bit_error_rate` | 178 | Eq. 16 |
| flag test BER < eta | `detected` | 183 | Eq. 16 |
| **eta = mu + 3*sigma** | `calibrate_eta` | 188 | §IV-D-3 |
| anti-dominance ratio | `dominance_ratio` | 202 | Eq. 6/10 |

**Key fact:** `calibrate_eta(benign_bers, floor)` takes μ+3σ over **whatever list you
pass it**. In the live server that list is the per-round *mean* honest BER (see §4).

---

## 2. Honest client — `faremark/client.py` (64) + `faremark/wm_client.py` (186)

- `client.Client` (26): plain FedAvg. `produce_update` (42) = load global -> `_local_train` (53) -> submit.
- `wm_client.WatermarkClient` (19): honest + watermark.
  - `produce_update` (42): load global -> `_local_train_wm` -> `_memory_update` -> submit.
  - `_local_train_wm` (52): `L = CE + wm_lambda * watermark_loss` on trigger images; meters each batch.
  - `_memory_update` (76): Eq. 14 blend `W = beta*(memory+delta) + (1-beta)*global`, persists `self.memory`.
- `build_watermarked_clients` (94): **the factory**. Assigns per-cid
  `trigger_class = cid % num_classes`, key & bits seeded by cid (this is the
  client's "position"), registers them, and dispatches free-rider slots to the
  autopilot or a baseline. Honours `cfg.free_rider_ids` via `resolve_free_riders`.

---

## 3. The attack: AUTOPILOT — `faremark/attacks_adaptive.py` (374)

`AutopilotFreeRider(_AdaptiveMixin, WatermarkClient)` — an honest client with a
different control flow. Reuses key/bits/lambda/alpha/beta/memory/`_local_train_wm` verbatim.

**`_AdaptiveMixin` (38):**
- `_ensure_triggers` (44): gather this shard's trigger images once; hold out a probe
  slice; build the reduced (data-ablation) tap loader from `autop_common_per_class`.
- `_probe_ber_current_model` (91) / `_probe_ber_state` (104): the FR's private BER self-probe.
- `_embed_loop` (112): the training loop for a tap. `scope` picks params
  (full/block2/block/head -> freezes backbone tensors); loader = reduced shard
  (cpc>=0) else full shard; `early_stop` gates the probe (off in taps -> effort fix).

**`make_autopilot_attack` (176) -> `AutopilotFreeRider` (181):**
- `__init__` (185): all `autop_*` knobs (see §7).
- `_eta_est` (235): ORACLE (if `autop_oracle_eta>0`) -> FROZEN (set after honest phase)
  -> best estimate μ+kσ over converged forced-honest BERs (trimmed) -> fallback.
- `_record_clean` (255): keep post-embed BERs (<=0.30) as the eta anchor.
- `_coast_state` (260): fresh global + frozen mark-direction (re-inject mark ~free).
- `_update_mark_delta` (275): memory - global = the mark direction.
- `produce_update` (283): **the controller** —
  1. `autop_honest_clone` bypass -> pure honest every round (control).
  2. WARMUP / honest phase: train honestly, detect convergence (`autop_conv_eps`),
     calibrate + FREEZE eta once, then end warmup (this is "know when the server
     stops forcing honesty").
  3. POST-WARMUP: if `autop_stay_min` and coast probe safely under target -> COAST
     (no training); else TAP = `_embed_loop` at fixed budget (cost = data x scope).

**Baselines — `faremark/attacks.py` (143):** `PreviousModelsFreeRider` (51, Eq. 17),
`GaussianNoiseFreeRider` (63, Eq. 18), `choose_free_riders` (93),
`resolve_free_riders` (107, honours `free_rider_ids`), `build_clients` (116, non-wm path).

---

## 4. The detector — `faremark/wm_verify.py` (157)

- `WatermarkRegistry` (22): cid -> (trigger_class, key, bits, kind, alpha, exclude).
- `build_trigger_bank` (47): N_T test-set images per trigger class (held-out).
- `make_verifier` (65) -> `verify_hook` (81), per round:
  1. Extract every client's mark on the trigger bank -> **one BER per client** (`measured`).
  2. `benign_now` = this round's honest BERs; **append their MEAN to `benign_history`**
     (the round-mean series). `calib_on_all=True` includes free-riders in the pool.
  3. `eta_round = calibrate_eta(benign_history)` = μ+3σ over the round-MEAN series
     (paper_faithful=cumulative; else windowed+capped at 0.25).
  4. Flag each client individually iff `ber >= eta_round`.
  5. Emit `wm_benign_ber`, `wm_fr_ber`, `wm_eta_round`, `wm_fpr`, `wm_fr_recall`,
     `wm_benign_ber_list`, `wm_fr_ber_list`, and `wm_per_client`
     = `[{cid, trigger_class, ber, is_free_rider, flagged}]`.

> This is the crux of the whole project: eta is built from **per-round means** (tight)
> but applied to **individual clients** (spread). See STATUS.md §Finding.

---

## 5. Orchestration + effort — `scripts/run_experiment.py` (304), `compute_meter.py` (168)

- `parse_args` (34): CLI overrides (all `--autop_*`, `--free_rider_ids`, `--attack`,
  partition, watermark, `--calib_on_all`, manifest). `_OVERRIDABLE` (99) applies them onto cfg.
- `main` (149): build data -> model -> clients (wm or baseline path) -> Server.run ->
  assemble `result.json` (config, manifest, wm summary, compute, history).
- `collect_compute` (112): per-client + summarized effort ->
  `honest_mean_gpu_ms/samples`, `fr_mean_gpu_ms/samples`, `effort_ratio_gpu/samples`.
- `compute_meter.ComputeMeter` (43): `record_batch` (82, training = samples+fwd+bwd+opt),
  `record_forward_only` (89, probe = fwd only, NOT counted as training samples -> the
  effort-inflation fix), `end_round` (95, gpu_ms via CUDA events), `summary` (117).

---

## 6. Server, data, plots

- `server.Server.run` (53): each round calls every client's `produce_update`, runs the
  verify hook, then `Aggregator.aggregate` (19, weighted FedAvg), evaluates test acc.
- `datasets.build_data` (101): `iid_partition` (63) or `dirichlet_partition` (70, Hsu 2019).
- `thresholds.eta_series` (39): offline eta variants — `frozen` (post-convergence window,
  the HEADLINE), `converged` (last-C), `windowed`, `cumulative`, `fixed`. **Note:** these
  are for plotting; the LIVE detector uses `watermark.calibrate_eta` over the round-mean series.
- `plot_tests.py`: `test1_fpr` (58) — per-client honest BER vs two eta definitions + FPR;
  `test_data` (140) — per-FR & per-honest BER + GPU/samples effort over the data sweep.
- `plotstyle.py`: `apply` (46), `stacked_panels` (76), `finish` (87).
- `manifest.build_manifest` (25): family/note/sweep_var/sweep_level + interpretation key.

---

## 7. TUNABLES / HYPERPARAMETERS — all in `config.py` `ExpConfig` (lines 16-77)

Override any of them at run time via `--flag` (run_experiment) or `ENV=val`
(submit_experiment.sh); defaults live in the dataclass.

### FL / training (17-26)
| field | line | default | meaning |
|---|---|---|---|
| rounds | 21 | 50 | communication rounds |
| local_epochs | 22 | 5 | local passes/round |
| lr / momentum / weight_decay | 23/25/26 | 0.01 / 0.9 / 5e-4 | SGD |
| batch_size | 24 | 16 | local batch |
| base_seed | 27 | 1000 | seed = base_seed + repeat |

### Free-rider selection (31-38)
| field | line | default | meaning |
|---|---|---|---|
| attack | 31 | none | none / previous_models / gaussian / autopilot |
| num_free_riders | 32 | 0 | how many FRs |
| **free_rider_ids** | 33 | "" | pin cids, e.g. "3,6" (overrides seeded choice) |
| noise_sigma / noise_decay | 35/36 | 0.1 / 0.0 | gaussian baseline |
| partition | 37 | iid | iid / dirichlet |
| dirichlet_alpha | 38 | 0.5 | non-IID skew (small=severe) |

### Autopilot (46-62)
| field | line | default | meaning |
|---|---|---|---|
| autop_oracle_eta | 46 | 0.0 | >0 => FR is GIVEN eta (testing). 0 => estimate. |
| autop_honest_until | 47 | 12 | behave honestly until convergence or this round (forced-honest window) |
| autop_conv_eps | 49 | 0.02 | convergence = BER improves < eps for 2 rounds |
| autop_honest_extra | 50 | 3 | stay honest N rounds after convergence |
| autop_eta_k | 51 | 3.0 | k in the frozen estimate mu+k*sigma |
| autop_protect_until | 52 | 8 | never defect before this round |
| autop_warmup_cap | 53 | 15 | hard warmup cap |
| autop_max_batches | 54 | 250 | budget for the non-honest warmup transition |
| autop_margin0 | (55) | 0.06 | target BER = eta - margin |
| autop_floor | 56 | 0.05 | "mark is good" bar |
| **autop_common_per_class** | 57 | -1 | DATA/tap: -1=full shard, 0=triggers-only, N=+N/common-class |
| **autop_scope** | 58 | full | PARAMS/tap: full / block2 / block / head |
| autop_stay_min | 59 | False | coast when safe, tap only when needed (else tap every round) |
| autop_holdout_ratio | 61 | 0.5 | probe holdout fraction |
| autop_honest_clone | 62 | False | DIAGNOSTIC: pure honest every round (the floor control) |

### Watermark (66-77)
| field | line | default | meaning |
|---|---|---|---|
| watermark | 66 | False | enable the scheme |
| wm_bits (m) | 67 | 0=auto | number of bits |
| wm_lambda | 68 | 5.0 | embed-loss weight (Eq. 11) |
| wm_alpha | 69 | 0.4 | smoothing exponent (Eq. 8) |
| wm_beta | 71 | 0.6 | memory coefficient (Eq. 14) |
| wm_num_triggers (N_T) | 73 | 50 | extraction trigger count (Eq. 15) |
| wm_eta | 74 | 0.25 | eta floor / cap |
| paper_faithful | 76 | True | random keys, no exclusion, cumulative mu+3sigma |
| **calib_on_all** | 77 | False | calibrate eta over ALL clients (poisoning) vs benign-only |

Sweep grids live in the runners: `CPC_HOPS="0 5 10 20 50 -1"`, `POS_A/POS_B`,
`SEEDS`, `ORACLE=0.09` in `run_tests.sh`.

---

## 8. EXPERIMENTS -> where they live

| experiment | driver | config | key flags |
|---|---|---|---|
| Test 1: all-honest FPR | run_tests.sh (TEST 1) | 14 | `ATTACK=none` |
| Test 2: full-scope data sweep x2 positions | run_tests.sh (TEST 2) | 14 | `AUTOP_SCOPE=full AUTOP_COMMON_PER_CLASS in CPC_HOPS FREE_RIDER_IDS=POS_{A,B}` |
| Test 3: block2 data sweep x2 positions | run_tests.sh (TEST 3) | 14 | `AUTOP_SCOPE=block2 ...` |
| floor control (honest-clone) | any | 14 | `AUTOP_HONEST_CLONE=1` |
| eta-poisoning | any | 14 | `CALIB_ON_ALL=1 NUM_FREE_RIDERS=5` |
| non-IID | any | 14 | `PARTITION=dirichlet DIRICHLET_ALPHA=...` |
| reproduction (Table I / Fig 7) | submit_sweep.sh | 1-13 | see config names |
| robustness (Figs 9-10) | run_robustness.py | 11 | finetune/prune/quantize |

Plot: `RES=/path ./run_tests.sh PLOT` -> figs/test1_fpr.png, test2_full_pos{A,B}.png,
test3_block2_pos{A,B}.png.
