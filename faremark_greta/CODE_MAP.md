# CODE_MAP — complete technical reference

```
faremark/
  client.py           honest FedAvg client (base class)                 
  server.py           FedAvg aggregation + round loop + verify hook      
  datasets.py         IID / Dirichlet shards + trigger test set          
  models.py           build_model (resnet18/alexnet/smallcnn)            
  robustness.py       finetune/prune/quantize ops                        
  manifest.py         self-describing run metadata for result.json       
  utils.py            set_seed, get_logger, evaluate_accuracy            
  watermark.py        the watermark math (Eq. 1-16): key/bits/embed/extract/BER   §1
  wm_client.py        WatermarkClient (honest embed + Eq.14 memory) + factory      §2
  attacks.py          crude baselines (previous_models, gaussian) + FR selection   §3a
  attacks_adaptive.py SUBMARINE adaptive free-rider (the attack under study)        §3
  wm_verify.py        server: extract mark -> BER -> FROZEN eta -> flag + diagnostics §4
  compute_meter.py    per-client effort (samples, gpu_ms, flops, duty cycle)       §6
  plotstyle.py        shared matplotlib style (colour-blind palette, panels)
scripts/
  run_experiment.py   orchestrates one (config, repeat); writes result.json        §6
  threshold.py        ALL threshold code: canonical eta + calibration CLI          §5
  plots.py            ALL plotting: derivation, class dynamics, timeline, ...       §7
  run_all.sh          honest -> calibrate -> attacks -> PLOTALL                     §9
  submit_experiment.sh  one RunAI job (env -> CLI flags)
```

---

## 0. Threshold 

**The detection threshold eta is a PRE-CALIBRATED CONSTANT.** It is computed
**once**, on honest-only multi-seed runs (`threshold.py calibrate`), frozen
to `eta_calibrated.json`, and passed into every experiment as `WM_ETA_FIXED`. 

Threshold definition:
```
per seed s (one run, 10 honest clients, last `tail`=20 rounds):
   m_r   = mean BER over the 10 clients IN round r     (mean over clients)
   mu_s  = mean_r(m_r)                                 (mean over the 20 rounds)
   sigma_s = std_r(m_r)
   eta_s = mu_s + 3*sigma_s
final:  eta = AVERAGE of eta_s over the seeds        (+ eta_std_across_seeds reported)
```

---

## 1. Watermark — `faremark/watermark.py` 

Measurement: **BER** (bit-error-rate) of a client's m-bit mark, extracted from its trigger-class softmax

| Symbol / step | Function | Line | Paper |
|---|---|---|---|
| smoothing `f(p)=p^alpha` | `smooth` | 48 | Eq. 7-9 |
| secret +/-1 key M [m,l] | `make_key` | 67 | §IV-A |
| same-sign (stuck) key rows | `unembeddable_fraction` | 91 | floor diagnostic |
| target bits B in `{0,1}^m` | `make_bits` | 106 | Eq. 2 |
| group size `l = n//m` | `grouping` | 120 | §IV-A |
| project probs -> per-bit z | `project_logits` | 132 | Eq. 1/13 |
| embed loss `BCE(z, B)` | `watermark_loss` | 157 | Eq. 11-12 |
| extract: `mean over N_T`, sign | `extract_bits` | 170 | Eq. 15 |
| `BER = mean(bits != B)` | `bit_error_rate` | 178 | Eq. 16 |
| flag test `BER < eta` | `detected` | 183 | Eq. 16 |
| `mu + 3*sigma` helper | `calibrate_eta` | 188 | §IV-D-3 |
| anti-dominance ratio | `dominance_ratio` | 202 | Eq. 6/10 |

---

## 2. Honest client — `faremark/wm_client.py` 

- `WatermarkClient(Client)` (19): honest client that also embeds.
  - `produce_update` (42): load global -> `_local_train_wm(round)` -> `_memory_update` -> submit.
  - `_local_train_wm(round_idx)` (52): `L = CE + wm_lambda * watermark_loss` on trigger
    images; meters each batch. logs per-round `self.wm_stats[round]` =
    {`cls_loss`, `wm_loss`, `total_loss`, `trig_train_acc`, `trigger_class`} — the
    client-side evidence for "which classes are hard to embed / have fuzzy boundaries".
  - `_memory_update` (105): Eq. 14 blend `W = beta*(memory+delta) + (1-beta)*global`; persists `self.memory`.
- `build_watermarked_clients` (123): `trigger_class = cid % num_classes`,
  key & bits seeded by cid (the client's "position"), registers them, and dispatches
  free-rider slots to the autopilot or a baseline. `paper_faithful` chooses m/l/exclude
  (random keys, full softmax) vs our balanced-key/excluded-column variant. Honours
  `cfg.free_rider_ids`. Passes ALL `autop_*` knobs (incl. the new
  `autop_eta_mode`/`autop_num_clients_est`/`autop_safety`/`autop_max_coast`) into the attack.

---

## 3. The attack: SUBMARINE — `faremark/attacks_adaptive.py`

`SubmarineFreeRider(_AdaptiveMixin, WatermarkClient)` — an honest client with a
different control flow. Reuses key/bits/lambda/alpha/beta/memory/`_local_train_wm`.

**`_AdaptiveMixin` (53):**
- `_ensure_triggers` (59): gather this shard's trigger images once; hold out a probe
  slice (n_probe=64 for a steadier self-BER estimate); build the reduced
  (data-ablation) tap loader from `autop_common_per_class`.
- `_probe_ber_current_model` (105) / `_probe_ber_state` (118): the FR's private BER
  self-probe (its own held-out trigger images — not the server's test bank).
- `_embed_loop` (126): the tap training loop. `scope` picks params
  (full/block2/block/head -> freezes backbone tensors); loader = reduced shard
  (cpc>=0) else full shard; `early_stop` gates the probe. accumulates and logs
  per-tap `cls_loss`/`wm_loss`/`trig_train_acc` via `_log_tap_stats` (206)

`make_submarine_attack` (222) -> `AutopilotFreeRider` (238):
- `__init__` (242): all `autop_*` knobs (§8), incl.
  `autop_eta_mode`, `autop_num_clients_est`, `autop_safety`, `autop_max_coast`.
- `_converged` (315): FR's own-BER convergence test (dynamic warmup).
- `_eta_target` (325): ORACLE (if `autop_oracle_eta>0`) -> the FR's FROZEN estimate.
- `_freeze_own_eta` (330): reconstruct the SERVER'S threshold from the FR's own honest
  BER stream (valid in IID). `autop_eta_mode` selects which:
  - `"tight"` (default) = `mu + k*sigma/sqrt(N)` — the round-mean (strict) eta; 
  - `"loose"` = `mu + k*sigma` (per-client)
- `_coast_state` (366): fresh global + frozen mark-direction (re-inject the mark ~free).
- `_update_mark_delta` (374): `memory - global` = the mark direction.
- `produce_update` (381):
  1. `autop_honest_clone` -> pure honest every round (floor control).
  2. WARMUP -> CALIB (state machine, `autop_warmup_mode` fixed|dynamic): train honestly,
     probe own BER; at convergence (dynamic) or round W (fixed) enter the K-round
     calibration window, freeze eta, defect.
  3. FREE-RIDE: `target = max(floor, eta - margin0 - safety)`.
     - **COAST** iff `autop_stay_min` AND predicted coast BER `<= target` AND the coast
       streak `< autop_max_coast`.
     - else **TAP** = `_embed_loop` (cost = data x scope); trace records `tap_reason`.

Per-round decisions are recorded in `self.trace` (action, eta, target, tap_reason,
coast_streak, ber_after) for the timeline plot.

### 3a. Baselines — `faremark/attacks.py`
`PreviousModelsFreeRider` (Eq. 17), `GaussianNoiseFreeRider` (Eq. 18),
`resolve_free_riders` (honours `free_rider_ids`), `build_clients` (non-wm path).

---

## 4. The detector — `faremark/wm_verify.py` 

- `WatermarkRegistry` (20): cid -> (trigger_class, key, bits, kind, alpha, exclude).
- `build_trigger_bank` (44): N_T test-set images per trigger class (held-out).
- `make_verifier` (63) -> `verify_hook` (86), per round:
  1. Extract every client's mark on the trigger bank -> one BER per client, and the client
     diagnostics `pmax`, `entropy`, `dominance` (Eq.6/10), `trig_acc`.
  2. THRESHOLD = the frozen constant `eta_fixed` (from `WM_ETA_FIXED`)
  3. Flag each client individually iff `ber >= eta_round`.
  4. Emit `wm_benign_ber`, `wm_fr_ber`, `wm_eta_round`, `wm_fpr`, `wm_fr_recall`,
     `wm_benign_ber_list`, `wm_fr_ber_list`, `wm_per_client`
     = `[{cid, trigger_class, ber, is_free_rider, flagged, pmax, entropy, dominance, trig_acc}]`,
     and round-level means `wm_benign_pmax/entropy/dominance/trig_acc`.

---

## 5. threshold code — `scripts/threshold.py` 

| purpose | function | line |
|---|---|---|
| mu+3sigma helper | `mu3s` | 24 |
| load result.json globs | `load` | 37 |
| honest-only run test | `is_honest_run` | 52 |
| calibration window [lo,hi] (dynamic tags / config) | `calib_window` | 74 |
| first free-riding round W | `freeride_start` | 84 |
| m_r = mean-over-clients per round (converged tail) | `round_means` | 97 |
| (eta, mu, sigma) from round-means | `eta_from_round_means` | 113 |
| canonical eta recomputed from runs (plots) | `frozen_eta` | 123 |
| read the frozen constant | `load_fixed` | 134 |
| find eta_calibrated.json near a dir | `find_fixed` | 142 |
| calibrate once, write eta_calibrated.json | `calibrate` | 153 |
| CLI (`python threshold.py calibrate ...`) | `_cli` | 193 |
| **double-check: recompute eta + confirm frozen use** | `verify` | - |

`calibrate(inp, honest_family, tail, out)` pools per-round means across seeds, applies
mu+3sigma once, and writes `{eta, grand_mean, grand_std, window, n_seeds, per_seed,
eta_all_rounds_for_reference}`. Default window = converged tail (last 20); `tail=0` =
all rounds (prints the warmup-inflated value for reference).

---

## 6. Orchestration + effort — `scripts/run_experiment.py`, `compute_meter.py` 

- `parse_args` (34): CLI overrides
- `_OVERRIDABLE` (108): applies the flags onto cfg 
- `collect_compute` (124): per-client + summarized effort
- `main`: build data -> model -> clients -> Server.run -> assemble result.json; passes
  `cfg.wm_eta_fixed` into `make_verifier(eta_fixed=...)`.
- `evaluate_per_class(model, loader, num_classes, device)`: per-class TEST acc + CE loss
  of the FINAL global model -> `result["per_class"]` (self-checks vs final_acc)
- `compute_meter.ComputeMeter` (43): `record_batch` (82, training), `record_forward_only`
  (89, probe = fwd only, NOT counted as training -> effort-inflation fix),
  `end_round` (95, gpu_ms via CUDA events), `summary` (117).

---

## 7. ALL plotting — `scripts/plots.py` 

Subcommand CLI (`python plots.py <cmd> --in '<glob>' [--family F] [--out DIR|PREFIX]`):

| cmd | shows | out style |
|---|---|---|
| `thresholds` | intuitive derivation of the ONE eta + where it lands (FPR/recall) | dir |
| `class_dynamics` | per-class L_wm / trig acc / BER-vs-confidence / loss curves | dir |
| `sanity` | TEXT report: flag flat/zero BER, non-frozen eta, missing loss | - |
| `class_difficulty` | confirm harder class ids: per-class acc/loss vs BER (+ Pearson r) | dir |
| `positions` | per-trigger-class-id BER (easy vs hard) + BER-vs-p_max panel | dir |
| `fidelity` | global accuracy + per-client BER (honest vs FR) + effort | dir |
| `timeline` | BER over rounds, taps/coasts, eta lines, calib window | prefix (+.png) |
| `honest_fpr` | honest false-positive rate vs the frozen eta | prefix |
| `threshold` | (legacy) two-distribution soundness view | dir |
| `frontier`/`scorecard`/`test_data` | legacy sweep plots (kept for reuse) | prefix/dir |

`all` runs the headline set (thresholds, class_dynamics, positions, fidelity).

---

## 8. TUNABLES — `faremark/config.py` `ExpConfig`

Override via `--flag` (run_experiment) or `ENV=val` (submit_experiment.sh).

### FL / training
| field | line | default |
|---|---|---|
| rounds / local_epochs / batch_size | 16/17/19 | 50 / 5 / 16 |

### Free-rider selection
| field | line | default | meaning |
|---|---|---|---|
| attack | 26 | none | none / previous_models / gaussian / autopilot |
| num_free_riders | 27 | 0 | how many FRs |
| free_rider_ids | 28 | "" | pin cids, e.g. "3,6" |
| partition / dirichlet_alpha | 32/33 | iid / 0.5 | non-IID skew |

### Submarine
| field | line | default | meaning |
|---|---|---|---|
| autop_oracle_eta | 41 | 0.0 | >0 => FR GIVEN eta (testing) |
| autop_warmup_mode | 43 | fixed | fixed = warmup ends at W; dynamic = at own-BER convergence |
| autop_honest_min / warmup_cap / conv_eps / conv_patience | 48/49/50/53 | 6/15/0.03/2 | dynamic-warmup controls |
| autop_honest_until (W) / autop_calib_rounds (K) | 54/55 | 12 / 4 | fixed window `[W-K,W-1]` |
| autop_eta_k | 58 | 3.0 | k in the FR's mu+k*sigma estimate |
| autop_eta_mode | 59 | tight | which server eta the FR reconstructs: tight/loose/cumulative |
| autop_num_clients_est | 64 | 10 | N for the sqrt(N) shrink in tight mode |
| autop_margin0 | 65 | 0.06 | deliberate headroom below eta |
| autop_safety | 66 | 0.02 | guard for probe(own)/server(test) mismatch |
| autop_max_coast | 67 | 4 | force a re-tap after this many coasts |
| autop_floor | 68 | 0.05 | "mark is good" bar |
| autop_common_per_class | 69 | -1 | DATA/tap: -1=full, 0=triggers-only, N=+N/common |
| autop_scope | 70 | full | PARAMS/tap: full/block2/block/head |
| autop_stay_min | 71 | False | coast-when-safe (else tap every round) |
| autop_holdout_ratio | 73 | 0.5 | probe holdout fraction |
| autop_honest_clone | 74 | False | DIAGNOSTIC: pure honest every round |

### Watermark
| field | line | default | meaning |
|---|---|---|---|
| watermark / wm_bits (m) | 78/79 | False / 0=auto | enable / bit count |
| wm_lambda / wm_alpha / wm_beta | 80/81/83 | 5.0 / 0.4 / 0.6 | embed weight / smoothing / memory |
| wm_num_triggers (N_T) | 85 | 50 | extraction trigger count |
| wm_eta_floor | 86 | 0.05 | degenerate guard only |
| wm_eta_fixed | 89 | 0.0 | >0 => use this PRE-CALIBRATED constant threshold |
| calib_on_all | 95 | False | (circularity demo) calibrate over all clients |

---

## 9. EXPERIMENTS — `scripts/run_all.sh`

One threshold, three focused experiments.

```
./run_all.sh honest      # all-honest, multi-seed (calibration + baseline)
./run_all.sh calibrate   # -> $RES/eta_calibrated.json  (threshold.py calibrate)
./run_all.sh attacks     # tap_every (+5/common, full) + tap_stay (coast-to-stay)
./run_all.sh PLOTALL      # timeline + class_dynamics + positions + thresholds + fidelity + honest_fpr
```
Families: `honest_iid`, `tap_every_iid`, `tap_stay_iid`. Attack runs read the frozen
eta via `read_eta` and pass it as `WM_ETA_FIXED`. `POS` (default `3,6`) pins the FR
positions (hard classes); `SEEDS` controls seed count.