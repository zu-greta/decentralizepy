# FareMark free-rider study — project doc 

## 1. What this is

Re-implementation of **FareMark** (Li et al., IEEE IoT-J 2025) — a box-free,
output-space watermark free-rider detector for FL — plus an adaptive free-rider
("autopilot submarine") built to find where the detector breaks.

Setup: CIFAR-100 · ResNet-18 · 10 clients · 2 free-riders · 50 rounds · 5 local
epochs · batch 16 · paper-faithful (random keys, full softmax, cumulative μ+3σ).

Detector: each round the server extracts every client's watermark on `N_T=50`
held-out trigger images → per-client BER; flags client `i` iff `BER_i ≥ η`, with
`η = μ + 3σ` calibrated on honest clients.

## 2. The finding (one paragraph)

Every client embeds its **own** watermark, set by its slot `cid`
(`trigger_class = cid % num_classes`; key & bits seeded from `cid`). This is the
client's **position**, and each position has a **BER floor** — the lowest BER that
embedding problem allows, reached with modest effort and immovable after (like
Bayes error for the embedding sub-task). Easy positions floor at ~0.00; hard
positions at ~0.10–0.20. A full-effort free-rider **is** an honest client
(`honest_clone` proves the code path is identical) and lands on its position's
floor. **η is calibrated on the round-AVERAGE honest BER (tight, ~0.09) but
applied per individual client**, so any client — honest or free-rider — on a hard
position is flagged. Result: the free-rider "fails to evade" and honest clients are
false-positived at ~30% of client-rounds *by the same mechanism*. FareMark's
security therefore reduces to η-calibration, which faces a dilemma: tight η catches
the free-rider but false-positives honest hard positions; loose η spares honest
clients but lets any embedding free-rider pass. Under plain IID with a fair η the
free-rider cannot get under; the exploitable weaknesses are all on the **threshold/
pool** side (see §6).

## 3. Concepts in plain terms

- **Floor** = irreducible best-case BER for a client's assigned watermark task.
  Effort/data/scope don't move it (proven by `honest_clone` and the data ablation).
- **Position (`cid`)** = which (trigger class, key, target bits) the server assigned.
  Determines the floor. Server-assigned; the attacker cannot choose it.
- **Threshold η** = `μ+3σ`. In code it's calibrated on the per-round *mean* benign
  BER (`wm_verify.py` L114) then applied to *individual* clients (L131). That
  average-vs-individual mismatch is the whole story.

## 4. Clean architecture — files to KEEP

Core (unchanged, correct):
```
client.py          honest FedAvg client (base)
server.py          FedAvg + per-round verify hook
wm_client.py       WatermarkClient (honest embed + Eq.14 memory) + client factory
watermark.py       key/bits/project/embed/extract/BER/calibrate_eta (Eq.1–16)
wm_verify.py       server extraction + η calibration + per-client BER records
compute_meter.py   per-client effort (samples, gpu_ms, flops, duty cycle)
thresholds.py      η variants for offline plots (frozen/converged/…)
datasets.py        IID / Dirichlet shards           [NOT UPLOADED — verify]
utils.py           evaluate_accuracy                [NOT UPLOADED — verify]
config.py          experiment configs (trim, see §5)
run_experiment.py  orchestration + result.json      (trim unused args, see §5)
plotstyle.py       shared plot style
submit_experiment.sh  cluster submit bridge
```
Attack (keep only the autopilot):
```
attacks_adaptive.py   KEEP _AdaptiveMixin + make_autopilot_attack; DELETE the rest (§5)
```
New (this cleanup):
```
run_tests.sh       3-test suite (replaces run_full_sweep.sh)
plot_tests.py      plots for the 3 tests (replaces most of plot_thresholds.py)
PROJECT.md         this file
```

## 5. Cleanup — what to DELETE / TRIM

**`attacks_adaptive.py`** — keep the autopilot, remove the dead submarine + legacy paths:
- Delete `make_submarine_attack(...)` (the entire `SubmarineFreeRider` factory).
- Delete `_blend_states(...)` and `from .attacks import _extrapolate` (submarine-only).
- In `AutopilotFreeRider.produce_update`, delete the legacy **DYNAMIC predictive-tap
  block** (the `coast with predictive, adaptive taps` section after `if stay:` —
  roughly the `_predict_cross`/`must_tap` tail). The tests always run under
  oracle/stay-under, so that branch is dead. Keep: honest phase → freeze η, the
  stay-under fixed-tap, `stay_min` coast-when-safe, `honest_clone` (useful control).
- Optionally drop `_predict_cross` and the `_ber_trend` bookkeeping once the dynamic
  block is gone.

**`attacks.py`** (NOT UPLOADED) — keep only `choose_free_riders` (used by the client
factory) and, if you keep the paper's crude baselines for a comparison figure,
`GaussianNoiseFreeRider` + `previous_models`. Delete `make_trigger_only`,
`make_mixed_attack`, `make_random_round_attack`, `make_train_then_attack`,
`make_submarine_attack` wiring, `_extrapolate`.

**`wm_client.py`** — in `build_watermarked_clients`, delete the `elif attack ==`
branches for `train_then_attack`, `trigger_only`, `random_round`, `mixed`,
`submarine`, `memory_exploit`, `reembed`. Keep `none`, `autopilot`, and (optional)
`previous_models`/`gaussian` for the baseline. **Add** the `free_rider_ids` override
(see §7) so positions are deterministic.

**`config.py`** — delete configs idx 14 (submarine), 15 (memexploit), 16 (reembed).
Keep idx 17 (autopilot) as the single attack config. Remove now-unused fields:
`sub_*`, `reembed_*`, `warmup_rounds`, `mem_blend_global`, `sub_coast_mode`,
`blend`, `full_trigger_class`, `honest_prob`, `attack_round`, `n_trigger_samples`,
`n_common_samples`, `autop_stay_under`(auto-on)/`autop_honest_clone`/`autop_stay_min`
only if you also drop those code paths. Keep: `partition`, `dirichlet_alpha`,
`autop_oracle_eta`, `autop_common_per_class`, `autop_scope`, `autop_honest_until`,
`autop_honest_extra`, `autop_conv_eps`, `autop_eta_k`, `autop_protect_until`,
`autop_margin0`, `calib_on_all`, `paper_faithful`, `wm_*`.

**`run_experiment.py` / `submit_experiment.sh`** — delete arg forwarding for the
removed attacks/knobs; add `--free_rider_ids` (§7).

**`plot_thresholds.py`, `seedband.py`** — superseded by `plot_tests.py` for the 3
tests. Keep `plot_thresholds.py` only if you still want the η-variant timeline
figure; otherwise archive. `plot_adaptive.py` is referenced but NOT uploaded — if it
exists, archive it.

**Delete/archive runners:** `run_full_sweep.sh` (replaced by `run_tests.sh`).

## 6. What the autopilot already does (maps to your 4-feature spec)

The existing `AutopilotFreeRider` already implements exactly what you described — no
rewrite needed, just the trim above:

| Your requirement | Where it lives |
|---|---|
| Uses the same modules as honest for the whole run | subclass of `WatermarkClient`; reuses `_local_train_wm`, `_memory_update`, key/bits/λ/α/β/exclude verbatim |
| Estimate the threshold (oracle optional for testing) | `_eta_est()`: `autop_oracle_eta>0` → given η; else μ+kσ over converged forced-honest rounds, frozen once |
| Know when the server stops forcing honesty | forced-honest phase (`autop_honest_until`) + convergence detector (`autop_conv_eps`) → freezes η, sets `_warm_done`, then defects |
| Train trigger + N common OR full shard | `autop_common_per_class`: `0`=triggers-only, `N`=+N/common-class, `-1`=full shard (`_reduced_loader`) |
| Tap when under threshold; tap strength = #samples | post-warmup: coast when probe safely under `η−margin` (`autop_stay_min`), else re-embed; strength = data (`autop_common_per_class`) × scope (`autop_scope`) |

## 7. Needed one-line addition — pin free-rider positions

Positions currently come from `choose_free_riders(n, k, seed)` (varies by seed),
which confounds every result with position luck. Add a deterministic override:

```python
# run_experiment.py (argparse):
p.add_argument("--free_rider_ids", type=str, default=None)   # e.g. "3,6"
# config.py (ExpConfig):
free_rider_ids: str = ""
# wm_client.build_watermarked_clients, right after choosing fr_idx:
if getattr(cfg, "free_rider_ids", ""):
    fr_idx = set(int(i) for i in cfg.free_rider_ids.split(","))
# submit_experiment.sh (with the other overrides):
[ -n "${FREE_RIDER_IDS:-}" ] && PY_EXTRA="$PY_EXTRA --free_rider_ids ${FREE_RIDER_IDS}"
```

## 8. The three tests (see `run_tests.sh` + `plot_tests.py`)

All tests: **3 seeds**, std shown (error band / points). Converged window = last 20 rounds.

- **TEST 1 — honest false-positive check.** 10 honest clients, no free-rider. Plot
  each honest client's BER (by trigger class) + overall mean, against **two** η lines:
  μ+3σ over round-means (as coded) and μ+3σ over per-client BERs (alternative). Report
  FPR under each. *Shows whether every honest client can sit under η, and exposes the
  calibration dilemma directly.*
- **TEST 2 — full-scope data sweep, ×2 positions.** 2 free-riders, `scope=full`,
  `autop_common_per_class ∈ {0,5,10,20,50,-1}`. Run at two position sets (`POS_A`,
  `POS_B`) to separate mechanism from position luck. Plot **each** free-rider's BER +
  FR mean and **each** honest BER + honest mean (distinguishable), plus effort in
  **GPU-ms and samples**, with the honest baseline on the same axes.
- **TEST 3 — same as TEST 2 but `scope=block2`.** Backbone frozen. Expect samples ≈
  unchanged but GPU-ms lower (backbone backprop skipped) — the scope-vs-data cost
  distinction becomes visible by comparing TEST 3 to TEST 2.

Run: `./run_tests.sh` then `RES=/path ./run_tests.sh PLOT`.

## 9. Status & next steps

Done: reproduction (Stages 1–4); adaptive autopilot; effort meter fix (~1.0×);
`honest_clone`, `all_honest`, data-ablation, non-IID, poisoning controls.

Open / next:
1. Run the 3 cleaned tests (positions pinned) → the false-positive dilemma is the paper.
2. **η-poisoning**, clean: log each FR's position; separate "η rose" from "FRs sat on
   easy classes." (`calib_on_all`, sweep FR count.)
3. **Collusion via a shared trigger class** — untested; stresses the paper's capacity
   mechanism (Table IX). Highest-value unexplored attack.
4. Non-IID: report under BOTH frozen and converged η (they disagree); only severe
   α=0.1 gives genuine cover, and there the honest watermark fails too. Don't headline α=0.5.

## 10. Missing files to upload (referenced but not provided)

Needed to finish/verify the cleanup: `attacks.py`, `datasets.py`, `utils.py`,
`inspect_results.py`, `plot_adaptive.py` (if it exists), and `watermark.py` is
present. Send `attacks.py` especially — the client factory imports `choose_free_riders`
and the attack registry from it, so the delete-list in §5 needs its exact contents.