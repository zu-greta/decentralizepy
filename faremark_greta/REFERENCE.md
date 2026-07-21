# FareMark limitations study — reference

Detection rule everywhere: **flag client i as a free-rider iff BER_i ≥ η** (`wm_verify`
uses `detected = ber < eta`). "Looser" = higher η = fewer flags = lower false-positive
rate but lower free-rider recall. "Tighter/stricter" = lower η = more flags.

---

## 1. Thresholds

### 1a. The live threshold η (the one that actually flags)

Frozen once by `threshold.py calibrate` on honest-only runs, reused every round via
`WM_ETA_FIXED`:

1. per round *r*, per seed: `m_r = mean over honest clients of BER` (`round_means`)
2. keep last `tail=20` rounds (converged)
3. per seed *s*: `μ_s = mean_r(m_r)`, `σ_s = std_r(m_r)` (population std), `η_s = μ_s + 3·σ_s`
4. `η = mean_s(η_s)`

Because `m_r` is a mean over *N* clients, its spread `σ_s ≈ σ_perclient/√N`. So the live
η is built on a **shrunk** variance — the tightest reasonable threshold — which is exactly
why honest *per-client* BERs poke above it (false positives, finding F1).

Also reported by `calibrate` (not used to flag): `eta_pooled` (pool all seeds' round-means,
then μ+3σ once) and `eta_all_rounds` (tail=0, warmup-inflated → larger). Off the live path:
`watermark.calibrate_eta` (μ+3σ over a flat benign list, floored at 0.05, cumulative) and the
commented-out cumulative / sliding-15-round variants in `wm_verify`. `eta_floor=0.05` is only
a degenerate guard.

### 1b. The regime (`separability.py`, post-hoc, honest converged-tail BER)

| # | rule | computed over | formula | tight ↔ loose |
|---|------|---------------|---------|----------------|
| 1 | **coded (μ+3σ round-mean)** | round-means | `mean_r(m_r) + 3·std_r(m_r)` (= live η) | **tightest** (variance ÷√N) → most flags, highest FPR |
| 2 | **loose (μ+3σ per-client)** | every per-client BER | `mean(H) + 3·std(H)` | ~√N **looser** than #1 → fewest flags |
| 3 | **median + 3·MAD** | per-client | `median + 3·1.4826·median(|x−median|)` | robust; near the bulk, ignores tails |
| 4 | **trimmed-10 μ+3σ** | middle 80% | drop 10% each tail, then `mean+3σ` | robust; between coded and loose |
| 5 | **adaptive-clip (iter μ+3σ)** | per-client | iterative σ-clip: drop `x > μ+3σ`, recompute, repeat to fixpoint; η = μ+3σ of survivors | tightens onto the bulk; **clipped honest → guaranteed FPs** |
| 6 | **honest p95 / p99** | per-client | 95th / 99th percentile of H | empirical "worst 5%/1% honest is the line"; p99 > p95 |
| 7 | **equal-error-rate** | H and F | η where FPR = FNR | data-driven balance point |
| 8 | **Youden-optimal (best)** | H and F | `argmin_η (FPR+FNR)/2` | the single best scalar η that exists |

Structural order (typical): coded < adaptive-clip ≈ trimmed ≈ median < loose < p99;
EER and Youden land wherever the two clouds cross.

### 1c. Rule-independent bounds (the headline numbers)

| metric | formula | meaning |
|---|---|---|
| **overlap coefficient (OVL)** | `Σ_bins min(density_H, density_F)` | 1.0 = honest & FR BER clouds identical |
| **best-possible balanced error** | `min over all η of (FPR+FNR)/2` | 0 = some η separates perfectly; ~0.5 = no η beats a coin |

### 1d. Adaptive clipping — what it does

The "clip-and-adapt during calibration" idea (`threshold.adaptive_clip_eta`): start from all
honest BER, drop everything above μ+3σ, recompute μ/σ on survivors, repeat until the inlier set
stops changing. Each pass discards the hard-class upper tail, so η converges onto the *bulk* of
honest clients. Example on a realistic bimodal honest sample (bulk ≈ 0, hard-class tail ≈ 0.11):
plain μ+3σ = 0.134 (keeps all); adaptive-clip = 0.021, kept = 0.90. The catch: the clipped 10%
now sit **above** η → they are guaranteed false positives. A tighter, better-behaved η on the
bulk buys itself a fixed set of honest FPs — that is the separability point, not a bug.

### 1e. Can these be computed AFTER the runs? — yes

Every honest-only rule (coded, loose, median+MAD, trimmed, p95/p99, adaptive-clip) needs only
**honest BER**, which is logged per client per round in `result.json`
(`history[*].wm_per_client[*].ber`, `is_free_rider=false`). EER and Youden additionally need the
free-rider BER, also logged. So the **entire regime is post-hoc**: to try a new threshold you
re-run `separability.py`, never the experiment. The only value that must be fixed *before* a run
is the single frozen η used for live flagging — but even those flags can be recomputed offline
for any η, since BER is stored.

---

## 2. Experiment matrix

`H` = honest seeds (default 6: `0 1 2 3 4 5`); `A` = attack seeds (default 3: `0 1 2`).
Unbalanced keys everywhere except the `balanced` leg. Run all with `./run_everything.sh`; run
one leg with `LEGS=<name> ./run_everything.sh`.

| leg → folder | dataset / partition / keys / bits / clients / smoothing | runs (family) | tests | look for |
|---|---|---|---|---|
| **iid** → `c100_iid` | CIFAR-100 / IID / unbal / m=10 / 10 / power | honest×H `honest_c100_bdef_iid`; reduced 1,7×A `…_c17`; reduced 3,6×A `…_c36`; sameclass 0→6×A `sameclass_…_c6` | easy-pos hides, hard-pos = floor, same-class inseparable | reduced 1,7: FR BER≈0 **below η**, ~30% effort. reduced 3,6: FR≈floor. sameclass: **OVL→1, best-err→~0.5** |
| **balanced** → `c100_iid_bal` | …/ **balanced** keys /… | honest×H; reduced 3,6×A; sameclass 0→6×A | does overlap survive removing the stuck-bit artifact (F6)? | compare honest spread & sameclass OVL vs `c100_iid` |
| **noniid** → `c100_niid` | CIFAR-100 / **Dirichlet(0.5)** /… | honest×H `…_niid`; reduced 3,6×A; sameclass×A | does label skew widen honest & worsen separability? | wider floor, larger η seed-std, OVL ≥ IID |
| **sin** → `c100_sin` | …/ **sin** smoothing (Eq.9) | honest×H `…_sin_iid`; reduced 3,6×A | does a different f() remove per-class floors? | floors shift, don't vanish |
| **bits20** → `c100_b20` | …/ **m=20** bits | honest×H `honest_c100_b20_iid`; reduced 1,7×A; reduced 3,6×A | does more capacity make BER separable? | finer BER, floor/overlap persists |
| **classes** → `c100_classes` | …/ trigger classes **9,19,…,99** (not 0–9) | honest×H (classes 9..99); reduced 3,6→classes 39,69×A | is class difficulty general, not specific to 0–9? | `class_probe`/`honest_lines`: spread of floors on new classes; per-class OVL on 39/69 |
| **capacity** → `c100_cap` | CIFAR-100 / **200 clients** | honest×H; reduced 106,107×A `…_c106107` | when clients MUST share classes, is same-class overlap systemic? | per-class OVL on 6,7 (honest 6/7 vs FR 106/107). watch data starvation |
| **capacity10** → `c10_cap` | **CIFAR-10 / 50 clients** | honest×H `honest_c10_bdef_iid`; reduced 16,17×A `…_c1617` | capacity without the thin-data confound | clean same-class overlap (5 clients/class, ~100 trigger imgs each) |

Each folder gets its own `eta_*.json` and `figs/<tag>/` (`honest_lines`, `timeline_*`,
`separability_*`, `class_difficulty`, `thresholds`, `fidelity`).

---

## 3. What the seed varies (and why)

`seed = base_seed + repeat`; one number re-rolls every random choice. Trigger class is **NOT**
among them (it's `cid % num_classes`, fixed).

| varied by seed | why it's random | variance impact |
|---|---|---|
| data partition (which images each client gets) | FL doesn't control who has what | moderate (easier/harder slice of a client's class) |
| batch shuffle order | standard SGD practice | small |
| model initialization | nets start from random weights | small–moderate |
| **secret key M** (`seed+1000·cid+1`) | keys must be unique & secret per client | **large** — a new random projection = a different question; unbalanced keys add the stuck-bit lottery |
| **target bits B** (`seed+1000·cid+1`) | messages must be unpredictable/unique | **large** — changes how hard the same class is to mark |
| trigger-image selection | random sample of the class | small |

Healthy variance to average over: partition / shuffle / init. Task-changing variance: key +
bits (and, avoidably, the unbalanced stuck-bit lottery — subtract it with the `balanced` leg).

---

## 4. Original task list — status audit

| area | task | status |
|---|---|---|
| housekeeping | check what seeds vary | ✅ done (§3; diagnose via `class_probe` + `wm_unembeddable_frac`) |
| housekeeping | fix experiment tagging/naming | ✅ done (self-identifying `RUN_TAG` from `FAMILY`) |
| housekeeping | cleanup logging in code & `result.json` | ❌ not touched |
| housekeeping | merge files (all plotting together) | ✅ done (merged into `plots.py`; delete the two probe scripts) |
| threshold | stress-test threshold calcs + prove non-separable | ✅ done (`separability.py` regime + OVL/best-error) |
| threshold | adaptive clipping in warmup rounds | ✅ done (`adaptive_clip_eta`) |
| threshold | median | ✅ done |
| threshold | trimmed mean | ✅ done |
| threshold | regime of thresholds | ✅ done |
| difficulty | try sin smoothing (paper Eq.9) | ✅ done (`WMF=sin`, formula verified) |
| detection | define consequence of crossing threshold | ❌ not touched |
| detection | how many warnings before flagging | ❌ not touched |
| detection | window of detection instead | ❌ not touched |
| experiments | show no threshold works | ✅ done (best-error metric) |
| experiments | rotate trigger class per round + average | ❌ not touched (differs from the static per-run spread in `classes`) |
| experiments | more clients than classes | ✅ done (`capacity`, `capacity10`) |
| experiments | different classes have different BER, high variance | ✅ done (`class_probe`, `honest_lines`, `classes` leg) |
| experiments | test all thresholds, all fail | ✅ done |
| experiments | same trigger class → same BER (FR vs honest) | ✅ done (`sameclass` leg) |
| experiments | FR spectrum: sweep +N/common, source classes, limits | ⚠️ partial (`AUTOP_COMMON_PER_CLASS` exists; no sweep wired) |
| experiments | non-IID | ✅ done (`noniid` leg) |
| theory | no threshold can work (noise/overlap) | ⚠️ empirically supported (OVL, best-error), not formalized |
| theory | not enough freedom in output logits | ❌ not formalized (dominance/entropy diagnostics gesture at it) |
| theory | class difficulty sacrifices honest for FR | ⚠️ supported (`sameclass`, `class_probe`), not formalized |
| theory | output-layer watermarking impossible | ⚠️ empirically supported, not proven |
| next | hint of a solution | ❌ not touched |
| next | show impossible | ⚠️ empirical only |

**Cleanly untouched, actionable next:** (1) `result.json` / logging cleanup; (2) the whole
**detection-policy** block — consequence of crossing, k-warnings-before-flag, detection window
(biggest gap; natural next edit in `wm_verify.py`); (3) per-round trigger-class rotation +
averaging; (4) FR-spectrum sweep over +N and source classes; (5) theory write-up + solution hint.
