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

### 1b. Provenance — where each rule comes from

| rule | origin | standard? |
|---|---|---|
| **μ+3σ** | **The FareMark paper itself**, §IV-D3: "η is set to the value of μ + 3σ, where μ is the mean error and σ is the standard deviation." Underneath it is the **3-sigma rule / Shewhart control limit** (statistical process control, 1920s). | Yes — textbook. Assumes roughly normal, unimodal data; covers 99.87% of a normal one-sided → ~0.13% nominal FPR |
| **median + k·MAD** | **Robust statistics** (Hampel). MAD = median absolute deviation; the **1.4826** factor makes MAD a consistent estimator of σ for normal data | Yes — the standard robust drop-in for μ+3σ. Breakdown point 50% (half the data can be garbage) |
| **trimmed mean + 3σ** | **Trimmed/Winsorized statistics** (Tukey) | Yes. Breakdown point = trim fraction (10% here) |
| **adaptive clipping** | **Iterative σ-clipping** — standard in astronomy source detection (`astropy.sigma_clipped_stats`); the "adapt the clip to a target quantile" idea also appears in DP-SGD adaptive clipping | Yes, as a robust-estimation technique |
| **percentiles p95 / p99** | **Non-parametric empirical quantile** | Yes. Makes *no* distributional assumption — directly sets nominal FPR (p95 ⇒ 5% FPR by construction on the calibration sample) |
| **equal-error-rate (EER)** | **Biometrics / verification** (speaker, face): the operating point where FAR = FRR | Yes — the conventional single-number operating point |
| **Youden-optimal** | **Youden's J** (1950), `J = sensitivity + specificity − 1`; maximizing J = minimizing balanced error | Yes — standard ROC operating-point selection |
| **overlap coefficient (OVL)** | **Weitzman's overlapping coefficient**; relates to total-variation distance and hence the **Bayes error** of any 1-D threshold classifier | Yes |

Rules 1–5 use **honest BER only** (they define "normal"). Rules 6–7 (EER, Youden) need
**both** honest and free-rider BER — they are *oracle* rules, not deployable, included
precisely to answer "what if you tuned η perfectly?"

### 1c. The regime (`separability.py`, post-hoc, honest converged-tail BER)

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

### 1d. Rule-independent bounds (the headline numbers)

| metric | formula | meaning |
|---|---|---|
| **overlap coefficient (OVL)** | `Σ_bins min(density_H, density_F)` | 1.0 = honest & FR BER clouds identical |
| **best-possible balanced error** | `min over all η of (FPR+FNR)/2` | 0 = some η separates perfectly; ~0.5 = no η beats a coin |

### 1e. Adaptive clipping — what it does

The "clip-and-adapt during calibration" idea (`threshold.adaptive_clip_eta`): start from all
honest BER, drop everything above μ+3σ, recompute μ/σ on survivors, repeat until the inlier set
stops changing. Each pass discards the hard-class upper tail, so η converges onto the *bulk* of
honest clients. Example on a realistic bimodal honest sample (bulk ≈ 0, hard-class tail ≈ 0.11):
plain μ+3σ = 0.134 (keeps all); adaptive-clip = 0.021, kept = 0.90. The catch: the clipped 10%
now sit **above** η → they are guaranteed false positives. A tighter, better-behaved η on the
bulk buys itself a fixed set of honest FPs — that is the separability point, not a bug.

### 1f. Exact inputs, timing, and why the numbers differ

**The one quantity everything is built from:** `BER_{c,r,s}` = bit-error-rate of client *c*
in round *r* of seed *s*, logged at `history[r].wm_per_client[c].ber`. Two reductions of it
are used, and the difference between them is the single biggest source of confusion:

| symbol | definition | spread | used by |
|---|---|---|---|
| `H` | the flat list of **per-client** BERs (all honest c, all r in tail, all s) | full σ | loose, median+MAD, trimmed, adaptive-clip, p95/p99, EER, Youden, OVL |
| `m_r` | **mean over clients** within one round: `mean_c BER_{c,r,s}` | ≈ σ/√N | the coded/live η |

With N = 10 clients, `std(m_r) ≈ std(H)/√10 ≈ std(H)/3.16`. So the coded rule adds
`3 × (σ/3.16) ≈ 0.95σ` above the mean, while the loose rule adds `3σ`. **That factor of ~3 is
why the coded η is the tightest rule in the table and why honest clients trip it** — it is
calibrated on an averaged quantity but applied to individual clients. This is a genuine
specification bug in the scheme, not a tuning choice.

**Window:** `tail=20` — the last 20 of 50 rounds, the converged region (the paper's Fig. 8
saturates ~round 30). `tail=0` uses all rounds and is warmup-inflated (BER starts near 0.5),
which is why `eta_all_rounds_for_reference` is always the largest number `calibrate` prints.

**Seed handling:** the live η is computed **per seed then averaged** (`mean_s(μ_s + 3σ_s)`),
*not* pooled. Pooling across seeds would fold seed-to-seed variation into σ and inflate η;
averaging per-seed etas keeps σ within-seed. `calibrate` prints the pooled value as
`eta_pooled_for_reference` so you can see the gap, and `eta_std_across_seeds` quantifies
how unstable the calibration itself is (your finding F2).

**σ convention:** `np.std` = **population** std (÷N), not sample std (÷N−1). At n=20 that
makes σ ~2.6% smaller than the sample convention — negligible, but it means the numbers
won't match a hand calculation done with `ddof=1`.

**When each is computed:**

| | computed | on what | frozen? |
|---|---|---|---|
| live η | **before** the attack runs, by `threshold.py calibrate` | honest-only runs | yes — passed as `WM_ETA_FIXED`, constant for every round of every downstream run |
| flags in `result.json` (`flagged`, `wm_fpr`, `wm_fr_recall`) | **during** each run, per round | that round's BER vs the frozen η | — |
| every regime rule + OVL + best-error | **after** everything, by `separability.py` | logged BER | no — recompute freely |

**Why they differ, in one line each:** *coded* averages first (√N-shrunk σ → tightest);
*loose* doesn't (→ ~3× wider); *median+MAD* and *trimmed* ignore the hard-class tail by
construction (→ land near the bulk); *adaptive-clip* iteratively removes that tail (→ tighter
still, and manufactures its own false positives); *p95/p99* fix the FPR instead of the σ
multiple (→ track the empirical tail regardless of shape, the right choice for the bimodal
honest distribution you actually have); *EER/Youden* peek at the free-riders (→ not
deployable, but they bound what any η could achieve).

### 1g. Can these be computed AFTER the runs? — yes

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
Unbalanced keys everywhere except the `balanced` leg. All runs land in run_all's flat
results dir, distinguished by a **unique family** per leg (submit writes to
`$MOUNT/home/zu/results/<RUN_TAG>`; nothing is mixed because each leg has its own family
and its own `eta_*.json`). Run it **staged**, fire-and-forget (nothing waits on the cluster):

```
./run_everything.sh honest     # submit every leg's honest jobs, return
# wait for the cluster (runai/kubectl), then:
./run_everything.sh attacks    # calibrate each leg's eta + submit its attacks, return
# wait, scp results to local, then:
RES=~/local/results ./run_everything.sh plot     # separability tables + figures
```

Run one leg with `LEGS=<name> ./run_everything.sh <phase>`.

| leg (eta file) | dataset / partition / keys / bits / clients / smoothing | families | tests | look for |
|---|---|---|---|---|
| **iid** `eta_c100_bdef` | CIFAR-100 / IID / unbal / m=10 / 10 / power | honest×H `honest_c100_bdef_iid`; reduced 1,7×A `reduced_c100_bdef_iid_c17`; reduced 3,6×A `…_c36`; sameclass 0→6×A `sameclass_c100_bdef_iid_c6` | easy hides, hard = floor, same-class inseparable | reduced 1,7: FR BER≈0 **below η**, ~30% effort. reduced 3,6: FR≈floor. sameclass: **OVL→1, best-err→~0.5** |
| **balanced** `eta_c100_bdef_bal` | …/ **balanced** keys (VTAG=bal) | honest×H `honest_c100_bdef_bal_iid`; reduced 3,6×A; sameclass 0→6×A | overlap survive removing stuck-bit artifact (F6)? | compare honest spread & sameclass OVL vs `iid` |
| **noniid** `eta_c100_bdef_niid` | CIFAR-100 / **Dirichlet(0.5)** | honest×H `honest_c100_bdef_niid`; reduced 3,6×A; sameclass×A | does skew widen honest & worsen separability? | wider floor, larger η seed-std, OVL ≥ IID |
| **sin** `eta_c100_bdef_sin` | …/ **sin** smoothing (Eq.9) | honest×H `honest_c100_bdef_sin_iid`; reduced 3,6×A | does a different f() remove floors? | floors shift, don't vanish |
| **bits20** `eta_c100_b20` | …/ **m=20** bits | honest×H `honest_c100_b20_iid`; reduced 1,7×A; reduced 3,6×A | more capacity → separable? | finer BER, floor/overlap persists |
| **classes** `eta_c100_bdef_spread` | …/ trigger classes **9,19,…,99** (VTAG=spread) | honest×H (classes 9..99); reduced→classes 39,69×A | class difficulty general, not just 0–9? | `class_probe`/`honest_lines` spread of floors; per-class OVL on 39/69 |
| **capacity** `eta_c100_bdef_nc200` | CIFAR-100 / **200 clients** (VTAG=nc200) | honest×H `honest_c100_bdef_nc200_iid`; reduced 106,107×A | clients MUST share classes → systemic overlap? | per-class OVL on 6,7 (honest 6/7 vs FR 106/107). watch data starvation |
| **capacity_paper** `eta_c100_bdef_nc200_tmtrain` | CIFAR-100 / 200 clients / **paper §V-F3 trigger-sample consistency** (`TRIGMODE=client_train`) | honest×H `honest_c100_bdef_nc200_tmtrain_iid`; reduced 106,107×A | reproduce the paper's capacity protocol; memorisation vs generalisation | detection should look much BETTER than `capacity`; that gap is the memorisation artifact (paper Table V) |
| **capacity10** `eta_c10_bdef` | **CIFAR-10 / 50 clients** | honest×H `honest_c10_bdef_iid`; reduced 16,17×A | capacity without thin-data confound | clean same-class overlap (5 clients/class, ~100 trigger imgs) |

Plot with `RES=<results> ./run_everything.sh plot` (per-family `figs/`: `honest_lines`,
`timeline_*`, `separability_*`, `class_difficulty`, `thresholds`, `fidelity`).

---

## 2b. Capacity / oversubscription vs paper §V-F3

**What the paper says.** When clients outnumber classes, multiple clients share a trigger
class. The paper resolves the conflict with two mechanisms: (i) the projection matrix **M is
3-D and indexed by client** — `M_{i,k,j}` is "the jth element of the projection vector for the
kth bit of client i" (§IV-A), so every client reads the output through its own secret matrix;
and (ii) **trigger-sample consistency** — "the trigger samples used during testing are
identical to those employed in training", each client using 50 trigger samples, so that
"clients sharing the same trigger class ... remain distinguishable through client-specific
trigger variations" (§V-F3).

**Your reading was half right.** The per-client projection matrix is real and we already had
it — but it is *not* the capacity mechanism, it's the general design (every client has a unique
`M^i` and bits `B^i` in every experiment, seeded by cid). The capacity-specific addition is the
**trigger images**: per client, and identical between training and verification.

**What our code did before this change:** `build_trigger_bank(test_dataset, classes, …)` built
**one bank per CLASS from the held-out TEST set**, and the verifier looked it up by trigger
class. So two clients sharing class 6 were verified on **identical images they had never
trained on**, distinguished only by `M^i`/`B^i`. That is a *stricter, generalisation* reading
of the watermark — not the paper's protocol.

**Now implemented — `wm_trigger_mode` (`TRIGMODE`):**

| mode | verification images | matches paper | what it measures |
|---|---|---|---|
| `class` (default) | one shared held-out bank per trigger class | ✗ (stricter) | does the mark **generalise** to unseen images of the class |
| `client` | per-client **disjoint** slice of held-out test images | partial (client-specific variations, still held-out) | generalisation **+** per-client image variation |
| `client_train` | each client's own **training** images (test == train) | ✓ paper §V-F3 | **memorisation** on those exact samples |

Verified: with 200 clients on CIFAR-100, cid 6 and cid 106 both sit on class 6 and receive
**disjoint** 50-image slices; all 200 clients get a bank.

**Why this matters for your argument.** `client_train` should make the mark look excellent —
each client is graded on images it memorised — which is likely how the paper reaches >95%
detection at 50 clients (Table IX). But the paper itself concedes the failure mode in Table V:
a mark fitted to specific trigger samples "becomes overfitted to those specific samples and
cannot be generalized to other trigger-class samples". So `client_train` measures memorisation,
not participation — and a free-rider that trains briefly on its own 50 trigger images passes
just as well. Running `capacity` (held-out) against `capacity_paper` (`client_train`) turns that
into a measured gap rather than an assertion.

**Hard constraint — CIFAR-100 test set has only 100 images per class** (10,000 / 100). In
`client` mode you need `N_T × clients_per_class ≤ 100`:

| setup | need | fits? |
|---|---|---|
| `CAP_NC=200` (2/class), `N_T=50` | 100 | exactly at the limit, zero slack |
| `CAP_NC=300` (3/class), `N_T=50` | 150 | ✗ slices wrap → **not disjoint** (drop to `N_T=33`) |
| CIFAR-10, `CAP10_NC=50` (5/class), `N_T=50` | 250 of 1000 | comfortable |

`client_train` has no such limit (it draws from each client's own shard), which is another
reason it's the right mode for the paper-faithful reproduction. The runner logs the bank mode
and warns if any client failed to get one.

Run the comparison:
```
LEGS="capacity capacity_paper" ./run_everything.sh submit      # held-out vs paper-faithful
```
Families: `…_nc200_iid` (held-out) vs `…_nc200_tmtrain_iid` (paper), each with its own eta.

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