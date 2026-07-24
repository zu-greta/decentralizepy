# FareMark limitations study — reference

Detection rule everywhere: **flag client *i* iff `BER_i ≥ η`** (`wm_verify.detected`
returns `ber < eta`). "Looser" = higher η = fewer flags = lower FPR, lower recall.

Companion to `README.md` (goals, findings, codemap) and `RUNPLAN.md` (what to run now).

---

## 1. Thresholds

### 1a. The live threshold η

Frozen once by `detection.py calibrate` on honest-only runs, injected as `WM_ETA_FIXED`:

1. per round *r*, per seed: `m_r = mean over honest clients of BER`
2. keep the last `tail = 20` rounds (converged)
3. per seed *s*: `μ_s = mean_r(m_r)`, `σ_s = std_r(m_r)` (population std), `η_s = μ_s + 3σ_s`
4. `η = mean_s(η_s)`

**Two structural problems, both now measured (README F8, F9).**

**(a) The √N bug.** `m_r` is a mean over *N* clients, so its spread is ≈ σ_per-client/√N,
but the test is applied to *individual* clients. Measured `(η − μ)/σ_per-client`:

| leg | N | claimed | delivered | honest FPR |
|---|---|---|---|---|
| iid | 10 | 3σ | **0.42σ** | 27% |
| spread | 10 | 3σ | 0.42σ | 28% |
| bits20 | 10 | 3σ | 0.31σ | 17% |
| noniid | 10 | 3σ | 0.26σ | 25% |
| capacity10 | 50 | 3σ | 0.23σ | 7.4% |
| sin | 10 | 3σ | 0.51σ | 13.5% |
| capacity | 200 | 3σ | **0.07σ** | **53%** |

Never above 0.51σ, in any configuration. Empirically worse than 3/√N because round
means are autocorrelated (the same clients recur every round).

**(b) Quantisation degeneracy.** BER only takes multiples of `1/m`. If the mark is good,
μ→0 and σ→0 so η→0, and once **η < 1/m** the rule `BER ≥ η` is exactly `≥1 bit wrong` —
every η in `(0, 1/m)` is the same detector.

| leg | η | 1/m | η in bits | reduces to |
|---|---|---|---|---|
| iid | 0.063 | 0.100 | 0.63 | **≥1 bit; η irrelevant** |
| spread | 0.063 | 0.100 | 0.63 | **≥1 bit; η irrelevant** |
| capacity10 | 0.068 | 0.500 | 0.14 | **≥1 bit; η irrelevant** |
| balanced | 0.000 | 0.100 | 0.00 | **flags 100%** |
| noniid | 0.161 | 0.100 | 1.61 | ≥2 bits |
| capacity | 0.384 | 0.100 | 3.84 | ≥4 bits |
| bits20 | 0.178 | 0.050 | 3.56 | ≥4 bits |

**(c) The paper's own text is ambiguous.** §IV-D3 says only *"we test the similarity
metric across many rounds … η is set to μ + 3σ"* — never whether μ,σ are over clients,
over rounds, or over round-means. Those readings differ by √N. The ambiguity is itself
a finding, and §V-A3b/§V-C make FPR the paper's *primary* stated objective.

### 1b. Provenance of every rule

| rule | origin | standard? |
|---|---|---|
| **μ+3σ** | the FareMark paper §IV-D3; underneath, the Shewhart 3-sigma control limit (1920s) | yes — assumes unimodal ~normal data; ~0.13% nominal one-sided FPR |
| **median + k·MAD** | robust statistics (Hampel). 1.4826 makes MAD consistent with σ under normality | yes — 50% breakdown point |
| **trimmed mean + 3σ** | Tukey | yes — breakdown = trim fraction |
| **adaptive σ-clipping** | iterative σ-clip (`astropy.sigma_clipped_stats`); also DP-SGD adaptive clipping | yes |
| **percentiles p95/p99** | non-parametric empirical quantile | yes — no distributional assumption; sets nominal FPR directly |
| **equal-error-rate** | biometrics (FAR = FRR) | yes |
| **Youden-optimal** | Youden's J (1950); maximising J = minimising balanced error | yes |
| **overlap coefficient (OVL)** | Weitzman's overlapping coefficient; relates to total variation and hence the Bayes error of any 1-D threshold | yes |

Rules 1–6 use **honest BER only** and are deployable. EER and Youden need the
free-riders too — they are **oracle** rules, included to bound what any η could ever
achieve. The sentence you want is always: *"even the oracle rule, which is not
implementable, only reaches balanced accuracy X."*

### 1c. The regime (`detection.py separability`, post-hoc)

| # | rule | computed over | formula |
|---|---|---|---|
| 1 | coded (μ+3σ round-mean) | round-means | `mean_r(m_r) + 3·std_r(m_r)` = live η |
| 2 | loose (μ+3σ per-client) | every per-client BER | `mean(H) + 3·std(H)`; ~√N looser |
| 3 | median + 3·MAD | per-client | `median + 3·1.4826·MAD` |
| 4 | trimmed-10 μ+3σ | middle 80% | drop 10% each tail, then μ+3σ |
| 5 | adaptive σ-clip | per-client | iterate: drop `x > μ+3σ`, recompute to fixpoint |
| 6 | honest p95 / p99 | per-client | empirical quantile |
| 7 | equal-error-rate | H and F | η where FPR = FNR — **oracle** |
| 8 | Youden-optimal | H and F | `argmin_η (FPR+FNR)/2` — **oracle** |

**Degenerate rows.** Any row reading `eta = 0.0, fpr = 1.0, recall = 1.0, bal_acc = 0.5`
is *not* a finding — it is the rule collapsing because the honest median (or trimmed
mean) is exactly 0, so `BER ≥ 0` flags everyone. Report as `degenerate`, not as numbers.

### 1d. Rule-independent bounds — the headline numbers

| metric | meaning | perfect-for-us value |
|---|---|---|
| `overlap_coefficient` (OVL) | shared area of the honest and free-rider BER histograms | **1.0** = the distributions are identical |
| `best_threshold_balanced_error` | min over *all* η of (FPR+FNR)/2 | **0.5** = no threshold anywhere beats a coin |

Current tally: **11 of 18 valid per-class comparisons are inseparable** (best balanced
error ≥ 0.40), **6 of them at exactly 0.500**. In 8 of 18 the free-rider's mark is
*cleaner* than an honest client's at the identical trigger class.

### 1e. Read `per_class`, not `global`

The `global` block compares free-riders at hard classes against a *mixture* of honest
clients across all classes, manufacturing separation out of class heterogeneity the
detector would also see in an all-honest population. The server knows each client's
trigger class (`cid % n`), so it *could* condition — meaning global separation is not
even a defence the paper can mount. The gap between the two blocks is itself a result.

---

## 2. Experiment matrix

### 2a. Current batch — see `RUNPLAN.md` for the full rationale

| label | what | seeds | proves |
|---|---|---|---|
| **R1** | CIFAR-100, **100 clients**, honest, m=10, N_T=100 | 3 | paper Table I+II; also class difficulty for **all 100** classes (cid%100) |
| **R2** | CIFAR-10, 10 clients, honest, **m=1**, N_T=100 | 3 | paper Table I+II; m=1 is the only value that reaches 99.72% |
| **R3** | crude `previous_models` (Eq. 17) | 1 | detector sanity — **must** be caught cleanly |
| **R4** | crude `gaussian` (Eq. 18) | 1 | same |
| **R5** | 200 clients honest, seeds 3–5 | 3 | calibration source for R6 (η seed-std ≈ 40%) |
| **R6** | 200 clients, `reduced` at cid 106,107 → classes 6,7 | 3 | more clients than classes: forced sharing with honest cid 6,7 |
| **R7** | `tap_oracle` coast/tap | 3 | the adaptive free-rider |
| **R8** | balanced keys, FR pinned to class 6 | 3 | "even at BER=0 for everyone, the free-rider matches" |

```bash
./run_now.sh                                                   # writes jobs.tsv only
unset DRYRUN
RUNAI_EXTRA="--node-pools a100-80" PODS=2 WORKERS=6 ./submit_pool.sh
```

### 2b. Trigger modes (capacity / oversubscription, paper §V-F3)

| `WM_TRIGGER_MODE` | verifier images | what it tests |
|---|---|---|
| `class` (default) | one shared held-out bank per trigger class | **generalisation** — strictest. Clients sharing a class differ only by M^i, B^i |
| `client` | per-client disjoint held-out slice | paper's "client-specific trigger variations", still held-out |
| `client_train` | the client's **own training images** | paper §V-F3 "trigger sample consistency" — **memorisation** |

The paper's Table IX capacity result uses `client_train`. Table V simultaneously says a
mark fitted to specific samples "cannot be generalized to other trigger-class samples".
**Those two claims are in tension** (README F14).

**Confirmed saving:** all three banks are used only inside `verify_hook`, never in
training, so **one training run can be extracted three ways** — 27 capacity jobs become 9.

### 2c. Non-IID — ⚠️ blocked

Dirichlet(α). **Do not quote any non-IID number yet.** Floors span 0.007–0.297 with
σ 0.179 > mean 0.115, far above the binomial prediction 0.101 — but under Dirichlet a
client can be assigned trigger class *c* while holding almost no images of *c*, giving
BER ≈ 0.5 for a reason unrelated to the watermark. Log `n_trigger_samples` per client
per round in `wm_verify.py`, split the honest population by it, then re-emit the tables.
α ladder: 0.1 (severe) · 0.5 (benchmark default) · 100 (≈IID null). **Skip α=1.0** — no
information between 0.5 and 100.

### 2d. +N free-riding spectrum — 🆕 highest-value untouched experiment

```bash
DS=c100 SEEDS='0 1 2' POS=3,6 NS='-1 0 1 2 5 10 25 50' \
        USE_FIXED_ETA=1 FIXED_ETA=0.063 ./run_all.sh sweep     # 24 runs
python scripts/plots.py sweep --in "$RES/*/result.json" --eta 0.063 --out figs/sweep.png
```

N = images kept per *common* class; the attacker always keeps all of its trigger class.
`N=0` is the cheapest embedder; **`N=-1` is a full shard — a client doing 100% of the
work but still labelled free-rider. If its BER is indistinguishable from honest, BER
measures data spent rather than honesty**, which is the argument in a single run.
`KCLS=K` adds the class-diversity axis (same N, only K random common classes).

### 2e. Bit count — settled, m=20 archived

A stuck row caps watermark accuracy at `100·(1 − 0.5·2^(1−l))`, and no number of trigger
samples can fix a stuck bit (Eq. 15 averages noise, not bias).

| dataset | m | l | stuck | ceiling | paper | verdict |
|---|---|---|---|---|---|---|
| CIFAR-100 | **10** | 10 | 0.20% | 99.90% | 99.71 | ✅ use |
| CIFAR-100 | 20 | 5 | 6.25% | 96.88% | — | 🗄️ **archived — cannot be faithful** |
| CIFAR-10 | 2 | 5 | 6.25% | 96.88% | 99.72 | ❌ 3pp short |
| CIFAR-10 | **1** | 10 | 0.20% | 99.90% | 99.72 | ✅ use — **requires the `make_bits` fix** |

**m = 1 caveat.** The old `make_bits` computed `half = num_bits // 2`, so at m=1 it
returned `B = [0]` for every client — a constant, not a secret. Patched to draw
uniformly for m < 4. Without that patch the CIFAR-10 reproduction is meaningless.

### 2f. Keys — unbalanced is the main line, balanced is one control

- **unbalanced** (`BALANCED=0`, default): the paper's random ±1 M (Eq. 1, Fig. 5).
  Reproduces the paper on CIFAR-100 at m=10 *and* preserves the honest floor that
  produces the FPR story. **Use this everywhere.**
- **balanced** (`BALANCED=1`): zero-sum rows, no stuck bits, honest BER → 0. Not what
  the paper says, but it is the paper's claimed *operating point* (>99.7%). One control
  run (R8): if the free-rider also reaches 0 there, `OVL = 1.000` and
  `best_threshold_balanced_error = 0.500` — the strongest possible non-separability
  result, at the point where the scheme works best.

Balanced works at odd `l` too (`make_key` builds `[1]*(l//2) + [-1]*(l−l//2)`, never
all-one-sign), so it is runnable at any m.

---

## 3. What the seed varies (and why the variance is so large)

`seed = base_seed + repeat`. It re-rolls: the data split, batch shuffling, model init,
the trigger-bank sample, and — critically — **the key `M^i` and target bits `B^i`**,
generated from `seed + 1000·cid + 1`. The trigger class is **not** re-rolled
(`cid % n`).

Changing the data split is *nuisance* variance: the same quantity measured more or less
precisely, and averaging over seeds helps. Changing `M` and `B` is **task-changing**
variance: you are measuring a *different watermark*. "How hard is class 6?" has no
single answer — it depends which decoder matrix and which message you drew. Averaging
over seeds averages over a *population of different questions*.

Three concrete lotteries:
1. **Stuck bits.** An all-same-sign row is permanently wrong: `P = 2^(1−l)`. At l=5 that
   is 6.25% per bit, so at m=20 a client expects ~1.25 dead bits — but some draw 0 and
   some draw 3.
2. **Bias strength.** Even without a fully stuck row, `s_k = Σ_j M_kj` sets how hard the
   bit is to flip. ~25% of rows sum to 0 (easiest); the rest spread out.
3. **Message alignment.** Some target bits agree with their row's bias, some fight it.

**On top of all that: pure counting noise.** BER is "how many of m bits are wrong ÷ m",
so even with every bit failing independently at the same rate it jumps by
`√(p(1−p)/m)`:

| leg | m | mean BER | noise from counting alone | observed |
|---|---|---|---|---|
| capacity10 | 2 | 0.037 | 0.134 | **0.131** (all of it) |
| iid | 10 | 0.036 | 0.059 | 0.064 |
| bits20 | 20 | 0.152 | 0.080 | 0.086 |

So most of what looks like "clients differ in how hard their position is" is "we only
measured m bits". This is why η itself has ~40% seed std (F2) — it inherits every
lottery above. **Report per-(class, key), or the ranking and its entropy correlation,
not exact BER values.**

---

## 3a. CLI reference

### A. The current batch — `run_now.sh` / `submit_pool.sh`

```bash
./run_now.sh                    # DRYRUN: appends 20 rows to jobs.tsv, submits nothing
wc -l jobs.tsv
unset DRYRUN
PODS=2 WORKERS=6 ./submit_pool.sh          # exactly 2 runai jobs
POOL_TAG=<same> ./submit_pool.sh           # resume after preemption (skips finished)
```

`PODS` = runai jobs = GPUs. `WORKERS` = concurrent runs **inside one pod on one GPU**
(6 on A100-80, 4 on A100-40). Rows are dealt round-robin so each pod gets a mix of legs.
`RUN_TAG` is deterministic, so a run whose `result.json` exists is skipped.

### B. Ad-hoc legs — `run_all.sh`

```bash
DS=c100|c10  [BITS=n] [BALANCED=0|1] [PART=iid|niid] [DIRICHLET_ALPHA=a]
[VTAG=tag] [TRIGMODE=class|client|client_train] [TCMAP="cid:class,..."]
[SEEDS='0 1 2'] [POS=3,6] ./run_all.sh <target>

targets: honest | calibrate | reduced | sweep | sameclass | noniid | separability | PLOTALL
```

Prefix with `DRYRUN=1 JOBS_FILE=jobs.tsv` to append to the pool manifest instead of
submitting. Attack targets need an η — pass `USE_FIXED_ETA=1 FIXED_ETA=<v>` if the eta
file is not on the submitting host. **η only drives live flagging, which `detection.py`
recomputes offline, so the value constrains nothing in the analysis.**

### C. Paper rows — `paper_check.sh`

```bash
ROW=c10|c100|t9 [SEEDS='0 1 2'] [BALANCED=0|1] [HELDOUT=1] [FAM=<name>] ./paper_check.sh submit
RES=<local> ROW=<r> ./paper_check.sh check
```

**Do not use the `sanity10`/`sanity100` legs in `run_everything.sh`** — they write
families like `honest_c10_bdef_sanity_iid`, but `paper_check.py` searches for
`paper_<row>_nc<N>_<mode>` and reports "no runs found". R1/R2 replace them.

### D. Analysis — local, after `scp`

```bash
RES=~/local/results ./plot_now.sh          # the four figure groups, nothing else

# individually:
python scripts/detection.py calibrate --in "$RES/*/result.json" \
       --honest-family <fam> --tail 20 --out eta.json
python scripts/detection.py separability --honest-in "$RES/*/result.json" \
       --honest-family <hfam> --attack-in "$RES/*/result.json" \
       --attack-family <afam> --tail 20 --per-class --emit sep.json
python scripts/plot_all_thresholds.py --in "$RES/*/result.json" \
       --family <fam> --tail 20 --out figs/thresholds_all      # -> .png + .md table
python scripts/plots.py timeline --in "$RES/*/result.json" --family <afam> \
       --honest_in "$RES/*/result.json" --honest_family <hfam> --eta <η> --out figs/t
python scripts/plots.py class_probe --in "$RES/*/result.json" --family <hfam> --out figs/
python scripts/resultio.py digest --in "$RES/*/result.json"
```

### E. Environment knobs

| var | effect |
|---|---|
| `SMOOTH_EPS` | **1e-3 (default, legacy)** or 1e-8 (fixed). Changes every BER — never mix inside one family |
| `WM_ALPHA` | smoothing α. **`sin` requires 1.5708 (π/2)**; it now raises rather than silently no-op |
| `BALANCED` | 0 = paper-exact random keys (default), 1 = zero-sum rows |
| `WM_BITS` | m. 0/unset → `max(2, n//10)` |
| `WM_TRIGGER_MODE` | `class` / `client` / `client_train` |
| `WM_NUM_TRIGGERS` | N_T. 100 for Table I/II, 50 for Tables III/IX |
| `RUNAI_EXTRA` | `"--node-pools a100-80"` — pin the GPU type, the cluster is heterogeneous |

---

## 4. Task audit

| area | task | status |
|---|---|---|
| threshold | stress-test all threshold calcs | ✅ `detection.py` regime, 9 rules |
| threshold | prove non-separable | ✅ OVL + best-balanced-error; 6 comparisons at 0.500 |
| threshold | **all thresholds on one timeline + explanation table** | ✅ **`plot_all_thresholds.py` [NEW]** |
| threshold | median / trimmed / adaptive clip / percentiles | ✅ all implemented |
| reproduce | match the paper's published rows | 🔁 **in flight (R1, R2)** — never done before |
| reproduce | the paper's own crude free-riders | 🔁 **in flight (R3, R4)** — never done before |
| difficulty | class difficulty across all CIFAR-100 classes | 🔁 in flight (R1, 100 clients ⇒ every class) |
| difficulty | sin smoothing (Eq. 9) | ⚠️ **was broken (F12), fixed, needs rerun with `WM_ALPHA=1.5708`** |
| experiments | more clients than classes | ✅ honest / 🔁 attack in flight (R5, R6) |
| experiments | same trigger class → same BER | ✅ `sameclass`, unbalanced ✅ + balanced 🔁 (R8) |
| experiments | adaptive free-rider | 🔁 in flight (R7) |
| experiments | non-IID | ⚠️ ✅ run but **blocked on `n_trigger_samples`** (F11) |
| experiments | FR spectrum (+N sweep) | 🆕 **not run — highest-value remaining** |
| experiments | different number of free-riders | 🆕 not run |
| experiments | rotate trigger class per round | ❌ not touched |
| detection | consequence of crossing / k-warnings / detection window | ❌ **not touched — biggest untouched block**, natural edit in `wm_verify.py` |
| theory | discrimination bounded by class count | ⚠️ inequality stated, not formalised |
| theory | output-layer watermarking impossible | ⚠️ original mechanism **refuted** (F9), restated version empirical only |
| next | hint of a solution | ❌ not touched |
| infra | 2-pod worker pool | ✅ `submit_pool.sh` |
| infra | deterministic run tags | ✅ `submit_experiment.sh` |
| infra | GPU-resident loader | 🆕 `fast_data.py` written, **not wired** — test `--num_workers 0` first |
| infra | log `n_trigger_samples` | 🆕 **blocking F11** |
| infra | `rule=degenerate` in `detection.py` | 🆕 open |
| infra | plot fixes (green η line, timeline η source, band clipping, nc200 legend) | 🆕 open |