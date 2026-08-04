# Status & Plan — output-layer watermarking cannot detect free-riders

---

## 0. Current status

**Thesis.** For output-layer (box-free) watermarking using FareMark's proposed schema, there is no detection
threshold η that can separate honest clients from free-riders, and adaptive free-rider can stay under any such η at a fraction of the honest compute. 

**Done (3 seeds, table-ready):**
- **Group A** (A1 honest ×6; A2–A4, AK ×3) — class-difficulty floors, threshold calibration, η
  seed-instability, reduced-FR non-separability at easy/hard classes, same-class and same-key controls.
- **Group D** (D1 +N spectrum ×3; +50 = 2) — the price-of-invisibility curve.
- **Group E** (E1/E2 α=0.5; E3 α∈{0.1,1.0}) ×3 — non-IID *worsens* separability (starvation).

**Done (1 seed, shape only — do not table):**
- **Group I** (adaptive-tap single-knob sweep, 40 rounds) — the effort dial works; `resend` coast can't
  save compute.
- **Group J** (graft-coast suite J0–J4, seed 1000, 40 rounds) — **`graft` coast is shipped and works**;
  **J2 is a confirmed adaptive submarine** (gradual fade, live submissions).

**Done (the first 3-seed submarine):**
- **Group NOW** = **J2×3 + J5×3** (seeds 1000/1001/1002, 40 rounds). **J2 rep0 complete; J5 rep0 crashed
  at R39** (no result.json — likely pod eviction during finalization). The submarine **works on the easy
  class only**: J2 cid3 taps ~14 % / J5 cid3 ~7 %, but cid6 (hard) taps ~66–75 % and saves no compute —
  its 16-image self-probe over-reads (~0.30) so it taps every round, though the *server* holds cid6 at
  ~0.217 < η_loose, i.e. cid6 **evades** but at full reduced-attack cost. See RESULTS_INDEX → Group NOW,
  and **§10 for the wrap-up** (completion, compute breakdown, hardcoded list, best config, what to finish).

**Not yet run:** **H5** (crude previous-models positive control on c100), **V2** (Table V trigger-sample
attack), **F1/F2/F3** (capacity, 200 clients), **C1** (sin smoothing ablation — unblocked, low priority).

**Deep dives (new):** the model's layers and what a tap trains (§11), the `graft` coast mechanism in code
+ math (§12), the exact compute-cost math (§10.3), and whether this is a paper-worthy attack (§13).

---

## 1. Thesis & threat model

**System.** FedAvg over *N* clients, *n* classes, *R* rounds. Ownership/effort is protected by an
**output-layer, box-free, multi-bit watermark** (FareMark family): each client *i* holds a private key
matrix `M_i` (shape `[m,l]`, entries ±1) and a message `B_i ∈ {0,1}^m`, and embeds its mark by adding
`λ·L_wm` to its local loss. The mark is read **only from softmax outputs** — the verifier pushes `N_T`
held-out trigger-class images through the model, smooths the softmax with `f`, projects onto `M_i`, and
thresholds the sign to recover `m` bits. Detection is a single scalar test: flag client *i* iff
`BER_i ≥ η`, with η calibrated on honest clients.

**Free-rider (the attacker).** A client that wants the aggregated global model without paying the full
training cost, while keeping `BER < η`. It holds a valid `(M_i, B_i)`, sees the global each round, and
can (a) train on a reduced shard, (b) train intermittently ("tap"), (c) submit a mark-carrying model
between taps ("coast"), and (d) estimate η from its own honest-phase BER. 

**Claim.** The watermark cannot function as a free-rider detector because, for every η a server could
deploy (calibrated on honest data alone), the honest-BER and free-rider-BER distributions overlap enough
that no η separates them at a useful operating point.
1. **BER does not measure effort.** A reduced free-rider concentrates its shrunken data on the trigger
   class and embeds *at least as cleanly* as honest (A2/A4: FR BER ≤ honest).
2. **Apparent separability is a key/class lottery the server cannot observe** (A3/A4/AK: a ~3× BER swing
   at one fixed class from the key draw alone).
3. **An adaptive attacker closes any residual gap** by tracking η and coasting under it (Groups I/J/NOW),
   at a fraction of the honest compute.

**Non-goal.** We do **not** claim the mark can't be embedded — it can, robustly (that's the Group-H
positive control). We claim (a) no deployable η separates honest from free-rider, and (b) an adaptive
free-rider exploits this to evade at minimal effort.

---

## 2. Glossary — measurement definitions

**BER — bit-error-rate.** wrong bits ÷ `m`. Takes only the values 0, 1/m, 2/m, … **0 = perfect mark,
0.5 = coin flip = no mark**, 1 = all flipped. (A random model gets half its bits right by luck, so 0.5,
not 1, is "no watermark".)

**η (eta) — the detection threshold.** One number pre-calibrated on honest clients; flag if `BER ≥ η`.

**FPR — false-positive rate.** Fraction of *honest* clients wrongly flagged. Best near 0.

**Recall (TPR).** Fraction of *free-riders* caught. Best near 1. FPR/recall trade off with η.

**Balanced accuracy** `= ((1−FPR) + recall)/2`; **balanced error** `= 1 − that`. 0.5 = useless, 0 = perfect.

**best_threshold_balanced_error.** Lowest balanced error any η achieves (incl. oracle rules that peek at
FRs). **= 0.5 ⇒ the two populations are inseparable — no threshold works.**

**OVL — overlap coefficient.** Shared area of the honest-BER and FR-BER histograms (Weitzman's
coefficient). **1.0 = identical distributions** (nothing can separate them); 0 = disjoint.

**Headroom (σ).** `(η − mean honest BER) / (per-client std of honest BER)` — standard deviations of
safety above the honest average. A Shewhart 3σ limit false-alarms ~0.13 %; we measure **0.1–0.6σ** for
the paper's own rule.

**Degenerate threshold.** η < 1/m. Because BER is quantised, any η in (0, 1/m) yields the *identical*
detector ("flag if ≥1 bit wrong") — the calibrated number does no work. **η tight 0.064 < 1/m 0.10 is
degenerate.**

**Entropy / dominance (of the softmax).** `H = −Σ p·ln p` (high = flat/unsure); `dominance = f(p_max)/Σf(p)`
(the paper's Eq. 10 quantity; high = peaky). A flat softmax has a rich tail to hide the mark in; a peaky
one does not — so entropy/dominance predict **class difficulty** (§6.1).

**N_T (trigger samples).** How many held-out trigger-class images the verifier averages over when
extracting (Eq. 15). Here **50**.

**cpc (common-per-class, the `+N` of the reduced/tap attacker).** Images per non-trigger class the FR
keeps. `cpc=5` = all trigger images + 5 random/other class ≈ **31 %** of an honest client's data. `cpc=-1`
= full honest shard (100 % effort, still labelled FR — the effort anchor).

**Effort / "data used %".** FR cumulative image-passes ÷ an honest client's. On a submarine this is
**warmup-dominated** — report attack-phase-only effort separately (§6.4).

---

## 3. What the seed randomizes (summary) - [detailed_seed_randomization](RESULTS_INDEX.md#seed-analysis--what-a-runs-seed-actually-randomizes)

The CLI "seed" is the **repeat index**; `seed = base_seed + repeat = 1000 + repeat` (`config.py:228-229,22`),
so 3 seeds = 1000/1001/1002. Everything derives deterministically from that one integer.

| re-rolled by the seed | why random | how much it moves the result |
|---|---|---|
| which images each client gets (IID shard) | nobody controls who holds what | moderate |
| minibatch shuffle order per client | standard SGD | small |
| model weight init | networks start random | small–moderate |
| **key matrix `M` per client** | keys must be secret/unique | **large** (the "key lottery") |
| **message `B` per client** | messages unpredictable | **large** |
| N_T verification images | it's a sample of the class | small |
| Dirichlet label split (non-IID only) | the skew pattern itself | **large, non-IID only** |

**Held constant across seeds:** trigger-class assignment (`cid % n`), N, m, rounds, epochs, λ, β, α, the
smoothing `f`, the dataset, the frozen thresholds, and the free-rider set (`FREE_RIDER_IDS=3,6`).

**Key/bits provenance.** `M` is seeded `S + 1000·cid + 1` (`clients.py:261`); `B` adds a `+7919` offset
inside `make_bits` (`watermark.py:206`) so the message is decoupled from the key. The `1000·cid` term
makes every client differ within a run; `S` makes the same client differ across runs.

---

## 4. The watermark scheme — theory (FareMark, `watermark.py`)

The mark of client *i* is an `m`-bit string `B_i` embedded into the model's **softmax on trigger-class
inputs**. Nothing is read from weights ("box-free"). Pipeline (paper Eq. numbers in code comments):

1. **Group** the `n`-dim softmax `P` into `m` groups of `l = n//m` (uses the first `m·l` outputs).
2. **Smooth** each probability with `f(p)=p^α`, α=0.4 (`smooth`, Eq. 7–9). Cross-entropy makes the
   softmax peaky (argmax ≈ 1, rest ≈ 0); `f` with 0<α<1 **amplifies the tail** so the projection isn't
   decided by the argmax alone. (Diagnostic `smoothing_gain`: power@0.4 = **4.91**; sin can only reach
   1.23 — §6.5.)
3. **Project** each group onto the secret ±1 key row: `z_k = Σ_j f(p_kj)·M_kj` (`project_logits`, Eq. 13).
4. **Bit** `k = sign(z̄_k)`: `≥0 → 1`, `<0 → 0` (Eq. 2), averaging `z` over the `N_T` trigger images first
   (`extract_bits`, Eq. 15).
5. **Embed** by adding `λ·L_wm`, `L_wm = BCE(z, target_bits)` (`watermark_loss`, Eq. 11–12).
6. **Detect** via `BER = (1/m)Σ|b̂−b|` (Eq. 16); flag if `BER ≥ η`.

**Two structural facts that create BER floors:**
- **Key lottery (stuck bits).** Probabilities are ≥0 and `f(p)≥0`, so a key row that comes out
  **all-same-sign** forces its bit to a fixed value regardless of input — structurally unembeddable.
  `P(same-sign row) = 2^(1−l)`: `l=10 → 0.2 %` (negligible, our CIFAR-100 case), `l=5 → 6.25 %`,
  `l=2 → 50 %`. Even non-stuck rows have a bias set by their row-sum, so difficulty is a per-client
  mixture over the rows it drew (`make_key`, `unembeddable_fraction`).
- **`m·l = n` squeeze.** More bits (`m↑`) ⇒ shorter projections (`l↓`) ⇒ noisier bits and more stuck
  rows. Fewer bits ⇒ coarser BER (m=2 → BER ∈ {0, .5, 1}). CIFAR-100 uses **m=10, l=10** as the sweet
  spot (§6.2).

---

## 5. Implementation walkthrough — the code

### 5.1 `datasets.py` — data & the client partition
- `build_data` loads torchvision CIFAR-100 (`_load_raw`), applies **RandomCrop(32,pad4)+HFlip** on train
  and normalization on both (`_build_transforms`), then partitions.
- **IID** (`iid_partition`): shuffle all 50 000 indices with `np.random.default_rng(seed)` and
  `np.array_split` into `N` near-equal shards → **5000 imgs/client**, class balance uniform (so per-class
  floors barely move across seeds).
- **Non-IID** (`dirichlet_partition`, Hsu et al. 2019): for each class draw `props ~ Dirichlet(α·1_N)`
  over clients and hand out that class in those proportions. Small α -> severe skew; large α -> ≈IID (but
  even α=1.0 is *not* equal shards). Trigger-class assignment is drawn **independently** of this split,
  so a client is usually data-starved on its own trigger class — the Group-E mechanism.
- Each client gets its own `DataLoader(Subset(...), batch_size=16, shuffle=True, generator=seed+cid)`
  (line 116/124) — the `seed+cid` offset is what per-client-shuffle stream #4 in the seed table uses.

### 5.2 `clients.py` §1 — honest `Client`
`produce_update(global_state, prev_global_state, round_idx)` is **the seam** every attacker overrides.
Honest behaviour: `load_state_dict(global) → _local_train() → return (cpu_state, num_samples)`.
`_local_train` builds a fresh SGD optimizer (lr 0.01, mom 0.9, wd 5e-4) and runs `local_epochs=5`
passes of cross-entropy over the loader. `num_samples` is the FedAvg weight.

### 5.3 `clients.py` §2 — `WatermarkClient` (embed + memory)
Honest client that also embeds. Two additions over `Client`:
- **`_local_train_wm`** (Eq. 11–12): same SGD loop, but on trigger-class samples it adds
  `λ·watermark_loss` (`λ=5`). It logs per-round `wm_stats` (cls/wm/total loss, trigger-class train
  accuracy, `n_trigger_samples`) — the client-side counterpart to the server's softmax diagnostics.
  Two guards keep the non-standard embedding term from exploding: skip a batch whose loss is non-finite,
  and clip grad-norm to 5.0.
- **`_memory_update`** (Eq. 14): after SGD, `W_new = β(memory + Δ) + (1−β)·W_g`, `Δ = W_sgd − W_g`,
  β=0.6, memory initialized to the global on round 1 and updated to `W_new` each call. This keeps the
  client's watermarked trajectory alive through FedAvg averaging (otherwise 9 honest neighbors would wash
  the mark out). **Important for the submarine:** the memory only advances when `produce_update` runs —
  i.e. on honest/warmup/tap rounds, **never on a coast** (§5.5), so a coasting FR carries its last
  *tapped* trajectory.

**Factory `build_watermarked_clients`.** Sets `m = max(2, n//10) = 10`, `l = grouping(100,10) = 10`,
`exclude_col = None` (**full 100-way softmax — the trigger class is *not* dropped**). For each cid it
draws `(M, B)` seeded from the cid, registers `(trigger_class, key, bits, exclude=None)` with the
verifier, and builds either an honest `WatermarkClient` or, if `cid ∈ FREE_RIDER_IDS`, the selected
attacker. Trigger class = `cid % n` unless `TRIGGER_CLASS_MAP` overrides; key/bits can be borrowed from a
twin via `WM_KEY_TWINS` (the AK effort-only control). `unembeddable_frac` is computed and warned on
(0.0 here).

### 5.4 `clients.py` §3 — the attackers

**Crude baselines (no embedding — the positive control the detector should catch):**
- `PreviousModelsFreeRider` (Eq. 17): submit `2·W_t − W_{t−1}` (resubmit `W_t` on round 1); norm buffers
  copied, never extrapolated (`_is_norm_buffer` prevents negative running-var → NaN).
- `GaussianNoiseFreeRider` (Eq. 18): submit `W_t + N(0,σ²)`. Neither trains → BER ≈ 0.5 → caught.
  *(Seeded by a hardcoded `1234+cid·1000+round`, not `S` — its noise is identical across experiment
  seeds.)*

**`_SimpleFRMixin` — shared reduced-loader + self-probe** (used by reduced and adaptive_tap):
- `_prepare(cpc, holdout)`: scans the shard once, splits trigger vs common images, holds out up to
  `holdout` trigger images for the self-probe (but **never below `MIN_TRAIN_TRIG=8`** to embed on — on
  scarce classes it skips the holdout, `_probe_x=None`, and threshold-tapping falls back to always-tap),
  and builds a reduced `TensorDataset` = `trig_train + cpc images × each common class`.
- `_probe_ber(state)`: the FR's own BER estimate on its **held-out** trigger images — a generalisation
  probe mirroring how the server tests. This is the quantity the submarine's tap decision uses; its gap
  from the server's 50-image bank is the central weakness (§ NOW/results).
- `_phase_action(round)`: `honest` (r < W−K), `calib` (W−K ≤ r < W), `freeride` (r ≥ W). With W=12, K=4:
  honest 1–7, calib 8–11, defect at 12.

**`make_reduced_attack` (`attack=reduced`) — the thesis's static attacker.** Honest through warmup, then
every freeride round trains the reduced shard (or the full shard if `cpc<0`) via the honest embed path.
Re-embeds every round but with less data -> the Group-A/D non-separability results.

**`make_adaptive_tap_attack` (`attack=adaptive_tap`) — the submarine.** Inherits
`(_SimpleFRMixin, WatermarkClient)`, so a tap is the honest embed path — the mark it writes is
genuine. Freeride logic (`produce_update`, `clients.py:881-932`):
1. **Freeze η once** at defection via `_resolve_eta`: `oracle` -> the given `AUTOP_ORACLE_ETA` (0.264 here
   — an oracle gift; `self` -> μ+k·σ over the FR's own calib probe BERs, **unused in J/NOW**). `target = η
   − margin`.
2. **Probe the coast candidate, not the raw global** (line 916 — the key fix). `_coast_candidate`:
   - `resend` → the received global (mark decays to ~0.5 in one round → probe says tap → **degenerates to
     always-tap**);
   - `decay` → the FR's own last-tapped weights verbatim (mark never fades, but **byte-identical replay**);
   - `graft` → **fresh global body + frozen last-tapped mark head** (`_graft_keys` = the last 2 params
     under `scope=head`). The body tracks the global (no replay tell); the frozen head's projected bits
     degrade **gradually** as drifting features flow through giving the sawtooth pattern
3. **Decide:** `threshold` → tap iff `probe > target`; `always` → tap; `every_k` → tap on the period;
   a `max_coast` streak forces a tap.
4. **Tap** (`_do_tap`): freeze scope (`_freeze_scope`; `head`=2 params, `block`=8, `block2`=20, `full`=all
   → cheaper backward), train the reduced loader through the honest path, restore scope, store
   `_last_submit` (the graft head source), record `ber_after`. **Coast** (`_do_coast`): submit the coast
   candidate, meter `trained=False` (zero compute), **bypasses the memory update**.

### 5.5 `watermark.py` — the math
`smooth` (§4 step 2), `make_key`/`make_bits` (§4 key/bits; balanced=False → paper-faithful ±1),
`project_logits` (Eq. 13; drops the `exclude` column if set — here `None`), `watermark_loss` (BCE),
`extract_bits` (avg-then-sign, Eq. 15), `bit_error_rate` (Eq. 16), `detected(ber,η) = ber < η`, and
`calibrate_eta` (μ+3σ with a floor). Two documented degeneracies live here: `detected` notes that a
perfect mark gives μ=σ=0 → η=0 → *everyone* flagged (why balanced-key runs report FPR 1.0), and that any
η<1/m is the same detector. `SMOOTH_EPS` defaults to the legacy 1e-3 (env-switchable to 1e-8 for a clean
re-run — never mix within a family).

### 5.6 `server.py` — aggregation + round loop
`Aggregator.aggregate`: **sample-weighted FedAvg**, float params accumulated in float64 then cast back,
integer buffers (e.g. `num_batches_tracked`) copied not averaged. Under equal IID shards this equals the
paper's simple mean. `Server.run`: each round, collect every client's `produce_update` (passing the
current and previous global — the previous is what `previous_models` FRs need), run the `verify_hook`
**before** aggregating, then aggregate, evaluate test accuracy, and record. It also announces free-rider
phase transitions (honest→calib→tap/coast) for the timeline bands.

### 5.7 `wm_verify.py` — extraction, threshold, flagging
`make_verifier` returns a `verify_hook(server, round, updates)`:
- **Pass 1** — for each client load its submitted state, softmax the shared per-class trigger bank
  (`build_trigger_bank`, N_T=50 held-out **test** images), `extract_bits`, compute BER, and log per-class
  difficulty diagnostics (trigger-class accuracy, p_max, entropy, dominance).
- **Threshold** — η is the **frozen** `WM_ETA_FIXED` (`eta_source="fixed"`); the legacy live μ+3σ recalcs
  are dead-commented (a live threshold judging the round it's calibrated on is circular). `eta_source` is
  logged so a properly-frozen run is distinguishable from a silent `eta_floor` fallback.
- **Flagging** — `flag iff BER ≥ η`; emit FPR, FR recall, per-client rows, and honest BER percentiles
  (p90 = hard-class tail, max = worst honest client). Cost: `N·N_T = 500` forward passes/round, all
  `@torch.no_grad`.

*Trigger-bank variants exist for the capacity work:* `build_trigger_bank` (shared per-class, held-out —
the generalisation test used everywhere), `build_trigger_bank_per_client` (disjoint held-out slices),
and `build_trigger_bank_from_train` (verify on the client's **training** images = paper Table IX's
"trigger sample consistency" = memorisation, §6.3).

### 5.8 config → env → flag plumbing
`./submit_experiment.sh <CONFIG_IDX> <REPEAT>` → **always CONFIG_IDX 14** (`submarine_resnet18_cifar100`,
`config.py:208-213`), `seed = 1000 + REPEAT`. Config 14's own `attack="submarine"` is dead; every run
sets `ATTACK=` and knobs via env, mapped to `--flags` and applied through `_OVERRIDABLE`. All `tap_*`
fields default to inert values ignored by non-tap attacks, so adding the submarine can't affect other
runs.

---

## 6. The mechanisms behind the results

### 6.1 Class difficulty
Each client hides its mark in the softmax **tail** on its trigger class. A **flat** (high-entropy)
softmax has a rich, shapeable tail → the mark embeds cheaply → low BER (easy class). A **peaky**
(high-dominance) softmax has a structureless tail → bits become coin flips → high BER (hard class). IID
CIFAR-100 floors span **~0.00 (cls 7,8,9) → ~0.21 (cls 6)**; entropy/dominance track BER at |r|≈0.6–0.7
while **classification accuracy tracks it only weakly** (|r|≈0.05–0.4). That gap is the point: difficulty
is a property of the *output distribution shape*, not model quality — which kills the "your hard classes
are just badly-classified classes" objection. The free-rider hides under this honest ceiling.

### 6.2 `m` — the number of bits
`m·l = n`, so `m ≤ n`. Small m → coarse, lumpy BER; large m → shorter/noisier projections and more stuck
rows (`P(stuck)=2^(1−l)`). **CIFAR-100: m=10, l=10** (reliable bits, 0.2 % stuck, paper-reachable 99.9 %
wm-accuracy). **CIFAR-10: m=1 forced** — m=2 gives l=5 → 6.25 % stuck → caps below the paper's 99.72 %,
so the paper's own headline number requires a *single-bit* watermark (BER = one coin flip), itself a
finding.

### 6.3 More clients than classes (capacity / Table IX)
With N > n, clients share trigger classes. FareMark's Table IX (ResNet-18/CIFAR-10, 50 clients, 5/class)
reports high capacity — **but under trigger-sample consistency**: verification images are the *same
images the client trained on* (`build_trigger_bank_from_train`). That is **memorisation, not
generalisation** (the paper's own Table V says a mark fitted to specific samples "cannot be generalised").
Our F3 reproduces that row *and* a held-out twin; the **memorisation gap** = paper-mode − held-out
watermark-acc. Group F (200 clients, held-out banks) is the stricter test the paper avoids. Note the
paper never states its data partition (images/client, IID?), which alone makes it non-reproducible on
that axis (our R1 hit ~46 % vs the paper's 75 % without guessing the split).

### 6.4 Effort is warmup-dominated
A submarine's forced-honest warmup (11 rounds × full shard) dwarfs its cheap attack phase. J2 rep0:
overall FR effort **30.6 %**, but subtract the 275 000 warmup samples and the **attack-phase cost is
~4.2 %** of an honest client's. **Always report attack-phase-only effort** — the headline understates the
marginal saving. (Per-round tallies are in `result.json → compute.per_client[cid].per_round`.)

### 6.5 Smoothing `f`: power vs sin (the corrected C1)
`smoothing_gain` measures how much `f` amplifies a tail probability vs a peak one (gain 1 = does
nothing). **power@0.4 = 4.91**; **sin@0.4 = 1.01** (no smoothing — the old silent bug), and sin's best
monotone value **sin@π/2 = 1.23**. `watermark.py:smooth` now **validates** the sin branch and refuses the
degenerate regime, so C1 is no longer a crash — the finding is that **Eq. 9 (sin) is a structurally far
weaker smoother than Eq. 8 (power) at any α**, which is why sin runs sat at BER≈0.4. C1 is a low-priority
ablation, not a blocker.

### 6.6 The threshold sweep (why every rule fails)
All rules computed offline from the honest tail (`detection.py`). On A1 (6 seeds) the candidates span:

| rule | η | honest FPR | headroom | degenerate? |
|---|---|---|---|---|
| median + 3·MAD | 0.000 | 100 % | −0.59σ | yes (collapses; >½ honest at BER 0) |
| coded (paper, round-mean μ+3σ) | 0.084 | 31 % | +0.55σ | **yes** (< 1/m) |
| pooled μ+3σ | 0.108 | 10 % | +0.87σ | no |
| trimmed-10 % μ+3σ | 0.160 | 10 % | +1.57σ | no |
| honest p95 | 0.200 | 10 % | +2.12σ | no |
| adaptive σ-clip | 0.224 | 2 % | +2.45σ | no |
| **loose (per-client μ+3σ)** | **0.264** | 2 % | **+3.00σ** | no |
| honest p99 | 0.300 | 2 % | +3.48σ | no |

The paper's "coded" rule lands at +0.55σ (not 3σ) and is degenerate; the only true-3σ variant (**loose
0.264**) is so high it catches nothing. **The two frozen reference lines used on every timeline are η
tight = 0.064 (10-seed round-mean μ+3σ, degenerate) and η loose = 0.264 (per-client μ+3σ).** The tight
value wobbles ~30 % seed-to-seed from the key/bits lottery alone, which is exactly why it's frozen from a
10-seed honest calibration rather than recomputed live.

---

## 7. Related work — output-layer watermarking lineage

- **Uchida et al. (2017)** — first DNN watermark; bits in *weights* via a projected regulariser
  (white-box). The "project onto a pseudorandom key" template. [arxiv 1701.04082]
- **Adi et al. (2018)** — backdoor watermark, proved from *outputs* on triggers (black-box, single-bit).
  [arxiv 1802.04633]
- **BlackMarks (2019)** — first *multi-bit* black-box scheme; signature in the output-activation
  distribution. [arxiv 1904.00344]
- **Universal BlackMarks (IEEE SPL 2023)** — **FareMark's reader ancestor**: power-function-on-softmax +
  projection onto a pseudorandom key. FareMark's Eq. 8 + Eq. 10 moved into FL. [ieee 10025674]
- **FedIPR (TPAMI 2022)** — the FL free-rider angle: client-side secret marks that can *identify*
  free-riders. The "watermark ⇒ free-rider detection" claim FareMark inherits. [arxiv 2109.13236]
- **WAFFLE (SRDS 2021)** — server-side FL watermark; can't police free-riders (the FR's model is just the
  global), which is why detection falls on client-side schemes. [arxiv 2008.07298]
- **FedSMW (2024)** and kin reuse the same power-on-softmax + projection reader, so the peaky-softmax
  failure we document is a property of the **reader family**, not a FareMark-specific bug. [ieee 10827540]

**The gap we fill.** None of these evaluate an *adaptive, effort-minimising insider* that holds a valid
key and tunes its behaviour to sit under η; their free-rider is the crude Gaussian/previous-models
attacker (caught trivially). Our threat model (§1) and the adaptive-tap attacker (§5.4) are exactly that
missing evaluation.

---

## 8. Every setting used (canonical config)

Base = **CONFIG_IDX 14** (`submarine_resnet18_cifar100`), overridden per run by env.

| setting | value | code | why |
|---|---|---|---|
| model / dataset | ResNet-18 (32×32) / CIFAR-100 | `config.py:210`, `datasets.py:_load_raw` | paper's c100 row; reproduces embedding (~73 % acc) |
| num_clients N | 10 | `config.py:211` | one client per used trigger class → no forced sharing in the base |
| rounds | per group (base 50; **J/NOW = 40**) | `config.py:16`, `ROUNDS` env | 40 for the graft suite |
| local_epochs | **5** ⚠ | `config.py:17` | **paper uses 2** — a deliberate deviation; flag before any FareMark-table comparison |
| lr / batch / mom / wd | 0.01 / **16** / 0.9 / 5e-4, SGD | `config.py:18-21`, `datasets.py:116` | config defaults giving the reproduced accuracy; frozen for comparability |
| aggregation | sample-weighted FedAvg (= mean under equal IID) | `server.py:20-40` | |
| watermark / λ / β / α / f | on / 5.0 / 0.6 / 0.4 / power | `config.py:121-124` | paper Eq. 11/14/8 |
| m / l | **10 / 10** | `clients.py:220-222` | reliable bits, 0.2 % stuck, paper-reachable |
| 1/m | **0.10** | derived | any η < 0.10 is degenerate |
| N_T | 50 | `config.py:126` | paper fidelity; V sweeps 1/10/50 |
| wm_trigger_mode | `class` (shared held-out **test** bank) | `config.py:127`, `wm_verify.py` | strict generalisation test |
| wm_balanced_keys | **False** (paper-faithful ±1) | `config.py:115` | keeps the stuck-bit artifact; here `unembeddable_frac=0.0` anyway |
| exclude column | **None** (full 100-way softmax) | `clients.py:223` | trigger class not dropped |
| **attack** | none / reduced / **adaptive_tap** / previous_models / gaussian | `clients.py:308-352` | submarine/autopilot DISABLED |
| FREE_RIDER_IDS | **3,6** | `config.py:29`, `clients.py:226` | pins the two hard trigger classes; fixed across seeds |
| warmup W / calib K | **12 / 4** | `config.py:65-66` | honest 1–7, calib 8–11, defect 12 |
| η tight / loose (frozen refs) | **0.064 / 0.264** | `WM_ETA_FIXED`; refs in `plots.py` | degenerate / the real operating line |
| adaptive-tap decision η | `AUTOP_ORACLE_ETA` (0.264) else `WM_ETA_FIXED` | `clients.py:329` | the FR aims just under 0.264 |
| tap knobs (defaults) | `cpc=5, scope=full, coast=resend, when=threshold, margin=0.02, max_coast=999, probe_holdout=16` | `config.py:95-110` | J2/J5 override scope=head, coast=graft, and the margin/coast/holdout trio |

**Same-class controls:** `TRIGGER_CLASS_MAP="0:6"` pins a FR onto an honest client's class; `WM_KEY_TWINS="0:6"`
also hands it that client's key+message (AK = effort-only isolation).

**Throughput (all statistically identical to the slow path):** batch-16 ResNet-18 barely uses an A100, so
speed is **concurrency** — `PODS×WORKERS=2×6` runs/GPU under CUDA MPS (`MPS=1`), `FAST_DATA=1`
(GPU-resident loaders, not a data reduction), `DETERMINISM=0` (cuDNN autotuner, ~1.3–2×). `runbook.sh:28-36`.

---

## 9. Experiment plan & status

**Legend:** ✅ done (3 seeds) · ◑ done (1 seed, shape only) · ▶ running · ⏳ to run · ❌ deprioritised.

| # | st | proves | family(ies) | run | notes |
|---|----|--------|-------------|-----|-------|
| A1 | ✅ | class-difficulty floors; η calibration; η seed-instability | `A1_honest_c100` ×6 | `run_now.sh A` | floors 0.00→0.21; η tight wobbles ~30 % |
| A2 | ✅ | non-sep at EASY classes (FR cleaner than honest) | `A2_reduced_c100_c17` ×3 | ″ | classes 1,7; FR→0.00 |
| A3 | ✅ | non-sep at HARD classes; catch only by flagging honest | `A3_reduced_c100_c36` ×3 | ″ | c3 inseparable; c6 catchable at 40 % FPR |
| A4 | ✅ | same class, own key | `A4_sameclass_c100_c6` ×3 | ″ | FR ≈ 0.067 ≤ honest floor 0.114 |
| AK | ✅ | same class + SAME key (effort-only) | `AK_sameclass_samekey_c6` ×3 | ″ | FR indistinguishable from honest twin |
| D1 | ✅ | price-of-invisibility (+N spectrum) | `D1_reduced_c100_c36_n{-1,0,1,2,5,10,25,50}` ×3 | `run_now.sh D` | +1/class already at plateau; trigger-only caught |
| E1 | ✅ | non-IID honest floor (skew widens BER) | `E1_honest_niid_c100` ×3 | `run_now.sh E` | span 0.007→0.255; coded η 0.150 @ 24 % FPR |
| E2 | ✅ | non-sep under non-IID (per-class OVL/best-error table done) | `E2_reduced_niid_c36` ×3 | ″ | FR 0.18–0.20 inside honest band 0.17–0.26; `E2_niid_sep.json` @3-seed: per-class best-error 0.50 |
| E3 | ✅ | α severity sweep {0.1, 1.0} | `E3_{honest,reduced}_niid_*_{a01,a10}` ×3 | ″ | more skew → wider floors; FR stays in own-class band (α=0.3 dropped: ≡α=0.1) |
| I_* | ◑ | adaptive-tap, one knob at a time | `I_<knob>_<val>_c36` ×1 | `BATCH=I ./runbook.sh` | effort dial + `when=always` evade; `resend` can't save compute |
| J0–J4 | ◑ | **graft-coast suite** (gate/persistence/sawtooth/coast A-B/scope) | `J{0..4}_*_c36` ×1 (seed 1000) | `BATCH=J` | **graft shipped; J2 = confirmed submarine** |
| **NOW** | ▶ | **first 3-seed submarine** (J2×3 + J5×3) | `J2_saw_graft_head_c36`, `J5_submarine_head_c36` ×3 | `BATCH=NOW ./runbook.sh manifest submit` | J2 rep0 done; J5 rep0 partial. See RESULTS_INDEX → NOW |
| H5 | ⏳ | crude previous-models FR on c100 IS caught (positive control) | `H5_prevmodel_c100` ×3 | `run_now.sh H` | feeds `operating_point` |
| V2 | ⏳ | Table V: few-trigger-sample FR overfits → caught | `V2_tableV_attack_c36_tn*` ×3 | `BATCH=V` | attack-side complement to D's trigger-only |
| F1/F2 | ⏳ | capacity (200 clients), forced sharing | `F1_honest_nc200`, `F2_reduced_nc200_c67` ×2–3 | `run_now.sh F` | most expensive per run |
| F3 | ⏳ | Table IX repro (`client_train`) + held-out twin → memorisation gap | `F3_tableIX_c10_nc50` ×3 | `PAPER_OK=1 BATCH=F` | thesis-fidelity, lowest priority |
| C1 | ❌ low | sin vs power smoothing | `C1_honest_sin_c100` ×3 | `run_now.sh C` | unblocked; sin is just a weak smoother (§6.5) |

**Priority order:** (1) finish **NOW** at 3 seeds + the attack-phase-effort metric + resolve the R33–40
tail instability; (2) **H5** (completes `operating_point`); (3) **V2**; (4) **F**; (5) **C1** last.

**Findings at a glance:** the empirical record — Groups A/D/E/I/J/NOW, the seed table, and the compute
numbers — lives in **RESULTS_INDEX.md**. This file owns setup, code, and plan; that file owns results.

---

## 10. Completion status 

### 10.1 Where the project actually is

**The thesis (detection is impossible) is DONE and defended — 3 seeds, table-ready.**
Groups A + D + E are the whole argument and they are complete at 3 seeds:
- A/D (IID): no deployable η separates honest from reduced free-rider; the FR is *cleaner* at easy
  classes and catchable at hard classes only at ~40 % honest FPR. `operating_point` shows recall ≤ 0.17
  at any usable FPR.
- E (non-IID, 3 seeds, plots in `figs_2/`): skew makes it **worse**, not better — kills the "real
  heterogeneous FL would separate them" rebuttal.
- The only loose end on the thesis side is the **H5 positive-control bar** (crude FR caught at ~1.0),
  which makes the money-plot contrast complete. (The E2 per-class OVL/best-error table is now done.)

**The constructive attack (a working submarine) is PARTIAL — a proof-of-concept with a crutch.**
- The **reduced attacker** (the thesis's static free-rider) is complete and is what all of A/D/E use.
- The **adaptive submarine** (`adaptive_tap`, graft/head/threshold) *exists and half-works*: on the easy
  class it genuinely coasts (J2 cid3 ~14 % taps, J5 cid3 ~7 %); on the hard class it does not (see 10.2).
  It is 1 seed (J2) / partial (J5) and leans on an **oracle η** (10.4). It is **not yet a complete,
  self-sufficient attack** and **not yet tabled**.

### 10.2 Is the Group-J attack working? — honest answer

**On the easy class: yes.** cid3 coasts through most of the attack phase on a graft mark that fades
gradually, submissions track the global (no replay tell), and its BER stays under η_loose.

**On the hard class: it evades, but saves no compute** (corrected after separating the FR's self-probe
from the server's BER — see RESULTS_INDEX → NOW §4.3 and the `tap_perfr` plot):
- cid6 **taps 19/29 freeride rounds (~66 %)** — nearly the full reduced-attack cost, *no* saving.
- **Why:** its 16-image self-probe over-reads (**~0.30** every round) while the **server** measures cid6
  at **~0.217**. Probe 0.30 > target 0.234 → the threshold fires every round. The taps *do* work at the
  server (they hold cid6 at ~0.217); the FR just can't tell, because its probe is noisy/pessimistic on the
  hard class. This is a **probe-calibration** problem, not a caught mark.
- **At the server, cid6 evades:** tail BER 0.217 < η_loose 0.264. That 0.217 is exactly the plain
  reduced-attack value for class 6 (A3/D1: FR ≈ 0.22), so on the hard class the submarine **≡ the reduced
  attack in both cost and BER** — and catching it there still needs ~40 % honest FPR.

So the submarine is a real, cheap, hidden attacker **on easy classes**; on hard classes it evades too but
collapses to the reduced attack's cost (no saving). That's the finding to state plainly — and it does not dent
the thesis (to hide on a hard class you must do enough real work to be indistinguishable from honest).

### 10.3 Exact compute-cost breakdown (J2 rep0, seed 1000, 40 rounds)

Numbers straight from `result.json → compute` (verified against the trace):

| quantity | honest client | free-rider (mean) | cid3 (easy) | cid6 (hard) |
|---|---|---|---|---|
| sample-passes, whole run | 1,000,000 | 305,600 | 285,660 | 325,540 |
| effort ratio vs honest | 100 % | **30.6 %** | 28.6 % | 32.6 % |
| **warmup** (R1–11, forced-honest full shard) | — | 275,000 (both) | 275,000 | 275,000 |
| **attack-phase** passes (R12–40) | 725,000 | 30,600 | 10,660 | 50,540 |
| **attack-phase effort** vs honest | 100 % | **~4 %** | **1.5 %** | **7.2 %** |
| taps / freeride rounds | — | — | 4/29 (14 %) | 19/29 (66 %) |
| duty_cycle (incl. warmup) | 1.0 | — | 0.375 | 0.75 |

Per-round unit costs (batch 16, 5 epochs — full derivation in §1.1 of RESULTS_INDEX → NOW):
- honest / warmup round = 5000 imgs → **1565 SGD steps**, 25,000 passes.
- one cpc=5 head tap = ~533 imgs → **170 steps** on 2 params, 2,665 passes (**10.7 %** of a round).
- a coast = **0**.

**The math, exactly.** The compute meter records **image-passes**: `record_batch(len(x))` summed over
every batch and epoch (`drop_last=False` ⇒ each image seen once/epoch), so `passes = images × epochs`.
- **Honest:** `5000 × 5 = 25,000`/round ⇒ `40 × 25,000 = 1,000,000` (matches `honest_mean_samples`).
- **Free-rider:** `reduced_n = 533` (confirmed = 38 trigger-train + 5×99 common), so
  `FR total = 275,000 (11 warmup rounds × 25,000) + n_tap × (533 × 5) + n_coast × 0 = 275,000 + n_tap·2665`.
  Check (rep0): cid3 `275,000 + 4·2665 = 285,660` ✓; cid6 `275,000 + 19·2665 = 325,635 ≈ 325,540` ✓.
- **Overall effort** = `fr_mean_samples / honest_mean_samples = 305,600 / 1,000,000 = 30.6 %` — but 275k of
  that is warmup.
- **Attack-phase effort** (the number to quote) divides only post-defection work by an honest client's
  post-defection work. Honest attack phase = `29 × 25,000 = 725,000`. One tap = `533/5000 = 10.7 %` of one
  honest round (epochs cancel). So:
  $$\text{attack-phase effort} = \frac{n_{\text{tap}}\times 2665}{29\times 25000} = \text{tap-fraction}\times\frac{533}{5000} = \text{tap-fraction}\times 10.7\%.$$
  cid3 (14 % taps) → 1.5 %; cid6 (66 % taps) → 7.0 %; 3-seed cid3 10 % → ~1.1 %, cid6 43 % → ~4.6 %.
- **Why `effort_ratio_gpu` ≠ `effort_ratio_samples` for J4 (block2):** 0.36 vs 0.32. Same image count, but
  a block2 tap backprops through **20 params (deeper)** vs head's 2 → more GPU-ms *per image*. For J2/head
  the two ratios match (0.305/0.306) because the head backward pass is cheap. "36 % GPU vs 30 %" is the
  **depth of the tap's backward pass**, not more data. (See §11 for what "20 params / 2 params" means as
  actual layers, and §12 for the graft mechanism the coasts use.)

**How to quote it:** lead with **~4 % attack-phase effort** (the real marginal cost), then note the
**30.6 % headline is warmup-dominated** (90 % of FR compute is the 11 forced-honest rounds). And flag
that **the saving is asymmetric** — cid3 1.5 % vs cid6 7.2 %, because cid6 taps constantly for nothing.
The `compute.per_client[cid].per_round` field already has the per-round tallies for an attack-phase-only
inset — that inset is the missing piece that makes the win legible.

### 10.4 What is hardcoded and must change for a *complete* attack

The current submarine works only because it is *handed* things a real attacker wouldn't have, and because
two safety knobs are fixed constants rather than derived. To claim a complete, self-sufficient attack,
remove these — in priority order:

1. **Oracle η → self-estimated η (the biggest crutch).** J2/J5 set `TAP_ETA_SOURCE=oracle` +
   `AUTOP_ORACLE_ETA=0.264`, i.e. the FR is *told* the true loose threshold. A real server never
   publishes η. The self path already exists — `_resolve_eta` (`clients.py:794-800`) computes
   `μ + k·σ` over the FR's own calibration-window probe BERs — but it is **untested at scale**. **Do:**
   run `TAP_ETA_SOURCE=self`, verify the FR's estimate lands near 0.264 from its own 4-round window and
   still evades. Until this passes, every evasion claim carries an asterisk.
2. **Fixed margin / safety → derived from estimation uncertainty.** `tap_margin` (0.03/0.05) and the
   disabled submarine's `autop_safety` are hand-tuned constants. The config even flags it:
   `autop_margin0` / `autop_safety` carry *"TODO hardcoded guard: should be DERIVED from estimation
   uncertainty, not fixed"* (`config.py:76-77`). **Do:** set `target = η̂ − k·σ(η̂)` where `σ(η̂)` comes
   from the calib-window spread, so the margin is principled, not a magic number per class.
3. **Fixed warmup schedule → dynamic (convergence-based).** `AUTOP_HONEST_UNTIL=12` / `CALIB_ROUNDS=4`
   are fixed; a real submarine on a hard, slow-converging class should warm up **longer** and defect when
   its *own* probe converges. The dynamic machinery exists but is **commented out** (the disabled
   `SubmarineFreeRider`: `autop_warmup_mode="dynamic"`, `autop_conv_eps/patience`, `clients.py:937-960`,
   `config.py:54-64`). **Do:** revive it, or accept fixed warmup as a stated limitation.
4. **The self-probe/server gap on the hard class.** Not a constant but a design gap: the FR probes on
   `holdout` images (16→48, capped to ~25 by `MIN_TRAIN_TRIG=8`, `clients.py:552-560`) while the server
   uses 50. On cid6 the probe under-reads, so the FR coasts (or taps uselessly) on optimistic reads.
   **Do:** either close the gap (probe on ≥ the server's N_T with generalisation, not memorisation) or
   accept that hard classes are un-submarine-able and scope the claim to easy/medium classes.

*Lower-stakes hardcodes (fine to leave, but know they're there):* `PF_GROUP=10` (m = n//10,
`clients.py:220`), `MIN_TRAIN_TRIG=8`, `SMOOTH_EPS=1e-3` legacy default (`watermark.py:62`), the fixed
free-rider set `3,6` (comparability, not a real attacker constraint), and the Gaussian-FR's hardcoded
seed (`clients.py:450`).

### 10.5 If you have limited time left — do this, in this order

1. **Lock the thesis (½ day).** Run **H5** (3 seeds) so `operating_point` shows the ~1.0 positive-control
   bar next to the ≤ 0.17 insider bars. (E2 per-class separability is already done.) The negative result is
   then fully closed and defensible.
2. **Finish + honestly scope the submarine (1–2 days).** Re-run **NOW** (J2×3 + J5×3) to completion
   (delete stale rep dirs first), add the **attack-phase-only effort inset**, fix `tap_dynamics` to plot
   **per-cid** (not majority), and write the verdict as *"a genuine low-effort submarine on easy/medium
   classes (~4 % attack-phase compute); on hard classes it collapses to an expensive, detectable reduced
   attack"*. That is a true, publishable claim **without** removing the oracle crutch.
3. **Only if time remains: attempt the self-η run (10.4 #1).** One `TAP_ETA_SOURCE=self` family at 3
   seeds converts the proof-of-concept into a self-sufficient attack. If it doesn't land, document it as
   the clear next step — the thesis does not depend on it.
4. **Do not start** F/C1 or the dynamic-warmup revival unless everything above is done; they are
   fidelity/ablation, not core.

**What to hand over:** this file (setup + code + plan + this wrap-up), RESULTS_INDEX (findings + NOW +
seeds + compute), and the one-paragraph honest verdict in 10.2/10.5#2. That is a complete, defensible
story: *output-layer watermarking cannot separate honest from free-rider at any usable threshold, and a
low-effort adaptive free-rider provably hides on the classes where hiding is possible.*

### 10.6 The best config right now — keep J2, don't rerun J5 as-is

**Winner: `J2_saw_graft_head_c36` (margin 0.03).** It is the configuration to build on and to quote.
Exact knobs (from `run_now.sh:411-420`, all on top of CONFIG_IDX 14):

```
ATTACK=adaptive_tap        FREE_RIDER_IDS=3,6        ROUNDS=40
AUTOP_HONEST_UNTIL=12      AUTOP_CALIB_ROUNDS=4      (warmup 1–11, defect 12)
TAP_WHEN=threshold         TAP_ETA_SOURCE=oracle     AUTOP_ORACLE_ETA=0.264   WM_ETA_FIXED=0.064
TAP_COAST_MODE=graft       TAP_SCOPE=head            TAP_DATA_CPC=5
TAP_MARGIN=0.03            TAP_MAX_COAST=12          TAP_PROBE_HOLDOUT=16
```

Why this is the best working point:
- **graft + head + cpc=5** is the only coast that both evades and keeps submissions live (resend
  degenerates to always-tap; decay is a replay). Confirmed in Group J.
- **Both free-riders evade at the server** (cid3 tail 0.190, cid6 0.217, both < η_loose 0.264).
- **Real compute win on the easy class** (cid3 ~14 % tap fraction ≈ 1.5 % attack-phase effort); the hard
  class costs ~66 % but is no worse than the reduced attack.
- The one thing J2 leans on is the **oracle η** (§10.4 #1) — the remaining crutch, not a knob to retune.

**J5 verdict — do not rerun `J5_submarine_head_c36` as-is.** Its levers were `TAP_MARGIN=0.05`,
`TAP_MAX_COAST=20`, `TAP_PROBE_HOLDOUT=48`. The logic backfires: a **bigger margin lowers the target to
0.214**, which makes the (already pessimistic) hard-class self-probe fire *more*, so cid6 still taps
~75 %; and the bigger holdout didn't fix cid6's over-reading. Net effect over J2: cid3 coasts a little
more (~7 % vs 14 %), cid6 unchanged, **plus** the run hit the R33–35 instability and crashed at R39. So
J5's specific knob combo is not an improvement and isn't worth re-running. **If you want the hard-class
saving, the right lever is a better *probe* (match the server: probe on ≥ N_T=50 held-out images, or
smooth/average the probe), *without* enlarging the margin** — that attacks the actual cause (probe
over-reads at 0.30 vs server 0.217). That's a one-line change to `_prepare`'s holdout handling plus a
`TAP_MARGIN=0.03`, not a new family. Everything else worth trying is in §10.4.

### 10.7 The fixed Group-J plot (per-free-rider) — how to run it

The old `tap_dynamics` single-family plot showed only the **first** free-rider of the **first** seed
while titling it with the two-FR average — which is why J looked like "all coast" (it was drawing cid3).
The new **`tap_perfr`** command in `scripts/plots.py` draws **one panel per free-rider cid**, each with:
its **server-measured BER** (the ground truth that gets flagged), its **self-probe** (`ber_before`, what
drives the tap/coast decision — so the probe-vs-server gap is visible), **tap ▼ / coast ▢ markers** per
round, the warmup/calib/free-ride bands, and the η_tight / η_loose / target lines. It aggregates over
seeds and prints a per-cid table (attack-phase tap-fraction, tail server-BER, evade/caught verdict).

Run it (local, after `fetch`; `$ALL` = the results glob, `$OUT` = figs dir):

```bash
# single family (what you want for J2 / J5):
python scripts/plots.py tap_perfr --in 'results/*/result.json' \
    --family J2_saw_graft_head_c36 --out figs/tap_perfr_J2
python scripts/plots.py tap_perfr --in 'results/*/result.json' \
    --family J5_submarine_head_c36 --out figs/tap_perfr_J5     # once/if J5 has a result.json

# via the runbook plot phase (add next to the tap_dynamics loop, runbook.sh ~L200):
for fam in J2_saw_graft_head_c36 J5_submarine_head_c36; do
  run "$PL tap_perfr --in '$ALL' --family $fam --out $OUT/tap_perfr_${fam}"
done
```

Files to run it on: any completed `result.json` from an `adaptive_tap` family — i.e. **J2 now** (its
`result.json` exists), and **J5 only once it finishes** (currently it has none, so `tap_perfr` will skip
it). Optional override flags: `--eta_tight`/`--eta_loose` (default 0.064 / 0.264). The companion
`tap_perfr_<fam>.md` is the paste-ready per-cid table.

**Depends on:** `scripts/plotstyle.py`, `scripts/detection.py` (for `calib_window`), `scripts/resultio.py`
— all already in `scripts/`. No new dependencies.

### 10.8 Submarine knobs — what each does and how varying it changes the result

All are `tap_*` fields on CONFIG_IDX 14, set per family in `run_now.sh`. Evidence = Groups I (single-knob
sweep) and J (graft suite) + the J2/J4 traces.

| knob | values (default) | what it controls | ↑ increase → | ↓ decrease → | best setting |
|---|---|---|---|---|---|
| `tap_coast_mode` | resend / decay / **graft** | how a coast round is built | — | — | **graft** — only mode that evades *and* isn't a replay (resend → always-tap; decay → byte-identical replay, caught on every coast) |
| `tap_scope` | head(2) / block(8) / block2(20) / full | how many params a tap unfreezes | deeper/cleaner re-embed (block2 → BER 0.0 each tap, crisp sawtooth) **but** more GPU/tap **and** mark fades faster → **more taps** | shallower re-embed (head sometimes 0.2→0.2, can't fix hard class) but **cheapest tap, fewest taps** | **head** for cheapest stealth; **block2** for the clean demonstrable sawtooth (J4) |
| `tap_data_cpc` | 0 / 1 / 2 / **5** / 25 / −1 | images per common class in the reduced tap set (0 = trigger-only, −1 = full shard) | more data/tap → cleaner embed, higher cost | ≤1 = cheaper but 0 (trigger-only) **overfits → caught** (Table V / D1) | **5** (≈31 % data; sweet spot). Try 1–2 to halve tap cost if it still embeds |
| `tap_margin` | **0.03** (J2) / 0.05 (J5) | safety gap below η: `target = η − margin` | lowers target → FR taps **more** (J5's mistake on the hard class) | raises target → coasts longer, risks drifting over η | **0.03**. Should be *derived* from probe σ, not fixed (§10.4 #2) |
| `tap_max_coast` | **12** (J2) / 20 (J5) | forced tap after this many coasts | longer coasts allowed → fewer forced taps (helps easy class) | more forced taps (safer, costlier) | **12**; only helps the class that can already coast (cid3) |
| `tap_probe_holdout` | **16** (J2) / 48 (J5) | held-out trigger images for the self-probe | bigger = less noisy probe → fewer *unnecessary* taps on the hard class | small (16) → probe over-reads on cid6 (0.30 vs server 0.22) → taps every round for nothing | **as large as the class allows** (match server N_T=50) — this is the real fix for cid6, *without* raising margin |
| `tap_when` | **threshold** / always / every_k | tap decision rule | every_k: predictable, caught on off-beats; always = the reduced attack | — | **threshold** (adaptive), once the probe is trustworthy |
| `tap_eta_source` | **oracle** / self | where the FR gets η | oracle = *given* 0.264 (a crutch) | self = μ+k·σ from own calib probes (untested) | **must become `self`** for a complete attack (§10.4 #1) |

**One-line reading:** `graft` + `head` + `cpc=5` + `threshold` + `margin 0.03` is the cheapest evading
config (**= J2, the winner, §10.6**). `block2` (J4) buys a prettier, deeper sawtooth and slightly lower
BER at ~2× the tap cost. The **hard class (cid6) is limited by `tap_probe_holdout`, not by any of the
other knobs** — a bigger/server-matched probe is the lever that would let it coast; raising `tap_margin`
(J5) makes it worse.
---

## 11. The model, its layers, and what a tap actually trains (ML background)

*Written for someone who wants to understand exactly what "train only the final layers" means. If you
know CNNs, skip to 11.3.*

### 11.1 What a neural network is, in one paragraph
A classifier is a stack of **layers**. Each layer is a function with tunable numbers (**parameters** /
"weights") that transforms its input into a more useful representation. Early layers turn raw pixels into
simple features (edges, textures); later layers combine those into complex features (shapes, object
parts); the very last layer turns the final feature vector into one score per class. **Training** =
feeding labelled images, measuring a **loss** (how wrong the scores are), and nudging every parameter a
little in the direction that lowers the loss (**gradient descent**, one **step** per minibatch). To
**freeze** a layer is to *not* nudge its parameters (`requires_grad_(False)`), so only the un-frozen
layers learn that round.

### 11.2 ResNet-18 specifically (what we use)
ResNet-18 is a convolutional network with **~11 M parameters** grouped into **62 parameter tensors**
(that's what `named_parameters()` returns, in order):

```
conv1, bn1                      # stem: first 3×3 conv on the 3×32×32 image  (tensors 0–2)
layer1  (2 residual blocks)     # 64 channels                                (tensors 3–14)
layer2  (2 residual blocks)     # 128 channels                              (tensors 15–29)
layer3  (2 residual blocks)     # 256 channels                              (tensors 30–44)
layer4  (2 residual blocks)     # 512 channels  ← last feature stage        (tensors 45–59)
── global average pool ──       # 512-D FEATURE VECTOR  z   (no parameters)
fc = Linear(512 → 100)          # the CLASSIFIER HEAD → 100 logits           (tensors 60–61)
                                #   softmax(logits) = the 100 probabilities
```

The convolutional stack (`conv1 … layer4`) is the **feature extractor** ("the body"): it maps an image to
a 512-dimensional **feature vector `z`**. The final `fc` layer (the **"head"**) is a single linear map
`logits = W_fc · z + b_fc` (`W_fc` is 100×512, `b_fc` is 100), and `softmax(logits)` gives the class
probabilities. **The watermark is read only from those softmax probabilities on trigger-class images**
(§4), so the watermark is entirely a property of `fc` applied to `z`.

### 11.3 What `tap_scope` freezes — "the final layers" in exact terms
`_SCOPE_KEEP = {"head": 2, "block": 8, "block2": 20, "full": None}` (`clients.py:744`). A tap freezes all
but the **last `keep` parameter tensors** (`_freeze_scope`, `clients.py:778-787`). Mapped onto the list
above:

| scope | keeps (unfreezes) | which layers | ~params trained | what you're "putting in" |
|---|---|---|---|---|
| `head` | last **2** | `fc.weight`, `fc.bias` only | ~51 K | re-tune **only the linear readout** `z → logits`; features `z` untouched |
| `block` | last **8** | `fc` + the **last residual block** of layer4 | ~2.4 M | readout + a little late-feature reshaping |
| `block2` | last **20** | `fc` + **all of layer4** (the last 512-channel stage) | ~8.9 M | readout + the whole final feature stage |
| `full` | all 62 | the entire network | ~11 M | a normal honest training step |

So **"train only the final layers" = update only `fc` (head), or `fc` + layer4 (block2), and leave the
rest of the feature extractor exactly as the current global model has it.** Because the watermark lives in
`softmax(fc(z))`, `head` is the *minimal* set of parameters that can move the mark — you are re-fitting the
100×512 matrix that turns fixed features into class scores so that the projected bits line back up with
your key. That's why a head tap is cheap (≈51 K params, 170 SGD steps) yet can re-embed the mark. `block2`
also lets the late features `z` themselves shift, which re-embeds *more strongly* (BER→0 every tap → the
crisp J4 sawtooth) but costs ~2× because the backward pass runs through millions more parameters (§10.3's
GPU-vs-samples gap).

**Key intuition for the attack:** the free-rider never has to retrain the expensive feature extractor —
the aggregated global model gives it good features `z` for free every round; it only occasionally re-tunes
the cheap readout `fc` to keep its watermark alive. That is the whole economic basis of the submarine.

---

## 12. Deep dive: the `graft` coast — code + theory

*The persistence mechanism that lets the free-rider submit a mark-carrying model without training.*

### 12.1 The code path
Each free-ride round (`produce_update`, `clients.py:904-932`) the FR **probes the model it *would*
submit**, then taps or coasts:

```python
ber_now = self._probe_ber(self._coast_candidate(global_state))   # probe the COAST candidate, not the raw global
tap = (ber_now is None) or (ber_now > target)                    # target = η − margin
return self._do_tap(...) if tap else self._do_coast(..., ber_now)
```

The graft candidate (`_coast_candidate`, `clients.py:837-858`):

```python
# graft: fresh global body + FR's frozen last-tapped mark head
out = {k: v.clone() for k, v in global_state.items()}   # start from THIS round's global (body + head)
for k in self._graft_keys():                            # the mark-carrying head param names
    if k in self._last_submit:
        out[k] = self._last_submit[k].clone()           # overwrite ONLY the head with the last-tapped head
return out
```

- `_graft_keys()` (`clients.py:828-835`) = the **last `keep` parameter names** — for `scope=head`, the two
  names `fc.weight`, `fc.bias`.
- `_last_submit` (`clients.py:813`) = the full state the FR uploaded on its **last tap** (its freshly
  embedded mark).
- `_do_coast` (`clients.py:861-872`) submits this candidate and calls the meter with `trained=False` →
  **0 image-passes** recorded.

Net: a graft coast uploads `{ body ← this round's fresh global ; fc.weight, fc.bias ← FR's last-tapped
head }`, with no training.

### 12.2 The theory (mathematically)
The watermark bit is `b̂_k = sign( Σ_j f(p_j)·M_kj )` with `p = softmax(W_fc·z + b_fc)`, `f(p)=p^0.4`
(§4). Grafting **freezes `(W_fc, b_fc)`** at the mark-carrying values and lets **only `z`** change (the body
tracks the fresh global). So the *only* thing moving the watermark on a coast round is the feature drift:

- At tap round `t₀`, features `z_{t₀}` gave projections `z_k(t₀) = Σ_j f(p_j)·M_kj` with the correct signs
  → BER ≈ 0.
- On coast round `t`, the body has drifted (other clients trained it, FedAvg averaged it), so
  `z_t ≈ z_{t₀} + Δ`, with `‖Δ‖` growing roughly linearly in `(t − t₀)` (FedAvg averages small updates).
  The frozen head maps the drifted features; a bit **flips only when its projection `z_k(t)` crosses zero**,
  i.e. after drift `≈ |z_k(t₀)| / ‖∇_z z_k‖`.
- **Easy class:** projections sit far from zero (large `|z_k|`) → many rounds of drift to flip a bit →
  **slow fade, few taps** (cid3 ≈ 10 %). **Hard class:** `|z_k|` small → tiny drift flips bits → **fast
  fade, frequent taps** (cid6 ≈ 43 %). That margin *is* the sawtooth period.

### 12.3 Why graft and not the other two coast modes
- `resend` → submit the raw global (no frozen head): the mark washes out in **one** FedAvg step →
  probe reads ~0.5 → the FR always-taps (no saving). This is the Group-I `resend` failure.
- `decay` → submit the **whole** last-tapped model frozen: the mark never fades, **but every submission is
  byte-identical** → a trivially detectable replay/staleness tell.
- `graft` → **body moves every round (tracks the global, no staleness tell) while the tiny frozen head
  keeps the mark alive.** The server never compares weights round-to-round; it only reads the watermark —
  which the frozen head still carries. This is the only mode that both **evades** *and* looks **live**.
- Pairing with `scope=head` is self-consistent: the tap trains exactly the params graft freezes, so
  `_last_submit`'s head *is* the freshly embedded mark. (If a tap trains `full`, `_graft_keys` falls back
  to the last 2 params so the body is still left free to follow the global — `clients.py:833`.)

---

## 13. Is this a real free-rider attack? — the paper framing

**Your stated idea:** *"always train reduced (some form of free-riding); coast when possible (best method
= keeps the mark longest); tap when needed (train the reduced set, best scope, estimated threshold); back
to coasting."*

**Verdict: yes, this is the right idea and it is paper-worthy — with two things made explicit.** It is a
genuine, novel *attack* on watermark-based free-rider **detection**, not just the static reduced attack:
the adaptive coast/tap loop **maintains a valid watermark while training on <5 % of the honest workload
in the attack phase**, and the graft coast makes each submission *live* (not a replay). What turns the
idea into a defensible paper:

1. **Estimate the threshold yourself — this is the contribution, and your idea already flags it (`?`).**
   Right now η is handed to the FR (oracle). A real attacker never sees η. Build the `eta_source="self"`
   path (μ + k·σ over the FR's own calibration-window probe BERs, `clients.py:794-800`) and show the FR's
   self-estimate lands near the true 0.264 and still evades. Until this works the attack has a crutch
   (§10.4 #1); once it works, the attack is **self-sufficient** — the paper's headline.
2. **State the threat model crisply.** The attacker (a) is a legitimate participant, so it legitimately
   holds its own key/bits; (b) knows the *form* of the detector (μ+3σ over honest BER — a reasonable
   published assumption) but **not** the exact η; (c) sees the global model each round (standard FL). Under
   these assumptions it evades detection at a fraction of honest cost. That is a clean, reviewable claim.

**Two honesty requirements a reviewer will check (you already have the answers):**
- **"Isn't this just the reduced attack with a schedule?"** No — on easy/medium classes it trains ~10 %
  of rounds (vs the reduced attack's 100 %) and still evades; the graft coast is a distinct, non-replay
  mechanism. *But* on **hard classes it collapses to the reduced attack** (cid6 ~43–66 % taps, no saving)
  because the noisy self-probe over-reads. Report this plainly — the attack still *evades* on hard classes,
  it just doesn't *save* there — and it doesn't weaken the paper.
- **Position against FareMark's own attacker tests.** FareMark shows the *train-then-attack* free-rider
  (Table IV) and the *few-trigger-sample* free-rider (Table V) are caught. Your submarine is different:
  it **adaptively re-embeds on a schedule tied to an estimated threshold**, which is exactly the case the
  paper does not test. That contrast is your novelty statement.

**The strongest paper is two-sided** (and you have both halves):
- **(Negative result)** output-layer BER is not a valid free-rider detector — per-class best-balanced-error
  = 0.50, the free-rider is often *cleaner* than honest, the calibrated η is degenerate, and separability
  is a pooling illusion (Groups A/D/E).
- **(Constructive attack)** even taking the detector at face value, an adaptive free-rider (the submarine)
  evades it at ≈1.5 % attack-phase compute on easy classes, with a self-estimated threshold.
- **Together:** watermark-based free-rider detection is both **theoretically unsound** and **practically
  evadable**.

**One refinement to your wording:** "always train reduced" is cleaner than the current honest-full warmup
(it means the attacker never does a full honest step), but keep an early **embedding phase** — the mark
must be *present* before you start coasting, or you're caught immediately. So: *reduced-train until the
mark is embedded and you've estimated η, then coast, tapping (best scope, `graft` coast, self-estimated
target) only when your probe says the mark is fading.* That is the complete submarine.