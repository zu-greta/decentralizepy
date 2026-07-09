# STATUS 

Single source of truth for where the free-rider project stands. Results are
**preliminary**; the `run_full_sweep.sh` batch (incl. autopilot + re-embed) will
refine them. Read `wm_fr_ber` (below η = evades) and `final_acc` (~72 = healthy,
low = poisoned), NOT `wm_fr_recall` (η swings, so recall is noisy).

---

## 1. One-paragraph status

Part 1 (reproduce FareMark) done. Part 2 — **we have a working attack.** The
autopilot free-rider, run on CIFAR-100 with the fair frozen threshold, has a
**confirmed winner: `autop_scope=block`** (re-embed the last block only, not the
whole model and not just the head): **free-rider BER 0.075 < fair η ≈ 0.097
(evades ~50% of converged rounds), final accuracy 72.0% (healthy), at 18% of
honest effort.** It is the sweet spot on the scope curve — `head` fails (BER 0.42,
the mark won't generalise), `full` is caught (BER 0.155 > η) and costs about the
same, `block` wins. The honest framing: the watermark is a *periodically
re-embeddable* property, so a client that trains fully but only on the last block
and only intermittently rides under a per-round detector at ~1/5 honest cost. Two
caveats to carry: (a) single-seed — confirm `block` at 3 seeds; (b) CIFAR-10 is
degenerate (only ~2 watermark bits → honest BER itself swings to 0.5), so its
"evasion" is a detector-coarseness finding, not an attack win. Read every result
against the FAIR frozen/converged η, never the swingy cumulative one.

## 2. Experimental setup (the exact configuration every result uses)

| Item | Value |
|---|---|
| Task / dataset | image classification, **CIFAR-100** (100 classes) |
| Model | **ResNet-18** (CIFAR stem) |
| Federated setup | **10 clients**, IID split, **FedAvg** aggregation |
| Free-riders | **2 of 10** clients (indices 1 and 6) |
| Rounds | **50** rounds; **5 local epochs**/round; batch 16; SGD lr 0.01, momentum 0.9, wd 5e-4 |
| Watermark | **m = 10 bits**, group size l = 10, lambda = 5.0, memory beta = 0.6, 50 verification triggers |
| Detector | **paper-faithful**: eta = mu+3sigma of benign BER, **cumulative** over all rounds (why eta swings 0.35-0.88) |
| Threshold option | sweep uses `CALIB_ON_ALL=0`; some earlier runs used `=1` (attacker in the eta pool) |
| Honest baseline | benign BER ~0.05, final accuracy ~**72-73%** (healthy reference; the FareMark paper reports ~75% for CIFAR-100 — aligned) |
| Effort metric | `effort_ratio_samples` (and `_gpu`) = free-rider compute / honest compute |

One honest run ~3 h on one A100; ~30 runs in the full sweep.

## 3. Attacks implemented (all compile & wired; config idx in parentheses)

| Attack | idx | Mechanism | Preliminary result |
|---|---|---|---|
| `previous_models`, `gaussian` | 8/9/12/13 | never embed (static fabricators) | caught (baseline anchors) |
| `trigger_only`, `mixed`, ... | 12/13 | the paper's / forgery attacks | trigger-only overfits on CIFAR-100 (BER 0.55-0.63) |
| `submarine` | 14 | warmup embed -> coast -> **tap** (re-train) when the mark fades. Coast modes: transplant/blend/replay/noise/global; tap size = `sub_max_burst_batches` | **strong taps (150) WIN**: BER 0.17 < eta, acc 71%, effort 0.22. Weak taps (20) never embed -> caught |
| `memory_exploit` | 15 | train `warmup_rounds`, then replay the frozen model forever | evades BER **but poisons** (acc 72->37, honest BER->0.5) |
| `reembed` | 16 | each round: fresh global, **freeze backbone, re-train only the head** on trigger data | **running** — the cheap candidate |
| `autopilot` | 17 | fully self-tuning submarine: self-terminating warmup, predicts the eta crossing and taps just before, adaptive tap size, no poisoning. **`autop_scope` = head/block/full** picks how much to re-train | **running** — should match the bb=150 win more cheaply |

## 4. Results (updated 2026-07-07 from the full sweep — all single-seed unless noted)

### 4.0 CONFIRMED headline (2026-07-08, real result.json, single seed)

**REFINED FINDING (2026-07-08, from the per-round traces) — two independent effort
levers, and the winner uses the SECOND one:**
- **Lever 1 — coasting (temporal):** skip training rounds; re-embed occasionally.
  This is the classic "submarine". `scope=full` uses it (coasts 31/50 rounds, taps
  only 5×) — but it gets CAUGHT (BER 0.155 > η) because the mark decays too far
  between rare taps.
- **Lever 2 — cheap shallow re-embedding (structural):** retrain only the last
  block (backbone frozen), which is cheap PER ROUND, so you can re-embed almost
  every round and still be cheap. `scope=block` uses it (trains 37/50 rounds but
  each is ~1/6 the cost) → BER 0.075 < η, effort 18%. **This is the winner.**

So the honest thesis is sharper than "coasting evades": **the watermark is forgeable
by retraining a shallow SLICE of the network — the last block. The head alone is too
shallow (mark won't generalise, BER 0.42); the full model is unnecessary; the last
block is both cheap and sufficient.** This partially resurrects the original
"output layer is cheap to forge" intuition — one block deeper than the literal head.
Coasting (lever 1) helps cost but, used alone (full-scope), gets caught. The two
levers are visualised in the `submarine` plot (block = dives often but cheaply;
full = coasts then caught).


The autopilot sweep (CIFAR-100, 12 configs × the fair frozen/converged η). All
BERs are the free-rider's server BER; η is the FAIR threshold (post-convergence,
fixed). Effort = attacker samples ÷ honest samples. Accuracy ~72 = healthy.

| Config | BER | fair η | effort | acc | verdict |
|---|---|---|---|---|---|
| **`scope=block`** | **0.075** | 0.097 | **0.182** | **72.0** | **EVADES (winner)** |
| `margin=0.12` | 0.105 | ~0.095 | 0.156 | 72 | borderline (≈ η) |
| `full maxtap=250` | 0.155 | 0.096 | 0.147 | 70.9 | caught |
| `full maxtap=400` | 0.165 | 0.10 | 0.232 | 71 | caught (and pricier) |
| `full maxtap=120` | 0.29 | 0.10 | 0.104 | 72 | caught |
| `full maxtap=60` | 0.59 | 0.10 | 0.052 | 72 | caught (weak taps) |
| `scope=head` | 0.42 | 0.10 | 0.214 | 70 | caught (mark won't generalise) |

**The scope curve is the key finding:** BER falls head (0.42) → full (0.155) →
**block (0.075)**. `block` re-embeds enough of the network to generalise the mark
to the server's triggers, but skips most of the backbone's backward pass, so it's
both effective AND cheaper than `full`. `head` fails outright (overfits its own
probe triggers). This is a clean, monotonic, defensible result.

**Knob findings (from the fixed per-knob `knob` plots):**
- `autop_max_batches` (tap strength, on `full`): BER 0.59→0.29→0.155→0.165 as taps
  go 60→120→250→400; effort rises 0.05→0.23. More taps help until ~250 then
  plateau; `full` never clears η — that's why `block`, not bigger `full` taps, is
  the answer.
- `autop_margin0`: bigger safety margin → lower BER (0.16→0.155→0.105) but higher
  effort (0.129→0.156). margin=0.12 gets `full` to the edge of η.
- `autop_protect_until` {4,8,12}: **no effect** on BER/effort (flat) — the
  calibration-window length doesn't change the outcome in this regime. A clean
  null result; leave it at the default 8.

**CIFAR-10 is degenerate — NOT an attack win.** Every autopilot config shows BER
0.0 at 0.3% effort. Cause: m ≈ 2 watermark bits, so honest clients' own BER swings
to 0.5 (visible in the timeline) and the detector is too coarse to mean anything.
Report this as "the watermark/detector is unusable with few classes", a
defence-relevant observation, not evasion.

### 4.1 Earlier sweep results (context)

All BERs below are read against **η = converged ≈ 0.17** (the fair threshold).
The full sweep confirmed the mechanisms but moved the goalposts: most "wins" only
held under the swingy cumulative η.

- **SCOPE QUESTION ANSWERED — head-only fails (the key result).** reembed family
  by scope × steps (from the weak-point map): head 100 -> BER ≈ 0.55; block 40 ->
  ≈ 0.55; full 40 -> ≈ 0.45; full 100 -> ≈ 0.25 (best, still > η). reembed is
  cheap (effort ≈ 0.046) and healthy (acc ≈ 71.3 ± 0.2) but simply does not
  embed. => "output layer is cheap to forge on the free backbone" is NOT
  supported; the backbone matters. This is currently the cleanest finding.
- **The frontier leader is autopilot maxtap=250, not submarine bb=150.** Under
  converged η the only healthy (green) point near the fair line is autopilot
  maxtap=250 (BER ≈ 0.175, effort ≈ 0.14). autopilot self-tuning (0.225, 0.14),
  submarine tap=150 (0.22, 0.225) and reembed full 100 (0.25, 0.14) all just
  miss. The preliminary bb=150 "BER 0.17" was 1 seed vs a looser η.
- **The trilemma reconfirmed (coast modes).** replay = lowest BER (≈ 0.30) but
  poisons (acc 52); blend/transplant/global/noise stay healthy (acc ~72) but sit
  at BER ≈ 0.46–0.52 (caught). Nobody clears η. Pick-two holds.
- **memory_exploit is a dead end.** warmup_rounds trades poisoning for evasion:
  acc 23.5 -> 37 -> 53 -> 61 and BER 0.55 -> 0.25 -> 0.20 -> 0.20 as warmup goes
  2 -> 5 -> 8 -> 12. It plateaus at BER ≈ 0.20 (> η) and acc ≈ 61 (poisoned) —
  never healthy AND below η simultaneously.
- **Submarine levers.** Tap size is the real lever: BER 0.625 -> 0.465 -> 0.22 as
  taps go 20 -> 60 -> 150 (monotonic). sub_warmup is NOT a lever — U-shaped
  (0.555 -> 0.465 -> 0.50), bottoming near 8. Spend compute on tap strength.

**Plotting bug to fix:** `sweep__reembed_effort__*` collapsed to one x-value
"None" — the `reembed_effort` sweep_var is not being stamped into the manifest,
so plot_adaptive can't separate scope/steps (data exists; the weak-point map
resolves the points). Fix manifest tagging, then re-plot the scope frontier.

## 5. [RESOLVED 2026-07-07] Is full-shard/full-model worth it? — YES, head-only fails

**Answer from the sweep: head-only is NOT enough.** reembed head/block -> BER
≈ 0.55 (no mark); full is needed and even full+100 steps only reaches ≈ 0.25,
still above the fair η. So head-only does NOT "sit low-and-left" — it sits
low-effort but high-BER (cheap but caught). The theory "the output layer is cheap
to forge on the free backbone" therefore does NOT hold; the paper's implicit
"you must train the backbone" holds better. This is a clean (if inverted-from-
hypothesis) result and a natural anchor for the defense. Original framing of the
open question preserved below for context.

---

### Original open question (now answered above)

Two INDEPENDENT axes decide embedding cost vs quality:
- **Data source** — full shard (each batch mostly non-trigger; the watermark loss
  fires rarely, so it needs more batches but generalizes) vs trigger-heavy (fires
  every batch, embeds fast, but on the *whole model* it overfits — that's what E7
  showed, BER 0.55).
- **Parameter scope** — whole model (every batch backprops the backbone = most
  compute) vs **head-only** (freeze the backbone, train only the final layer =
  much cheaper per batch, and may still generalize because the backbone is
  already good and freely received).

What is **proven**: full-model + trigger-only overfits (E7). What is **assumed
but NOT yet proven**: that you must pay full-model + full-shard. The cheap escape
— **head-only** re-embedding — is exactly what `reembed` and `autopilot
autop_scope=head` test. The graph to make (from the sweep): **effort (x) vs
fr_ber (y), one point per scope in {head,block,full}**. If head-only sits
low-and-left (evades cheaply), full-shard/full-model is *not* worth it and the
theory ("the output layer is cheap to forge on the free backbone") holds. Plot
the `autopilot_scope` and `R_frontier` families.

## 6. Next steps (updated 2026-07-07 — attack angle, one more push)

Decision: keep pushing the ATTACK on the autopilot (memory_exploit dropped). All
results now judged against the FAIR headline threshold `frozen` (§10), not
cumulative. Code changes (§8, §10) are pushed but UNVALIDATED — the sweep below is
the real test; I could not run it here (no GPU/cluster).

1. **Re-run the autopilot family (3 seeds) with the fixes**, sweeping the effort
   dial that actually matters — warmup length + tap strength — since the fair bar
   (eta_frozen ~0.07-0.17) is high:
     `for R in 0 1 2; do ROUNDS=50 ATTACK=autopilot AUTOP_PROTECT_UNTIL=8 \
        AUTOP_SCOPE=full AUTOP_MAX_BATCHES=250 FAMILY=autopilot SWEEP_VAR=none \
        NOTE="autopilot full maxtap250 3seed" WAIT=0 ./submit_experiment.sh 17 $R; done`
2. **Make the credibility plot** `plot_thresholds.py evade_bars` — the go/no-go: does
   any autopilot config evade under `frozen`/`converged`, or only `cumulative`? If
   only cumulative, the honest attack claim is "beats the as-published detector";
   if frozen too, it's a real break.
3. **Read the sawtooth** `plot_thresholds.py overlay` on the best autopilot run: is
   the FR's server BER (not just its probe) staying under eta_frozen between taps?
   Watch the probe/server gap (§8 limitation).
4. Fix the `reembed_effort` manifest sweep_var and the weak-point map label overlap
   (carried over).
5. Only if the fair-threshold attack stalls: pivot the headline to the DEFENSE
   (frozen threshold + "output layer not cheap to forge") — that result is already
   in hand from last night's sweep.

## 7. Doc map

STATUS.md (this) . ADAPTIVE_ATTACKS.md (deep reference) . HYPERPARAMS.md (every
knob incl. autopilot/reembed/scope) . RUNSHEET_ADAPTIVE.md (commands) .
EXPERIMENTS.md (family registry) . DOCUMENTATION.md (code<->paper) .
PROJECT_PLAN.md (pillars) . README.md / GRETA.md (overview / log).

---

## 8. Autopilot mechanism (technical reference)

**[CODE CHANGES 2026-07-07] Autopilot is now the sole attack focus (memory_exploit
abandoned — it can't be both healthy and below eta).** Fixes/adjustments applied
to `attacks_adaptive.py`:
- **Coast = `transplant`** (fresh global + frozen mark-delta): the recommended
  coast. No poisoning (tracks the live global, acc stays ~72) and re-injects the
  mark for ~0 compute; the mark decays as the global moves, which the taps refresh.
  (replay poisons; blend dilutes; global/noise carry no mark — all worse.)
- **`autop_protect_until` (new knob, default 8):** the FR never defects before this
  round, so the detector's FROZEN eta is calibrated on a genuinely no-free-rider,
  post-convergence window (ties the attack to the §10 headline threshold) and the
  FR's own clean-BER eta anchor is only recorded once honest clients have converged.
- **Tap-growth bug fixed:** taps now grow x1.6 only when the PREVIOUS tap genuinely
  failed to reach floor (`_last_tap_undershot`), not on any clean sample >floor
  (which fired spuriously and could run tap size away).
- **Taps are "solid":** they early-stop at `autop_floor` on the FULL shard so the
  mark GENERALISES to the server's test triggers (not just the FR's 16-image probe)
  — driving BER down hard with margin, per the intended warmup->coast->tap loop.
- **`_embed_loop` now try/finally-restores requires_grad** on every exit path
  (a head/block tap that threw used to leave the backbone frozen for later rounds).
- CLI/wiring: added `--autop_protect_until`, `--autop_enriched`, `--autop_lookahead`
  (config.py / run_experiment.py / wm_client.py).

**KNOWN LIMITATION (honest):** the FR's probe is ~16 held-out shard trigger images;
the server verifies on 50 test-set triggers. The probe can under-predict the server
BER (train/test trigger gap — the paper's Table V overfitting mechanism). Mitigated
by full-shard taps + a safety margin, but this gap is why probe-says-floor can still
be server-caught, and is the thing to watch in the sawtooth overlay.

---

### Original mechanism reference

`AutopilotFreeRider` in `faremark/attacks_adaptive.py`. A fully self-tuning
submarine: nothing is a fixed schedule; every decision is computed live from its
own held-out BER probe (measuring is ~free; only training costs compute).

**The BER probe.** `_ensure_triggers` runs once, splitting the client's own
trigger-class images into a HELD-OUT slice `_probe_x` (16 images, never trained
on) and the rest for training. `_probe_ber_current_model` / `_probe_ber_state`
forward only those 16 images (no_grad) and return the BER. Held-out => predicts
the server's secret-trigger score without overfitting to it.

**Threshold estimate `_eta_est`.** The attacker can't see the server's eta, so it
estimates it as calibrate_eta (mu+3sigma) over its own recent CLEAN BERs (BERs
reached right after a real embed = its honest-pool proxy). `_record_clean` only
stores a BER as clean if <= 2x floor, so failed embeds don't inflate it. Falls
back to `sub_eta_fixed` (0.35) before it has data. It aims at
`target = eta_est - autop_margin`.

**Phase 1 — self-terminating warmup.** Calls `_embed_loop` (train), updates
memory, probes, records the clean BER. Sets `_warm_done=True` only when it truly
embedded (ber <= floor) AND has >=2 clean samples to anchor eta, or the
`autop_warmup_cap` safety cap is hit. So warmup lasts exactly as long as needed.

**Phase 2 — predictive, adaptive taps.**
  - Forms the coast candidate `_coast_state` = fresh global + frozen mark-delta
    (never stale replay => no poisoning); probes its BER; appends to `_ber_trend`.
  - WHEN to tap: `_predict_cross` linearly extrapolates the last 3 probe BERs to
    estimate rounds-until-target; taps if BER is already near target OR predicted
    to cross within `autop_lookahead` — i.e. JUST BEFORE being caught.
  - HOW HARD: tap size scales with drift; if the last tap undershot it grows the
    next x1.6 (self-corrects weak taps). Bounded [autop_min_batches, autop_max_batches].
  - MARGIN adapts: relax x0.98 when safe (coast more, cheaper), tighten +0.02
    after a miss (safer).

**Training `_embed_loop(..., scope, enriched)` — two independent dials.**
  - `scope`: full (whole model), block (last ~8 tensors), head (final linear
    only; freezes the backbone so its backward pass is skipped => much cheaper
    per batch). Restores requires_grad on every exit path.
  - `enriched`: data source — False = full shard, True = trigger-heavy loader.
  Both warmup and taps use `autop_scope` / `autop_enriched`.

**Knobs (config idx 17):** autop_floor, autop_margin0, autop_min_batches,
autop_max_batches, autop_lookahead, autop_warmup_cap, autop_scope, autop_enriched.
Everything it decides is logged to `self.trace` (action, ber_coast, eta_est,
target, predict_cross_in, tap_batches, ber_after).

## 9. Attack family — one mechanism, different embed strategies

All the "smart" attacks share the SAME idea (embed -> coast without poisoning ->
re-embed when it fades). They differ only in HOW they coast and HOW they re-embed:

| Attack | Coast (off-rounds) | Re-embed (taps) |
|---|---|---|
| memory_exploit | replay frozen old model (poisons) | never re-embeds |
| submarine | pick a coast mode (below) | full-shard, full-model, capped at sub_max_burst_batches |
| reembed | (re-embeds every round, no coast) | HEAD-ONLY on fresh global (cheapest) |
| autopilot | fresh global + mark-delta (no poison) | `autop_scope` = head/block/full, size auto-tuned |

**Coast modes (submarine `sub_coast_mode`):**
  - `replay` — submit the frozen embedded model unchanged. Mark is preserved but
    the model is stale => POISONS the global over time (acc drops, honest BER->0.5).
  - `blend` — submit mem_blend_global*global + (1-...)*memory. Fresher (less
    poisoning) but mixing the unmarked global in DILUTES the mark => it decays =>
    caught unless taps are strong.
  - `transplant` — submit global_now + (memory - global_at_embed), i.e. the fresh
    model plus the frozen "mark direction". No poisoning; but the mark direction
    doesn't hold as the global moves => decays => caught unless re-embedded.
  - `noise` — submit global + small Gaussian noise (a naive free-rider baseline;
    carries no mark => caught).
  - `global` — submit the received global unchanged (do-nothing baseline; no mark
    => caught).
  Expectation: replay = evades-but-poisons; blend/transplant = healthy-but-fade;
  noise/global = caught. None wins alone; the WIN is strong re-embedding on top.

---

## 10. Threshold (eta) variants — all computed post-hoc, all shown on every plot

eta only sets the detection LABEL, never the model/training, so all definitions
are recomputed from the benign-BER series in result.json (no re-runs). Source of
truth: `faremark/thresholds.py`. Overlay them on any run with
`scripts/plot_thresholds.py overlay`; the `worth` plot and the weak-point map can
annotate whichever variant.

**[UPDATED 2026-07-07] What the paper actually does (headline = `frozen`).**
FareMark Eq.16 sets eta = mu+3sigma of legitimate clients' BER over "many rounds
... to observe the TYPICAL error rate". Fig.8 shows benign BER is HIGH until
~round 30 and only then converges, so "typical error rate" = the CONVERGED error,
calibrated once and held FIXED — NOT a per-round cumulative recompute. So
`cumulative` (our old "paper_faithful") is the OVER-literal reading and is exactly
what let memory_exploit/replay "evade" (poisoning inflates the swingy eta). The
faithful, fair headline is **`frozen`, calibrated on a stable POST-convergence
window and frozen** — and because it's fixed before the FR defects, a poisoning
attack can't inflate it. NOTE (correction to an earlier plan): do NOT calibrate on
rounds 0-9 — honest clients haven't embedded yet there, so mu+3sigma is too loose.
`thresholds.py` `frozen` now skips the first `warmup` rounds and calibrates on the
next `converged` block. Verified on a synthetic series: a FR at BER 0.12 shows
100% evade under cumulative (eta 0.47) but 0% under frozen/converged (eta 0.07).

| Variant (flag `--eta`) | Definition | Character |
|---|---|---|
| `cumulative` | mu+3sigma of benign BER over ALL rounds so far | old paper_faithful; INFLATES when benign BER rises (swings 0.35-0.89); reference only |
| `frozen` | mu+3sigma over a stable post-convergence window `[warmup : warmup+converged]`, held fixed | **the faithful, fair HEADLINE**; cannot be inflated by a later defection |
| `windowed` | mu+3sigma over the last K rounds | adaptive, no long memory |
| `converged` | mu+3sigma over the last C rounds, fixed | fair backup, but its tail CAN be poisoned |
| `fixed` | a constant (0.25) | reference |

All five stay on every plot (overlay, evade_bars, weak-point map `--eta`).

**Important finding (from the overlay): memory_exploit only 'evades' the
cumulative eta.** On a replay run: cumulative eta = 0.89 -> evades 100%, but
frozen/windowed/converged eta = 0.48-0.50 -> evades 0% (caught), fixed 0.25 ->
caught. So its "evasion" was an artifact of the poisoning it caused inflating the
cumulative threshold. Under any fair threshold it is caught. Report evasion under
ALL variants (via `thresholds.evades_under`), not the swingy per-round recall.

## 11. Plots available
- `scripts/plot_thresholds.py submarine` **[NEW]** — the "submarine line": free-rider
  BER vs the fair η, TRAINING rounds shaded (dives) / coasting clear, per-tap cost
  bars + total effort in title. Shows WHY each tap happens. Contrast `scope=block`
  (dives often, cheaply — WINS) vs `scope=full` (coasts, caught) — the two levers (§4.0).
- `scripts/plot_thresholds.py timeline` **[NEW]** — FR+honest BER + all η lines (top)
  stacked with cumulative attacker-effort vs round (bottom). Interpretive per-run plot.
- `scripts/plot_thresholds.py knob` **[NEW]** — per-knob sweep filtered by family AND
  sweep_var (fixes merged sweeps): BER-vs-knob + effort-vs-knob, mean±std.
- `scripts/plot_thresholds.py decay` **[NEW]** — one autopilot run: watermark decay while coasting (BER climbs to η) + re-embed cost per tap (batches-to-floor), stacked panels vs round. The mechanism plot: coast-rounds-gained / tap-batches-spent is the effort frontier, read straight off it.
- `scripts/plot_thresholds.py overlay` — one run: fr_ber + benign_ber vs round,
  all eta variants as lines, % evade under each in the title. **[UPDATED]** now
  marks warmup-embed (squares) and tap (triangles) rounds from the FR trace, so
  the warmup->coast->tap->coast sawtooth is visible (proof of HOW it works).
- `scripts/plot_thresholds.py evade_bars` **[NEW — the credibility plot]** — per
  config, fraction of converged rounds evaded under EVERY eta variant, mean±std
  over seeds. High under frozen/converged = a real break; high only under
  cumulative = the artifact. Command:
    `python scripts/plot_thresholds.py evade_bars --in "$RES/*/result.json"
     --family autopilot autopilot_scope S_coast --out figs/evade_bars`
- `scripts/plot_thresholds.py worth` — across configs: grouped effort bars
  (effort_ratio_samples, effort_ratio_gpu, duty_cycle) + BER + accuracy, mean±std
  over seeds. The "worth/cheap" multi-metric figure.
- `scripts/plot_frontier.py` — weak-point map (fr_ber vs effort, color=acc). `--eta {cumulative|frozen|windowed|converged|fixed}` picks which threshold the reference line uses (post-hoc; no re-run).
- `scripts/plot_adaptive.py sweep|effort|duty` — per-knob sweeps.

## 12. Seeds
Broad sweeps run at 1 seed to FIND winners cheaply; then re-run the 2-3 winning
configs at 3 seeds (`SEEDS="0 1 2"`) for error bars. All plotters draw mean±std
across seeds automatically. Single-seed points are exploratory, not final.

## 13. Plotting standards (enforced via scripts/plotstyle.py)
All plotters import `plotstyle.py` and follow these rules (apply to every future plot too):
- **No dual / twin y-axes** — different-unit quantities go in **stacked panels**
  sharing the x-axis (e.g. the `worth` plot: effort / BER / accuracy = 3 panels).
- **Colour-blind-safe** Okabe-Ito categorical palette; sequential colormap =
  **viridis** (not RdYlGn/jet). Series also vary linestyle + marker for greyscale.
- Descriptive titles, axis labels with units, legends, light grid, despined axes.
- Semantic colours are consistent everywhere: honest=blue, free-rider=red,
  accuracy=green, eta=black.
---

## 14. HOW IT ALL WORKS — from scratch (documentation, read if new to the project)

This section assumes no background. It explains the model, training, the watermark,
and the three attack mechanisms in plain terms.

### 14.1 The model: backbone + head, and what "training" costs

A ResNet-18 image classifier is a stack of ~18 layers. Think of it in two parts:
- **Backbone** (the first ~17 layers): turns a raw 32×32 image into a compact
  "feature vector" — a list of numbers describing edges, textures, shapes, objects.
  This is where almost all the model's knowledge lives and it is expensive to learn.
- **Head** (the final single layer, a.k.a. the classifier / output layer): a simple
  linear map from the feature vector to 100 class scores (for CIFAR-100), turned into
  probabilities by softmax. Cheap; it only draws boundaries in the space the backbone
  already built.

**A training step has two halves.** (1) *Forward pass*: push a batch of images
through every layer to get predictions. (2) *Backward pass* (backpropagation):
compute how wrong the prediction was (the "loss") and push the error backwards
through every layer to nudge each layer's weights. The backward pass through the
backbone is the expensive part. **If you "freeze" the backbone (mark it
`requires_grad=False`), you skip its backward pass entirely** — you still forward
through it (cheap) but only update the head/last block. That is the whole trick
behind the cheap attack: freezing saves most of the compute.

**Cost units in this project.** "Effort ratio" = attacker samples ÷ honest samples.
An honest client trains on its full shard every round (~25,000 sample-passes/round).
A frozen-backbone tap of 250 mini-batches of 16 = 4,000 sample-passes and skips the
backbone backward, so it is far cheaper per round.

### 14.2 The watermark and BER (how FareMark detects free-riders)

FareMark forces honest training to leave a hidden mark. The server picks, per client,
a secret **trigger class** (a category of images) and a secret key. During training,
honest clients add a **watermark loss** on trigger-class images: it pushes the model's
softmax output on those images into a specific pattern that encodes `m` secret bits
(for CIFAR-100, m ≈ 49 bits; for CIFAR-10 only ≈ 2 — which is why CIFAR-10 is
degenerate). Total loss = classification loss + λ·watermark loss (λ=5).

Each round the server reads the bits back from every client's model using held-out
trigger images and computes the **Bit-Error-Rate (BER)** = fraction of the `m` bits
that come out wrong. **Low BER = the mark is present = the client did real training.**
If BER ≥ threshold **η**, the client is flagged as a free-rider. η is set to μ+3σ of
honest clients' BER — see §10 for why the FAIR reading (frozen/converged) is the one
to use, not the swingy cumulative one.

Key subtlety the whole attack rests on: a mark only reads back correctly if it
**generalises** — i.e. training on some trigger images makes the pattern appear on
*other* (the server's) trigger images too. Training on too few triggers, or with too
little of the network, **overfits**: the mark shows on the images you trained on but
not the server's → high BER → caught. (This is the paper's own Table V effect.)

### 14.3 The attack loop (autopilot) — warmup, coast, tap

The working attack is a self-tuning free-rider ("autopilot"). It measures its own BER
every round for free (forward-only on 16 held-out trigger images — no training cost)
and acts on it. Three phases:

1. **Warmup (embed honestly).** For the first rounds it trains like an honest client
   until the mark is solidly embedded (probed BER ≤ floor 0.05) AND past the protected
   calibration window (`autop_protect_until`, so the detector gets clean rounds to set
   η). This is the attacker's one unavoidable honest investment.
2. **Coast (free-ride).** It stops training and submits the fresh global model plus the
   frozen "watermark direction" (transplant coast — keeps the model healthy, no
   poisoning). With no training, the mark slowly **decays** as aggregation moves the
   global model, so BER drifts upward toward η.
3. **Tap (re-embed just in time).** It predicts (from its recent BER trend) when BER
   will cross η and, just before, does a short burst of real training to drive BER back
   down. Tap size self-adjusts (grows if the last tap undershot). Then it coasts again.

The controller's numbers: it estimates η from its own recent clean BERs (μ+3σ), aims
at `target = η_est − margin` so it acts before being caught, and predicts the crossing
`autop_lookahead` rounds ahead. All decisions are logged to the per-client `trace`.

### 14.4 The THREE re-embed mechanisms (the scope lever) — the core result

When the attacker taps, HOW MUCH of the network it retrains is `autop_scope`. This is
the decisive knob. The three mechanisms:

- **`head` — retrain only the final linear layer (backbone frozen).** Cheapest. But
  the head can only use the features the frozen backbone already provides; it cannot
  reshape them. It overfits its own trigger images and the mark does NOT generalise to
  the server's → **BER ≈ 0.42, caught.** *Mechanism of failure: too few degrees of
  freedom.* (This falsifies the naive "the output layer is cheap to forge" claim.)
- **`full` — retrain the whole network (backbone + head).** Most capacity, mark
  generalises → low BER *when freshly embedded*. But every batch backprops the whole
  backbone = expensive, so the attacker can only afford a few taps and must coast a lot;
  between rare taps the mark decays too far → **BER ≈ 0.155, caught under the fair η.**
  *Mechanism of failure: too expensive to refresh often enough.*
- **`block` — retrain only the last block (~8 tensors), rest of the backbone frozen.**
  Enough capacity to reshape features so the mark generalises, but skips most of the
  backbone's backward pass → cheap per round. So it re-embeds almost every round and
  stays fresh, at low total cost → **BER 0.075 < η, effort 18%, healthy 72%. WINNER.**
  *Why it wins: cheap AND sufficient — the Goldilocks depth.*

The monotonic scope curve (BER: head 0.42 → full 0.155 → block 0.075) is the headline.

### 14.5 The TWO effort levers (why "cheap" happens two different ways)

- **Lever 1 — coasting (temporal):** skip training rounds. `full` relies on this (trains
  19/50 rounds). Saves cost but the mark decays between rare taps → caught.
- **Lever 2 — cheap shallow re-embedding (structural):** make each re-embed cheap by
  freezing the backbone. `block` relies on this (trains 37/50 rounds but each is ~1/6
  cost) → stays fresh AND cheap → wins.

So the honest thesis: the vulnerability is structural, not temporal — **the watermark
is forgeable by retraining a shallow SLICE (the last block); the head is too shallow,
the full model unnecessary.** Coasting helps but, used alone, gets caught. The
`submarine` plot shows both: block dives often but cheaply (wins); full coasts then is
caught.

### 14.6 Why the detector is fooled (one paragraph)

FareMark assumes "has the watermark" ⇔ "did the expensive training". The attack breaks
that equivalence: the mark is an output-space property that a *shallow* re-embed can
forge on the freely-received backbone, so a client can have the mark without doing the
work. The fair threshold (frozen η) raises the bar — most cheap tricks are caught — but
`block`-scope re-embedding still clears it. The eventual DEFENCE follows from this: add
a check the shallow forgery can't pass (e.g. backbone-consistency), or a fair threshold,
which already neutralises the coasting/poisoning routes.

---

## 15. WATERMARKING MECHANISM & GLOSSARY (answers to the deck TODOs)

### 15.1 Exactly how FareMark's watermark works — the key and the secret message

Three secrets are agreed per client at REGISTRATION and stored by BOTH the client
and the server:
1. **Trigger class** — one image category (e.g. "cat") assigned to the client; the
   mark is read only from images of this class.
2. **Watermark message B** — the secret bit-string the client must embed,
   B = [b_1..b_m], m bits (CIFAR-100: m≈49; CIFAR-10: m≈2). Chosen at registration;
   **the server keeps a copy — that is how it knows what the bits "should" be.**
3. **Key M** — a secret pseudorandom projection matrix; each bit k has its own random
   projection vector (row M_k).

**Embed (client).** For a trigger image the model outputs softmax vector P (probs over
classes). Bit k is read as: b̂_k = 1 if (P·M_k) ≥ 0 else 0 (a smoothing f(x)=x^α is
applied to P first so it isn't too spiky). Training adds a watermark loss that pushes
each projected sign to match the target b_k. So "embedding" = shaping the softmax on
trigger images until its projections onto the secret directions spell out B.

**Verify (server).** Take ~50 held-out trigger-class images, run them through the
submitted model, recover B̂ the same way (project onto M, take signs, average over the
50 images), compare to the registered B. **BER = fraction of the m bits where B̂ ≠ B.**
Low BER ⇒ the client shaped the output as only real training can ⇒ not a free-rider.
Flag if BER ≥ η.

**How the attacker estimates the threshold.** The attacker is a registered client, so
it holds its OWN B and M and can compute its own BER every round for free. It does NOT
know the server's η, so it estimates η as μ+3σ of its recent clean BERs (or a fallback
0.25 before it has data) and aims below that. Its estimate can be far from the true
fair η — see the `estimate` plot; in our block run eta_est sat at 0.25 while the actual
fair η was ~0.10 (false confidence).

**The probe≠server gap.** The attacker probes BER on 16 of ITS OWN shard's trigger
images; the server verifies on HELD-OUT trigger images. Same B and M, different images
→ the attacker can read a lower BER than the server sees. This gap is why FR BER can be
above the server's line even when the attacker thinks it is safe.

### 15.2 Box-free vs black-box vs white-box — and why the attack is box-free-specific

Watermarking methods differ by WHAT the verifier needs and WHERE the signal lives:
- **White-box (feature-based, e.g. FedIPR):** signal is IN THE WEIGHTS. Verify by
  reading bits from the parameters directly. Needs full parameter access.
- **Black-box (backdoor-based, e.g. Adi et al.):** signal is a hidden INPUT→LABEL rule
  (trigger image → planted label). Verify by querying. Recognizing the trigger is a
  FEATURE-DETECTION job → lives in the BACKBONE.
- **Box-free (FareMark):** signal is the SHAPE of the softmax OUTPUT on trigger-class
  images (bits via projection onto M). Verify by querying. Signal is SHALLOW — a
  property of the output mapping.

**Why our shallow-slice attack is box-free-specific:**
- Box-free: the mark is a shallow output property, so a cheap shallow re-embed (last
  block) reaches it → our attack works.
- Black-box: forging the label rule needs the network to RECOGNIZE the trigger, a
  backbone job → a last-block-only tap can't build a trigger detector → you're pushed
  to full-model cost (i.e. toward doing the real work). HARDER for our method.
- White-box: verification reads weights, not outputs, so riding under an output BER is
  irrelevant; you'd have to reproduce a weight-space signature. HARDEST for our method
  — BUT white-box needs the server to hold every client's weights, which FL privacy
  (secure aggregation) often forbids. That assumption is exactly WHY FareMark chose
  box-free.

**Sharper thesis:** the weakness is intrinsic to putting the mark in a SHALLOW
output-space location. Box-free is the only method that works under FL privacy, but its
shallowness is what makes it cheaply forgeable. This directly motivates the defence:
anchor the mark deeper (force backbone dependence, like black-box) or add a
backbone-consistency check so a shallow forgery is caught. CAVEAT: black/white-box
resistance is reasoned from mechanism, not yet tested — a clean future experiment.

### 15.3 GLOSSARY (plain definitions for the deck TODOs)

- **Weights / parameters:** the numbers inside each layer that get adjusted during
  training; "the model" is essentially its weights.
- **Features / feature vector:** the compact numeric description of an image produced by
  the backbone (edges→textures→shapes→objects); the head classifies from these.
- **Layer:** one processing stage. ResNet-18 ≈ 18 such stages: a conv stem, 4 stages of
  residual blocks (the backbone), then a final linear layer (the head).
- **ResNet-18 / CNN:** ResNet-18 is one specific Convolutional Neural Network (CNN)
  architecture — CNNs are the standard family for image classification. We use ResNet-18
  because the paper does; the attack idea is architecture-agnostic (any backbone+head
  model), but all our NUMBERS are ResNet-18 on CIFAR-100/10.
- **Batch / mini-batch:** training processes images in small groups called mini-batches
  (here 16 images). One "batch" in our tap counts = one mini-batch of 16. "250 batches"
  = 250×16 = 4,000 image-passes.
- **Effort from batches:** effort_ratio = attacker image-passes ÷ honest image-passes.
  Honest ≈ 25,000 passes/round (full shard × 5 epochs). A frozen-backbone tap of 250
  batches = 4,000 passes AND skips the backbone's backward pass, so it is much cheaper
  per round. Sum over all rounds → the ~18% figure.
- **Forward / backward pass:** forward = compute the prediction; backward
  (backpropagation) = compute how to nudge weights. The backbone's backward pass is the
  expensive part; freezing the backbone skips it.
- **"Knee at 250":** on the tap-strength sweep, BER stops improving much past
  max_batches=250 (the curve bends like a knee) — bigger taps cost more for little gain,
  so 250 is the efficient choice.
- **Safety margin (autop_margin0) and how it auto-adjusts:** the attacker aims at
  target = eta_est − margin so it acts before being caught. It RELAXES the margin (×0.98,
  coast longer, cheaper) after safe rounds and TIGHTENS it (+0.02, safer) after a tap
  undershoots. Starts at 0.08.
- **Sawtooth:** the shape of the free-rider's BER over rounds — it rises while coasting
  (mark decaying) then drops sharply at a tap (re-embed), repeatedly, like saw teeth.
- **Slope (decay plot):** how fast BER climbs per coasting round = how fast the mark
  decays = how many rounds you can coast before nearing η. Read it as rise-over-run on
  the BER curve during a clear (non-shaded) stretch.
- **Duty cycle:** fraction of rounds the free-rider actually trains (block: 37/50; full:
  19/50).

---

## 16. HONEST-ROUND ETA CALIBRATION + adaptive convergence (2026-07-09)

**Why:** the 3-seed confirm sweep showed the block attack rides ON the fair eta
(BER ~0.10-0.13 vs eta ~0.09), not under it. Root cause: the attacker estimated eta
(mu+3sigma) from its own COAST/TAP probe, which reads high (~0.15-0.25), so the
estimate was ~0.25 and it aimed too high. The `blk_honest` arm evaded best because a
deep full-model warmup buys a long coast — but it still used the wrong eta target.

**Fix (in `attacks_adaptive.py`, autopilot):** during the forced-honest warmup the FR
IS an honest client, so its BER there samples the SAME distribution the server's fair
eta is calibrated on. It now:
  1. Trains fully-honest (full model, full epoch) at the start;
  2. **Auto-detects convergence** = the honest BER curve FLATTENS (two consecutive
     round-to-round improvements < `autop_conv_eps`, default 0.02 — a RATE test, not a
     hand-tuned BER cutoff, so it is dataset-agnostic);
  3. Calibrates eta = mu+3sigma over the CONVERGED honest rounds only (~0.09), not the
     pessimistic probe (~0.25);
  4. **Auto-stops the honest phase** at convergence (>=2 calibration samples), with
     `autop_honest_until` now just a SAFETY CAP (not an exact schedule).

**IMPORTANT — which numbers are computed vs assumed (for the write-up):**
  * COMPUTED from the run: all eta values (mu+3sigma), FR/honest BER, effort, accuracy,
    tap timing, and NOW the convergence round (auto-detected) and the eta estimate
    (calibrated on the converged honest rounds).
  * ASSUMED (fixed hyperparameters, not measured): `autop_conv_eps=0.02` (flattening
    rate), `autop_honest_until` cap (default 10 in the sweep), `autop_margin0`,
    `autop_max_batches`. The old hard-coded `ber<=0.15` convergence gate has been
    REMOVED and replaced by the rate test. mu+3sigma itself is from the FareMark paper.

**Test:** `run_honestcal_sweep.sh` — arms hc_block / hc_block2 / hc_full (all do the
honest warmup + converged-round calibration) + hc_block_nocal (honest_until=0, A/B
baseline). Read `honestcal_evade.png` first; then the seed-bands — the tell is the grey
ESTIMATED-eta line dropping from ~0.25 to ~0.09, and red (FR) dipping under green
(actual eta). Wiring: config.py/run_experiment.py/wm_client.py/submit_experiment.sh
forward `AUTOP_HONEST_UNTIL`; `autop_conv_eps` is tunable via config.