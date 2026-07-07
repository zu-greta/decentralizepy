# STATUS 

Single source of truth for where the free-rider project stands. Results are
**preliminary**; the `run_full_sweep.sh` batch (incl. autopilot + re-embed) will
refine them. Read `wm_fr_ber` (below η = evades) and `final_acc` (~72 = healthy,
low = poisoned), NOT `wm_fr_recall` (η swings, so recall is noisy).

---

## 1. One-paragraph status

Part 1 (reproduce FareMark) done. Part 2 (can a client free-ride past the
watermark detector cheaply?): the pipeline is solid, five attacks are built, and
**the full sweep is in (2026-07-07).** Read under the *fair* threshold
(η = converged ≈ 0.17, the one HANDOFF says to headline) rather than the swingy
cumulative one, the sweep **sharpens and partly reverses** the preliminary story.
Two headline results: **(a)** *head-only re-embedding does NOT work* — freezing
the backbone and retraining only the output head leaves BER ≈ 0.55 (caught), and
even *full-model* re-embedding at 100 steps only reaches ≈ 0.25, still above η.
So the thesis hypothesis "the output layer is cheap to forge on the free
backbone" is **NOT supported** by this data; the backbone matters. **(b)** Under
converged η the earlier submarine bb=150 "win" no longer clears the bar (it now
sits ≈ 0.22, marginally caught); the only healthy point flirting with the fair
line is **autopilot maxtap=250** (BER ≈ 0.175, effort ≈ 0.14) — confirm at 3
seeds. Net: the cheap attacks beat only the paper's *inflated cumulative*
threshold; a fair threshold largely neutralizes them — which is itself the
emerging **defense** result (attack claim survives only as "evades the detector
*as published*", not "the mark is cheap to forge in general").

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