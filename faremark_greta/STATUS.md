# STATUS 

Single source of truth for where the free-rider project stands. Results are
**preliminary**; the `run_full_sweep.sh` batch (incl. autopilot + re-embed) will
refine them. Read `wm_fr_ber` (below η = evades) and `final_acc` (~72 = healthy,
low = poisoned), NOT `wm_fr_recall` (η swings, so recall is noisy).

---

## 1. One-paragraph status

Part 1 (reproduce FareMark) done. Part 2 (can a client free-ride past the
watermark detector cheaply?): the pipeline is solid and five attacks are built.
**Key finding so far:** *re-using* a stored watermark fails (a trilemma: cheap /
keep-mark / don't-poison — pick two), but *re-making* the watermark each round on
the fresh model **works** — the submarine with strong taps hit BER 0.17 (< η),
accuracy 71%, at 22% of honest effort. Two attacks are now running to push the
cost down and confirm robustness: `reembed` (re-train only the output head) and
`autopilot` (a fully self-tuning submarine, now with a training-scope dial).

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
| Honest baseline | benign BER ~0.05, final accuracy ~**72-73%** (the "healthy" reference) |
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

## 4. Results so far (PRELIMINARY)

- **The trilemma** (re-using a stored mark fails): replay evades but poisons
  (acc 55, benign->0.5); blend/transplant stay healthy (acc 72) but the mark
  decays / won't transfer -> caught. Confirmed across independent runs.
- **Re-embedding works** (re-making the mark): submarine bb=150 -> BER 0.17 < eta,
  acc 71%, effort 0.22 — first point in the "cheap + evades + healthy" corner.
  (n=1 seed; being repeated.)
- **Poisoning breaks the detector**: a stale replay drags honest BER to 0.5, so
  honest clients get flagged and the free-rider ends up looking *more* honest —
  a separate detector-fragility finding.
- **Clean supporting result**: a generalizing mark needs full-shard training;
  trigger-only shortcuts overfit (BER 0.55-0.63). See section 5 for the nuance.

## 5. Open question to settle with a graph: is full-shard/full-model worth it?

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

## 6. Next steps

1. Push the current code (5 files). Run `./scripts/run_full_sweep.sh` (~30 runs,
   reembed + autopilot first), then `... PLOT`.
2. Read the **weak-point map** (`figs/weakpoint_all.png`) and the **scope graph**
   (section 5): does head-only evade cheaply?
3. Confirm the bb=150 winner at 3 seeds.
4. Deck: drop autopilot + reembed points onto the map; add the scope graph.
5. Write-up: "re-using a watermark is hard (trilemma); re-making it — especially
   head-only on the free backbone — defeats the detector cheaply."

## 7. Doc map

STATUS.md (this) . ADAPTIVE_ATTACKS.md (deep reference) . HYPERPARAMS.md (every
knob incl. autopilot/reembed/scope) . RUNSHEET_ADAPTIVE.md (commands) .
EXPERIMENTS.md (family registry) . DOCUMENTATION.md (code<->paper) .
PROJECT_PLAN.md (pillars) . README.md / GRETA.md (overview / log).