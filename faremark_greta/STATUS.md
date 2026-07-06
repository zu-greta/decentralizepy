# STATUS — read this first

Last updated for: the effort/cost pillar (free-rider attacks vs FareMark's
output-layer watermark detector). This file is the single source of truth for
**where the project stands right now**. Results here are **preliminary** — a new
full sweep (`scripts/run_full_sweep.sh`) is about to run and may change the
conclusions. Nothing below is final.

---

## 1. One-paragraph status

Part 1 (reproduce FareMark) is done. Part 2 is the free-rider work: can a client
skip most of the training, still pass the watermark detector (BER < η), and keep
receiving the global model? The infrastructure is solid and three "coast"
attacks have been built and run at a real 50-round horizon. **Preliminary
finding: none of the three coast attacks is simultaneously cheap, evasive, and
harmless** — they hit a three-way trade-off ("the trilemma"). A fourth,
theoretically-motivated attack (`reembed`) has just been implemented to target
the actual weak point but is **not yet run**. The immediate next action is the
sweep that measures every attack's effort-vs-evasion frontier.

## 2. What the detector does (so the metrics make sense)

Server holds each client's secret key + trigger class. Each round it runs every
submitted model on its own **held-out** trigger images, extracts bits from the
softmax, and computes **BER** (bit-error-rate). It flags a free-rider when
**BER ≥ η = μ+3σ** of the benign BER distribution. Key consequence: the mark is a
property of the *weights*; you can't attach it, you must train it in AND have it
generalize to triggers you never see.

**Read results by `wm_fr_ber` and `final_acc`, NOT `wm_fr_recall`.** The
paper-faithful η is cumulative and swings (~0.35–0.88) depending on the benign
pool, so `recall` (caught rate) is a noisy, threshold-dependent readout. `fr_ber`
below η = evades; `final_acc ≈ 72%` = model healthy; low acc = poisoned.

## 3. Attacks implemented (code state — all compile, all wired)

| Attack | config idx | Idea | Status |
|---|---|---|---|
| `previous_models`, `gaussian` | 8/9/12/13 | static fabricators, never embed | done — caught (baseline anchors) |
| `train_then_attack`, `trigger_only`, `random_round`, `mixed` | 12/13 + flag | the paper's / forgery attacks | done — `trigger_only`/`mixed` overfit on CIFAR-100 (E7) |
| `submarine` | 14 | warmup a mark, then **coast** (memory replay), tap only when BER drifts up. Coast modes: `transplant` (default), `blend`, `replay`, `noise`, `global` | built & run — **caught or poisons** (see §4) |
| `memory_exploit` | 15 | train `warmup_rounds`, then **replay** frozen mark forever | built & run — **evades but poisons** (see §4) |
| `reembed` | 16 | **NEW / not yet run.** Each round: take the fresh global (free backbone), freeze it, cheaply fine-tune only the **head** on trigger data → fresh + marked + cheap. Scopes: `head`/`block`/`full` | built, wired, **awaiting the sweep** |

Effort is measured per client (`compute.summary.effort_ratio_samples/gpu`,
duty cycle) and each free-rider logs a per-round decision `trace`.

## 4. Results so far (PRELIMINARY — 50-round CIFAR-100, paper-faithful)

The headline picture is the **trilemma**: pick two of {cheap, keep-your-mark,
don't-poison}. With std over seeds:

| Strategy | effort | fr_ber | acc | verdict |
|---|---|---|---|---|
| memory_exploit (frozen **replay**, warmup 8) | 0.16 | 0.15 ± 0.04 | **55 ± 1** | evades η **but poisons** the model (honest BER→0.5, FPR↑) |
| submarine **blend** (mem_blend 0.3, warmup 8) | 0.14 | 0.44 ± 0.06 | 70 ± 2 | healthy **but mark decays → caught** |
| submarine **transplant** (global + frozen mark-delta) | 0.20 | 0.48 ± 0.04 | 72 ± 0 | healthy **but mark won't transfer (nonlinear) → caught** |
| honest reference | 1.00 | 0.05 | 73 | — |

Supporting result that is **clean and holds** (E7): embedding a *generalizing*
mark needs the **full shard** — trigger-only shortcuts overfit (fr_ber 0.55–0.63
regardless of sample count) while the full shard reaches ~0.05. This is *why* the
coast attacks can't cheaply re-make the mark by the naive routes.

**Interpretation (tentative):** the coast attacks all try to *reuse* a stored
mark, which either goes stale (poisons) or decays/doesn't transfer (caught). The
untested `reembed` attack instead *cheaply re-makes* the mark on the freely-
received backbone — this is the route that should reach the empty "cheap + evades
+ healthy" corner, if any does. The sweep decides.

## 5. The theoretical angle (why a weak point should exist)

FareMark secures free-riding with an **output-layer** watermark. The mark is the
map (trigger → softmax), produced by the **last layer(s)** on top of the backbone
— and the free-rider **receives a good backbone for free** every round. Shaping
the output on its own trigger class is low-dimensional and cheap (fine-tune the
head), not full-task training. So the cost asymmetry the scheme assumes ("honest
embedding is expensive, a free-rider can't afford it") may **collapse**. `reembed`
is the direct test; the effort-vs-evasion frontier is the evidence.

Honest caveat: this is about **cost**, not undetectability. Even if `reembed`
wins, a defender could add a staleness / accuracy / backbone-consistency check
outside the watermark. The likely thesis claim is "output-layer watermarking
*alone* cannot detect free-riders cheaply."

## 6. Next steps (in order)

1. **Run `./scripts/run_full_sweep.sh`** (see §7) — measures every attack's
   frontier and, critically, tests `reembed`.
2. **Plot** with `./scripts/make_sweep_figs.sh` → per-attack `fr_ber`/`acc` vs
   knob, plus the combined **weak-point map** (fr_ber vs effort, colored by acc).
3. **Locate the weak point**: the lowest-effort config that puts `fr_ber` under η
   while `acc` stays ~72. Re-run that config at 3 seeds as the clean result.
4. **Write up**: either "output-layer watermarking is cheaply defeatable" (if
   reembed wins) or "the trilemma — cheap free-riding is structurally hard" (if
   it doesn't). Both are defensible.
5. Later: non-IID (E4) and CIFAR-10 (E6) regimes where a weak mark passes anyway;
   the theoretical proposition; collusion.

## 7. Should you run `run_full_sweep.sh` now? — YES

Push the current code first (the pod clones fresh), then run it. It is
fire-and-forget (`WAIT=0`), priority-ordered (reembed + memory finish first), ~23
runs at 1 seed. On ~4 GPUs it finishes overnight; on 1–2 GPUs it won't fully
drain, but the priority order means you'll still get the reembed frontier and
memory sweep. Command:

```bash
./scripts/run_full_sweep.sh          # 1 seed; SEEDS="0 1" for tighter bands
DRY=1 ./scripts/run_full_sweep.sh    # preview only
```

When it finishes:
```bash
RES=/mnt/nfs/home/zu/results ./scripts/make_sweep_figs.sh
```
Then upload the PNGs for analysis + deck.

## 8. Map of the docs (what to read for what)

- **STATUS.md** (this file) — where you stand, read first.
- **ADAPTIVE_ATTACKS.md** — deep reference: threat model, each attack's mechanism
  and failure mode, the mixed-attack explainer, the experiment catalog.
- **HYPERPARAMS.md** — every knob and what it does (incl. the adaptive/reembed
  knobs + compute metrics glossary).
- **RUNSHEET_ADAPTIVE.md** — copy-paste cluster commands.
- **EXPERIMENTS.md** — the family registry (what each FAMILY tag means).
- **DOCUMENTATION.md** — code↔paper map (modules, equations, tables).
- **PROJECT_PLAN.md** — the thesis pillars and overall plan.
- **README.md** / **GRETA.md** — project overview / personal log.
