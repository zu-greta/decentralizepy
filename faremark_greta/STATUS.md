# STATUS — analysis, results, conclusions (with provenance)

Dated snapshot. Each claim lists where it comes from and the setup that produced it.

Standard setup unless noted: **CIFAR-100, ResNet-18, 10 clients, 2 free-riders,
50 rounds, 5 local epochs, batch 16, paper-faithful (random keys, full softmax,
cumulative μ+3σ), 3 seeds.** Detector flags client i iff BER_i >= η, with η = μ+3σ.

---

## The finding (one paragraph)

FareMark's detector cannot separate an adaptive free-rider from an honest client at
a "hard" trigger position, because **η is calibrated on the per-round MEAN honest
BER (tight, ~0.09) but applied to individual clients (whose BER is position-
dependent and ranges ~0.00-0.20).** A free-rider that trains identically to an
honest client lands on its position's BER floor; if that floor is above η it is
flagged — and so is an honest client at the same position. This is a dilemma, not a
tuning issue: **any η tight enough to catch an embedding free-rider also false-
positives honest hard-position clients; any η loose enough to spare them lets the
free-rider pass.**

---

## Concepts

- **BER floor** = the lowest bit-error-rate a client can reach for its assigned
  watermark, no matter how much it trains. Analogous to irreducible/Bayes error, but
  for the embedding sub-task. Immovable by effort.
- **Position** = the (trigger_class, key, target_bits) triple assigned to a client
  slot: `trigger_class = cid % num_classes`, key & bits seeded from `cid`
  (wm_client.build_watermarked_clients). Different position -> different floor
  (easy ~0.00, hard ~0.10-0.20). Server-assigned; the attacker cannot choose it.
- **Per-round mean honest BER** = average of the honest clients' BERs in one round
  (one number/round). Tight, because averaging smooths the position spread.
- **Per-client BER** = the individual grid of (client, round) BERs. Wide, because it
  contains the hard-position clients.
- η = μ+3σ over the **round-mean series** => ~0.09 (as implemented, wm_verify.py).
  η = μ+3σ over the **per-client pool** => much higher (σ inflated by hard positions).

---

## Results and provenance

### R1. The floor is structural, not the attacker's embedder — ESTABLISHED
- **Evidence:** `seedband_clone_full` (honest-clone control): a free-rider embedding
  via the exact honest path plateaus at ~0.10-0.11, same as the autopilot.
- **Setup:** config 14, `AUTOP_HONEST_CLONE=1`, oracle η, full scope, 3 seeds.
- **Conclusion:** effort/data/scope do not move the floor.

### R2. Honest false positives at hard positions — ESTABLISHED (headline)
- **Evidence:** `fpr_all_honest` (attack=none): ~30% of honest client-rounds flagged
  under η=0.090; honest clients at classes 0/2/3/4/5 sit ~0.10 and class 6 ~0.20,
  while 1/7/8/9 sit ~0.00.
- **Setup:** config 14, `ATTACK=none`, 3 seeds.
- **Conclusion:** the tight η flags honest hard-position clients. This is the
  strongest, most defensible result.

### R3. Effort meter fixed — ESTABLISHED
- **Evidence:** submarine/timeline arms show effort ~100-105% (probe forward-passes
  no longer counted as training samples).
- **Setup:** oracle, full scope, 3 seeds.
- **Provenance:** compute_meter.record_forward_only vs record_batch;
  _embed_loop probes only when early_stop=True.

### R4. Data ablation reproduces paper Table V — ESTABLISHED
- **Evidence:** `data_oracle_full`: triggers-only -> BER ~0.5 (overfits, fails);
  +5/class -> ~0.10 at ~14% cost; full shard -> ~0.09 at 100% cost.
- **Setup:** config 14, oracle, full scope, `autop_common_per_class in {0,5,10,20,50,-1}`.
- **Conclusion:** the mark generalizes with a few common images; floor reached cheaply.

### R5. The eta-calibration dilemma — ANALYZED, not yet MEASURED both ways
- **Evidence:** code path in wm_verify.py (round-mean calibration) + the fpr plot.
  The per-client-η arm has NOT been run yet.
- **Status:** OPEN. `plot_tests.test1_fpr` computes BOTH η definitions and both FPRs
  from one all-honest run — running Test 1 closes this.

### R6. Easy-class selection is off the table — ESTABLISHED (threat model)
- **Evidence:** `trigger_class = cid % num_classes` (server-assigned); paper §IV-D /
  Fig. 4 Stage I. The free-rider inherits its class and never reassigns it.
- **Conclusion:** the attacker cannot pick a low-floor position under the paper's model.

### R7. Non-IID — REVISED (weaker than first thought)
- **Evidence:** seedband_noniid_a{0.1,0.5,1.0} vs the two η definitions.
- **Status:** the α=0.5 "evasion" depends on which η you read (frozen-on-early-window
  ~0.23 vs converged ~0.11). Under converged η the FR is caught at α=0.5 and α=1.0;
  only severe α=0.1 gives genuine cover, and there honest watermarks also fail.
- **Action:** report under BOTH η definitions; do not headline α=0.5.

### R8. eta-poisoning (calib_on_all) — PRELIMINARY
- **Evidence:** seedband_poison_eta: FR converges under η with 5 FRs + calib_on_all,
  but the FRs sat near ~0.06 (below the usual floor) -> success partly position luck.
- **Action:** re-run logging each FR's position to separate "η rose" from "easy position".

---

## Current conclusion

Under plain IID with a fair η, the adaptive free-rider does not evade — but that
"failure" is the result: it is indistinguishable from an honest client at a hard
position, and the threshold that catches one flags the other. FareMark's security
reduces to η-calibration, which is fragile in specific ways (tight vs loose dilemma;
poisoned/untrusted pool; slow honest convergence under heterogeneity). The
attacker-side knobs (effort, data, scope, oracle) are exhausted; the remaining
exploitable levers are all on the threshold/pool side.

---

## Caveats on the current numbers

- The specific figures above (0.09, ~0.11, ~30%) come from runs BEFORE the cleaned,
  position-pinned harness. Position luck (which cids were free-riders) confounded
  them. Re-run the cleaned `run_tests.sh` (pinned POS_A/POS_B, 3 seeds) before
  quoting any number in a writeup.
- The per-client-η arm (R5) has not been run; the dilemma is currently argued from
  code + the FPR plot, not measured end-to-end.

## What to run next (in order)

1. Cleaned Test 1/2/3 (pinned positions, 3 seeds) -> confirm R2, R4 and MEASURE R5.
2. eta-poisoning with position logging -> firm up R8.
3. Non-IID reported under both η definitions -> correct R7.
4. (Design direction) a per-position / per-client-calibrated threshold as the
   "fix" that resolves the dilemma — the natural next contribution.
