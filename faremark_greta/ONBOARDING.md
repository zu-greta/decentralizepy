# Free-Rider Attacks on FareMark — Onboarding & Storyline

*A get-up-to-speed doc for a labmate joining mid-project. Read top to bottom; ~10 min.*

---

## 1. The research question

**Can a client in federated learning obtain the shared model without doing the
training, and evade FareMark's watermark-based free-rider detector — at a cost far
below an honest client's?** And, as the mirror question that leads to a defense:
**under what threshold policy does the detector remain robust to such an attack?**

---

## 2. Background in five sentences

- **Federated learning (FL):** many clients train one shared model; each trains
  locally on its own data, a server averages the updates, repeat for ~50 rounds.
- **Free-rider:** a dishonest client that wants the model but skips the expensive
  training.
- **FareMark's defense:** it forces honest training to leave a hidden **watermark**
  in the model's *softmax outputs* on secret *trigger images*. Each round the server
  reads the watermark bits from every client and computes the **Bit-Error-Rate
  (BER)**; low BER = mark present = did the work. BER above a threshold **η** ⇒
  flagged as a free-rider.
- **η is set as μ + 3σ** of honest clients' BER (the paper's rule).
- **Our project:** build the attack that beats this, characterize when it works,
  then turn it into a better defense.

---

## 3. The storyline so far (what we found, in order)

1. **The threshold is the crux.** η can be computed several ways. Read *literally*
   as a running (cumulative) average it **inflates** when honest BER drifts up, so a
   poisoning attacker can push the bar above itself — the attack "works" but it's an
   artifact. Read *fairly* (calibrate once on the converged honest BER, then freeze),
   η ≈ 0.09 and is a much harder, honest bar. **We judge everything against the fair
   (frozen) η.**

2. **A naive shortcut fails.** You cannot forge the mark by retraining only the
   output layer (`head`): the frozen backbone's features can't be reshaped, so the
   mark overfits the attacker's own trigger images and doesn't generalize to the
   server's ⇒ BER ≈ 0.42, caught.

3. **The autopilot attack.** A self-tuning free-rider that trains as little as
   possible: **warm up** honestly, then **coast** (submit the fresh global model + a
   frozen copy of the watermark direction, no training), and **tap** (a short real
   re-embed) just before its BER would cross η. It probes its own BER for free each
   round (forward pass on 16 held-out trigger images) to decide.

4. **Two effort levers.** (a) *Coasting* (skip rounds) — the classic submarine;
   `full` scope uses it but is caught because the mark decays too far between rare
   taps. (b) *Cheap shallow re-embed* — `block` scope retrains only the last stage
   (backbone frozen), cheap per round, so it can re-embed almost every round. `block`
   gets closest.

5. **The honest current result (3 seeds).** Under the fair η, `block` **rides on the
   threshold** (BER ≈ 0.10–0.13 vs η ≈ 0.09) at ~15% of honest effort — *close, but
   not reliably under*. Deeper `block2` (last two stages) helps slightly. So far this
   reads as: **a fair threshold makes FareMark roughly robust to cheap forgery; the
   free-rider gets to the boundary but not cleanly under.**

6. **Current experiments (why it grazes the line).** The attacker was aiming at a
   *wrong* η (it estimated ~0.25 from its pessimistic self-probe). Fix: during the
   forced-honest warmup the free-rider **is** an honest client, so it calibrates η on
   *those* rounds (→ ~0.09) — the same distribution the server uses. Plus a diagnostic
   **oracle** run (give it the true η) to see if staying under is even possible
   cheaply, and a **data-shard ablation** (how little data the re-embed needs).

---

## 4. Key terms (glossary)

| Term | Meaning |
|---|---|
| **Backbone / head** | Backbone = most layers, turns image → feature vector (expensive). Head = final linear layer, features → class scores (cheap). |
| **Weights** | The numbers inside layers, adjusted by training. |
| **Forward / backward pass** | Forward = predict; backward = compute weight updates. Freezing the backbone skips its (expensive) backward pass. |
| **Watermark / BER** | Hidden bit-pattern in the softmax on trigger images; BER = fraction of bits wrong vs the registered message. |
| **η (threshold)** | μ+3σ of honest BER; BER ≥ η ⇒ flagged. We use the *frozen/fair* version (~0.09). |
| **scope** | Which params a re-embed trains: `head` / `block` (last stage) / `block2` (last two) / `full`. |
| **coast / tap** | coast = submit global + frozen mark, no training. tap = a short real re-embed. |
| **effort ratio** | attacker image-passes ÷ honest image-passes. < 1 = cheaper than honest. |
| **margin** | safety gap below the (estimated) η the attacker aims for. |

---

## 5. Algorithm (pseudocode)

```
# One free-rider client, per communication round.
# State kept across rounds: honest_cal[], clean_hist[], margin, warm_done, honest_done

function PRODUCE_UPDATE(global_model, round):
    eta_est = ESTIMATE_ETA()                 # see below
    target  = max(floor, eta_est - margin)   # aim a safety margin under the line

    # ---- PHASE 1: warmup (behave like an honest client) ----
    if not warm_done:
        honest = (honest_until > 0) and (not honest_done)
        model  = TRAIN(global_model,
                       scope = "full" if honest else attack_scope,
                       data  = full_shard,           # (or reduced shard, if ablating)
                       budget= full_epoch if honest else max_batches)
        submit = MEMORY_UPDATE(global_model, model)  # keep global knowledge
        ber    = PROBE_BER(submit)                   # free: forward on 16 held-out triggers
        if honest:
            honest_seq.append(ber)
            if FLATTENED(honest_seq):                # convergence = rate of change < eps
                honest_converged = true
            if honest_converged: honest_cal.append(ber)   # calibrate eta on THESE
            if honest_converged and |honest_cal| >= 2: honest_done = true
        if ber <= floor and past_protected_window and honest_done: warm_done = true
        return submit

    # ---- PHASE 2: coast / tap ----
    coast_model = global_model + frozen_mark_delta      # no training
    ber_coast   = PROBE_BER(coast_model)
    if ber_coast < target and NOT PREDICTED_TO_CROSS(target, lookahead):
        margin = relax(margin)                          # safe -> coast longer, cheaper
        return coast_model                              # COAST (zero training cost)
    else:
        nb     = TAP_SIZE(drift, last_tap_undershot)    # adaptive: bigger if drifted more
        model  = TRAIN(global_model, scope=attack_scope, data=shard, budget=nb)
        submit = MEMORY_UPDATE(global_model, model)
        if PROBE_BER(submit) > floor: margin = tighten(margin)   # missed -> be safer
        return submit                                   # TAP (short re-embed)

function ESTIMATE_ETA():
    if oracle_eta > 0:            return oracle_eta                 # DIAGNOSTIC only
    if |honest_cal| >= 2:         return mean(honest_cal) + 3*std(honest_cal)  # THE fix
    if |clean_hist| >= 3:         return mean(clean_hist)+ 3*std(clean_hist)
    return fixed_fallback (0.25)
```

---

## 6. Schematic of the attack (per round)

```
                        ┌─────────────────────────────────────────────┐
   global model ───────▶│  PROBE own BER (free: forward on 16 triggers)│
   (from server)        └───────────────────┬─────────────────────────┘
                                             │
                          ┌──────────────────┴───────────────────┐
                          ▼                                       ▼
                 round < honest_until?                    warmup done?
                   (warmup phase)                        (coast/tap phase)
                          │                                       │
          ┌───────────────┴──────────┐              ┌─────────────┴─────────────┐
          ▼                          ▼               ▼                           ▼
   TRAIN full model,          (after convergence)  BER < target?          BER near η?
   full epoch, like            calibrate η on       │                      │
   an honest client            these honest rounds  ▼                      ▼
          │                    (η ≈ 0.09)         COAST:                  TAP:
          ▼                          │            submit global +         short re-embed
   submit + record BER               │            frozen mark             (scope=block/…)
   into honest_cal ──────────────────┘            (no training)           to drive BER↓
          │                                          │                      │
          └──────────────────────────────────────────┴──────────────────────┘
                                             │
                                             ▼
                            server reads BER; flags if BER ≥ η
                            (free-rider aims to keep BER just under the FAIR η
                             at a total effort ≪ an honest client)
```

**Timeline intuition (BER vs round):**

```
 BER
 0.5 │■ warmup (honest, full model) — BER falls like an honest client
     │ ■■
 0.2 │    ■■■■   ╱╲    ╱╲   ╱╲     coast: BER drifts UP (mark decays)
 η≈0.09│────────╳──╲──╳──╲─╳──╲────── fair threshold ────────────────
     │          tap  ╲tap ╲tap ╲     tap: short re-embed drives BER DOWN
 0.0 │_____________________________________________________________ round
        0    10          25          40          50
```

---

## 7. What to read / run first

- **Status & full detail:** `STATUS.md` (§14 mechanics from scratch, §15 watermark +
  glossary, §16 the eta-calibration fix).
- **Plots:** `scripts/plot_thresholds.py` — `seedband` (BER±std vs round),
  `evade_bars` (does it beat the FAIR η?), `submarine` (coast/tap), `estimate`
  (believed vs actual η), `knob` (a swept parameter).
- **Run an experiment:** `run_honestcal_sweep.sh` (the eta-calibration test) and
  `run_meeting_sweep.sh` (oracle-η + data-shard ablation).

## 8. Open questions (good places to contribute)

1. Does honest-round eta-calibration pull `block2` cleanly under the fair η, or does
   the attack fundamentally graze the boundary?
2. If the free-rider is *given* η (oracle), can it stay under at < honest effort? If
   not even then, the detector is robust and we pivot to the defense.
3. How little data does the re-embed need (trigger-only vs +N/common class)?
4. Would black-box / white-box watermarks resist the shallow-slice attack? (Reasoned
   yes; untested.)
5. The defense: a fair frozen threshold + a backbone-consistency check.
