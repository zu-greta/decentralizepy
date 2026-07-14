# STORYLINE — goals, diagrams, pseudocode

## 0. Goal

FareMark (Li et al., IEEE IoT-J 2025) detects **free-riders** in federated learning
by having honest clients embed a private **watermark** into the global model's
softmax on a trigger class; free-riders can't reproduce it, so their bit-error-rate
(BER) is high and they're flagged. **Our goal: characterize when FareMark can be
evaded and where it fails** — not to break it blindly, but to map its limits and
motivate a better detector.

The result flipped from "we built a cheap evader" to "the detector cannot separate an
embedding free-rider from an honest client at a hard trigger position" — a
false-positive / threshold-calibration limitation.

---

## 1. The system (FareMark)

```
                          ┌─────────────────────── SERVER ───────────────────────┐
   client i               │  for each submitted model W_i:                        │
   ┌────────────┐  W_i     │    bits = extract(W_i, trigger_bank[t_i], key_i)      │
   │ trigger t_i│ ───────► │    BER_i = mean(bits != target_bits_i)                │
   │ key   M_i  │          │  eta = mu + 3*sigma  over honest BERs                 │
   │ bits  B_i  │          │  flag client i  iff  BER_i >= eta                     │
   │ embed W_wm │ ◄─────── │  FedAvg aggregate the (unflagged) models -> W_global  │
   └────────────┘  W_global└───────────────────────────────────────────────────────┘
```

- Each client is assigned a **position**: `t_i = cid % num_classes`, and key/bits
  seeded from `cid`. Honest clients embed with `L = CE + lambda*BCE(z, B)` and keep
  the mark alive through aggregation with a **memory update** (Eq. 14):
  `W = beta*(memory + delta) + (1-beta)*global`.
- Detection is a single scalar threshold η applied per client.

---

## 2. The watermark pipeline (one client)

```
 trigger images ─► model ─► softmax P ─► f(p)=p^alpha (smooth tail)
      │                                     │
      │                        group into m blocks of size l
      │                                     │
      │                        z_k = sum_j f(p) * M[k,j]        (project, Eq.1/13)
      │                                     │
   embed: minimize BCE(z, B)         extract: bits_k = sign(mean_over_N_T z_k)  (Eq.15)
      (drives sign(z_k) -> B_k)             │
                                      BER = mean(bits != B)      (Eq.16)
                                      honest -> ~0 ; no mark -> ~0.5
```

**Why some positions are "hard":** for certain (class, key, bits) the softmax tail
can't be bent far enough to flip every bit without hurting classification, so a few
bits stay wrong -> BER floors at ~0.10-0.20. That irreducible value is the **floor**.

---

## 3. The attack: AUTOPILOT free-rider

Idea: be an honest client while the server calibrates η, then coast/tap to hold the
mark just where it needs to be — at minimum cost.

```
 rounds:   1 ....... ~12            13 ................................ 50
          │ FORCED HONEST │        │ FREE-RIDE (coast / tap)            │
          │ train fully,  │        │ coast: submit carried mark (~free) │
          │ watch honest  │        │ tap:   re-embed on fresh global,   │
          │ BER converge, │        │        cost = data x scope         │
          │ FREEZE eta    │        │ target = eta - margin              │
          └───────────────┘        └────────────────────────────────────┘
```

### Pseudocode (autopilot `produce_update`)

```
ensure_triggers()                          # probe holdout + reduced tap loader (once)

if honest_clone:                           # DIAGNOSTIC control
    return honest.produce_update()         # pure honest every round

eta    = oracle_eta if given else estimate()
target = max(floor, eta - margin)

# ---- WARMUP: honest until eta is calibrated, then defect ----
if not warm_done:
    train honestly (full model, full shard)         # like an honest client
    ber = probe(submit)
    if honest_BER has flattened (rate < conv_eps for 2 rounds):
        collect converged BERs; after honest_extra rounds -> eta_frozen = estimate(); honest_done
    if (mark good AND past protect_until AND honest_done) or round >= warmup_cap:
        warm_done = True
    return submit

# ---- POST-WARMUP: coast when safe, else tap ----
if stay_min and probe(coast_state) <= target:
    return coast_state                     # COAST: no training, re-inject mark
else:
    _embed_loop(scope=autop_scope,         # TAP: cost = data (common_per_class) x params (scope)
                data=reduced or full,       #   -1=full shard, 0=triggers-only, N=+N/common-class
                early_stop=False)
    return memory_update(global, trained)
```

### eta estimate (`_eta_est`)

```
if oracle_eta > 0:              return oracle_eta          # testing shortcut
if eta_frozen is not None:      return eta_frozen          # frozen after honest phase
if >=2 converged honest BERs:   return mu + k*sigma (trimmed)   # the FR's estimate of the fair eta
else:                           fall back to recent post-embed BERs / 0.35
```

---

## 4. The detector's flaw (the story's turn)

```
 per-(client,round) BER grid (converged):
                r1    r2    r3
   client0    0.10  0.10  0.10   ┐
   client1    0.00  0.00  0.00   │  DOWN a column: wide spread (position floors)
   client6    0.20  0.20  0.20   │
   client7    0.00  0.00  0.00   ┘
   round mean 0.055 0.055 0.055  <- ACROSS a row of means: tight

 eta = mu+3sigma over {0.055, 0.055, ...}  ->  ~0.09   (built from MEANS: tight)
 decision: flag client i iff BER_i >= 0.09  (applied to INDIVIDUALS: spread)

 => client6 (0.20) and client0 (0.10) flagged though HONEST  (false positives)
 => a free-rider on a hard position (~0.11) also flagged
```

The dilemma:

```
 tight eta (over round-means, ~0.09):  catches the free-rider  BUT flags honest hard positions
 loose eta (over per-client BERs, ~0.25): spares honest clients BUT the free-rider passes too
```

No single scalar η separates an *embedding* free-rider from an honest hard-position
client, because after the free-rider embeds, the two live in the same BER band.

---

## 5. The three experiments (evidence)

```
 TEST 1  10 honest, no FR ─► per-client BER vs TWO eta defs ─► FPR under each
         goal: show the tight eta flags honest hard positions (and the dilemma)

 TEST 2  2 FRs, full scope ─► sweep data (triggers-only -> +N/class -> full shard)
         x TWO pinned positions ─► per-FR & per-honest BER + GPU/samples effort
         goal: FR = honest at same position; floor reached cheaply; effort scales with data

 TEST 3  same as TEST 2 but scope=block2 (backbone frozen)
         goal: samples flat, GPU drops -> separate "data cost" from "param cost"
```

---

## 6. Contribution arc

```
 reproduce FareMark  ->  build adaptive free-rider  ->  observe it "fails" under IID
        │                                                     │
        └─────────────────────────────►  realize the failure = FALSE POSITIVE  ◄──┘
                                                 │
             FareMark security reduces to eta-calibration, which is fragile:
               (a) tight vs loose dilemma (this project's core result)
               (b) poisoned / untrusted calibration pool
               (c) slow honest convergence under heterogeneity
                                                 │
                          NEXT: a per-position / per-client-calibrated
                          threshold that resolves the dilemma (a better detector)
```
