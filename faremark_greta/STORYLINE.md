# STORYLINE — goals, diagrams, pseudocode

## 0. Goal
FareMark (Li et al., IEEE IoT-J 2025) detects free-riders in FL by having honest
clients embed a private watermark into the global model's softmax on a trigger class;
free-riders can't reproduce it, so their bit-error-rate (BER) is high and they're
flagged. **Goal: characterize when FareMark can be evaded and where it fails.** The
result flipped from "we built a cheap evader" to "the detector cannot separate an
embedding free-rider from an honest client at a hard trigger position" — a
false-positive / threshold-calibration limitation.

## 1. The system (FareMark)
```
   client i               SERVER (each submitted model W_i):
   trigger t_i            bits   = extract(W_i, trigger_bank[t_i], key_i)   (Eq.15)
   key   M_i   --W_i-->   BER_i  = mean(bits != target_bits_i)              (Eq.16)
   bits  B_i             flag i  iff  BER_i >= eta            (eta = FROZEN constant)
   embed W_wm <--W_g--   FedAvg aggregate -> W_global
```
- Position: `t_i = cid % num_classes`; key/bits seeded from cid.
- Honest embed: `L = CE + lambda*BCE(z,B)`; memory update (Eq.14)
  `W = beta*(memory+delta) + (1-beta)*global` keeps the mark through aggregation.
- Detection = ONE scalar eta, **now a pre-calibrated constant** (threshold.py).

## 2. The watermark pipeline (one client)
```
 trigger imgs -> softmax P -> f(p)=p^alpha (smooth tail) -> group into m blocks of l
   -> z_k = sum_j f(p)*M[k,j]           (project, Eq.1/13)
   embed: minimize BCE(z,B)             extract: bits_k = sign(mean_over_N_T z_k) (Eq.15)
                                        BER = mean(bits != B)  (honest ~0; no mark ~0.5)
```
**Why some positions are hard:** a confidently-predicted class has a peaky softmax
(high `pmax`, low `entropy`) -> almost no tail to bend -> some bits stuck -> BER floors
at ~0.10-0.20. Diagnostics now logged: server `pmax`/`entropy`/`dominance`/`trig_acc`
(wm_verify), client `wm_loss`/`trig_train_acc` (wm_client.wm_stats).

## 3. The attack: SUBMARINE free-rider  (`attacks_adaptive.make_submarine_attack`)
Be an honest client while the server calibrates eta, then coast/tap to hold the mark
just under eta at minimum cost.
```
 rounds:   1 ...... W                W+1 .............................. 50
          | WARMUP + CALIB |        | FREE-RIDE (coast / tap)            |
          | train fully,   |        | coast: submit global+mark_delta    |
          | watch own BER  |        |        (re-inject mark, no train)   |
          | converge,      |        | tap:   re-embed on fresh global,    |
          | FREEZE eta     |        |        cost = data x scope          |
          └────────────────┘        | target = eta - margin0 - safety     |
```

### Pseudocode (`SubmarineFreeRider.produce_update`)
```
ensure_triggers()                       # probe holdout + reduced tap loader (once)
if honest_clone:  return honest.produce_update()          # DIAGNOSTIC control

# WARMUP -> CALIB: honest until own probe BER converges, then K calib rounds, freeze eta
if phase in ("warmup","calib"):
    submit = honest.produce_update(); ber = probe(submit); honest_hist.append(ber)
    if phase=="warmup" and (round>=honest_min and converged()) or round>=warmup_cap:
        phase="calib"; calib_start=round
    if phase=="calib":
        own_calib.append(ber)
        if round-calib_start+1 >= K:  freeze_own_eta(); phase="freeride"
    return submit
# fixed mode forces the transition at round W-K (deterministic [W-K,W-1] window).

# FREE-RIDE
eta    = oracle_eta if given else eta_frozen
target = max(floor, eta - margin0 - safety)          # safety gap below eta
if stay_min:
    coast = global + mark_delta                      # mark_delta = memory - global
    if probe(coast) <= target and coast_streak < max_coast:
        coast_streak++; return coast                 # COAST (no training, ~free)
    coast_streak = 0                                 # else fall through to TAP
tap = embed_loop(scope=autop_scope, data=reduced|full, early_stop=False)  # cost=data x scope
return memory_update(global, tap)
```

### eta estimate (`_freeze_own_eta`) — reconstruct the SERVER's threshold from own BER
```
src = own_calib (tight/loose)  or  full honest history (cumulative)
mu, sd = mean(src), pstdev(src)
tight (default):  eta = mu + k*sd/sqrt(N)     # server averages N clients -> sd shrinks by sqrt(N)
loose:            eta = mu + k*sd
cumulative:       eta = mu + k*sd over all honest rounds
eta_frozen = max(eta, floor)
```

## 4. The detector's flaw (the turn)
```
 per-(client,round) BER (converged):        round-mean
   client0  0.10 0.10 0.10                    |
   client1  0.00 0.00 0.00   wide DOWN a col  |  tight ACROSS the row of means
   client6  0.20 0.20 0.20                     -> eta = mu+3sigma over MEANS ~ 0.09
   client7  0.00 0.00 0.00
 decision: flag iff BER_i >= eta  (applied to INDIVIDUALS: spread)
   => client6 (0.20) and client0 (0.10) flagged though HONEST (false positives)
   => a free-rider on a hard position (~0.10) also flagged
```
No single scalar eta separates an embedding free-rider from an honest hard-position
client, because after the FR embeds, both live in the same BER band.

## 5. The three experiments (see STATUS.md "EXPERIMENTS TO RUN")
```
 STEP 0  honest (multi-seed) -> calibrate the ONE eta -> plots prove it
 KNOB 1  position (hard {3,6} vs easy {1,7})   -> evasion is position-dependent
 KNOB 2  data/tap (cpc 0/5/-1)                 -> effort floor
 KNOB 3  coast vs tap-every (stay_min)         -> submarine coasting saves effort
```

## 6. Contribution arc
```
 reproduce FareMark -> build submarine FR -> it "fails" under IID
        -> realize the failure = FALSE POSITIVE (position-dependent floor)
        -> FareMark security reduces to eta-calibration (fragile)
        -> NEXT: per-position / per-client-calibrated threshold (a better detector)
```
