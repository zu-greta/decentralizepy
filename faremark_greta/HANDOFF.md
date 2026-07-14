# NEW CONVERSATION PLAN — build the FareMark results deck

Paste this whole file into a fresh chat, upload the files in §12, then hand over the
new result plots. The new Claude's job: **place every plot into the slide deck's
appendix (with the full description template in §11) and refresh the body findings.**

---

## 1. Mission for the new Claude
1. Take the existing deck (`FareMark_attack.pptx`) + `experiment_rundown.pptx`.
2. For every uploaded result plot, add an appendix slide using the template in §11
   (what it is · setup · every symbol/line/acronym · thresholds & how computed · why ·
   data choice · units + theoretical meaning · analysis · conclusion).
3. Update the body findings/numbers where the new (post-warmup-fix) results change them.
4. Keep the fairness discipline: every attack stays inside FareMark's protocol; the one
   exploited gap is the η-population ambiguity, where multiple readings fail.

---

## 2. Context & the finding (one paragraph)
We stress-test **FareMark** (Li et al., IEEE IoT-J 2025): honest FL clients embed a
private output-space watermark on their assigned trigger class; the server extracts it
on N_T=50 held-out trigger images → per-client **bit-error-rate (BER)**; flags a client
if **BER ≥ η**, η = μ+3σ over legitimate BERs. We built an adaptive free-rider (the
"autopilot"). **Finding:** the honest BER floor is *position-dependent*; no single scalar
η both catches an embedding free-rider and spares honest hard-position clients — tight η
→ ~32% honest false positives, loose η → the free-rider passes. It's a false-positive /
threshold-calibration limitation, not a cheap-evasion result. Setup: CIFAR-100 · ResNet-18
· 10 clients · 50 rounds · 5 local epochs · m=10 bits · N_T=50 · λ=5, β=0.6, α=0.4 ·
paper-faithful · 3 seeds.

---

## 3. Algorithm (FareMark)
**Honest client, per round:** load global → train `L = CE + λ·BCE(project(softmax,key), bits)`
on trigger-class images → memory update (Eq.14) `W = β(memory+Δ) + (1−β)·W_global` to keep
the mark alive through FedAvg → submit.
**Server, per round:** extract each client's bits on the trigger bank → BER = mean(bits≠target)
→ η = μ+3σ over honest BERs → flag BER≥η → FedAvg the unflagged.
**Position** = the (trigger_class, key, bits) triple assigned by `cid`
(`trigger_class = cid % num_classes`, key/bits seeded from cid). Some positions have an
irreducible BER floor 0.10–0.20 (hard); others ~0.00 (easy). Floor is like Bayes error for
the embedding sub-task — effort/scope don't move it (proven by honest-clone).

---

## 4. The attack (autopilot) — schedule & knobs
```
warmup (dynamic)    forced-honest: train FULL-SHARD honestly (== honest client) until the
                    FR's OWN probe BER converges (flat for conv_patience+1 rounds within
                    conv_eps, after honest_min, capped at warmup_cap). Position-dependent:
                    hard positions converge later => longer warmup, defect later.
CALIBRATION window  the next K converged honest rounds: η frozen here (server: all clients;
                    FR: its own BER). Tagged "calib".
free-ride           TAP (re-embed, cost = data × scope)  or  COAST (resubmit carried mark,
                    ~0 cost) when safely under target = η − margin.
autop_warmup_mode="fixed" (DEFAULT: warmup=[1,W-1], calib=[W-K,W-1], defect@W, W=autop_honest_until=12
-- position-independent, so warmup cost does not confound the position comparison)
            | "dynamic" (ends at own-BER convergence; position-dependent) -- robustness cell.
K = autop_calib_rounds (4).
```
The FR reuses the honest key/bits/λ/α/β/memory/training loop verbatim; only control flow
differs. It estimates η from its OWN calibration-window BER (`_own_calib_bers` →
`_freeze_own_eta` at the end of the window), or is given the oracle η (testing). Trace tags
each round `honest`/`calib`/`tap`/`coast` so plots read the (dynamic) window from the trace.

Free-rider options (all wired): full/block2 (`autop_scope`) · triggers→+N/class→full shard
(`autop_common_per_class`) · tap/coast (`autop_stay_min`) · oracle/estimate
(`autop_oracle_eta`) · hard/easy position (`free_rider_ids`) · IID/non-IID
(`partition`,`dirichlet_alpha`).

---

## 5. Thresholds — canonical window + the 7 definitions
`eta_calib.py` is the single source of truth. Window `[W-K, W-1]` (from `"calib"` trace
tags, else config). The **7-threshold plot** (`plot_analysis.py all_thresholds`) computes:
1. **SPEC** — μ+3σ over per-ROUND-MEAN BER of all clients in the calib window (the fair η).
2. **longer honest window** — same but over the last 20 rounds, honest clients.
3. **per-client** — μ+3σ over individual client BERs in the window (bigger σ, looser).
4. **incl. full warmup** — includes non-converged early warmup rounds (inflated).
5. **cumulative** — μ+3σ over all honest round-means (the swingy live paper-faithful one).
6. **FR-inflated** — all clients incl. defected free-riders, post-warmup (poisoned).
7. **all-honest EASY vs HARD** — from the all-honest run, split by per-class mean BER.
The tight/loose split (1 vs 3) IS the dilemma.

---

## 6. Code file map (what each does)
```
faremark/
  client.py           honest FedAvg client (base)
  server.py           FedAvg + per-round verify hook
  wm_client.py        WatermarkClient (embed + Eq.14) + build_watermarked_clients factory
  watermark.py        Eq.1–16: key/bits/project/embed/extract/BER/calibrate_eta
  attacks.py          paper baselines (previous_models, gaussian) + choose/resolve_free_riders
  attacks_adaptive.py AUTOPILOT (warmup→calib→tap/coast; the attack under study)
  wm_verify.py        server extraction + η + per-client BER records (wm_per_client)
  compute_meter.py    per-client effort (samples, gpu_ms, flops)
  eta_calib.py        canonical calibration window + frozen_eta + all_thresholds  ★shared
  datasets.py         IID / Dirichlet shards           utils.py  seed/logger/eval
  config.py           ExpConfig + CONFIGS (autopilot = idx 14, AUTOPILOT_IDX)
  models.py           build_model (resnet18/alexnet/smallcnn)     manifest.py  result metadata
scripts/
  run_experiment.py   one (config,repeat) → result.json (manifest+compute+history)
  run_all.sh          THE sweep: matrix over all knobs + PLOT per partition
  submit_experiment.sh  cluster submit bridge (env → --flags)
  plot_tests.py       test1_fpr (FPR) + test_data (per-FR/honest BER + effort)
  plot_analysis.py    timeline · frontier · scorecard · all_thresholds  (uses eta_calib.py)
  plotstyle.py        shared palette/helpers (C_HONEST blue, C_FR orange, OKABE)
```
`result.json` schema the plots read: `manifest.{family,sweep_level}` · `history[t].{round,
wm_per_client:[{cid,trigger_class,ber,is_free_rider,flagged}], wm_benign_ber, wm_eta_round}`
· `compute.summary.{honest/fr_mean_gpu_ms, honest/fr_mean_samples, effort_ratio_*}` ·
`compute.per_client[cid].{is_free_rider, total, trace:[{round,action,...}]}`.

---

## 7. All tuning knobs (config.py ExpConfig → CLI `--flag` / env `VAR`)
| knob | default | meaning |
|---|---|---|
| autop_warmup_mode | **fixed** | "fixed" (DEFAULT, headline): warmup ends at W, calib `[W-K,W-1]`, position-independent. "dynamic": ends at own-BER convergence (position-dependent) — run as robustness cell (tier5, `_dyn` families) |
| autop_honest_min | 6 | dynamic: never defect before this round |
| autop_warmup_cap | 15 | dynamic: hard stop if never converges |
| autop_conv_eps / _patience | 0.03 / 2 | dynamic: flat for (patience+1) rounds within eps => converged |
| autop_honest_until (W) | 12 | fixed-mode warmup end / dynamic fallback |
| autop_calib_rounds (K) | 4 | K converged rounds calibrate η (dynamic `[conv,conv+K-1]`; fixed `[W-K,W-1]`) |
| autop_oracle_eta | 0.0 | >0 → FR given η (testing); 0 → FR estimates |
| autop_eta_k | 3.0 | k in FR's own μ+kσ estimate |
| autop_margin0 | 0.06 | target = η − margin |
| autop_floor | 0.05 | "mark good" bar |
| autop_common_per_class | −1 | data/tap: −1 full shard, 0 triggers-only, N +N/common-class |
| autop_scope | full | params/tap: full · block2 · block · head |
| autop_stay_min | False | coast when safe (submarine) vs tap every round |
| autop_honest_clone | False | DIAGNOSTIC: pure honest every round (floor control) |
| free_rider_ids | "" | pin FR cids e.g. "3,6" (position control) |
| partition / dirichlet_alpha | iid / 0.5 | IID vs Dirichlet non-IID skew |
| calib_on_all | False | calibrate η over ALL clients (η-poisoning) |
| watermark, wm_lambda, wm_beta, wm_alpha, wm_num_triggers, paper_faithful | on/5/0.6/0.4/50/True | scheme |

---

## 8. Current status
- ESTABLISHED (IID): position-dependent floor (honest-clone); ~32% honest FPR; data
  ablation reproduces paper Table V (triggers-only overfits → caught) and the cost/evasion
  tradeoff; easy-class selection ruled out by threat model (server-assigned).
- CODE: warmup fixed to full-shard-honest (FR pays honest warmup → cost floor ~24%);
  canonical calibration window used by attack + plots + FR estimate; 7-threshold plot;
  full-matrix run_all.sh. All prior runs must be RE-RUN (warmup fix changes effort + η).
- RUNNING NEXT: the full matrix (§9). Non-IID, coast/submarine, oracle-vs-estimate are new.

---

## 9. Experiment matrix (what's being run)
partition {iid, dir0.1, dir0.5, dir1.0} × η {oracle, estimate} × scope {full, block2} ×
mode {tap, coast} × data {0,5,10,20,50,−1} × position {posA 3,6 / posB 1,7} × 3 seeds,
plus one all-honest run per partition. Family names: `{part}_{eta}_{scope}_{stay}_{pos}`,
all-honest `t1_{part}`, sweep_level = cpc. Commands: `./run_all.sh quick` (sanity),
`./run_all.sh matrix` (or slice with PARTS=…), `RES=… ./run_all.sh PLOT <partition>`.

---

## 10. Expected results (to sanity-check plots against)
- IID all-honest → ~41% FPR tight η (quick slice; was ~32% pre-clean-harness), ~0-4% loose. posA hard → FR 0.11–0.20 above fair η →
  caught at every cpc. posB easy → FR ≈0.00 from +5/class → passes. triggers-only → ~0.45
  (overfits, caught). block2 → same BER, ~35% less GPU. coast → cheaper at easy, still
  caught at hard. Effort floor: triggers-only FR ~24% of honest (NOT ~0 — honest warmup).
- Non-IID → higher/noisier fair η; non-monotonic in α; calibration-timing matters; report
  under the frozen-window η, not cumulative.

---

## 11. APPENDIX PLOT TEMPLATE (use for EVERY uploaded plot)
Each plot → one appendix slide: image left, description right with these bold-labeled
fields (keep ~10pt, ~10 fields; verify no overflow):
- **What it is** · **Setup** (config, partition, positions, scope, cpc, η mode, seeds)
- **Lines/symbols** (every colour/marker/band) · **Thresholds** (which η lines, HOW each is
  computed, and the VALUES) · **Acronyms** (BER, η, cpc, FR, block2)
- **Why we ran it** · **Data choice** (why these cpc levels / positions)
- **Units** (x and y; theoretical meaning — e.g. "image-passes TOTAL over the run: honest
  1.25×10⁶ = 25k/round×50; GPU-ms ×10⁶ ≈ 21 min/client") · **Analysis** · **Conclusion**
Plot types to expect (from run_all.sh PLOT): test1_fpr, test_data ×4, timeline, frontier,
scorecard, all_thresholds — per partition. See §10 for expected shapes.

Deck style (pptxgenjs, LAYOUT_WIDE): NAVY 1E2761 headers, ICE EAF0FA cards, honest=BLUE
0072B2, free-rider/attack=ORANGE D55E00, GREEN 2C7A3F = fair/safe, RED B23A2E = limitation.
Fonts Cambria (head) + Calibri (body). Validate with the pptx skill, render to check overflow.

---

## 12. Files to upload to the new conversation
**Docs:** this file · `STATUS.md` · `CODE_MAP.md` · `STORYLINE.md` (context).
**Decks:** `FareMark_attack.pptx` (extend its appendix) · `experiment_rundown.pptx`.
**Code (the consistent set — deploy/inspect together):** `config.py · attacks_adaptive.py ·
wm_client.py · attacks.py · watermark.py · wm_verify.py · client.py · server.py ·
compute_meter.py · eta_calib.py · datasets.py · utils.py · manifest.py · models.py ·
run_experiment.py · submit_experiment.sh · run_all.sh · plot_tests.py · plot_analysis.py ·
plotstyle.py`.
**Reference:** the FareMark paper PDF.
**Results:** the new `figs/**/*.png` from `run_all.sh PLOT`, and a few `result.json` files
so the new Claude can read exact numbers for the appendix descriptions.

---

## 13. Ground rules (carry over)
Be honest about negative results (the false-positive limitation IS the finding). Keep the
free-rider modular (reuses honest modules; only control flow differs). Judge evasion vs the
frozen calibration-window η, never the cumulative. Distinguish per-round-mean vs per-client η
(the dilemma). Pin positions + average ≥3 seeds before quoting numbers. Trigger class is
server-assigned — easy-class selection is off-limits. Every attack stays inside the protocol.