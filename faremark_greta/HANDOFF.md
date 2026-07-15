# FareMark results deck — build brief (handoff to a fresh conversation)

**Your job in the new conversation:** build a fresh slide deck presenting these results, in the
same visual style/structure as the existing `FareMark_attack.pptx` (which the user will upload as
the style reference), on the APPROVED palette below. Body slides tell the story; the appendix has
one slide per figure using the appendix template in §7. Pull exact numbers from any uploaded
`result.json`; otherwise use the numbers embedded here (they were read off the figures).

Read the pptx skill (`/mnt/skills/public/pptx/SKILL.md`) before building. Validate + render to
check overflow before finishing.

---

## 1. The contribution (the one thing the deck must land)

FareMark (Li et al., IEEE IoT-J 2025) detects free-riders in federated learning by having honest
clients embed an m-bit output-space **watermark** on their assigned trigger class; the server
extracts it, computes each client's **bit-error-rate (BER)**, and flags a client if `BER ≥ η`,
with `η = μ + 3σ` over legitimate clients' BER.

**Finding (a negative result — this IS the contribution):** the honest BER floor is
**position-dependent** — it depends on which trigger class / bit-position a client is assigned.
Easy positions sit at BER ≈ 0.00; hard positions sit far higher (≈0.10–0.20 under IID, up to
≈0.87 under non-IID). But `η` is calibrated on the per-round **MEAN** honest BER (tight) and then
applied to **INDIVIDUAL** clients (wide spread). **No single scalar η both catches an embedding
free-rider and spares honest hard-position clients.** Tight η → many honest false positives; loose
(per-client) η → the free-rider passes. It is a **false-positive / threshold-calibration
limitation**, not a cheap evasion trick — and it gets dramatically worse as data becomes non-IID.

An **adaptive "autopilot" free-rider** demonstrates the gap: it trains honestly during a warmup,
estimates η from its own BER, then re-embeds ("taps") just enough to keep its BER under η.

---

## 2. Experimental setup (state on an early slide + every appendix slide)

CIFAR-100 · ResNet-18 · 10 clients · 50 rounds · 5 local epochs · watermark bits m=10 ·
N_T=50 trigger samples · λ=5 · β=0.6 · α=0.4 · **paper-faithful** (random keys, no exclusion,
live η = cumulative μ+3σ) · config index 14 (AUTOPILOT_IDX).

**Free-rider schedule = FIXED warmup (default):** forced-honest full-shard training for rounds
1–11; **calibration window = rounds [8,11]** (η frozen here, all clients honest); free-riding from
round 12. Fixed (not dynamic) so warmup length is constant across positions and the position effect
isn't confounded by warmup cost. (A dynamic convergence-based mode exists and was run as a
robustness cell → the `*_dyn` families.)

**"fair η" in the plots = the offline frozen-window η** (μ+3σ over the round-MEANS inside the
calibration window). This is the correct yardstick for "caught vs hidden" — NOT the live cumulative
η (which is inflated by noisy early rounds). Verdicts are judged against this frozen η.

---

## 3. Results — numbers to put in the body (read off the figures)

### 3.1 The headline dilemma (IID)
- Honest mean BER = **0.057**. Positions split hard vs easy: cls6 ≈ 0.225 (hard), cls0/2/4/5 ≈ 0.10,
  cls3 ≈ 0.07 (with outliers to 0.40), cls1/7/8/9 ≈ 0.00 (easy).
- **Tight η (round-mean) = 0.087–0.091 → honest FPR ≈ 41%.**
- Loose η (per-client) = 0.28–0.31 → FPR ≈ 0–4% (but a free-rider hides under it).
- Free-rider by training data (posA = hard {3,6}): triggers-only BER 0.63 (overfits → CAUGHT),
  **+5/class BER 0.10**, full-shard 0.14 — all above fair η 0.087 → **hard position always caught (IID)**.
- posB = easy {1,7}: +5/class and full-shard reach BER 0.00 → **easy position EVADES from +5/class**.
- Effort floor: triggers-only ≈ **23% of honest** image-passes (the warmup cost it must pay);
  +5/class ≈ 31%; full-shard ≈ 100%.
- `all_thresholds` (IID): SPEC 0.087 (FPR 41%) · longer-window 0.099 (41%) · per-client 0.282 (4%) ·
  incl-full-warmup 0.614 (0%) · cumulative-live 0.343 (0%) · FR-inflated 0.233 (4%) ·
  all-honest EASY 0.000 (**FPR 100%**, the degenerate η=0 case) · all-honest HARD 0.315 (0%).

### 3.2 Non-IID makes it worse (the money table)

| partition | seeds | honest mean BER | tight η (SPEC) | FPR@tight | per-client η (FPR) | hard posA @ +5/class | detector status |
|-----------|-------|-----------------|----------------|-----------|--------------------|----------------------|-----------------|
| IID       | 1*    | 0.057           | 0.087          | **41%**   | 0.31 (0–4%)        | 0.10 → **caught**    | dilemma present |
| dir1.0    | 1     | 0.190           | 0.315          | 20%       | 1.11 (0%)          | 0.74 → caught†       | dilemma present |
| dir0.5    | 3     | 0.106           | 0.270          | 12%       | 0.84 (0%)          | 0.19 → **EVADES**    | weakening       |
| dir0.1    | 1     | 0.375           | 0.431          | **60%**   | 1.09 (0%)          | 0.41 → **EVADES**    | **fully broken**|

\* IID is the quick slice (1 seed).  † dir1.0 posA is caught only because that position is
pathologically hard for the free-rider too (BER ~0.74, and honest cls6 ≈ 0.87 there) — not because
detection works.

**The trend that matters:** as data becomes more non-IID, the honest floor rises → fair η rises →
free-riders hide more easily, while the tight-η false-positive rate stays high or grows. Under
**dir0.1** the tight η flags **60% of honest clients AND still lets the free-rider hide at
+5/class** — there is no usable threshold.

Per-partition detail (fair η ; FR BER trigonly / +5cls / full ; verdicts):
- **dir0.1** fair η 0.43: posA 0.65(caught)/0.41(hidden)/0.40(hidden); posB 0.42/0.20/0.25 (mostly hidden). FPR 60%.
- **dir0.5** fair η 0.27: posA 0.45(caught)/0.19(hidden)/0.21(hidden); posB 0.43(caught)/0.03(hidden)/0.10(hidden). FPR 12%.
- **dir1.0** fair η 0.32: posA 0.74/0.74/0.73 (all caught, pathological); posB 0.35(caught)/0.00(hidden)/0.02(hidden). FPR 20%.

---

## 4. Caveats / threats to validity (needs its own slide + a line on affected appendix slides)
1. **IID = quick slice, 1 seed, pre-fix code** (fixed schedule [8,11]; the old inert 0.25 floor —
   which did NOT affect paper-faithful η, so fair η 0.087 stands). Treat as directionally solid,
   numbers provisional.
2. **dir0.1 and dir1.0 are single-seed** (tier3 ran them at SEEDS=0); **dir0.5 has 3 seeds**. This is
   why honest floor is non-monotonic (dir1.0 0.190 > dir0.5 0.106): a 1-seed artifact. The clean
   monotonic trend holds across the seeded/extreme points IID < dir0.5 < dir0.1.
3. **dir1.0 posA flat ≈0.74 regardless of data** is a non-IID shard-imbalance effect (those clients
   barely have trigger-class samples) — real but seed-sensitive; confirm with more seeds before leaning on it.
4. **Effort ratios exceed 100%** in some non-IID cells (FR shard larger than the honest average), so
   keep "cheaper than honest" claims on the **IID** numbers.
5. block2 (cheaper-scope) and coast-mode cells were not in this batch — the block2 columns are blank
   in the scorecards. If the user uploads them, add the scope/mode story; else omit.

---

## 5. Approved palette (use verbatim)
- **Chrome:** ink/headers `#16324F` · accent teal `#2E6E8E` · card paper `#EEF3F8` · rule `#C9D6E3` · body text `#223244`
- **Data (Okabe–Ito, colour-blind-safe, matches the plots):** honest `#0072B2` · free-rider `#D55E00` ·
  fair/safe/hidden `#009E73` · caution/limitation `#E69F00` · spare 5th series `#CC79A7`
- **Fonts:** Cambria (headers) · Calibri (body). (Swatch: `palette_swatch.png`.)

---

## 6. Proposed slide order (body) — adjust as needed
1. Title + one-line contribution.
2. TL;DR: the negative result in 3 bullets (position-dependent floor → no single η works → worse under non-IID).
3. FareMark in one slide: embed → extract → per-client BER → flag if BER ≥ η=μ+3σ.
4. The adaptive free-rider: honest warmup → freeze η on calibration window → tap under η.
5. Core problem: honest floor is position-dependent  → `test1_fpr_iid`.
6. The dilemma: tight round-mean η (41% FPR) vs loose per-client η (FR hides)  → `all_thresholds_iid`.
7. What the FR does: effort/BER frontier — triggers-only overfits, +5/class is the cheap hidden point → `frontier_iid` + `scorecard_iid`.
8. Position matters: hard posA vs easy posB → the two `iid_*_tap_pos{A,B}` sweeps.
9. Non-IID makes it worse: the §3.2 table + dir0.1 catastrophe → `test1_fpr_dir01`, `all_thresholds_dir01`, `scorecard_dir0*`.
10. Effort cost: FR pays a ~23% warmup floor then coasts (IID only).
11. Conclusion / implications for watermark-based FR detection.
12. Limitations & threats to validity (§4).

---

## 7. Appendix slide template (one per figure)
For each plot, one slide covering, in order: **what it shows · setup (§2) · lines/symbols legend ·
thresholds shown + their values · acronyms (BER, FPR, η, cpc, posA/posB) · why it matters ·
data/partition choice · units (BER unitless 0–1; effort = image-passes or GPU-ms, ratio to honest) ·
analysis (what the reader should notice) · conclusion (one sentence)**. Keep verdict language tied
to the **fair frozen η**.

Figure inventory per partition (naming): `test1_fpr_<p>`, `all_thresholds_<p>`,
`<p>_estimate_full_tap_posA`, `..._posB` (BER + effort sweeps), `timeline_<p>`, `frontier_<p>`,
`scorecard_<p>`, where `<p>` ∈ {iid, dir01, dir05, dir10}. Dynamic-warmup robustness = `*_dyn`
families (figs/iid_dyn). Some slices (coast, oracle, block2) may arrive later — add if uploaded.

---

## 8. What to upload to the new conversation
Essential: **this file** · `FareMark_attack.pptx` (style reference) · `palette_swatch.png` ·
all result PNGs you have (IID 5 + non-IID 20 + whatever else you generate) · a few `result.json`
files per family (lets the new Claude pull exact numbers instead of trusting the read-offs here).
Helpful: `NEW_CONVERSATION_PLAN.md` (has the appendix template + matrix), `STORYLINE.md`
(diagrams/pseudocode), `STATUS.md`, `CODE_MAP.md`. Not needed for the deck: the .py/.sh code.

Tell the new conversation: "Build the FareMark results deck per DECK_HANDOFF.md; here are the
figures and the existing deck for style. Palette is approved. Ask me before inventing any number
not in a result.json or the handoff."
