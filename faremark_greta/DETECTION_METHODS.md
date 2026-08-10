# Free-Rider Detection: How to Detect, When to Flag, and What Happens Next

> A standalone section for the paper on the *mechanism* of detection: how the server turns a watermark reading into a decision, at what moment it should flag, and what the consequence of crossing the threshold is. Written for a reader with zero background. Grounded in the FareMark paper and the related detectors it compares against.

---

## 0. The decision the server has to make

Every round, each client sends a model update. The server wants a rule that sorts clients into **"contributed"** (keep) and **"free-rode"** (act on). A detection method is fully specified by answering four questions:

1. **What signal** does the server read from the update? (a watermark BER, a gradient-similarity score, an anomaly score, …)
2. **What threshold / rule** turns that signal into a binary flag?
3. **When** in training is the flag applied and trusted? (early rounds are unreliable)
4. **What consequence** follows a flag? (exclude, downweight, deny the model, forfeit reward, …)

This project's negative result is precisely that for the *watermark-BER* signal, **no answer to (2) works** — there is no threshold with both low false-positives and useful recall (Section 3 of the wrap-up). This document lays out the space of possible answers so the paper can say clearly *which* detection designs it defeats and which remain open.

---

## 1. How detection works in FareMark (the method we attack)

**Signal.** For each client *i*, the server pushes N_T trigger-class images through the client's submitted model, extracts the m-bit watermark from the softmax output, and computes the **bit-error-rate** `BER_i = (1/m)·Σ|b̂_k − b_k|` against the client's registered message (FareMark Eq. 15–16). BER = 0 is a perfect mark; BER = 0.5 is a coin flip (no mark).

**Rule.** Flag client *i* as a free-rider iff **`BER_i ≥ η`** (FareMark Eq. 16). One global threshold η, the same for every client.

**How η is chosen.** FareMark sets **`η = μ + 3σ`**, where μ and σ are the mean and standard deviation of the *honest* bit-error-rate observed "across many rounds of federated training" (FareMark, Sec. IV-D3). The stated intent is a 3-sigma control chart: honest clients should almost never exceed μ+3σ, so anyone who does is anomalous.

**When it fires.** FareMark verifies **every round** (it defines one "communication round" as uploading every ten local training epochs, and runs verification each such round). But it reports (Fig. 8, Sec. V-C) that detection is *unreliable early*: for roughly the first ~30 rounds the honest watermark itself has not converged (honest BER is high, watermark extraction accuracy ~40% at the start), so early flags are meaningless. FareMark's own finding is that only **after ≈30 rounds** do honest clients reach >98% extraction while free-riders stay below 40% — i.e. the flag is only trustworthy *after the honest mark has converged*.

**Consequence in FareMark.** FareMark is framed as *detection + IPR*: a flagged client is one the server can prove did not contribute (it cannot produce the watermark on demand), so it can be **denied ownership / excluded from the model's beneficiaries**. FareMark does not itself run a removal-and-retrain loop; the flag is evidentiary.

**The three ways FareMark's own text lets a free-rider fail (all crude):**
- *Fabrication* (previous-models / Gaussian, Eqs. 17–18): no real training → no mark → BER ≈ 0.5 → caught.
- *Train-then-attack* (Table IV): trains early, then coasts; the mark decays after it stops → caught once it decays.
- *Few-trigger-sample* (Table V): trains on too few trigger images → the mark overfits those images and fails on the server's held-out bank → caught.

Our attackers (reduced-data, and the adaptive submarine) are exactly the case FareMark never tests: a client that **holds a valid key and does just enough real, generalising work to keep BER < η at minimal cost**.

---

## 2. How the *related* detectors decide (from the papers in the wrap-up)

FareMark positions itself against non-watermark detectors. Understanding how *they* answer questions (1)–(4) shows why the field still lacks a working rule, and gives us the right baselines.

### 2.1 Contribution-evaluation detectors — signal = gradient similarity

- **RFFL (Xu & Lyu, 2020).** *Signal:* cosine similarity between each client's uploaded gradient and the aggregated gradient, accumulated into a per-client **reputation** score. *Rule:* clients whose reputation falls below a share threshold are flagged and **removed**. *When:* reputation accumulates over rounds; low-reputation clients are dropped progressively. *Consequence:* removal from aggregation **and** proportional denial of the final model ("fairness" — you get model quality proportional to contribution).
- **DSGMF / Xu et al. (2022) "double security guarantee".** *Signal:* a contribution score from gradient statistics. *Rule + consequence:* score below a threshold → identified as a free-rider and removed to protect the global model.

*Why they can be evaded / why FareMark critiques them:* they judge **similarity, not effort**. A free-rider that submits a plausibly-aligned update (e.g. trained on a little real data, or a well-crafted extrapolation) scores as "similar" and passes. They also need the honest gradients as a reference population, which degrades as the free-rider fraction grows.

### 2.2 Anomaly-detection detectors — signal = reconstruction error

- **STD-DAGMM / DAGMM (deep autoencoding Gaussian mixture, Lin et al. 2019 line).** *Signal:* an autoencoder is trained on *benign* updates; a client's **reconstruction error + likelihood under a Gaussian mixture** is its anomaly score. *Rule:* score beyond a cutoff → free-rider. *When:* needs a pretraining phase on enough benign clients before it can score. *Consequence:* flag / remove.

*Failure mode (FareMark's stated critique):* accuracy **collapses as the free-rider proportion rises** — the "benign" population the autoencoder assumes becomes contaminated, so the anomaly boundary drifts toward the attackers. This is the majority-adversary regime.

### 2.3 Hybrid — FRAD (Wang et al., 2024)

*Signal:* contribution evaluation **+** reputation folded into a DAGMM anomaly mechanism. *Claim:* better behaviour when free-riders are the majority. *Consequence:* flag / remove. It is the state of the art among *non-watermark* detectors and is the natural third baseline alongside RFFL and STD-DAGMM.

### 2.4 Server-side watermark (WAFFLE) — why it can't police free-riders at all

*Signal:* a watermark the **server** embeds into the global model after aggregation. *Why it's irrelevant here:* a free-rider's submission is just (a copy of) the global model, which already carries the server's mark — so the server's own watermark can never distinguish a free-rider from an honest client. This is *why* free-rider detection needs **client-side** marks (FedIPR, FareMark) — and therefore why our attack, which targets the client-side reader, hits the only viable design.

---

## 3. The space of thresholding rules (question 2, in full)

The single most important design choice is how the scalar signal becomes a flag. For the BER signal, we enumerated every reasonable rule (wrap-up Section 3.4; `plot_all_thresholds.py`). They fall into families:

| Family | Rule | What it controls | Our finding |
|---|---|---|---|
| **Parametric μ+kσ** | `η = μ + 3σ` of honest BER | assumes Gaussian honest tail | On CIFAR-100 lands **below 1/m → degenerate**; even so flags 31% of honest clients. FareMark's own rule. |
| **Robust location/scale** | median + 3·MAD | outlier-immune | Collapses to 0 when >half of honest BER = 0 → flags everyone. |
| **Trimmed / clipped** | drop tails, then μ+3σ | excludes hard classes from their own calibration | Non-degenerate but only reaches ~10% FPR at best recall. |
| **Percentile** | honest p95 / p99 | fixes FPR by construction | p95 → 5% FPR but recall ≤ 0.17 on our attackers. |
| **Per-client vs round-mean** | σ measured on the population the test is applied to | the "3σ" only means 99.7% if σ matches the tested population | The only *true* 3σ rule (per-client) sits at η≈0.264, which catches **nothing**. |

**The structural trap — the `1/m` floor.** BER can only take the values 0, 1/m, 2/m, …. Any η in the open interval (0, 1/m) produces the **identical** detector as "flag if ≥ 1 bit is wrong." On CIFAR-100 (m=10), 1/m = 0.10, and FareMark's μ+3σ recipe lands at ~0.064–0.084 < 0.10 — so the carefully-calibrated threshold does **no work at all**. On CIFAR-10 the paper's headline needs m=1, where BER is a single coin flip and the "threshold" is trivial.

**Per-class thresholds don't rescue it.** One might give each trigger class its own η (since class difficulty spans 100×). But the free-rider's own class is exactly where its BER matches the honest floor (best-balanced-error = 0.500 per class, wrap-up Section 6.3), so a per-class oracle threshold collapses recall to 0.00. Averaging across classes only *looks* separable because of the pooling illusion.

---

## 4. When to flag (question 3)

Timing is a real design axis, not a detail:

- **Not before the honest mark converges.** FareMark's own Fig. 8 shows honest extraction is ~40% early and only exceeds 98% after ≈30 rounds. Flagging during warmup produces false positives against honest clients whose mark hasn't formed. **Rule of thumb: only trust the flag on the converged tail** (we read the last 20 of 50 rounds). This is also why the submarine's *warmup* is nearly free of detection risk — the server isn't yet acting.
- **The calibration-vs-application circularity.** η must be frozen from a calibration window and then applied to *later* rounds. A threshold recomputed on the very round it judges is circular (and a free-rider in the calibration pool poisons it — the `calib_on_all` demo). Our pipeline freezes η offline from honest-only seeds and draws it as a fixed line.
- **Single-round vs persistence.** Flagging on a single round's BER is noisy (BER is quantised and seed-sensitive). A **persistence rule** — flag only if BER ≥ η for R consecutive rounds — cuts false positives but *widens the coasting window* an adaptive free-rider can exploit (it only has to surface briefly). The graft submarine is designed against exactly a per-round rule; a persistence rule makes it *easier*, not harder, to hide.
- **Consequence timing.** If crossing η triggers **immediate exclusion**, one unlucky honest round removes a good client. If it triggers a **strike/reputation decrement** (RFFL-style), the system tolerates noise but a free-rider that stays just under η never accrues strikes.

---

## 5. What crossing the threshold should *do* (question 4)

The consequence is a policy choice with different attack surfaces:

| Consequence | Description | Attacker's counter |
|---|---|---|
| **Hard exclusion** | drop the flagged client from aggregation this round | one honest false-positive loses a real contributor; free-rider just needs BER < η |
| **Reputation / strikes** | decrement a running score; remove after k strikes (RFFL) | free-rider stays a hair under η → never strikes; slow to catch |
| **Proportional reward** | model quality / payment scaled by contribution score | free-rider with a valid cheap mark scores as "contributed" |
| **Model denial / IPR** | flagged client cannot claim ownership or receive the final model (FareMark's framing) | evidentiary only; needs the flag to be *correct*, which is what we break |
| **Audit trigger** | flag escalates to a heavier, offline check (e.g. compute attestation) | the honest escape hatch — see Section 6 |

**The key point for the paper:** *every* consequence above is only as good as the flag that triggers it, and the flag is a BER-vs-η test. Because no η separates honest from free-rider (Sections 3–6 of the wrap-up), **no consequence policy built on this flag can be both fair (low FPR) and effective (high recall).** The consequence design is downstream of a detector that does not work.

---

## 6. Proposed detection directions that could actually work (and why they leave the watermark behind)

Our result is specifically about **output-layer BER thresholding**. It suggests where a *working* detector would have to look — none of which is the FareMark reader:

1. **Effort attestation, not output inspection.** Detect free-riding by measuring **work done**, not the mark left behind: verifiable compute (trusted-execution attestation of the training loop), or challenge-response on data the client claims to hold. This sidesteps BER entirely. Cost: infrastructure the FL setting usually lacks.
2. **Contribution via influence, measured on held-out data.** Score each client by the *marginal test-accuracy improvement* its update contributes (leave-one-out / Shapley-style), not by gradient cosine similarity (which the reduced free-rider mimics). Cost: expensive; degrades with many clients.
3. **Staleness / replay forensics.** The `decay` coast (resubmitting identical weights) is trivially caught by a replay check — but the `graft` coast defeats it because the body tracks the fresh global. A detector that models *how much of the submission is genuinely new* (not just whether it changed) is a more promising signal than BER, and is the one direction the submarine specifically stresses.
4. **Majority-robust anomaly detection (FRAD line).** If watermarking is kept, pair it with a contribution+reputation+anomaly hybrid that does not assume benign-majority. This is the honest baseline to compare against — and the paper should show the submarine evades *these* too, or concede they catch it.

**The paper's stance.** We do **not** claim free-rider detection is impossible in general. We claim the specific, popular design — a client-side, output-layer, multi-bit watermark read as BER and thresholded at μ+kσ — is unsound (no separating η) and cheaply evadable (the submarine). A working detector must measure effort or genuine novelty, which the output softmax does not encode.

---

## 7. One-paragraph summary for the paper body

> FareMark and its lineage detect free-riders by embedding a client-side watermark and flagging any client whose recovered bit-error-rate exceeds a threshold η = μ + 3σ of the honest error, applied every round once the honest mark has converged, with the consequence that a flagged client is denied contribution/ownership. We show this decision rule has no viable operating point: on a 100-class problem the μ+3σ recipe falls below the 1/m quantisation floor and is degenerate, the only genuinely 3σ rule sits so high it catches nothing, and per-class the honest and free-rider BER distributions are inseparable (best balanced-error 0.5). Because the consequence — exclusion, reputation loss, reward scaling, or IPR denial — is downstream of this flag, none of them can be simultaneously fair and effective. Detectors that instead measure effort (compute attestation), genuine marginal contribution (held-out influence), or submission novelty (replay/graft forensics) remain open directions; the output-layer watermark is not one of them.
