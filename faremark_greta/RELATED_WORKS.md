# Related Work — Technical Breakdown for the Submarine Free-Rider Attack on FareMark

**Purpose of this document.** Context-setting and literature grounding for a paper arguing that
*output-layer (box-free) watermarking cannot reliably detect free-riders in federated learning*,
demonstrated via a "submarine" attack that trains *just enough* to keep its watermark bit-error
rate (BER) inside the honest band and below the detection threshold η.

Each section covers one paper: (1) a technical breakdown of what it does and how, (2) a compact
summary, and (3) how it connects to the attack and the impossibility thesis. FareMark is included
first as the anchor/target; the other three are the surrounding literature. A final synthesis
section ties everything to the thesis and to your current experimental groups (A/D/E/I).

**One-line thesis framing to keep in mind while reading.** Ownership verification asks a *binary*
question the *defender controls* ("did *someone* embed my mark?"). Free-rider detection asks a
*graded* question the *attacker controls* ("did *this client* contribute *enough*?"). The second
question forces a threshold on a quantity — watermark recoverability — that is a smooth,
monotone-ish function of training effort. A submarine attacker tunes effort so that quantity lands
inside the honest distribution. Every paper below either (a) supplies the mechanism you are
attacking, (b) supplies the threat model / free-rider definitions you inherit, or (c) supplies the
capacity theory and taxonomy you generalize over.

---

## 1. FareMark — *Model-Watermark-Driven Free-Rider Detection in Federated Learning* (Li et al., IEEE IoT-J 2025) — **the target**

### 1.1 Technical breakdown

**Setting and goal.** N clients train a shared classifier by FedAvg-style parameter averaging.
FareMark serves two functions: free-rider detection (during training) and IPR/ownership
verification (after deployment). The novelty relative to prior FL watermarking is that instead of
one watermark for the whole model, *every* client embeds its *own* private watermark, and a client
whose submitted model lacks a valid watermark is flagged as a free-rider.

**Box-free, output-layer watermark representation (Sec. IV-A, Fig. 5).** This is the crux of what
you are attacking. The watermark is read out of the model's *softmax output vector*, not its
parameters and not a fixed trigger-label mapping:

- The output vector `P̂ = [p̂₁,…,p̂ₙ]` is split into `m` groups `{P̂¹,…,P̂ᵐ}`.
- Each group is projected onto a pseudorandom direction given by a row of a secret projection
  matrix `M`: `z_k = Σ_j p̂ᵏ_j · M_{i,k,j}`.
- The k-th watermark bit is `b̂ᵏ = δ(z_k)` with `δ(z)=1` if `z≥0` else `0` (Eq. 2).
- Because cross-entropy makes softmax outputs steep (one class ≈ 1, the rest ≈ 0), the raw output
  can be dominated by its max and carry no bit information. FareMark therefore passes outputs
  through a smoothing function `f(x)` before projection — a power function `x^α` (α<0, or 0<α<1) or
  a periodic `sin(αx)` — with a constraint `f(max(P̂))/Σ f(P̂) < 0.5` so no single class dominates
  (Eqs. 7–10). α trades off watermark sensitivity vs. robustness.

**Trigger-class assignment (Sec. IV-B).** To avoid inter-client conflict (different clients pulling
the same decision boundary in opposite directions), the *server assigns each client a unique
"trigger class."* Watermarking loss is applied only on samples of that client's trigger class;
common (non-trigger) classes keep the ordinary cross-entropy loss. Conceptually this partitions the
decision boundary into per-client subspaces so each client "owns" one class's output geometry. This
is why your experiments are organized *per trigger class* (iso_c1, iso_c3, iso_c6, etc.) — class
difficulty is a first-order variable in this scheme.

**Training loss (Sec. IV-B, Eqs. 11–13).** `L = L_cl + λ·L_wm`, where `L_cl` is standard CE and
`L_wm` is a sigmoid/BCE-style term pushing each projected `z̃ᵏ` to the correct sign for the target
bit. λ balances main-task accuracy against watermark fidelity.

**Memory-enhanced local updating (Sec. IV-C, Eq. 14).** Aggregation averages away the delicate
watermark signal. FareMark counters with `W^j_{i+1} = β(W^j_i + lr·∂L/∂W^g_i) + (1−β)·W^g_i`, mixing
the previous global knowledge back into each local update. The authors note this **slows
convergence** — relevant to you because it means the mark takes several rounds of genuine training
to consolidate, which is exactly the window a submarine exploits.

**Free-rider detection (Sec. IV-D, Eqs. 15–16).** Detection is a *threshold on watermark
recovery*. Over `N_T` trigger samples the server extracts `b̂` (Eq. 15), then computes the mean
absolute bit difference against the client's *registered* watermark and flags a free-rider when it
is **not** below η:

```
(1/m) Σ_k |b̂ᵏ − bᵏ|  <  η        (Eq. 16)   →   "watermark present" (honest)
```

**η is set to μ + 3σ**, where μ, σ are the mean and standard deviation of the *legitimate* client
BER measured across training rounds. So η is calibrated on the honest population and is meant to sit
just above honest noise. This single scalar is the entire security surface of the scheme, and its
placement is the whole game for your operating-point analysis.

**Free-rider models considered (Sec. V-A2).** Two, both inherited from the FedIPR/Fraboni lineage:
"Free-riding with Previous Models" `W_free = Free(W^t, W^{t−1})` (Eq. 17), and "Free-riding with
Gaussian Noise" `W_free = W^t + ε` (Eq. 18). Both are *zero-effort* fabrications — no training at
all — which is the regime FareMark is designed to catch and which it does catch.

**The admissions that seed your attack (Tables IV–V).**
- **Table V — Train-Trigger-Sample-Only free rider.** If an attacker trains on only a *small number
  of trigger samples*, the watermark **overfits to those samples and fails to generalize** to the
  broader trigger class, so it is *not detected*. FareMark presents this as a limitation. This is
  the seed: it is already a proof-of-existence that a *training* free-rider can evade, and the
  failure mode is over-specialization, not under-effort.
- **Table IV — Train-then-attack.** A client that trains for a few early rounds then defects is
  *easy* to detect (the early mark decays); the more rounds it participates, the harder detection
  becomes and the more compute it spends. This sketches the effort/detectability trade-off curve
  your Group D formalizes.

**Other evaluation.** Fidelity within ~2% of FedAvg (Table I); watermark detection ≈100% for honest
clients vs. <0.40 for zero-effort free-riders after ~30 rounds (Fig. 8); robustness to DP (Opacus),
fine-tuning (λ=0), pruning (tolerates up to ~50%), quantization; capacity study up to 50 clients on
CIFAR-10 by oversubscribing trigger classes (Table IX). Note the capacity trick: when #clients >
#classes, clients *share* a trigger class and are disambiguated by trigger-sample consistency —
which is precisely the "same-class insider" situation your A4/AK and class-averaging-illusion
analysis probes.

### 1.2 Summary

FareMark is a box-free, output-layer, per-client watermarking scheme for FL that doubles as a
free-rider detector. Each client owns a unique trigger class and embeds a private multi-bit mark
read from smoothed, projected softmax outputs; detection thresholds the per-client BER at μ+3σ of
honest error. It catches *zero-effort* free-riders (previous-model, Gaussian) well, but its own
Tables IV–V concede that *training* free-riders — those who do minimal or overfit-y training — can
slip through.

### 1.3 Relation to your project

This is the mechanism your impossibility claim is about, and it hands you three levers:

1. **The threshold is calibrated on honest BER (μ+3σ).** Your entire operating-point argument is
   that this scalar cannot be placed to separate "trained enough" from "trained just enough."
   Group A's `operating_point.png` is the direct empirical statement of that: at any usable FPR
   (≤5%) insider recall ≤0.17 and no bar hits 0.9. The per-class oracle collapsing same-class
   recall to 0.00 is the sharpest version — even an *unrealizable* per-class threshold can't do it.
2. **Table V is your existence proof, and you generalize its scope.** FareMark frames trigger-only
   evasion as overfitting. Your Group D reframes the *whole effort axis*: trigger-only overfits and
   is caught (BER≈0.44, your positive control), but the moment you add ~1 real image/class the BER
   collapses to the honest floor (~0.11–0.13) and *stays* there for every larger budget. So the
   evasion is not a narrow overfitting artifact — there is a broad effort plateau *inside the honest
   band*. That converts FareMark's caveat into a structural failure.
3. **Memory-enhanced updating slows convergence.** Because the mark consolidates over rounds, a
   coasting/tapping submarine (your Group I) has a natural place to hide: embed only when the mark
   is at risk of drifting above η, coast otherwise. (Group I is currently invalidated by the
   probe-holdout starvation bug — once the `_prepare` fix lands, the expected signature is BER
   hugging just *under* η, not blowing up.)

Scoping note for the paper: FareMark's detector is *behavioral/output-layer*. Keep your
impossibility claim scoped to that mechanism; the parameter-based schemes below behave differently
and are your contrast cases, not counterexamples.

---

## 2. FedIPR — *Ownership Verification for Federated Deep Neural Network Models* (Li, Fan, Gu, Li, Yang; IEEE TPAMI 2023)

### 2.1 Technical breakdown

**Goal.** Ownership verification (IPR) for FedDNN — not free-rider detection as the primary aim,
though it offers free-rider detection as a corollary (Sec. 6.9). Each client independently embeds
and later verifies private watermarks without disclosing private data or watermark parameters.

**Two watermark types.**
- **Feature-based (white-box).** An N-bit string `B ∈ {0,1}^N` is embedded into model *parameters*
  — specifically the normalization-layer scale parameters `W_γ` — via a hinge-like regularizer.
  Extraction: `B̃ = sgn(W_γ E)` with a secret embedding matrix `E`; ownership holds if Hamming
  distance `H(B, B̃) ≤ threshold` (white-box, needs parameter access).
- **Backdoor-based (black-box).** A trigger set of PGD-generated adversarial samples with designated
  labels; verified through the API by checking the designated-label return rate.

**Theorem 1 — the capacity result (the part you flagged).** For K clients each embedding N-bit
feature-based watermarks into M channels of the shared parameters:
- **Case 1 (KN ≤ M):** there exists W with detection rate `η_F = 1` — all marks coexist conflict-free.
- **Case 2 (KN > M):** conflict is unavoidable, but bounded: `η_F ≥ (KN + M)/(2KN)`.

The optimal per-client bit-length for the strongest (smallest-p-value) verification is
`N_opt = M/K`. M is concrete per architecture (e.g., 896 channels across AlexNet's last 3 layers,
2048 for ResNet-18). The scheme is cast as a hypothesis test: under H₀ ("not plagiarized") each
recovered bit is Bernoulli(1/C), giving a binomial p-value that quantifies confidence.

**Free-rider handling (Sec. 6.9).** FedIPR uses the *same* two free-rider constructions FareMark
later adopts: plain freerider `Free(W^t, W^{t−1})` and Gaussian-noise freerider `W^t + ξ`. In a
22-client setup with one of each, benign clients reach ≈100% feature-based detection within ~30
rounds while freeriders sit at ≈50% (random) — so freeriders are separable *because they never
embed anything*. Crucially this is a *zero-effort* threat model; FedIPR does not evaluate a partial
/ submarine trainer.

**Robustness / breadth.** Tested under DP, client selection, defensive aggregation (Krum, Bulyan,
Trim-mean), fine-tuning, pruning, and non-IID (Dirichlet β=0.1, 1). Feature-based marks in
normalization layers are notably robust; backdoor marks degrade more under defensive aggregation
(e.g., ~63% under Trim-mean) but stay statistically significant.

### 2.2 Summary

FedIPR is the reference client-side FL watermarking scheme: each client embeds private
feature-based (parameter, white-box) and backdoor-based (behavioral, black-box) marks, with a clean
capacity theory (Theorem 1: KN≤M ⇒ perfect coexistence; else a detection-rate floor) and a
hypothesis-testing significance framework. Free-rider detection is a side benefit and is
demonstrated only against zero-effort freeriders.

### 2.3 Relation to your project

- **Threat-model lineage — cite this for legitimacy.** Your free-rider definitions come through
  FareMark from FedIPR (and Fraboni et al.). Framing your submarine as the *missing* point on this
  lineage — the *partial-effort* trainer that neither FedIPR nor FareMark stress-tests — is a clean
  motivation. FedIPR's Sec. 6.9 is exactly the "freeriders are trivially separable" claim your
  operating-point figure contradicts *once the freerider trains a little*. Your personal note
  ("they consider the same free-riders") is right and worth foregrounding.
- **Theorem 1 is your capacity/impossibility scaffolding — but re-aim it.** FedIPR's theorem is
  about *coexistence* of many marks in finite channel capacity. Your impossibility is different in
  kind: it is about *separability* of one honest mark from one submarine mark under a *single
  deployable threshold*. Borrow the theorem's style (a clean capacity/lower-bound statement) but
  make explicit that (a) it is a parameter-capacity argument, whereas (b) yours is an
  output-distribution overlap argument. If you want a theoretical companion to your empirics,
  the analogue you'd prove is: honest and reduced-effort BER distributions overlap (OVL→1,
  best-balanced-error→0.5), so no threshold achieves both bounded FPR and useful recall — which is
  what your Group D `D1_sep_n5.json` measures numerically (class 6 catchable only at ~40% FPR).
- **White-box vs. output-layer contrast for scoping.** FedIPR's *feature-based* mark lives in
  parameters and is verified white-box; a submarine that trains minimally still nudges those
  parameters, so parameter-based detection is a *different* (and harder-to-evade) target. This is
  the boundary of your claim: your impossibility is strongest for the *behavioral/output-layer*
  detection route (FareMark; FedIPR's backdoor half). State that boundary rather than
  over-claiming over all FL watermarking.

---

## 3. FedTracker — *Furnishing Ownership Verification and Traceability for Federated Learning Model* (Shao et al., IEEE TDSC 2024)

### 3.1 Technical breakdown

**Goal.** Two properties at once: ownership verification *and* traceability (identify *which*
client leaked a model). It is **server-side**: the trusted server embeds everything; clients are
potential leakers, not embedders. This is the opposite design axis from FareMark/FedIPR
(client-side).

**Bi-level protection.**
- **Global watermark (black-box, backdoor).** Server embeds a shared trigger set (WafflePattern:
  per-class patterns + Gaussian noise) into the aggregated global model each round, for
  ownership verification via API.
- **Local fingerprint (white-box, parameter).** Server inserts a *unique* multi-bit fingerprint per
  client into BN-layer scale weights `W_γ` via a hinge-like loss `sgn(A·W_γ)=F` with a secret
  Gaussian key matrix A. This is what enables traceability.

**Continual-Learning embedding (Sec. IV-B).** Because the server has no natural task data, naively
retraining on an out-of-distribution trigger set wrecks utility (their Table I: WAFFLE drops BN
models >50%). FedTracker treats primitive task vs. watermark task as two domains and applies a
GEM-style constraint: it keeps a "global memory" = accumulated global gradients, and projects the
watermark gradient so it does not increase the primitive-task loss (`⟨g, m_t⟩ ≥ 0`, else QP
projection). This recovers utility (up to ~3.2% over no-CL).

**Fingerprint Similarity Score (FSS).** Traceability compares an extracted fingerprint against all
K client fingerprints. Instead of discrete Hamming distance (which rounds and can tie two
candidates), they use a *continuous* score `FSS = Σ min(δ, b_ij·f_ij)`; the max-FSS client is the
suspected leaker. Fingerprints are generated to *maximize the minimum pairwise Hamming distance*
(NP-hard, solved via genetic algorithm), and the paper leans on FedIPR's capacity relationship to
reason about how many distinct fingerprints the BN parameters can hold.

**Threat model / robustness.** Server trusted; some clients malicious leakers. Attacks: fine-tuning,
quantization, pruning (BN vs non-BN layers analyzed separately), overwriting, and backdoor
mitigation (Neural Cleanse, FeatureRE). Reports 100% traceability rate across settings, including
non-IID (Dirichlet ξ=0.5–0.9).

### 3.2 Summary

FedTracker is a server-side, bi-level scheme: a shared backdoor global watermark for ownership plus
a per-client parameter fingerprint (in BN scale weights) for traceability, embedded with a
continual-learning constraint to preserve utility and traced with a continuous FSS metric. It is
about catching *leakers after training*, not free-riders during training, and it assumes the
embedder (server) is honest.

### 3.3 Relation to your project

- **Contrast case that sharpens your scope.** FedTracker's *identity* signal (fingerprint) is in
  parameters and inserted by a trusted server — a free-rider cannot suppress it by under-training
  because the *server* embeds it after aggregation. So FedTracker is *not* vulnerable to your
  submarine in the same way, and that is useful: it demonstrates that the vulnerability you exploit
  is specific to (a) *client-side* embedding where effort is the attacker's dial, and (b)
  *behavioral/output* readout where recoverability is continuous in effort. Use it to argue your
  impossibility is a property of the *client-side output-layer free-rider-detection* design point,
  not of FL watermarking writ large.
- **FSS vs. threshold-on-BER.** FedTracker deliberately moved from a discrete Hamming threshold to
  a continuous relative-ranking score because a hard threshold was brittle. That is a
  literature-internal admission that *hard thresholds on watermark distance are fragile* — which is
  precisely the fragility FareMark's η depends on. You can cite FedTracker's FSS motivation as
  independent evidence that thresholding recovery is a weak primitive, then note that even a
  ranking metric wouldn't save free-rider detection, because ranking still needs a decision
  boundary between honest and submarine, and your Group D shows those two populations coincide.
- **Non-IID robustness comparison.** FedTracker (and FedIPR) report watermark survival under
  Dirichlet non-IID. Your Group E makes the *defensive* point that non-IID does not rescue the
  *detector* either: skew lifts honest floor, submarine BER, and η together, so separation is
  unchanged. Positioning Group E against these papers' non-IID sections preempts the "your
  non-separability is an IID lab artifact" referee objection. (Flag from your notes: earlier E runs
  were secretly IID due to the `PART` vs `PARTITION` env-var bug; the claim only holds once the
  re-run shows uneven Dirichlet shard sizes in `run.log`.)

---

## 4. Lansari et al. — *When Federated Learning Meets Watermarking: A Comprehensive Overview* (MDPI MAKE 2023)

### 4.1 Technical breakdown

**What it is.** A survey/taxonomy of FL watermarking for IP protection. No new algorithm; it
formalizes the problem space and reviews nine schemes (WAFFLE, FedIPR, FedTracker, Liu et al.,
FedCIP, FedRight, Yang et al., FedZKP, Merkle-Sign). *FareMark is not in it* (survey is 2023,
FareMark 2025), which is convenient — you get the taxonomy without the survey pre-empting your
target.

**Three watermarking scenarios (who embeds):** S1 server-side, S2 client-side, S3 collaborative.
FareMark and FedIPR are S2; FedTracker is S1 (with per-client fingerprints).

**Requirement set (their Table 2), specialized for FL.** Fidelity, Capacity, Reliability (low false
*negative* rate), Integrity (low false *positive* rate), Generality, Efficiency, Robustness,
Secrecy. For FL they add nuance to five — notably Capacity (multi-client bit conflicts), Secrecy
(watermarking updates must look like benign updates or defensive aggregation cancels them), and
Robustness reinterpreted as *traitor tracing*.

**White-box vs black-box.** White-box embeds in parameters/activations (Uchida-style
sign-of-projected-weights; passport layers). Black-box changes behavior via a trigger set
(content-, unrelated-, noise-, or adversarial-based). It gives the standard trigger-set accuracy
metric `acc = (1/|T|) Σ 1[M_wat(x)=y]`.

**Free-rider-relevant observations.** The survey explicitly notes FedIPR is *the only* reviewed
scheme tested under a federation with free-riders (Gaussian-noise / random-weight clients). It also
catalogs the open challenges: server-side trigger-set difficulty (no natural data → OOD triggers →
evasion via query detectors), interaction with robust aggregation (Krum/Bulyan can reject
watermark-carrying updates because they look outlier-ish), client selection effects, cross-device
scale, DP/HE compatibility, and non-IID.

### 4.2 Summary

The definitive map of FL watermarking circa 2023: it defines the S1/S2/S3 embedding scenarios, the
eight FL watermarking requirements (including Reliability=low FNR and Integrity=low FPR), the
white-box/black-box split, and the open problems. It flags that free-rider robustness is
under-tested (only FedIPR) and that hard, real-FL conditions (defensive aggregation, non-IID,
cross-device) are largely unaddressed.

### 4.3 Relation to your project

- **Requirement vocabulary = your operating-point axes.** The survey's *Reliability (low FNR)* and
  *Integrity (low FPR)* are exactly the two axes of your `operating_point.png`. Recasting your
  result in their language is strong: FareMark, as a free-rider detector, cannot simultaneously
  satisfy Integrity (don't flag honest clients) and the detection goal (catch free-riders =
  low FNR) — at any usable FPR budget, FNR is near 1. That is a requirements-level impossibility
  statement, not just an empirical curve, and it's phrased in the community's own terms.
- **"Only FedIPR tested free-riders" is your motivation gap.** The survey certifies that the
  literature has barely stress-tested free-rider robustness, and only against *zero-effort*
  attackers. Your submarine (partial-effort, adaptive) is the natural next adversary the field has
  not confronted. Use this sentence as the citation that justifies the paper's existence.
- **Generalization target.** Your goal is to generalize from FareMark to *output-layer watermarking
  in general*. The survey's white-box/black-box taxonomy lets you state the generalization
  precisely: your impossibility bites on *behavioral/black-box, client-side, free-rider-detection*
  watermarking, because that is the family where the readout is a smooth function of the attacker's
  training effort. White-box parameter marks and server-side embedding (per the survey's own
  categories) are explicitly *outside* the claim. That framing makes the generalization defensible
  rather than over-broad.
- **Secrecy/aggregation caveat to preempt.** The survey warns that watermark-carrying updates must
  resemble benign updates or defensive aggregation rejects them. Your submarine benefits from the
  opposite property — it *wants* to look benign — but note the honest clients in your simulation
  are not running defensive aggregation; if a referee asks "would Krum change this?", the honest
  answer is that defensive aggregation targets outlier updates, and a submarine's near-honest
  updates are the least outlier-ish of all, so it should if anything *help* the attacker.

---

## 5. Synthesis — how the four papers assemble your argument

**The mechanism you attack (FareMark) rests on one scalar.** Free-rider detection = `BER < η`,
η = μ+3σ of honest BER. Everything reduces to whether honest and submarine BER distributions are
separable by one deployable threshold.

**The literature gives you four things:**

| Need | Supplied by | Use |
|------|-------------|-----|
| The target mechanism + its own evasion caveat (Table V) | FareMark | Existence proof you generalize into a plateau (Group D) |
| The inherited zero-effort free-rider threat model | FedIPR (via Fraboni), echoed by FareMark | Legitimacy; you add the missing *partial-effort* point |
| Capacity/lower-bound proof *style* + white-box contrast | FedIPR Theorem 1 | Template for a separability lower-bound; scope boundary |
| Server-side / parameter fingerprint contrast; "hard thresholds are fragile" | FedTracker (FSS motivation) | Scopes your claim to client-side output-layer; independent fragility evidence |
| Requirement vocabulary (FNR/FPR), scenario taxonomy, "free-riders under-tested" | MDPI survey | Requirements-level phrasing of impossibility; motivation gap; precise generalization |

**The impossibility argument in one paragraph (for your intro/abstract).** Free-rider detection via
output-layer watermarking requires distinguishing "contributed enough" from "contributed just
enough" through a threshold on watermark recoverability. But recoverability is a smooth, monotone
function of training effort (FareMark's own memory-enhanced updating makes the mark consolidate over
rounds; its Table V shows minimal-effort training already breaks detection). A submarine attacker
therefore tunes effort so its BER lands inside the honest μ±σ band. Your `operating_point.png`
shows the consequence: at any usable false-positive budget (≤5% honest clients wrongly flagged),
free-rider recall is ≤0.17 and never reaches 0.9, and a per-class oracle threshold — an
*unrealizable* upper bound — collapses same-class insider recall to exactly 0.00. Group D shows the
evasion is not a knife-edge: adding ~1 real image per class drops BER onto a flat plateau (~0.11–
0.13) that coincides with the honest floor across the entire remaining effort axis. Group E shows
non-IID does not restore separation (skew lifts honest floor, attacker BER, and η together). This
is a property of the design point — client-side, behavioral, free-rider-detecting watermarking —
not of FareMark's specific hyperparameters, and it generalizes to any output-layer watermark whose
free-rider decision is a threshold on recovery.

**Where each results group plugs in.**
- **Group A (`operating_point`, iso plots):** the headline impossibility figure + the per-class
  isolation showing free-rider marks are often *cleaner* than honest ones at easy classes. This is
  the FareMark-mechanism section.
- **Group D (`D1_spectrum`, `D1_sep_n5`):** the effort→detectability curve; turns FareMark's Table V
  caveat into a plateau. This is your core generalization evidence.
- **Group E (non-IID):** the defensive section answering "IID artifact" — position against FedIPR's
  and FedTracker's non-IID robustness tables. (Only valid after the `PART`/`PARTITION` re-run.)
- **Group I (submarine tap):** the constructive attack — mark hugging just under η by tapping when
  near-threshold, coasting when safe. Currently invalid pending the `_prepare` probe-holdout fix;
  once fixed it is the paper's constructive centerpiece.

**Two scoping guardrails to state explicitly (so referees can't over-read the claim):**
1. The impossibility is about *output-layer / behavioral, client-side, free-rider-detection*
   watermarking. Parameter-based (FedIPR feature-based, FedTracker fingerprints) and server-side
   embedding are contrast cases, not counterexamples.
2. Ownership verification (the primary aim of FedIPR/FedTracker) is *not* refuted — a submarine that
   trains enough to evade free-rider detection may still carry a recoverable ownership mark. Your
   claim is specifically that the *free-rider* use of the mark is unsound, which is the narrower and
   defensible target.

---

## Suggested citation slots in your paper

- **Motivation / gap:** MDPI survey (free-rider robustness under-tested; only FedIPR) + FedIPR
  Sec. 6.9 (zero-effort free-riders only).
- **Threat model:** FedIPR / Fraboni free-rider definitions, inherited by FareMark Sec. V-A2.
- **Target mechanism:** FareMark Secs. IV-A–IV-D (box-free output watermark, η=μ+3σ), Tables IV–V
  (existing evasion caveats).
- **Theory template + scope boundary:** FedIPR Theorem 1 (capacity), FedTracker FSS motivation
  (hard-threshold fragility).
- **Non-IID rebuttal:** FedIPR Table 12/14 and FedTracker Sec. VI-C, contrasted with your Group E.
