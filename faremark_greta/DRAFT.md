# DRAFT PROPOSAL - Submarine Free-Riders: Why Output-Layer Watermarking for Free-Rider Detection in Federated Learning Fails

## goal/thesis
Output layer watermarking (box-free) for free-rider detection is impossible to threshold (no threshold possible due to class difficulty - no safe threshold possible that would not also flag hoenst clients) and can easily be evaded by a cheap free-rider (defined as a client that does minimal amount of work - using the least samples as possible, closest to 0 - while still embedding a watermark that when checked, stays under the defined threshold).
*NOTE: wording too strict?*

- **Negative / impossibility result** — the honest floor is a per-class band (0.001–0.114 IID; 0.008–0.34+ non-IID), so the μ+3σ threshold is either too tight (flags honest) or too loose (passes free-riders).
- **Constructive attack** — with minimal effort (<=22% of honest compute on certain classes) an adaptive free-rider evades detection while walking away with a full-quality global model.

---

## 1. Abstract

- FL intro + highlight free-rider problem (+def)
- intro watermarking as a proposed solution (faremark paper intro, basic one line summary of the method and threhsold to flag detection).
- show that their detector does not work by 
   1. impossibility argument with honest BER band that is class dependent and no threshold can spare hoenst and flag FR 
   2. submarine FR intro (brief intro of the attack and how it works in theory (the name) and summary results). 
- mention non-iid effects 
- generalize the results to show that the problem is not just faremark but the whole family of output-layer watermarking for free-rider detection


---

## 2. Introduction

- setup FR problem
   -> from Lin, Fraboni, etc.
   -> define free-rider formally - differs a bit from og def
   -> why it happens and why detection matters
- setup watermarking solution proposed by faremark
   -> watermarking in general
   -> faremark usage of watermarking for FR detection + threhsold detection rule
   -> define the threshold faremark uses and our adaptation of it (what values we use for reference)
- setup impossibility argument and submarine attack
   -> attack assumption that honest and FR BER are cleanly separable by a threshold (all other papers test are baseline FR, never anything a bit more advanced)
   -> contributions
      - re-implementation of faremark paper
      - honest BER band (per-class, per-seed, IID and non-IID) [plots group A]
      - submarine attack
      - non-iid makes it worse

**PLOTS**: 
- BER (y) vs effort/samples (x). honest band + reduced sweep + submarine + crude baseline, with the two η lines. shows that FR can reach the honest band at less effort and no threshold possible

---

## 3. Background & Related Work

- complete background on the output layer watermarking (faremark) -> see paper for details
   - include info about other watermarking schemes mentioned in faremark - related papers
- detection rules (based on other papers) 

TODO: read papers and fill in the table for reference
| Scheme | Metric | Decision rule | Free-rider tested? | Threshold specified? |
|---|---|---|---|---|
| **FareMark** (IoT-J'25) | BER on trigger-class softmax | hard, per-round, per-client: flag if `BER ≥ η = μ+3σ` | baseline (prev-model, Gaussian) + train-then-attack + trigger-only | μ,σ and η |
| **FedIPR** (TPAMI'23) | - | - | - | - |
| **FedTracker** (TDSC'24) | - | - | - | - |
| **Lansari review** (MAKE'23) | - | - | - | - |

---

## 4. Threat Model

- experimental setup (`K=10` clients, ResNet-18 on CIFAR-100, FedAvg, 50 rounds; each client assigned one trigger class, a secret key `M`, target bits `B`. Server verifies every round on `N_T=50` held-out trigger-class images)
- Adversary - FR: define FR goals, what it has access to and controls. how it works and what it is bound to
- Success criterion for the submarine FR

**PLOTS**:
- threat model diagram ?

---

## 5. Reproduction of FareMark

- establish the baseline and reproduction of the faremark paper base
- steup the honest runs and baseline free-rider runs to show everything is in place

**PLOTS**:
- baseline comaprison plots to compare with faremark reported results (config table + result table)
- all honest runs with the per-class BER band + thresholds (A1 per-class honest floors + A1 honest per-round with trig_acc panel)
- baseline free-rider runs to show faremark paper implementation

---

## 6. Threshold impossibility + FR intro (effort vs BER)

- setup the threshold (how it was defined and calculated) and how it is not possible to draw one in the honest band without flagging honest clients - hard classes are noisy and overlaps happen
- intro the effort scale and reduced FR idea (CPC) -> as opposed to table V from faremark. the static FR intro and compute cost calculations
- group D reduced free-rider runs to show how cheap FR can go and match the honest band/overlap already -> show the compute cost reduction already 

**PLOTS**:
- group D plot with data budget spectrum + reduced timelines + compute savings (combined)
- isolated hoenst and FR overlays per class
- ROC over threshold ?

### 7. Submarine Free-Rider Attack

- setup the submarine attack and how it works (adaptive, effort-minimizing)
- results of the submarine attack - the differnt things to be varied and effects of each

**PLOTS**:
- submarine per FR timelines with hoenst comparison + compute savings
- scope and data budget for the submarine attack
- global accuracy
- trigger class accuracy ???

---

## 8. Non-IID

- intro what non-iid is and how it is implemented (alpha value) and how the trigger class is assigned (random vs fair assignment)
- alpha varied values effects
- the effects of non-iid on honest only and then on the FR (reduced and submarine) 

**PLOTS**: 
- non-iid distribution (with diff alpha values and diff assignments)
- honest only for both assignemnt methods
- FR reduced and submarine for both assignment methods + comparison with honest + compute savings

---

## 9. Generality & Discussion

- generalize the results and argument to output layer watermarking as a whole
- potention solutions for mitigation ??? *NOTE: or not mentioning this would be better?*
- mention experimentation limitations here (or later ?) - models used, dataset used etc

---

## 10. Conclusion

- restate results and thesis

---
---
---

# Experiments Plan

ResNet-18 on CIFAR-100

all experiments to run for the paper:
- all honest no watermark run - establish basis
- honest client only runs - to establish honest BER, class difficulty and thresholds (run at multiple seeds with varying trigger class assignments -> so far only 0-9 done)
- baseline free-rider runs - for comparison with faremark paper results - match paper settings
- static reduced free-rider spectrum +N to show cheap free-rider
- submarine free-rider runs -> oracle and self + full and block2 + reduced data amount variations -> settle on a single best config but show some different variations to show the effect of each parameter
- non-iid with random and fair trigger class assignment for honest only and FR runs

plots to have for the paper:
- honest BER from multiple seeds - show the differnet class difficulties and threshold basis
- comparison with all paper results to establish baseline correct reproduction
- trigger class accuracy and global accuracy for all runs - proof FR and watermark is not altering fedavg model quality
- static reduced free-rider spectrum +N to show cheap free-rider BER - plot the timeline, the compute savings, overlap comaprison with honest client on that trigger class, the threshold overlaps -> multiple seeds. isolated plots to show just the honest on that trigger class
- table of all config variations for submarine
- submarine FR timeline plots (same as for static) + compute savings -> show every seed's tap/coasts and savings cumulative compared with the honest client
- couple of plots for submarine to show variations of params + different free rider index (hard/easy classes) + different amount of FR present 
- comparison for non-iid with random and fair trigger class assignment for honest only and FR runs - show the effect of non-iid on the honest band and the FR runs

extra experiments for sanity checking
- trigger class accuracy - on no watermark training to check if accruacy 0 is product of watermarking

---