# RUNBOOK - experiments

## Paths (set these once, keep them consistent)
```bash
export RES=/Users/gretazu/Documents/summer@epfl/watermarking_DL.nosync/decentralizepy/faremark_greta/results/threshold_calibrate           
                                       # the dir that CONTAINS the per-run subfolders
                                       # (must match where submit_experiment.sh writes:
                                       #  $MOUNT/home/zu/results). e.g. ../results/threshold_calibrate
export ETA_FILE=$RES/eta_calibrated.json
```
- **You submit from local**, so `run_all.sh` + `read_eta` run locally: the eta FILE only
  needs to exist locally at `$ETA_FILE`. Its VALUE is injected into each cluster job as
  `WM_ETA_FIXED` — the pod never reads the file. No upload needed.
- **Plotting** reads `result.json` from `$RES/*/result.json` and the eta from
  `$ETA_FILE` — so keep both reachable from wherever you plot (local is fine).
- Rule: `$RES` is ONE directory holding all run subfolders + `eta_calibrated.json`.

---

## PHASE 0 — threshold (once)

| command | runs / does | input | output | expect |
|---|---|---|---|---|
| `SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_all.sh honest` | submits 10 all-honest jobs (ATTACK=none), 10 clients, 50 rounds | config 14 | 10 `result.json` under `$RES` | 10 runs, `final_acc ~73%`, no free-riders |
| `./run_all.sh calibrate` | `threshold.py calibrate` on the honest runs | `$RES/*/result.json` | `$ETA_FILE` | prints `eta = <x> +/- <std>`, per-seed etas |
| `python scripts/threshold.py verify --in "$RES/*/result.json" --honest-family honest_iid --eta-file "$ETA_FILE"` | recompute eta + check frozen use | honest runs + eta file | PASS/FAIL text | `MATCH` on recompute |

**Threshold definition:** per seed, `eta_s = mu_s + 3*sigma_s` over the last 20
per-round (mean-over-clients) honest BERs; final `eta` = **average of the 10 eta_s**.

---

## PHASE 1 — confirm before trusting anything (all-honest)

| command | shows | input | output |
|---|---|---|---|
| `python scripts/plots.py sanity --in "$RES/*/result.json"` | TEXT: flags flat/zero BER, non-frozen eta, missing loss/per_class | all runs | console warnings |
| `python scripts/plots.py eta_stability --in "$RES/*/result.json" --family honest_iid --out "$RES/figs"` | per-seed BER curves + mean band + per-seed eta spread | honest runs | `eta_stability_honest_iid.png` |
| `python scripts/plots.py thresholds --in "$RES/*/result.json" --family honest_iid --out "$RES/figs"` | eta derivation + **honest FPR** at the used eta | honest runs (+eta file) | `thresholds_honest_iid.png` |
| `python scripts/plots.py class_difficulty --in "$RES/*/result.json" --family honest_iid --out "$RES/figs"` | per-class BER; + acc/loss vs BER IF `per_class` present | honest runs | `class_difficulty_honest_iid.png` |

**per_class note:** the acc/loss panels need runs made with the CURRENT
`run_experiment.py`. If your honest runs predate it, run **3** more honest seeds to get
them (does NOT change the frozen eta — do not re-calibrate):
```bash
SEEDS="0 1 2" ./../scripts/run_all.sh honest      # then re-run class_difficulty
python scripts/plots.py class_difficulty --in "$RES/*/result.json" --family honest_iid --out "$RES/figs"
```

---

## PHASE 2 — free-rider experiments (one knob per batch, >=3 seeds)

`POS` = the trigger CLASS IDs that free-ride. Families are auto-tagged by POS
(`tap_every_iid_c17`, `tap_stay_iid_c36`, ...) so nothing collides.

| command | tests | mechanism | expect |
|---|---|---|---|
| `USE_FIXED_ETA=1 POS=1,7 ./../scripts/run_all.sh attacks` | free-riding at EASY class ids | tap_every (+5/common) & tap_stay (coast) | FR BER ~0 < eta -> hides cleanly |
| `USE_FIXED_ETA=1 POS=3,6 ./../scripts/run_all.sh attacks` | free-riding at HARD class ids | same two | FR BER ~ honest floor at a fraction of the effort |
| `USE_FIXED_ETA=1 POS=3,6 AUTOP_COMMON_PER_CLASS=0 ./../scripts/run_all.sh tap_every` | data ablation: triggers-only | +0/common | overfits -> high BER -> caught |

`attacks` = `tap_every` + `tap_stay`. Each submits `$SEEDS` (default `0 1 2`).
All read the frozen eta via `read_eta` -> `WM_ETA_FIXED`.

---

## PHASE 3 — plot + re-verify (after attack jobs finish)

| command | does | expect |
|---|---|---|
| `RES=$RES ./scripts/run_all.sh PLOTALL` | minimal set: sanity, class_difficulty, thresholds (honest); timeline, fidelity, class_dynamics (each attack family) | figures in `$RES/figs` |
| `python scripts/threshold.py verify --in "$RES/*/result.json" --honest-family honest_iid --eta-file "$ETA_FILE"` | attack runs used the frozen eta? | flat `wm_eta_round == eta` -> PASS |

**Read `sanity` first.** If an FR BER is flat/zero or eta isn't flat, open
`class_dynamics` for that family before trusting the story.

---

## Plot reference (what each figure means)

- **eta_stability** — is the threshold stable? faint line per seed = that seed's honest
  BER over rounds; black = seed-mean + std band; green = final eta + std; right panel =
  the per-seed eta spread. Wide spread = fragile calibration (a finding in itself).
- **thresholds** — how eta is built (dots = per-round mean BER) + where it lands: panel
  (b) = honest per-client BER histogram with the used eta; title shows the **honest FPR**.
- **class_difficulty** — are some class ids harder? per-class BER bar; per-class test
  accuracy/loss; BER-vs-error and BER-vs-loss scatters with Pearson r (needs `per_class`).
- **timeline** — one attack family: BER vs round, eta line, tap/coast markers, calib window.
- **fidelity** — FR vs honest per-client BER + training effort (image-passes) ratio.
- **class_dynamics** — per-class watermark loss / trigger-class accuracy / loss curves
  (the DIAGNOSTIC when a BER looks suspicious).
- **sanity** — TEXT only; run first on every batch.

---

## Command index (all entry points)
```
scripts/run_all.sh    honest | calibrate | tap_every | tap_stay | attacks | PLOTALL
scripts/threshold.py  calibrate --in <glob> --honest-family honest_iid --tail 20 --out <file>
                      verify    --in <glob> --honest-family honest_iid --eta-file <file>
scripts/plots.py      <cmd> --in <glob> [--family F] [--out DIR|PREFIX]
   cmds: eta_stability sanity class_difficulty thresholds class_dynamics positions
         fidelity timeline honest_fpr threshold  (+ legacy: frontier scorecard test_data)
```
