#!/usr/bin/env bash
# TODO: remove?

# make_all_figs.sh — regenerate every figure for the deck from whatever finished.
# Safe to run repeatedly; missing families are skipped with a warning, not an error.
#   RES=/mnt/nfs/home/zu/results ./scripts/make_all_figs.sh
set -uo pipefail
RES="${RES:-/mnt/nfs/home/zu/results}"
OUT="${OUT:-figs}"
mkdir -p "$OUT"
P="python scripts/plot_adaptive.py"
ALL="$RES/cfg1*/*result.json"

run () { echo "== $* =="; eval "$*" || echo "   (skipped — no matching runs yet)"; }

# --- THE TRILEMMA (headline) --------------------------------------------------
# effort vs detection: who evades and at what cost
run $P effort --in "$ALL" --family E2_effort E1_coast E8_coast_mode \
      --effort samples --metric wm_fr_recall --out "$OUT/trilemma_effort_recall"
# the poison axis: accuracy collapse for frozen replay
run $P sweep  --in "$RES/cfg15_*/result.json" --family E2_effort \
      --sweep_var warmup_rounds --metric final_acc --out "$OUT/trilemma_mem_accuracy"
# the mark-decay axis: submarine recall vs warmup
run $P sweep  --in "$RES/cfg14_*/result.json" --family E2_effort \
      --sweep_var sub_warmup  --metric wm_fr_recall --out "$OUT/trilemma_sub_recall"
# fr_ber knee: mark embeds with enough warmup
run $P sweep  --in "$RES/cfg15_*/result.json" --family E2_effort \
      --sweep_var warmup_rounds --metric wm_fr_ber --out "$OUT/mem_frber_knee"

# --- E5 freshness sweep: the trilemma as one figure (acc AND recall vs blend) --
run $P sweep  --in "$ALL" --family E5_freshness --sweep_var mem_blend_global \
      --metric final_acc     --out "$OUT/e5_accuracy"
run $P sweep  --in "$ALL" --family E5_freshness --sweep_var mem_blend_global \
      --metric wm_fr_recall  --out "$OUT/e5_recall"

# --- E8 transplant: the escape attempt -----------------------------------------
run $P sweep  --in "$ALL" --family E8_coast_mode --sweep_var sub_coast_mode \
      --metric wm_fr_recall  --out "$OUT/e8_recall"
run $P sweep  --in "$ALL" --family E8_coast_mode --sweep_var sub_coast_mode \
      --metric final_acc     --out "$OUT/e8_accuracy"

# --- E1 coast mechanism: only memory-carrying coast evades ---------------------
run $P sweep  --in "$ALL" --family E1_coast --sweep_var attack \
      --metric wm_fr_recall  --out "$OUT/e1_coast_recall"

# --- E7 need-the-full-shard ----------------------------------------------------
run $P sweep  --in "$RES/cfg13_*/result.json" --family E7_embed_composition \
      --sweep_var n_trigger_samples --metric wm_fr_ber --out "$OUT/e7_need_full_shard"

# --- duty-cycle trace for one submarine (shows warmup->coast->tap) -------------
run $P duty   --in "$RES/cfg14_*/result.json" --family A7_submarine \
      --out "$OUT/submarine_duty"

echo; echo "Figures in $OUT/. Upload the PNGs tomorrow and I'll assemble the deck."
