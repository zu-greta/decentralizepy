#!/usr/bin/env bash
# Turn the full sweep into figures. Skips families with no runs (no error).
#   RES=/mnt/nfs/home/zu/results ./scripts/make_sweep_figs.sh
set -uo pipefail
RES="${RES:-/mnt/nfs/home/zu/results}"; OUT="${OUT:-figs}"; mkdir -p "$OUT"
ALL="$RES/*/result.json"
PA="python scripts/plot_adaptive.py"
run(){ echo "== $*"; eval "$*" || echo "   (skipped)"; }

# ---- THE weak-point map: every attack on one plane (fr_ber vs effort, color=acc)
run python scripts/plot_frontier.py --in "'$ALL'" \
    --family R_frontier M_warmup S_warmup S_coast S_samples --out "$OUT/weakpoint_all"

# ---- REEMBED frontier: fr_ber and acc vs effort
run $PA sweep --in "'$ALL'" --family R_frontier --sweep_var reembed_effort --metric wm_fr_ber  --out "$OUT/reembed_frber"
run $PA sweep --in "'$ALL'" --family R_frontier --sweep_var reembed_effort --metric final_acc  --out "$OUT/reembed_acc"

# ---- MEMORY: warmup vs fr_ber and acc (Q1/Q2)
run $PA sweep --in "'$ALL'" --family M_warmup --sweep_var warmup_rounds --metric wm_fr_ber --out "$OUT/memory_frber"
run $PA sweep --in "'$ALL'" --family M_warmup --sweep_var warmup_rounds --metric final_acc --out "$OUT/memory_acc"

# ---- SUBMARINE warmup (Q1): fr_ber + acc vs warmup
run $PA sweep --in "'$ALL'" --family S_warmup --sweep_var sub_warmup --metric wm_fr_ber --out "$OUT/sub_warmup_frber"
run $PA sweep --in "'$ALL'" --family S_warmup --sweep_var sub_warmup --metric final_acc --out "$OUT/sub_warmup_acc"

# ---- SUBMARINE coast type (Q3): fr_ber + acc by coast mode
run $PA sweep --in "'$ALL'" --family S_coast --sweep_var sub_coast_mode --metric wm_fr_ber --out "$OUT/sub_coast_frber"
run $PA sweep --in "'$ALL'" --family S_coast --sweep_var sub_coast_mode --metric final_acc --out "$OUT/sub_coast_acc"

# ---- SUBMARINE samples per tap (Q2)
run $PA sweep --in "'$ALL'" --family S_samples --sweep_var sub_max_burst_batches --metric wm_fr_ber --out "$OUT/sub_samples_frber"

# ---- effort readouts (duty/effort vs the knob) for the effort-vs-evasion question
run $PA sweep --in "'$ALL'" --family R_frontier --sweep_var reembed_effort --metric effort_ratio_samples --out "$OUT/reembed_effort"
run $PA sweep --in "'$ALL'" --family M_warmup --sweep_var warmup_rounds --metric effort_ratio_samples --out "$OUT/memory_effort"

echo; echo "Figures in $OUT/. Upload them and I'll rebuild the deck."
