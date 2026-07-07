#!/usr/bin/env python3
"""Weak-point map: fr_ber (y) vs effort (x), point color = accuracy, eta band shaded.
The weak point = low-effort points that fall BELOW eta AND stay green (healthy acc).
Usage: python scripts/plot_frontier.py --in "$RES/*/result.json" --out figs/weakpoint
"""
import argparse, glob, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys as _sys, os as _osp
_sys.path.insert(0, _osp.path.dirname(__file__))
import plotstyle as _ps; _ps.apply()
import os as _os
from importlib import util as _u
_sp=_u.spec_from_file_location('thresholds', _os.path.join(_os.path.dirname(__file__),'..','faremark','thresholds.py'))
thr=_u.module_from_spec(_sp); _sp.loader.exec_module(thr)

ap=argparse.ArgumentParser()
ap.add_argument("--in",dest="inp",nargs="+",required=True)
ap.add_argument("--out",required=True)
ap.add_argument("--family",nargs="+",default=None)
ap.add_argument("--eta",default="converged",choices=thr.ALL_VARIANTS,help="which threshold definition to draw as the reference line")
a=ap.parse_args()
files=[f for g in a.inp for f in glob.glob(g)]
rows=[]
for f in files:
    try: r=json.load(open(f))
    except: continue
    m=r.get("manifest",{}) or {}
    if a.family and m.get("family") not in a.family: continue
    cs=r.get("compute",{}).get("summary",{}) or {}
    rows.append((cs.get("effort_ratio_samples"), r.get("wm_fr_ber"),
                 r.get("final_acc"), r.get("wm_eta_used"),
                 m.get("note") or r.get("attack")))
rows=[x for x in rows if None not in x[:4]]
if not rows:
    print("no rows"); raise SystemExit
eff,frb,acc,eta,lab=zip(*rows)
# eta reference: recompute the chosen variant per run (post-hoc), average its converged value
_ev=[]
for f in files:
    try: _r=json.load(open(f))
    except: continue
    _m=_r.get("manifest",{}) or {}
    if a.family and _m.get("family") not in a.family: continue
    _b=[h.get("wm_benign_ber") for h in _r.get("history",[])]
    if _b: _ev.append(thr.eta_series([x or 0.5 for x in _b], a.eta)[-1])
eta_mean=float(np.mean(_ev)) if _ev else float(np.mean(eta))
fig,ax=plt.subplots(figsize=(8.5,5.8))
ax.axhspan(eta_mean,1.0,color="#fde8e8",zorder=0)
ax.axhline(eta_mean,color="#c0392b",lw=1,ls="--")
ax.text(0.02,eta_mean+0.01,f"η ({a.eta}) ≈ {eta_mean:.2f}  (above = caught)",color="#c0392b",fontsize=9)
sc=ax.scatter(eff,frb,c=acc,cmap="viridis",vmin=25,vmax=75,s=140,edgecolor="k",linewidth=0.6,zorder=3)
for x,y,l in zip(eff,frb,lab):
    ax.annotate(l,(x,y),textcoords="offset points",xytext=(6,4),fontsize=7)
cb=fig.colorbar(sc); cb.set_label("global accuracy % (green = healthy, red = poisoned)")
ax.set_xlabel("attacker effort / honest effort  (samples) →  more work")
ax.set_ylabel("free-rider BER at the server  (below η = evades)")
ax.set_title("Weak-point map: where does the attack evade cheaply AND keep the model healthy?\n"
             "target = bottom-left points that are GREEN and below the η line")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(a.out+".png",dpi=150)
print("wrote",a.out+".png")