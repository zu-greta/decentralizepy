import json, glob, sys, argparse
sys.path.insert(0, 'faremark'); sys.path.insert(0, 'scripts')
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plotstyle as ps, thresholds as thr
ps.apply()

# Load JSON files from glob patterns
def load(globs):
    out=[]
    for g in globs:
        for f in sorted(glob.glob(g)):
            try: out.append(json.load(open(f)))
            except: pass
    return out

# Extract a series of values for a given key from the history
def series(r, key):
    return [x.get(key) for x in r.get("history", [])]

# Extract the free-rider's trace of per-round events (including eta_est and whether it was poisoned)
def fr_trace(r):
    for _,c in r.get("compute",{}).get("per_client",{}).items():
        if c.get("is_free_rider") and c.get("trace"): return c["trace"]
    return []

ap=argparse.ArgumentParser()
ap.add_argument("--in",dest="inp",nargs="+",required=True)
ap.add_argument("--note",required=True, help="substring of manifest.note to select the config")
ap.add_argument("--title",default="")
ap.add_argument("--out",required=True)
a=ap.parse_args()

runs=[r for r in load(a.inp) if a.note in (r.get("manifest",{}) or {}).get("note","")]
if not runs:
    print("no runs matching note:", a.note); sys.exit(1)
n=min(len(series(r,"wm_fr_ber")) for r in runs)
rounds=np.arange(1,n+1)
def stack(key):
    M=np.array([[ (v if v is not None else np.nan) for v in series(r,key)[:n]] for r in runs],dtype=float)
    return np.nanmean(M,0), np.nanstd(M,0)
fr_m,fr_s=stack("wm_fr_ber"); ho_m,ho_s=stack("wm_benign_ber")
# fair eta (frozen) mean across seeds
frozen=np.nanmean([thr.eta_series(series(r,"wm_benign_ber")[:n],"frozen") for r in runs],0)
# attacker estimated eta mean across seeds
est=[]
for r in runs:
    d={t["round"]:t.get("eta_est") for t in fr_trace(r) if t.get("eta_est") is not None}
    est.append([d.get(int(rd),np.nan) for rd in rounds])
est_m=np.nanmean(np.array(est,dtype=float),0)

fig,ax=plt.subplots(figsize=(10.5,6))
# shaded std bands
ax.fill_between(rounds, ho_m-ho_s, ho_m+ho_s, color=ps.C_HONEST, alpha=0.18)
ax.fill_between(rounds, fr_m-fr_s, fr_m+fr_s, color=ps.C_FR, alpha=0.18)
ax.plot(rounds, ho_m, color=ps.C_HONEST, lw=2.6, label="honest BER (mean ± std over 3 seeds)")
ax.plot(rounds, fr_m, color=ps.C_FR, lw=2.6, label="free-rider BER (mean ± std)")
ax.plot(rounds, frozen, color=thr.STYLE["frozen"]["color"], lw=2.2, label="ACTUAL fair η (frozen)")
if np.isfinite(est_m).any():
    ax.plot(rounds, est_m, color=ps.OKABE["grey"], ls="--", lw=2, label="attacker's ESTIMATED η")
ax.set_xlabel("communication round")
ax.set_ylabel("bit-error-rate  (lower = watermark present)")
ax.set_ylim(0,0.7)
ax.set_title(a.title or "Free-rider vs honest BER, mean ± std over seeds")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(a.out+".png", bbox_inches="tight"); plt.close(fig)
print("wrote", a.out+".png", "from", len(runs), "seeds")
