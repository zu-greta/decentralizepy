#!/usr/bin/env python
"""backfill_holdings.py -- add summary.wm_trigger_holdings to OLD result.json files
that predate the holdings-logging patch (e.g. the existing E1/E2/E3 non-IID runs), so
the trigger_fairness plot works WITHOUT re-running them.

It replays the EXACT datasets.py::dirichlet_partition with the run's own seed/alpha/
num_clients, recomputes each client's class histogram, reads each client's trigger
class from the run itself (history[-1].wm_per_client[*].trigger_class), and writes
holdings = count of that client's trigger class in its shard. IID runs are handled too
(iid_partition). CIFAR labels are loaded from torchvision (needs the dataset available)
OR, if --labels_npy is given, from a cached label array so no download is needed.

    python backfill_holdings.py --in 'results/E1_honest_niid_c100_rep*/result.json'
    python backfill_holdings.py --in 'results/E*_*/result.json' --labels_npy cifar100_train_labels.npy

Idempotent: skips runs that already have holdings unless --force.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

def iid_partition(n, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [list(s) for s in np.array_split(idx, k)]

def dirichlet_partition(labels, k, alpha, seed):
    labels = np.asarray(labels); rng = np.random.default_rng(seed)
    ncls = int(labels.max()) + 1
    shards = [[] for _ in range(k)]
    for c in range(ncls):
        idx_c = np.where(labels == c)[0]; rng.shuffle(idx_c)
        props = rng.dirichlet(alpha * np.ones(k))
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for cid, part in enumerate(np.split(idx_c, cuts)):
            shards[cid] += part.tolist()
    return shards

def get_labels(dataset, data_root, labels_npy):
    if labels_npy and os.path.exists(labels_npy):
        return np.load(labels_npy)
    from torchvision import datasets as D
    ds = dataset.lower()
    if ds == "cifar100": t = D.CIFAR100(data_root, train=True, download=True)
    elif ds == "cifar10": t = D.CIFAR10(data_root, train=True, download=True)
    elif ds == "mnist":   t = D.MNIST(data_root, train=True, download=True)
    else: raise SystemExit(f"unknown dataset {dataset}")
    return np.asarray(t.targets)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--labels_npy", default=None, help="cached train-label array to avoid a download")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    files = [f for pat in a.inp for f in sorted(glob.glob(pat))]
    label_cache = {}
    n_done = n_skip = 0
    for f in files:
        try: d = json.load(open(f))
        except Exception as e: print(f"  (skip {f}: {e})"); continue
        summ = d.setdefault("summary", {})
        if summ.get("wm_trigger_holdings") and not a.force:
            n_skip += 1; continue
        cfg = d.get("config") or {}
        dataset = cfg.get("dataset") or (summ.get("dataset"))
        k = int(cfg.get("num_clients") or summ.get("num_clients"))
        seed = int(d.get("seed") or summ.get("seed"))
        partition = cfg.get("partition", "iid")
        alpha = float(cfg.get("dirichlet_alpha", 0.5))
        # trigger class per cid from the run itself
        hist = d.get("history") or []
        if not hist: print(f"  (skip {f}: no history)"); continue
        tclass = {int(p["cid"]): int(p["trigger_class"])
                  for p in (hist[-1].get("wm_per_client") or [])}
        if dataset not in label_cache:
            label_cache[dataset] = get_labels(dataset, a.data_root, a.labels_npy)
        labels = label_cache[dataset]
        if partition in ("dirichlet", "noniid"):
            shards = dirichlet_partition(labels, k, alpha, seed)
        else:
            shards = iid_partition(len(labels), k, seed)
        holdings, shard_sizes = {}, {}
        for cid in range(k):
            shard_labels = labels[np.asarray(shards[cid], dtype=int)]
            shard_sizes[str(cid)] = int(len(shard_labels))
            tc = tclass.get(cid)
            holdings[str(cid)] = int((shard_labels == tc).sum()) if tc is not None else None
        summ["wm_trigger_holdings"] = holdings
        summ["wm_shard_sizes"] = shard_sizes
        summ.setdefault("wm_trigger_assign", cfg.get("wm_trigger_assign", "roundrobin"))
        json.dump(d, open(f, "w"))
        n_done += 1
        print(f"  backfilled {os.path.basename(os.path.dirname(f))}: "
              f"holdings min={min(v for v in holdings.values() if v is not None)} "
              f"max={max(v for v in holdings.values() if v is not None)}")
    print(f"\ndone: {n_done} backfilled, {n_skip} already had holdings.")

if __name__ == "__main__":
    main()
