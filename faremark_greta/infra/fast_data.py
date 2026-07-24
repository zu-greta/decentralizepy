"""fast_data.py -- GPU-resident replacement for the DataLoader path.

WHY
---
The current setup builds one DataLoader per client with num_workers=2 and
non-persistent workers. Iterating a client's shard for one local epoch spawns
2 worker processes and tears them down again. Per run that is

    10 clients x 5 local_epochs x 50 rounds = 2500 spawn/teardown cycles

at roughly 0.3-1.0 s each -- i.e. 12-40 minutes of a typical run spent forking
processes, before any training happens. Under packing (6 runs per pod) it is
worse: 6 runs x 2 workers alive, plus the fork storms overlapping, on a pod that
probably has 8-16 cores.

CIFAR is 154 MB as uint8. It fits on the GPU with room to spare, so the whole
CPU pipeline can be deleted: hold the images on the device, index them, and do
crop/flip/normalise as batched GPU ops.

NUMERICAL EQUIVALENCE
---------------------
Same dtype (fp32), same ops, same augmentation *distribution* (RandomCrop with
4px zero padding + RandomHorizontalFlip, then per-channel normalise). But the
augmentation RNG stream differs from torchvision's, so a run is NOT bit-identical
to the old path -- it is equivalent to changing the seed. Do not mix old and new
runs inside one `manifest.family`.

Padding detail worth knowing: torchvision pads the PIL image with 0 in [0,255]
space and normalises afterwards, so padded pixels end up at -mean/std, not 0.
This module reproduces that (see `_fill`), otherwise your border statistics would
quietly differ from every run you have already done.

WIRING
------
In datasets.py, replace the loader construction in build_data:

    from .fast_data import FastLoader, GPUImageStore     # noqa

    store = GPUImageStore.from_torchvision(train, name, device)
    client_loaders = [FastLoader(store, shard, batch_size=batch_size,
                                 train=True, seed=seed + cid)
                      for cid, shard in enumerate(shards)]
    test_store = GPUImageStore.from_torchvision(test, name, device)
    test_loader = FastLoader(test_store, range(len(test)), batch_size=256,
                             train=False, seed=0)

and add `device` to the build_data signature (one extra arg at the call site in
run_experiment.py: `device=device`). Everything else -- `.dataset` for shard
sizes, iteration yielding (x, y) -- keeps working unchanged.

For the reduced attacker, replace its
    self.loader = DataLoader(Subset(train, idx), ...)
with
    self.loader = self.loader.subset(idx)
otherwise the attacker falls back to the slow CPU path and becomes the
bottleneck in every attack run.
"""
from __future__ import annotations

import numpy as np
import torch

_NORM = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


class GPUImageStore:
    """The whole dataset as one uint8 tensor on the GPU, plus its labels.

    Built once per run and shared by reference across every client loader, so
    N clients cost one copy of the data, not N.
    """

    def __init__(self, images_u8: torch.Tensor, labels: torch.Tensor,
                 mean, std, pad: int, device):
        self.images = images_u8.to(device, non_blocking=True)   # [N,C,H,W] uint8
        self.labels = labels.to(device, non_blocking=True)      # [N] int64
        self.device = device
        self.pad = pad
        c = self.images.shape[1]
        self.mean = torch.tensor(mean, device=device).view(1, c, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, c, 1, 1)
        # value a zero-padded pixel takes AFTER normalisation -- matches
        # torchvision's pad-then-normalise ordering
        self._fill = (-self.mean / self.std).view(1, c, 1, 1)

    @classmethod
    def from_torchvision(cls, ds, name: str, device, pad: int = 4):
        """Pull the raw uint8 array straight out of a torchvision dataset.

        Deliberately bypasses ds.transform -- we re-implement it on GPU.
        """
        name = name.lower()
        mean, std = _NORM[name]

        raw = getattr(ds, "data", None)
        if raw is None:
            raise TypeError(f"{type(ds).__name__} has no .data; cannot build a GPU store")

        arr = np.asarray(raw)
        if arr.ndim == 3:                      # MNIST: [N,H,W]
            arr = arr[:, None, :, :]
        else:                                  # CIFAR: [N,H,W,C] -> [N,C,H,W]
            arr = arr.transpose(0, 3, 1, 2)

        images = torch.from_numpy(np.ascontiguousarray(arr))
        labels = torch.as_tensor(np.asarray(_labels_of(ds)), dtype=torch.long)
        if name not in ("cifar10", "cifar100"):
            pad = 0                            # only CIFAR gets crop augmentation
        return cls(images, labels, mean, std, pad, device)

    def __len__(self):
        return self.images.shape[0]


def _labels_of(ds):
    for attr in ("targets", "labels"):
        if hasattr(ds, attr):
            return getattr(ds, attr)
    raise TypeError("dataset exposes neither .targets nor .labels")


class _IndexView:
    """Stands in for `loader.dataset` so `len(loader.dataset)` still works."""

    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


class FastLoader:
    """Iterable of (x, y) batches, already on the GPU, already augmented.

    Drop-in for a DataLoader over a Subset, for the operations this codebase
    actually uses: iteration, len(), and .dataset for the shard size.
    """

    def __init__(self, store: GPUImageStore, indices, batch_size: int,
                 train: bool, seed: int = 0, drop_last: bool = False):
        self.store = store
        self.indices = torch.as_tensor(list(indices), dtype=torch.long,
                                       device=store.device)
        self.batch_size = int(batch_size)
        self.train = bool(train)
        self.drop_last = bool(drop_last)
        self._gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.dataset = _IndexView(len(self.indices))

    # -- construction helpers -------------------------------------------------
    def subset(self, indices, seed: int | None = None) -> "FastLoader":
        """New loader over a subset of the SAME store (no data copied).

        This is what the reduced/tap attackers should call instead of building
        a fresh CPU DataLoader.
        """
        return FastLoader(self.store, indices, self.batch_size, self.train,
                          seed=self._gen.initial_seed() if seed is None else seed,
                          drop_last=self.drop_last)

    def __len__(self):
        n = len(self.indices)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    # -- the actual work ------------------------------------------------------
    def _prepare_epoch(self):
        """Normalise (and augment) the whole shard in one shot.

        Batch-at-a-time augmentation would launch a dozen kernels per 16-image
        batch, which at this batch size is pure overhead. Doing the entire epoch
        at once amortises it: for a 5000-image shard the transient is ~61 MB.
        """
        s = self.store
        idx = self.indices
        x = s.images.index_select(0, idx).float().div_(255.0)
        x = (x - s.mean) / s.std

        if not self.train or s.pad == 0:
            return x, s.labels.index_select(0, idx)

        n, c, h, w = x.shape
        p = s.pad
        # pad with the normalised value of a zero pixel (see module docstring)
        xp = x.new_empty((n, c, h + 2 * p, w + 2 * p))
        xp[:] = s._fill
        xp[:, :, p:p + h, p:p + w] = x

        span = 2 * p + 1
        off_i = torch.randint(0, span, (n,), generator=self._gen).to(s.device)
        off_j = torch.randint(0, span, (n,), generator=self._gen).to(s.device)

        rows = torch.arange(h, device=s.device).view(1, h, 1) + off_i.view(n, 1, 1)
        cols = torch.arange(w, device=s.device).view(1, 1, w) + off_j.view(n, 1, 1)
        flat = (rows * (w + 2 * p) + cols).view(n, 1, -1).expand(n, c, -1)
        x = torch.gather(xp.view(n, c, -1), 2, flat).view(n, c, h, w)

        flip = (torch.rand(n, generator=self._gen) < 0.5).to(s.device)
        x = torch.where(flip.view(n, 1, 1, 1), x.flip(-1), x)

        return x, s.labels.index_select(0, idx)

    def __iter__(self):
        x, y = self._prepare_epoch()
        n = x.shape[0]
        order = (torch.randperm(n, generator=self._gen).to(x.device)
                 if self.train else torch.arange(n, device=x.device))
        stop = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for i in range(0, stop, self.batch_size):
            sel = order[i:i + self.batch_size]
            yield x.index_select(0, sel), y.index_select(0, sel)


# ---------------------------------------------------------------------------
# Sanity check: distributions match the torchvision path even though the
# individual samples do not. Run this once before trusting the swap.
#
#   python -m faremark.fast_data /path/to/data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from torchvision import datasets as tvd, transforms as T

    root = sys.argv[1] if len(sys.argv) > 1 else "./data"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mean, std = _NORM["cifar100"]

    tv = tvd.CIFAR100(root, train=True, download=True, transform=T.Compose(
        [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
         T.ToTensor(), T.Normalize(mean, std)]))
    old = torch.stack([tv[i][0] for i in range(2000)])

    store = GPUImageStore.from_torchvision(
        tvd.CIFAR100(root, train=True, download=True), "cifar100", dev)
    new = next(iter(FastLoader(store, range(2000), 2000, train=True, seed=0)))[0].cpu()

    print(f"{'':12s} {'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s}")
    for tag, t in (("torchvision", old), ("fast_data", new)):
        print(f"{tag:12s} {t.mean():9.4f} {t.std():9.4f} {t.min():9.4f} {t.max():9.4f}")
    print("\nPer-channel means (should agree to ~0.01):")
    print("  torchvision", [round(float(v), 4) for v in old.mean((0, 2, 3))])
    print("  fast_data  ", [round(float(v), 4) for v in new.mean((0, 2, 3))])
