"""Watermarking end-to-end test on synthetic data (quick)

WatermarkClient, Server, Aggregator, WatermarkRegistry and
verifier over a few FL rounds and checks:
  (a) EMBEDDING + EXTRACTION : mean benign BER is low (watermark recovered).
  (b) FIDELITY               : global test accuracy climbs (task still learned).
  (c) DETECTION              : free-rider BER >> benign BER and it gets flagged.

Run:  python test_stage3.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "faremark_paper"))

import torch
from torch.utils.data import DataLoader, TensorDataset

from faremark.models import build_model
from faremark.server import Server
from faremark.wm_client import build_watermarked_clients
from faremark.wm_verify import WatermarkRegistry, build_trigger_bank, make_verifier
from faremark.config import ExpConfig
from faremark.utils import set_seed, get_logger

set_seed(0)
N_CLASSES, IN_CH, DEVICE = 10, 1, "cpu"


def synth(nsamp, seed):
    """Learnable 28x28 data: each class imprints a distinct bright row-band."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, N_CLASSES, (nsamp,), generator=g)
    x = torch.randn(nsamp, IN_CH, 28, 28, generator=g) * 0.2
    for i in range(nsamp):
        r = (int(y[i]) * 2) % 26
        x[i, 0, r:r + 2, :] += 2.0
    return x, y


# 10 IID client shards + a global test set
NUM_CLIENTS = 6
client_loaders = []
for cid in range(NUM_CLIENTS):
    x, y = synth(600, 100 + cid)
    client_loaders.append(DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True))
xte, yte = synth(1500, 999)
test_ds = TensorDataset(xte, yte)
test_loader = DataLoader(test_ds, batch_size=256)

cfg = ExpConfig("wm_synth", "smallcnn", "synth", num_clients=NUM_CLIENTS,
                rounds=8, local_epochs=1, lr=0.05,
                watermark=True, wm_lambda=5.0, wm_beta=0.6,
                attack="previous_models", num_free_riders=2,
                wm_num_triggers=60, wm_eta=0.25, wm_verify_every=1)

model = build_model(cfg.model, N_CLASSES, IN_CH).to(DEVICE)
registry = WatermarkRegistry()
clients, fr_idx = build_watermarked_clients(cfg, client_loaders, model, DEVICE,
                                            seed=0, num_classes=N_CLASSES,
                                            registry=registry)
print(f"clients={NUM_CLIENTS}  free-riders={fr_idx}  registered={len(registry)}")

verify_model = build_model(cfg.model, N_CLASSES, IN_CH)
classes = sorted({e["trigger_class"] for e in registry.entries.values()})
bank = build_trigger_bank(test_ds, classes, cfg.wm_num_triggers, seed=0)
verify_hook = make_verifier(registry, bank, verify_model, DEVICE, fr_idx,
                            eta=cfg.wm_eta, verify_every=1)

logger = get_logger()
server = Server(model, clients, test_loader, DEVICE, logger, verify_hook=verify_hook)
history = server.run(cfg.rounds)

last = history[-1]
print("\nfinal round metrics:")
for k in ("test_acc", "wm_benign_ber", "wm_fr_ber", "wm_detect_acc",
          "wm_fpr", "wm_fr_recall"):
    print(f"  {k:16s} = {last.get(k)}")

benign_ber = last["wm_benign_ber"]
fr_ber = last["wm_fr_ber"]
acc = last["test_acc"]
print("\nRESULT")
print(f"  (a) benign BER     = {benign_ber:.3f}   (want low)")
print(f"  (b) test accuracy  = {acc:.1f}%       (want climbing, fidelity)")
print(f"  (c) free-rider BER = {fr_ber:.3f}   (want >> benign)")

assert benign_ber <= 0.20, f"watermark not embedding (benign BER={benign_ber})"
assert fr_ber >= benign_ber + 0.15, "free-rider not separable from benign"
assert acc >= 60.0, f"fidelity too low (acc={acc})"
print("\nSTAGE 3 OK: embeds for benign, fails for free-riders, task still learned.")
