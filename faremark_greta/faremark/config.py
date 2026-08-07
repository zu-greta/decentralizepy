"""Experiment configs.

`config_idx` selects an experiment; `repeat` selects a seed.

`expected_acc` is a loose [low, high] band on final FedAvg test accuracy for reference
"""
from dataclasses import dataclass, field, asdict


@dataclass
class ExpConfig:
    name: str
    model: str
    dataset: str
    num_clients: int
    rounds: int = 50
    local_epochs: int = 5
    lr: float = 0.01
    batch_size: int = 16
    momentum: float = 0.9
    weight_decay: float = 5e-4
    base_seed: int = 1000
    expected_acc: tuple = (0.0, 100.0)      # correctness band

    # ---- free-rider selection / paper baselines ----
    attack: str = "none"                    # "none"|"previous_models"|"gaussian"|"reduced"
                                            # |"adaptive_tap" (aka "submarine"/"autopilot")
    num_free_riders: int = 0                # how many of num_clients are free-riders
    free_rider_ids: str = ""                # "3,6" pins which cids free-ride (overrides seeded choice). Empty => choose_free_riders(seed).
    noise_sigma: float = 0.1                # GaussianNoiseFreeRider std
    noise_decay: float = 0.0                # >0 -> sigma_t = sigma0 * t^(-decay)
    partition: str = "iid"                  # 'iid' or 'dirichlet' (non-IID)
    dirichlet_alpha: float = 0.5            # dirichlet skew; small=severe non-IID, large~=IID
    trigger_class_map: str = ""             # "cid:class,cid:class" overrides the default trigger_class = cid % num_classes

    # ---- shared free-rider schedule + data knobs ----
    autop_oracle_eta: float = 0.0           # >0 => the FR is handed the true eta (controlled test).
                                            # For the submarine, tap_eta_source="self" ignores this.
    autop_honest_until: int = 12            # W: fixed-warmup defect round (warmup=[1,W-1], calib=[W-K,W-1]).
                                            # For dynamic warmup (tap_warmup_mode="dynamic") W is the fallback.
    autop_calib_rounds: int = 4             # K: the K honest rounds that calibrate eta (server + FR self-est).
                                            # Tagged "calib" in the trace.
    autop_trigger_train_n: int = -1         # TABLE V: num of trigger imgs trained on (-1 = all)
    autop_common_per_class: int = -1        # DATA per tap: -1=full shard; 0=triggers-only; N=+N/common-class
    autop_n_common_classes: int = -1        # how many COMMON CLASSES the free-rider draws from:
                                            # -1/0 = all of them; K>0 = K randomly chosen classes

    # ---- adaptive tap free-rider  (attack="adaptive_tap", aka the "submarine")  ----
    # FR that trains only on rounds when its own BER nears the estimated eta.
    tap_eta_source: str = "oracle"   # which eta the FR aims under: "oracle" = the true server eta
                                     # (wm_eta_fixed / autop_oracle_eta, for controlled tests);
                                     # "self" = the FR estimates it from its own calib-window probe BER
    tap_eta_k: float = 3.0           # self mode: eta_hat = mu + k*sigma over the FR's own calib probe BERs
    tap_margin: float = 0.02         # aim this far below eta:  target = eta - margin
    tap_when: str = "threshold"      # when to tap: "threshold" (tap iff probe BER > target, else coast),
                                     # "always" (tap every post-warmup round), "every_k" (tap every tap_period)
    tap_period: int = 1              # period P for tap_when="every_k"
    tap_max_coast: int = 999         # force a tap after this many consecutive coasts (safety cap)
    tap_data_cpc: int = 5            # amount of data per tap: images/common-class (-1 full shard, 0 trigger-only, N=+N)
    tap_scope: str = "full"          # model scope a tap trains: "full" | "block2" (last 20 tensors) |
                                     # "block" (last 8) | "head" (last 2). Backbone frozen => cheaper tap
    tap_coast_mode: str = "resend"   # how the FR free-rides between taps: "resend" = submit the global
                                     # unchanged (zero compute); "decay" = resend its own last tapped
                                     # weights (mark fades slower, but it submits stale weights)
    tap_graft_decay: float = 0.0      # graft coast: blend frozen mark-head toward global head each coast (0=off, tail-spike fix)
    tap_probe_holdout: int = 16      # held-out trigger images for the FR's self-BER probe (generalisation)
    # ---- DYNAMIC adaptive-tap knobs (default = the prior FIXED behaviour) ----
    tap_margin_mode: str = "fixed"   # "fixed" = constant tap_margin; "derived" = eta - margin_k*sigma(calib probe BER)
    tap_margin_k: float = 1.0        # k for the derived margin (target = eta_hat - k*sigma)
    tap_warmup_mode: str = "fixed"   # "fixed" = defect at autop_honest_until; "dynamic" = defect when own probe converges
    tap_conv_eps: float = 0.03       # dynamic: converged when the last (patience+1) probe BERs are within this
    tap_conv_patience: int = 2       # dynamic: consecutive flat rounds required to declare convergence
    tap_honest_min: int = 6          # dynamic: never defect before this round (protect the calibration window)
    tap_warmup_cap: int = 15         # dynamic: hard stop -- defect by here even if the probe never converges

    # ---- watermarking ----
    watermark: bool = False
    wm_bits: int = 0                        # m; 0 -> auto
    wm_balanced_keys: bool = False          # False = paper-faithful random +/-1 keys 
                                            # True = sign-balanced rows (removes that artifact by
                                            # construction, still pseudo-random)
    wm_trigger_assign: str = "roundrobin"   # trigger-class -> client assignment policy:
                                            #   "roundrobin" = cid % num_classes (blind; the paper default)
                                            #   "distribution" = server assigns each client a class it holds a lot of
    wm_lambda: float = 5.0                  # weight of L_wm (Eq. 11)
    wm_alpha: float = 0.4                   # smoothing exponent (Eq. 8)
    wm_f: str = "power"                     # smoothing kind: "power" | "sin"
    wm_beta: float = 0.6                    # memory coefficient (Eq. 14)
    wm_label_smoothing: float = 0.1
    wm_num_triggers: int = 50               # N_T trigger samples for extraction (Eq. 15)
    wm_trigger_mode: str = "class"          # which trigger images the verifier uses:
                                            #  "class"  = one shared held-out bank per trigger
                                            #             class (default; clients sharing a class
                                            #             see identical images -> only M^i/B^i
                                            #             distinguish them). Generalisation test.
                                            #  "client" = per-client DISJOINT held-out slice
                                            #             (paper V-F3 "client-specific trigger
                                            #             variations", still held-out)
                                            #  "client_train" = per-client images taken from that
                                            #             client's OWN training shard (paper V-F3
                                            #             "trigger sample consistency": test imgs
                                            #             == train imgs). Memorisation, not
                                            #             generalisation -- see paper Table V.
    wm_eta_floor: float = 0.05              # small degenerate guard for eta only (not the threshold):
                                            # keeps eta = mu+3sigma strictly positive if every benign BER is
                                            # ~0. The operative threshold is always the computed mu+3sigma
    wm_eta_fixed: float = 0.0               # >0 => pre-calibrated constant threshold for every
                                            # round/experiment (from calibrate_eta.py). 0 => live calc.
    wm_verify_every: int = 1
    calib_on_all: bool = False              # calibrate eta over all clients (exposes circularity) vs benign-only

    def to_dict(self):
        return asdict(self)


CONFIGS = [
    # 0: fast smoke test to prove the pipeline learns
    ExpConfig("smoke_mnist_smallcnn", "smallcnn", "mnist", num_clients=5,
              rounds=5, local_epochs=1, batch_size=64, expected_acc=(95.0, 100.0)),

    # ---- Table I reproduction (FedAvg baseline) ----
    ExpConfig("resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              expected_acc=(88.0, 94.0)),
    ExpConfig("resnet18_mnist", "resnet18", "mnist", num_clients=10,
              expected_acc=(98.0, 99.7)),
    ExpConfig("resnet18_cifar100", "resnet18", "cifar100", num_clients=100,
              expected_acc=(70.0, 80.0)),
    ExpConfig("alexnet_cifar10", "alexnet", "cifar10", num_clients=10,
              expected_acc=(82.0, 90.0)),
    ExpConfig("alexnet_mnist", "alexnet", "mnist", num_clients=10,
              expected_acc=(88.0, 99.5)),
    ExpConfig("alexnet_cifar100", "alexnet", "cifar100", num_clients=10,
              expected_acc=(62.0, 74.0)),

    # 7: fast free-rider smoke (Fig. 7 trend)
    ExpConfig("fr_smoke_mnist", "smallcnn", "mnist", num_clients=10,
              rounds=10, local_epochs=1, batch_size=64,
              attack="previous_models", num_free_riders=0,
              expected_acc=(0.0, 100.0)),
    # 8: previous-models free-rider (Fig. 7a)
    ExpConfig("fr_prev_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              attack="previous_models", num_free_riders=2,
              expected_acc=(0.0, 100.0)),
    # 9: Gaussian-noise free-rider (Fig. 7c)
    ExpConfig("fr_gauss_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              attack="gaussian", num_free_riders=2, noise_sigma=0.1,
              expected_acc=(0.0, 100.0)),

    # ---- watermarking ----
    # 10: fast watermark smoke
    ExpConfig("wm_smoke_mnist", "smallcnn", "mnist", num_clients=10,
              rounds=10, local_epochs=1, batch_size=64,
              watermark=True, wm_lambda=5.0, wm_beta=0.6,
              expected_acc=(0.0, 100.0)),
    # 11: fidelity, all honest + watermarked
    ExpConfig("wm_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              watermark=True, wm_lambda=5.0, wm_beta=0.6,
              expected_acc=(86.0, 94.0)),
    # 12: detection, watermark + crude free-riders (Tables III-V)
    ExpConfig("wm_fr_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="previous_models", num_free_riders=2,
              expected_acc=(0.0, 100.0)),
    # 13: paper-faithful detection target, CIFAR-100
    ExpConfig("wm_fr_resnet18_cifar100", "resnet18", "cifar100",
              num_clients=10, watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="previous_models", num_free_riders=2,
              expected_acc=(0.0, 100.0)),

    # 14: CIFAR-100 attack base config. The runbook/run_now.sh always override
    #     --attack (reduced / adaptive_tap / previous_models / ...) and the free-rider
    #     ids per family, so the attack set here is just a non-crashing default.
    ExpConfig("attack_base_resnet18_cifar100", "resnet18", "cifar100",
              num_clients=10, watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="adaptive_tap", num_free_riders=2,
              expected_acc=(0.0, 100.0)),
]

ATTACK_BASE_IDX = 14   # convenience for scripts
SUBMARINE_IDX = ATTACK_BASE_IDX   # back-compat alias
AUTOPILOT_IDX = ATTACK_BASE_IDX   # back-compat alias


def get_config(idx: int) -> ExpConfig:
    if idx < 0 or idx >= len(CONFIGS):
        raise IndexError(
            f"config_idx {idx} out of range (have {len(CONFIGS)}): "
            + ", ".join(f"{i}:{c.name}" for i, c in enumerate(CONFIGS)))
    return CONFIGS[idx]


def seed_for(cfg: ExpConfig, repeat: int) -> int:
    return cfg.base_seed + repeat