"""Experiment configs

`config_idx` selects an experiment; `repeat` selects a seed (paper: 10). 

`expected_acc` is a loose [low, high] band on final FedAvg test accuracy
(optional for pass/fail reference). For example, the reference points are
the FedAvg(%) column of Table I in the paper:

    ResNet-18  CIFAR-10  (10 clients)  91.85
    ResNet-18  MNIST     (10 clients)  98.75
    ResNet-18  CIFAR-100 (100 clients) 76.54
    AlexNet    CIFAR-10  (10 clients)  86.35
    AlexNet    MNIST     (10 clients)  91.54
    AlexNet    CIFAR-100 (10 clients)  68.45
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
    expected_acc: tuple = (0.0, 100.0)  # correctness band

    # ---- free-rider attacks ----
    attack: str = "none"        # "none" | "previous_models" | "gaussian" | ...
    num_free_riders: int = 0    # how many of num_clients are free-riders
    noise_sigma: float = 0.1    # GaussianNoiseFreeRider std
    noise_decay: float = 0.0    # >0 -> sigma_t = sigma0 * t^(-decay)
    attack_round: int = 50      # train_then_attack: round at which the FR defects (Table IV)
    n_trigger_samples: int = 8  # trigger_only / mixed: # trigger samples the FR uses (Table V)
    honest_prob: float = 0.5    # random_round: per-round prob the FR trains honestly
    blend: float = 0.5          # mixed: weight on the FR's own (lightly-trained) weights
    full_trigger_class: bool = False  # mixed: train on ALL trigger-class samples (generalizing embed) vs capped n_trigger_samples
    n_common_samples: int = 0   # mixed + full_trigger_class: # random common-class samples added (disguise + stability)
    partition: str = "iid"      # data split: 'iid' or 'dirichlet' (non-IID)
    dirichlet_alpha: float = 0.5  # dirichlet skew; small=severe non-IID, large~=IID

    # ---- adaptive free-riders autopilot submarine ----
    # submarine: closed-loop controller that keeps its own BER just under its
    # estimate of eta, training a minimal burst only when needed.
    sub_warmup: int = 8                 # rounds of full-shard honest embedding up-front (CIFAR-100 needs ~8 to generalize)
    sub_warmup_batches: int = 150       # per-warmup-round batch budget (cycles the enriched set)
    sub_margin: float = 0.05            # target BER = eta_estimate - margin
    sub_floor: float = 0.05             # embed until held-out probe BER <= floor
    sub_eta_mode: str = "adaptive"      # "adaptive" (anchor to clean BER) | "fixed"
    sub_eta_fixed: float = 0.25         # eta guess when mode=fixed / no clean history yet
    sub_max_burst_batches: int = 60     # cap on a maintenance tap's mini-batches
    sub_probe_every: int = 3            # re-check probe BER every k burst batches
    sub_common_samples: int = 50        # common-class samples in an enriched burst (stability/disguise)
    reembed_scope: str = "head"         # reembed attack: which params to fine-tune (head|block|full)
    reembed_steps: int = 40             # reembed attack: max fine-tune steps (the EFFORT knob)
    reembed_floor: float = 0.05         # reembed attack: stop when held-out probe BER <= this
    autop_floor: float = 0.05           # autopilot: 'mark is good' bar
    autop_margin0: float = 0.08         # autopilot: initial safety gap below eta-estimate
    autop_min_batches: int = 20         # autopilot: smallest adaptive tap
    autop_max_batches: int = 200        # autopilot: largest adaptive tap
    autop_lookahead: int = 2            # autopilot: rounds ahead to predict the eta crossing
    autop_warmup_cap: int = 15          # autopilot: hard cap so self-terminating warmup can't run forever
    autop_protect_until: int = 8        # autopilot: never defect before this round (the detector
                                        # calibrates its frozen eta on this no-free-rider window;
                                        # keep it >= honest convergence, ~8 on CIFAR-100)
    autop_scope: str = "full"           # autopilot: which params to re-train (full|block|block2|head) — the effort dial
    autop_stay_under: bool = False      # autopilot STAY-UNDER: prioritise staying below eta over saving effort.
                                        #   Re-embeds every post-warmup round with a fixed honest-style budget
                                        #   (no probe early-stop, no dynamic tap sizing). Auto-ON when
                                        #   autop_oracle_eta>0. Cost set by autop_scope + autop_common_per_class.
    autop_eta_k: float = 3.0            # autopilot: k in the frozen estimate mu + k*sigma over converged honest
                                        #   rounds (lower => tighter/lower estimate, closer to the fair eta).
    autop_honest_until: int = 0         # autopilot: SAFETY CAP on honest-client rounds; FR trains fully
                                        #   honest until its BER FLATTENS (auto-detected) or this cap. 0=off.
    autop_oracle_eta: float = 0.0       # autopilot DIAGNOSTIC: if >0, FR is GIVEN the true eta (~0.09) not estimated
    autop_common_per_class: int = -1    # autopilot DATA-ABLATION: -1=full shard; 0=triggers only; N=+N/common class
    autop_honest_extra: int = 3         # autopilot: stay honest N rounds AFTER convergence (better frozen eta)
    autop_conv_eps: float = 0.02        # autopilot: convergence = honest BER improves < this for 2 rounds
    autop_enriched: bool = False        # autopilot: data source (False=full shard, True=trigger-heavy)
    # memory_exploit: train (embed) for warmup_rounds, then replay frozen memory.
    warmup_rounds: int = 1              # rounds of honest embedding up-front
    # shared: how much of the global to mix into a coast/replay (freshness vs mark)
    mem_blend_global: float = 0.3       # coast "freshness": fraction of the live global mixed into
                                        # a coast submission. THE TRILEMMA (validated 50-round):
                                        #   0.0 (frozen replay)  -> mark preserved BUT stale weights
                                        #        POISON the global (acc 72->53%, honest BER->0.5, FPR up)
                                        #   0.3 (blend)          -> no poisoning (acc 72%) BUT the mark
                                        #        decays -> the FR is caught by BER
                                        # Neither is a clean stealthy free-ride; see sub_coast_mode.
    sub_coast_mode: str = "transplant"  # "transplant" (global_now + frozen mark-delta; fresh+marked)

    # ---- watermarking ----
    watermark: bool = False     # honest clients embed an output-space watermark
    wm_bits: int = 0            # m; 0 -> auto = (num_classes - 1)//2 (l=2)
    wm_lambda: float = 5.0      # weight of L_wm in L = L_cl + lambda L_wm (Eq.11)
    wm_alpha: float = 0.4       # smoothing f() exponent (Eq. 8)
    wm_f: str = "power"         # smoothing kind: "power" | "sin" (Eq. 7-9)
    wm_beta: float = 0.6        # memory coefficient in the Eq. 14 update
    wm_label_smoothing: float = 0.1  # keeps the softmax tail movable
    wm_num_triggers: int = 50   # N_T trigger samples used for extraction (Eq.15)
    wm_eta: float = 0.25        # detection threshold on BER (Eq. 16)
    wm_verify_every: int = 1    # run verification every k rounds (cost control)
    # paper_faithful=True strips the three deviations so the run matches the bare
    # FareMark algorithm: (1) random (not sign-balanced) keys, (2) NO trigger-class
    # exclusion (full softmax, Eq.1/10 anti-dominance only), (3) threshold = the
    # paper's cumulative mu+3sigma over ALL rounds, no sliding window, no 0.25 cap.
    # Use with a high-class dataset (e.g. CIFAR-100) so the full-softmax projection
    # is embeddable.
    paper_faithful: bool = True
    # calib_on_all controls the attacker-vs-threshold relationship:
    #   False -> server calibrates eta on a trusted benign pool that excludes the
    #            attacker (idealized; attacker must GUESS eta). 
    #   True  -> eta is mu+3sigma over all clients incl. the attacker, computed
    #            each round during training (realistic; attacker poisons/inflates
    #            eta). 
    calib_on_all: bool = False

    def to_dict(self):
        return asdict(self)


# Index 0: fast smoke test to prove the pipeline learns
CONFIGS = [
    ExpConfig("smoke_mnist_smallcnn", "smallcnn", "mnist", num_clients=5,
              rounds=5, local_epochs=1, batch_size=64, expected_acc=(95.0, 100.0)),

    # ---- Table I reproduction configs (FedAvg baseline) ----
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

    # ---- free-rider attacks ----
    # The free-rider gate is the trend (main-task accuracy falls as the free-rider
    # fraction rises, cf. Fig. 7), not a single accuracy band. Sweep the number
    # of free-riders with the --num_free_riders override (or NUM_FREE_RIDERS in
    # submit_experiment.sh) and watch accuracy drop.

    # idx 7: fast smoke test for Fig.7 trend
    ExpConfig("fr_smoke_mnist", "smallcnn", "mnist", num_clients=10,
              rounds=10, local_epochs=1, batch_size=64,
              attack="previous_models", num_free_riders=0,
              expected_acc=(0.0, 100.0)),

    # idx 8: free-riding with previous models, ResNet-18 / CIFAR-10 (Fig. 7a).
    ExpConfig("fr_prev_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              attack="previous_models", num_free_riders=2,
              expected_acc=(0.0, 100.0)),

    # idx 9: free-riding with Gaussian noise, ResNet-18 / CIFAR-10 (Fig. 7c).
    ExpConfig("fr_gauss_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              attack="gaussian", num_free_riders=2, noise_sigma=0.1,
              expected_acc=(0.0, 100.0)),

    # ---- watermarking ----
    # idx 10: fast watermark smoke test
    ExpConfig("wm_smoke_mnist", "smallcnn", "mnist", num_clients=10,
              rounds=10, local_epochs=1, batch_size=64,
              watermark=True, wm_lambda=5.0, wm_beta=0.6,
              expected_acc=(0.0, 100.0)),

    # idx 11: fidelity run, ResNet-18 / CIFAR-10, all honest + watermarked
    ExpConfig("wm_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              watermark=True, wm_lambda=5.0, wm_beta=0.6,
              expected_acc=(86.0, 94.0)),

    # idx 12: detection run, watermark + free-riders (Tables III-V)
    ExpConfig("wm_fr_resnet18_cifar10", "resnet18", "cifar10", num_clients=10,
              watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="previous_models", num_free_riders=2,
              expected_acc=(0.0, 100.0)),

    # idx 13: PAPER-FAITHFUL detection target. CIFAR-100 (many classes -> many
    # bits, so the full-softmax projection is embeddable without our trigger-class
    # exclusion). Random keys + cumulative uncapped mu+3sigma threshold.
    ExpConfig("paper_faithful_resnet18_cifar100", "resnet18", "cifar100",
              num_clients=10, watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="previous_models", num_free_riders=2,
              paper_faithful=True, expected_acc=(0.0, 100.0)),

    # idx 14: SUBMARINE adaptive free-rider, paper-faithful CIFAR-100, 2 FR.
    # The headline "cheap evasion" config. Sweep the server-side calib_on_all
    # (option 1 vs 2) and sub_* knobs via CLI overrides.
    ExpConfig("submarine_paper_faithful_resnet18_cifar100", "resnet18", "cifar100",
              num_clients=10, watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="submarine", num_free_riders=2,
              paper_faithful=True, expected_acc=(0.0, 100.0)),

    # idx 15: MEMORY-EXPLOIT free-rider (train once, replay frozen mark forever),
    # paper-faithful CIFAR-100, 2 FR. The cheapest break; contrast its compute
    # (~1 round) with the submarine's.
    # warmup_rounds=5: on CIFAR-100 an honest client needs several rounds to
    # reach BER~0, so freeze the mark only after it has actually embedded. Raise
    # to ~8-10 for full 50-round runs; the amortized cost is warmup/total.
    ExpConfig("memexploit_paper_faithful_resnet18_cifar100", "resnet18", "cifar100",
              num_clients=10, watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="memory_exploit", num_free_riders=2, warmup_rounds=8,
              mem_blend_global=0.0, paper_faithful=True, expected_acc=(0.0, 100.0)),
    # idx 16: output-layer re-embed attack (the theoretically-motivated one).
    # Fresh global backbone + cheap head-only trigger fine-tune. Sweep reembed_scope
    # and reembed_steps to trace the effort-vs-evasion frontier (the weak-point demo).
    ExpConfig("reembed_paper_faithful_resnet18_cifar100", "resnet18", "cifar100",
              num_clients=10, watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="reembed", num_free_riders=2, reembed_scope="head",
              reembed_steps=40, paper_faithful=True, expected_acc=(0.0, 100.0)),
    # idx 17: autopilot — fully self-tuning submarine. No fixed schedule: it ends
    # warmup itself, predicts when BER will cross eta and taps just before, and
    # sizes each tap to the drift. Re-embeds on the fresh global (no poisoning).
    ExpConfig("autopilot_paper_faithful_resnet18_cifar100", "resnet18", "cifar100",
              num_clients=10, watermark=True, wm_lambda=5.0, wm_beta=0.6,
              attack="autopilot", num_free_riders=2, paper_faithful=True,
              expected_acc=(0.0, 100.0)),
]


def get_config(idx: int) -> ExpConfig:
    if idx < 0 or idx >= len(CONFIGS):
        raise IndexError(
            f"config_idx {idx} out of range (have {len(CONFIGS)}): "
            + ", ".join(f"{i}:{c.name}" for i, c in enumerate(CONFIGS))
        )
    return CONFIGS[idx]


def seed_for(cfg: ExpConfig, repeat: int) -> int:
    return cfg.base_seed + repeat