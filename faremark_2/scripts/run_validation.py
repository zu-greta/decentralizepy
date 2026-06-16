import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from faremark.utils import load_config
from faremark.simulator import run_simulation
import logging

def main():
    # List of configs that reproduce each table
    configs = {
        'table_i': 'configs/exp_fidelity.yaml',
        'table_ii': 'configs/exp_detection.yaml',
        'table_iii': 'configs/exp_free_rider.yaml',
        'table_iv': 'configs/exp_free_rider.yaml',  # with attack_round
        'table_v': 'configs/exp_free_rider.yaml',  # with trigger_only
        'table_vi': 'configs/exp_robustness.yaml',
        'table_vii': 'configs/exp_ablation.yaml',
        'table_viii': 'configs/exp_ablation.yaml',  # memory enhanced
        'table_ix': 'configs/exp_ablation.yaml',    # capacity
    }
    # For simplicity, we run each config with specific overrides
    # This is a placeholder; you would implement specific runs per table.
    logging.info("Running validation experiments (this may take a while).")
    # For demonstration, we run only the fidelity experiment.
    config = load_config('configs/exp_fidelity.yaml')
    run_simulation(config)

if __name__ == '__main__':
    main()