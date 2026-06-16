import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from faremark.utils import load_config
from faremark.simulator import run_simulation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    config = load_config(args.config)
    run_simulation(config)

if __name__ == '__main__':
    main()