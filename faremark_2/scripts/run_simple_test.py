import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faremark.simulator import run_simulation

def main():
    config = {
        'experiment_name': 'simple_test',
        'model': 'resnet18',
        'dataset': 'cifar10',
        'num_clients': 10,
        'num_rounds': 10,
        'local_epochs': 2,
        'batch_size': 16,
        'learning_rate': 0.01,
        'watermark_lambda': 0.1,
        'memory_mu': 0.01,
        'trigger_sample_count': 50,
        'alpha': 0.5,
        'free_rider_ratio': 0.1,
        'free_rider_strategy': 'previous_models',
        'seed': 42
    }
    run_simulation(config)

if __name__ == '__main__':
    main()