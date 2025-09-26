#!/usr/bin/env python3

import yaml
import os

# Fixed values
input_size = 1024
walk_eps = 0.5
pick_bias = 0.05
timelimit = 200

# Variable values (small batch for testing)
layer_nums = [5, 6, 7]
layer_sizes = [512, 1024]
seeds = [1, 2, 3, 4, 5]
pruned_densities = [0.3, 0.5, 0.8, 1.0]

# Create directory
os.makedirs('configs/mass_experiments', exist_ok=True)

config_id = 0
exp_id = 0

for layer_num in layer_nums:
    for layer_size in layer_sizes:
        for seed in seeds:
            exp_name = f'exp_{exp_id:04d}'

            for pruned_density in pruned_densities:
                config = {
                    'experiment': {'name': exp_name, 'method': 'RW'},
                    'network': {
                        'input_size': input_size,
                        'layer_num': layer_num,
                        'layer_size': layer_size,
                        'pruned_density': pruned_density
                    },
                    'algorithm': {
                        'walk_eps': walk_eps,
                        'pick_bias': pick_bias,
                        'random_seed': seed,
                        'timelimit': timelimit
                    }
                }

                filename = f'configs/mass_experiments/config_{config_id:04d}.yaml'
                with open(filename, 'w') as f:
                    yaml.dump(config, f)

                config_id += 1

            exp_id += 1

print(f'Generated {config_id} config files')
