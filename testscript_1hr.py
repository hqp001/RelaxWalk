import pandas as pd
from multi_layer_relaxation_walk import relaxation_walk_deep

#seed_list = [50, 51, 52, 53, 54]
seed_list = [51, 52, 53, 54]
#input_size_list = [10, 100, 1000]
input_size_list = [100]
layer_num_list = [3]
#layer_size_list = [100, 500]
layer_size_list = [100]
prune_amount_list = [0.0, 0.1, 0.2, 0.3, 0.5]
#prune_amount_list = [0.3, 0.5]
timelimit = 120

walk_eps = 0.01
pick_bias = 0.05
gap = 2/3


for input_size in input_size_list:
    for layer_num in layer_num_list:
        for layer_size in layer_size_list:
            for seed in seed_list:
                for prune_amount in prune_amount_list:
                    try:
                        tag = str([input_size] + layer_num * [layer_size] + [1])
                        print(f'[{input_size}, {layer_num} x {layer_size}, 1] with seed {seed} prune {prune_amount} relaxation waking start')
                        result = relaxation_walk_deep(input_size, layer_num, layer_size, seed, walk_eps, pick_bias,
                                                              timelimit, prune_amount)
                        print(f'max: {result[1]}')
                        print(f'-----------------------------------------')
                    except Exception as e:
                        print(f'ERROR in experiment [{input_size}, {layer_num} x {layer_size}, 1] with seed {seed} prune {prune_amount}:')
                        print(f'  {type(e).__name__}: {str(e)}')
                        print(f'Skipping this experiment and continuing with the next one...')
                        print(f'-----------------------------------------')
