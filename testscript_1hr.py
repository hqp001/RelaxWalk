import pandas as pd
from multi_layer_relaxation_walk import relaxation_walk_deep

seed_list = [50, 51, 52, 53, 54]
input_size_list = [10, 100, 1000]
layer_num_list = [1, 2, 3]
layer_size_list = [100, 500]
timelimit = 60

walk_eps = 0.01
pick_bias = 0.05
gap = 2/3


for input_size in input_size_list:
    for layer_num in layer_num_list:
        for layer_size in layer_size_list:
            for seed in seed_list:
                tag = str([input_size] + layer_num * [layer_size] + [1])
                print(f'[{input_size}, {layer_num} x {layer_size}, 1] with seed {seed} relaxation waking start')
                result = relaxation_walk_deep(input_size, layer_num, layer_size, seed, walk_eps, pick_bias,
                                                      timelimit)
                print(f'max: {result[1]}')
                print(f'-----------------------------------------')
