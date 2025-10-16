import pandas as pd
from gurobi_benchmark import *
from multi_layer_relaxation_walk import *
from main import *

seed_list = [50, 51, 52, 53, 54]
input_size_list = [10, 100, 1000]
layer_num_list = [1, 2, 3]
layer_size_list = [100, 500]
timelimit = 60

walk_eps = 0.01
pick_bias = 0.05
gap = 2/3

rw_df_exist = True
sm_df_exist = True
# rwd_df_exist = True
gr_df_exist = True


if os.path.exists(f'Gurobi_Benchmark.csv'):
    gr_df = pd.read_csv(f'Gurobi_Benchmark.csv')
else:
    gr_df_exist = False
    gr_df = pd.DataFrame()


if os.path.exists(f'RW_experiment_result_{timelimit}.csv'):
    rw_df = pd.read_csv(f'RW_experiment_result_{timelimit}.csv')
else:
    rw_df_exist = False
    rw_df = pd.DataFrame()
# print(rw_df.to_string())
if os.path.exists(f'SM_experiment_result_{timelimit}.csv'):
    sm_df = pd.read_csv(f'SM_experiment_result_{timelimit}.csv')
else:
    sm_df_exist = False
    sm_df = pd.DataFrame()

if os.path.exists(f'RWD_experiment_result_{timelimit}.csv'):
    rwd_df = pd.read_csv(f'RWD_experiment_result_{timelimit}.csv')
else:
    rwd_df_exist = False
    rwd_df = pd.DataFrame()

for input_size in input_size_list:
    for layer_num in layer_num_list:
        for layer_size in layer_size_list:
            for seed in seed_list:
                tag = str([input_size] + layer_num * [layer_size] + [1])
                # print(tag)
                # print(
                #     f'[{input_size}, {layer_num} x {layer_size}, 1] with seed {seed} dynamic relaxation walking start')
                # inHistory = False
                # if rwd_df_exist and rwd_df['model_size'].isin([tag]).any():
                #     results = rwd_df.index[rwd_df['model_size'] == tag].tolist()
                #     for result in results:
                #         if rwd_df.loc[result, 'seed'] == seed and rwd_df.loc[result, 'parameters'] == str([walk_eps]):
                #             result = rwd_df.loc[result, 'max_']
                #             print(f'max: {result}')
                #             print(f'-----------------------------------------')
                #             inHistory = True
                #             break
                #     if not inHistory:
                #         result = relaxation_walk_dynamic(input_size, layer_num, layer_size, seed, walk_eps, timelimit)
                #         print(f'max: {result[1]}')
                #         print(f'-----------------------------------------')
                # else:
                #     result = relaxation_walk_dynamic(input_size, layer_num, layer_size, seed, walk_eps, timelimit)
                #     print(f'max: {result[1]}')
                #     print(f'-----------------------------------------')

                print(
                    f'[{input_size}, {layer_num} x {layer_size}, 1] with seed {seed} Gurobi MIP')
                inHistory = False
                if gr_df_exist and gr_df['model_size'].isin([tag]).any():
                    results = gr_df.index[gr_df['model_size'] == tag].tolist()
                    for result in results:
                        if gr_df.loc[result, 'seed'] == seed:
                            max_ = gr_df.loc[result, 'max_']
                            time_count = gr_df.loc[result, 'time_count']
                            print(f'max: {max_}')
                            print(f'time: {time_count}')
                            print(f'-----------------------------------------')
                            inHistory = True
                            break
                    if not inHistory:
                        result = solve_with_gurobi(input_size, layer_num, layer_size, seed, timelimit)
                        print(f'max: {result[1]}')
                        print(f'time: {result[2]}')
                        print(f'-----------------------------------------')
                else:
                    result = solve_with_gurobi(input_size, layer_num, layer_size, seed, timelimit)
                    print(f'max: {result[1]}')
                    print(f'time: {result[2]}')
                    print(f'-----------------------------------------')


                print(f'[{input_size}, {layer_num} x {layer_size}, 1] with seed {seed} relaxation waking start')
                inHistory = False
                if rw_df_exist and rw_df['model_size'].isin([tag]).any():
                    results = rw_df.index[rw_df['model_size'] == tag].tolist()
                    for result in results:
                        if rw_df.loc[result, 'seed'] == seed and rw_df.loc[result, 'parameters'] == str([walk_eps, pick_bias]):
                            result = rw_df.loc[result, 'max_']
                            print(f'max: {result}')
                            print(f'-----------------------------------------')
                            inHistory = True
                            break
                    if not inHistory:
                        result = relaxation_walk_deep(input_size, layer_num, layer_size, seed, walk_eps, pick_bias,
                                                      timelimit)
                        print(f'max: {result[1]}')
                        print(f'-----------------------------------------')
                else:
                    result = relaxation_walk_deep(input_size, layer_num, layer_size, seed, walk_eps, pick_bias,
                                                  timelimit)
                    print(f'max: {result[1]}')
                    print(f'-----------------------------------------')

                print(f'[{input_size}, {layer_num} x {layer_size}, 1] with seed {seed} sampling mip start')
                inHistory = False
                if sm_df_exist and sm_df['model_size'].isin([tag]).any():
                    results = sm_df.index[sm_df['model_size'] == tag].tolist()
                    for result in results:
                        # print(sm_df.loc[result, 'parameters'])
                        if sm_df.loc[result, 'seed'] == seed and sm_df.loc[result, 'parameters'] == str([gap]):
                            result = sm_df.loc[result, 'max_']
                            print(f'max: {result}')
                            print(f'-----------------------------------------')
                            inHistory = True
                            break
                    if not inHistory:
                        result = sampling_mip(input_size, layer_num, layer_size, seed, gap, timelimit)
                        print(f'max: {result[1]}')
                        print(f'-----------------------------------------')
                else:
                    result = sampling_mip(input_size, layer_num, layer_size, seed, gap, timelimit)
                    print(f'max: {result[1]}')
                    print(f'-----------------------------------------')
