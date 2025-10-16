from walk import single_walk_with_timelimit
from io_csv import store_data
from util import (
    get_binary_activations,
    get_linear_relaxation,
    get_linear_relaxation_with_restriction,
    get_prob_list_with_bias,
    get_index_from_prob_list,
    solve_lp_pre_calc,
    update_x
)
from Network import Network
import numpy as np
import torch
import time

def relaxation_walk_deep(input_size, layer_num, layer_size, random_seed, walk_eps, pick_bias, timelimit, prune_amount):
    seed = random_seed
    eps = walk_eps
    bias = pick_bias

    layer_dims = layer_num * [layer_size] + [1]

    model_nn = Network(in_size=input_size, layer_dims=layer_dims, seed=seed, prune_amount=prune_amount)
    model_nn.use_original = True

    start = time.time()
    max_ = -1000
    x_max = None
    update_list = []

    np.random.seed(seed)

    # Use original model for initial solution search
    x, frac_z = get_linear_relaxation(model_nn, timelimit)
    if x is None:
        store_data({
            'method': 'RW',
            'model_size': [input_size] + layer_dims,
            'parameters': [walk_eps, pick_bias],
            'seed': random_seed,
            'prune_amount': prune_amount,
            'x_max': None,
            'max_': None,
            'first_max': None,
            'time_count': None,
            'start_count': None,
            'valid_start_count': None,
            'original_max': None,
            'update_list': None
        }, f'RW_experiment_result_{timelimit}.csv')
        return None, None, None, None, None, None, None
    int_z = get_binary_activations(model_nn, x)
    prob_list = []
    for i in range(layer_num):
        prob_list_layer = get_prob_list_with_bias(int_z[i], frac_z[i], bias)
        prob_list.append(prob_list_layer)


    # First walk for relaxation - use original model
    step_count = 0
    ap = get_binary_activations(model_nn, x)
    max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
    while max_lp is not None and max_lp > max_:
        # print(max_lp)
        step_count += 1
        max_ = max_lp
        x_max = x_new
        if time.time() - start > timelimit:
            break
        x = update_x(model_nn, x, x_new, eps)
        ap = get_binary_activations(model_nn, x)
        max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
        if max_lp is None:
            break

    first_max = max_
    record_ap = [int_z]
    ap = int_z
    now = time.time() - start
    update_list.append([first_max, x_max, 1, 1, now])

    # print(f'relaxation walk done by {now}')

    valid_start_count = 1
    start_count = 1

    # Generate a new points in a while loop
    while time.time() - start < timelimit:

        restriction = []
        picked_neurons = []

        for i in range(layer_num):
            while len(restriction) < len(int_z[i]) and len(picked_neurons) < len(
                    int_z[i]) and time.time() - start < timelimit:
                random_num = np.random.uniform(0, prob_list[i][-1])
                pick = get_index_from_prob_list(prob_list[i], random_num)
                while pick in picked_neurons:
                    random_num = np.random.uniform(0, prob_list[i][-1])
                    pick = get_index_from_prob_list(prob_list[i], random_num)
                picked_neurons.append(pick)
                restriction.append([i, pick])
                # print(restriction)

                # Generate a single new point - use original model for search
                x_vals, frac_ap = get_linear_relaxation_with_restriction(model_nn, int_z, restriction)
                if x_vals is None:
                    # In this case, infeasible, then try to pick another point
                    restriction.remove(restriction[-1])
                    continue
                ap = get_binary_activations(model_nn, x_vals)
                start_count += 1
                if ap in record_ap:
                    continue
                valid_start_count += 1
                record_ap.append(ap)
                time_remain = timelimit - (time.time() - start)
                if time_remain <= 0:
                    break
                else:
                    # now = time.time() - start
                    # print(f'{valid_start_count} walk start with time remain{time_remain} at {now}')
                    # Use pruned model for walking (limit to 5 steps)
                    model_nn.use_original = False
                    pruned_x_max, pruned_max, step_count1, time_consuming = single_walk_with_timelimit(model_nn, x_vals, eps,
                                                                                                     time_remain, max_steps=5)

                    # Now walk on original model using pruned model's solution
                    time_remain = timelimit - (time.time() - start)
                    if time_remain > 0 and pruned_x_max is not None:
                        model_nn.use_original = True
                        local_x_max, local_max, step_count2, time_consuming2 = single_walk_with_timelimit(model_nn, pruned_x_max, eps,
                                                                                                         time_remain, max_steps=5)
                    else:
                        # No time left or pruned walk failed
                        model_nn.use_original = True
                        local_x_max = model_nn.original_x_max
                        local_max = model_nn.original_max

                    # now = time.time() - start
                    # print(f'{valid_start_count} walk done by {now} with {step_count1} steps')
                    if local_max > max_:
                        max_ = local_max
                        x_max = local_x_max
                        now = time.time() - start
                        update_list.append([max_, x_max, start_count, valid_start_count, now])


    time_count = time.time() - start
    original_max = model_nn.original_max
    original_x_max = model_nn.original_x_max

    if max_ is None or x_max is None:
        store_data({
            'method': 'RW',
            'model_size': [input_size] + layer_dims,
            'parameters': [walk_eps, pick_bias],
            'seed': random_seed,
            'prune_amount': prune_amount,
            'x_max': x_max,
            'max_': None,
            'first_max': first_max,
            'time_count': time_count,
            'start_count': start_count,
            'valid_start_count': valid_start_count,
            'original_max': original_max,
            'update_list': update_list
        }, f'RW_experiment_result_{timelimit}.csv')
    else:
        store_data({
            'method': 'RW',
            'model_size': [input_size] + layer_dims,
            'parameters': [walk_eps, pick_bias],
            'seed': random_seed,
            'prune_amount': prune_amount,
            'x_max': x_max,
            'max_': model_nn(torch.FloatTensor(x_max)).item(),
            'first_max': first_max,
            'time_count': time_count,
            'start_count': start_count,
            'valid_start_count': valid_start_count,
            'original_max': original_max,
            'update_list': update_list
        }, f'RW_experiment_result_{timelimit}.csv')
    return x_max, max_, first_max, time_count, start_count, valid_start_count, update_list

