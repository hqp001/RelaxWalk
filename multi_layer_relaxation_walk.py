# REFACTORED: Using modular package imports with logging
import time
import numpy as np
import torch
from Network import Network
from configs.log_config import get_logger

logger = get_logger(__name__)
from walk import single_walk_with_timelimit
from utils import (
    get_linear_relaxation,
    get_binary_activations,
    solve_lp_pre_calc,
    update_x,
    get_prob_list_with_bias,
    get_index_from_prob_list,
    get_linear_relaxation_with_restriction
)
from utils.pruning import prune_model

def relaxation_walk_deep(input_size, layer_num, layer_size, random_seed, walk_eps, pick_bias, timelimit, pruned_density=1.0):
    """Main relaxation walk algorithm with comprehensive logging."""
    logger.info(f"Starting relaxation_walk_deep with params: input_size={input_size}, "
                f"layer_num={layer_num}, layer_size={layer_size}, seed={random_seed}, "
                f"eps={walk_eps}, bias={pick_bias}, timelimit={timelimit}")

    seed = random_seed
    eps = walk_eps
    bias = pick_bias

    layer_dims = layer_num * [layer_size] + [1]
    logger.info(f"Creating network with layer_dims={layer_dims}")

    try:
        original_model = Network(in_size=input_size, layer_dims=layer_dims, seed=seed)
        model_nn = prune_model(original_model, pruned_density) if pruned_density < 1.0 else original_model
        logger.info("Network created successfully")
    except Exception as e:
        logger.error(f"Failed to create network: {e}")
        raise

    start = time.time()
    max_ = -1000
    x_max = None
    update_list = []

    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")

    logger.info("Starting linear relaxation...")
    try:
        x, frac_z = get_linear_relaxation(model_nn, timelimit)
        if x is None:
            logger.warning("Linear relaxation returned None - problem may be infeasible")
            return None, None, None, None, None, None, None
        logger.info(f"Linear relaxation successful, x shape: {len(x) if x else 'None'}")
    except Exception as e:
        logger.error(f"Linear relaxation failed: {e}")
        raise
    try:
        int_z = get_binary_activations(model_nn, x)
        logger.info(f"Got binary activations for {len(int_z)} layers")
    except Exception as e:
        logger.error(f"Failed to get binary activations: {e}")
        raise

    prob_list = []
    for i in range(layer_num):
        prob_list_layer = get_prob_list_with_bias(int_z[i], frac_z[i], bias)
        prob_list.append(prob_list_layer)
    logger.info(f"Created probability lists for {len(prob_list)} layers")

    # First walk for relaxation
    logger.info("Starting first relaxation walk...")
    step_count = 0
    try:
        ap = get_binary_activations(model_nn, x)
        max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
        if isinstance(x_new, list):
            original_model(torch.FloatTensor(x_new))
        logger.info(f"Initial LP solve: max_lp={max_lp}")
    except Exception as e:
        logger.error(f"Failed initial LP solve: {e}")
        raise

    while max_lp > max_:
        step_count += 1
        max_ = max_lp
        x_max = x_new
        elapsed_time = time.time() - start
        logger.debug(f"First walk step {step_count}: max_lp={max_lp:.6f}, elapsed_time={elapsed_time:.3f}s")

        if elapsed_time > timelimit:
            logger.warning(f"Time limit reached during first walk at step {step_count}, elapsed={elapsed_time:.3f}s")
            break

        try:
            x = update_x(model_nn, x, x_new, eps)
            ap = get_binary_activations(model_nn, x)
            max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
            if isinstance(x_new, list):
                original_model(torch.FloatTensor(x_new))
            logger.debug(f"First walk step {step_count}: updated x, new max_lp={max_lp:.6f}")
        except Exception as e:
            logger.error(f"Error in first walk step {step_count}: {e}")
            break

    first_max = max_
    record_ap = [int_z]
    ap = int_z
    now = time.time() - start
    update_list.append([first_max, x_max, 1, 1, now])

    logger.info(f"First relaxation walk completed in {now:.3f}s with {step_count} steps, max={first_max}")

    valid_start_count = 1
    start_count = 1

    # Generate a new points in a while loop
    logger.info("Starting main point generation loop...")
    main_loop_iteration = 0
    while time.time() - start < timelimit:
        main_loop_iteration += 1
        elapsed_time = time.time() - start
        time_remaining = timelimit - elapsed_time
        logger.debug(f"Main loop iteration {main_loop_iteration}: elapsed={elapsed_time:.3f}s, remaining={time_remaining:.3f}s")

        restriction = []
        picked_neurons = []

        for i in range(layer_num):
            layer_restriction_count = 0
            while len(restriction) < len(int_z[i]) and len(picked_neurons) < len(
                    int_z[i]) and time.time() - start < timelimit:
                layer_restriction_count += 1
                logger.debug(f"Layer {i}, restriction attempt {layer_restriction_count}")

                random_num = np.random.uniform(0, prob_list[i][-1])
                pick = get_index_from_prob_list(prob_list[i], random_num)
                neuron_pick_attempts = 1
                while pick in picked_neurons:
                    neuron_pick_attempts += 1
                    random_num = np.random.uniform(0, prob_list[i][-1])
                    pick = get_index_from_prob_list(prob_list[i], random_num)
                    if neuron_pick_attempts > 100:  # Prevent infinite loop
                        logger.warning(f"Too many attempts to pick unique neuron in layer {i}")
                        break

                picked_neurons.append(pick)
                restriction.append([i, pick])
                logger.debug(f"Added restriction [layer={i}, neuron={pick}], total restrictions: {len(restriction)}")

                # Generate a single new point
                x_vals, frac_ap = get_linear_relaxation_with_restriction(model_nn, int_z, restriction)
                if x_vals is not None and isinstance(x_vals, list):
                    original_model(torch.FloatTensor(x_vals))
                if x_vals is None:
                    # In this case, infeasible, then try to pick another point
                    logger.debug(f"Infeasible restriction, removing last restriction. Remaining: {len(restriction)-1}")
                    restriction.remove(restriction[-1])
                    continue

                ap = get_binary_activations(model_nn, x_vals)
                start_count += 1
                logger.debug(f"Generated new point, total start_count: {start_count}")

                if ap in record_ap:
                    logger.debug("Point already exists in record, skipping")
                    continue

                valid_start_count += 1
                record_ap.append(ap)
                time_remain = timelimit - (time.time() - start)
                logger.info(f"Starting walk {valid_start_count} with {len(restriction)} restrictions, time_remain={time_remain:.3f}s")

                if time_remain <= 0:
                    logger.warning("No time remaining for walk")
                    break
                else:
                    local_x_max, local_max, step_count1, time_consuming = single_walk_with_timelimit(model_nn, x_vals, eps,
                                                                                                     time_remain, original_model)
                    now = time.time() - start
                    logger.info(f"Walk {valid_start_count} completed: {step_count1} steps, local_max={local_max:.6f}, time_used={time_consuming:.3f}s, total_elapsed={now:.3f}s")

                    if local_max > max_:
                        logger.info(f"New best maximum found: {local_max:.6f} > {max_:.6f}")
                        max_ = local_max
                        x_max = local_x_max
                        now = time.time() - start
                        update_list.append([max_, x_max, start_count, valid_start_count, now])


    time_count = time.time() - start

    # Calculate max_ value using the neural network if x_max exists
    if max_ is not None and x_max is not None:
        max_ = model_nn(torch.FloatTensor(x_max)).item()

    return x_max, max_, first_max, time_count, start_count, valid_start_count, update_list, original_model

