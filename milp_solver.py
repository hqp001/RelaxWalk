"""
MILP Solver for Pruned Model with Dense Evaluation using Custom Gurobi Formulation
"""

import torch
import time
import gurobipy as gp
from gurobipy import GRB
from gurobi_lib.torch2gurobi import add_predictor_constr, check_correct_formulation
from gurobi_lib.GurobiSolver import remove_region_callback, solve_lp_relaxation, get_warm_start_from_relaxation, dense_evaluation_callback
from Network import Network


def solve(network, time_limit, seed):
    """
    Solve MILP on pruned model and evaluate solution on dense model using custom Gurobi formulation.

    Args:
        network: Network object with both pruned and dense models
        time_limit: Time limit for Gurobi solver (seconds)
        seed: Random seed

    Returns:
        dict: Results with keys: method, model_size, parameters, seed, prune_amount,
              max_, original_max, original_max_time_elapsed
    """

    relaxed_input, lp_relax_time, first_solution = solve_lp_relaxation(network, time_limit=120, seed=seed)
    print(f"LP Relaxation input: {relaxed_input}, Time: {lp_relax_time:.2f}s, Dense output: {first_solution}")

    warm_start = get_warm_start_from_relaxation(network, relaxed_input, time_limit=120, seed=seed)
    print("Warm start obtained")

    start_time = time.time()

    # Subtract LP relaxation time from the main model's time limit
    #remaining_time_limit = max(10, time_limit - lp_relax_time)
    remaining_time_limit = time_limit

    # Create Gurobi model
    model = gp.Model("neural_network_optimization")
    model.setParam('OutputFlag', 1)  # Suppress output
    model.setParam('TimeLimit', remaining_time_limit)
    model.setParam('Seed', seed)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions quickly
    model.setParam('LazyConstraints', 1)  # Enable lazy constraints for region removal
    # model.setParam('PoolSearchMode', 1)  # Search for n best solutions
    # model.setParam('PoolSolutions', GRB.MAXINT)  # Store up to 1000 solutions in the pool

    # Calculate model_size as list of neuron counts per layer [input_size, layer1, layer2, ..., output]
    model_size = [network.in_size] + network.layer_dims

    # Calculate number of non-zero parameters in sparse model
    non_zero_params = 0
    for param in network.sparse.parameters():
        non_zero_params += torch.count_nonzero(param).item()

    # Track statistics
    stats = {
        'method': 'custom_gurobi',
        'model_size': model_size,  # List of neurons per layer
        'parameters': non_zero_params,  # Count of non-zero parameters
        'seed': seed,
        'prune_amount': network.prune_amount,
        'max_': float('-inf'),
        'original_max': float('-inf'),  # Will be updated from network.original_max at the end
        'original_max_time_elapsed': None,  # Will be updated from network.original_max_time at the end
        'time_limit': time_limit,
        'first_solution': first_solution
    }

    # Create input variables (bounded between 0 and 1) with shape (1, in_size) for batch dimension
    input_vars = model.addMVar((1, network.in_size), lb=0.0, ub=1.0, name="x")

    # Create output variable (unbounded for maximization)
    output_var = model.addMVar((1, 1), lb=-GRB.INFINITY, name="y")

    # Add the sparse neural network as predictor constraints using custom formulation
    # Use network.sparse as the trained model
    add_predictor_constr(
        model,
        network.sparse,
        network.dense,
        input_vars,
        output_var
    )

    print("Verifying formulation correctness...")
    is_correct = check_correct_formulation(model, network.sparse, input_vars, output_var)
    if is_correct:
        print("✓ Formulation verified successfully!")
    else:
        print("✗ Warning: Formulation verification failed!")
        raise ValueError("Formulation verification failed - neural network constraints may be incorrect")

    # Apply warm start from LP relaxation
    input_vars.Start = warm_start['input_vars']
    for i, binary_layer in enumerate(model._binary):
        binary_layer.Start = warm_start['binary_vars'][i]
    print("Warm start applied to model")

    # Initialize dense evaluation counter
    model._dense_eval_count = 0

    model.setObjective(output_var[0], GRB.MAXIMIZE)

    #model.optimize(dense_evaluation_callback)
    model.optimize(remove_region_callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            optimal_y = output_var.X[0][0]
            stats['max_'] = optimal_y
            stats['solve_time'] = time.time() - start_time
            stats['sol_count'] = model._dense_eval_count
        else:
            stats['max_'] = float('-inf')
            stats['solve_time'] = time.time() - start_time
            stats['sol_count'] = model._dense_eval_count
    else:
        stats['max_'] = float('-inf')
        stats['solve_time'] = time.time() - start_time
        stats['sol_count'] = model._dense_eval_count

    stats['original_max'] = model._original_max
    stats['original_max_time_elapsed'] = model._original_max_time

    return stats
