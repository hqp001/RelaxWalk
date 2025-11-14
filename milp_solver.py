"""
MILP Solver for Pruned Model with Dense Evaluation using Gurobi Machine Learning
"""

import torch
import time
import gurobipy as gp
from gurobipy import GRB
import gurobi_ml as gml
from Network import Network


def solve(network, time_limit, seed):
    """
    Solve MILP on pruned model and evaluate solution on dense model using Gurobi ML.

    Args:
        network: Network object with both pruned and dense models
        time_limit: Time limit for Gurobi solver (seconds)
        seed: Random seed

    Returns:
        dict: Results with keys: method, model_size, parameters, seed, prune_amount,
              max_, original_max, original_max_time_elapsed
    """

    start_time = time.time()

    # Create Gurobi model
    model = gp.Model("neural_network_optimization")
    model.setParam('OutputFlag', 0)  # Suppress output
    model.setParam('TimeLimit', time_limit)
    model.setParam('Seed', seed)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions quickly
    model.setParam('PoolSearchMode', 1)  # Search for n best solutions
    model.setParam('PoolSolutions', 10000)  # Store up to 1000 solutions in the pool

    # Calculate model_size as list of neuron counts per layer [input_size, layer1, layer2, ..., output]
    model_size = [network.in_size] + network.layer_dims

    # Calculate number of non-zero parameters in sparse model
    non_zero_params = 0
    for param in network.sparse.parameters():
        non_zero_params += torch.count_nonzero(param).item()

    # Track statistics
    stats = {
        'method': 'gurobi_ml',
        'model_size': model_size,  # List of neurons per layer
        'parameters': non_zero_params,  # Count of non-zero parameters
        'seed': seed,
        'prune_amount': network.prune_amount,
        'max_': float('-inf'),
        'original_max': float('-inf'),  # Will be updated from network.original_max at the end
        'original_max_time_elapsed': None,  # Will be updated from network.original_max_time at the end
        'time_limit': time_limit
    }

    # Create input variables (bounded between 0 and 1)
    input_vars = model.addMVar(network.in_size, lb=0.0, ub=1.0, name="x")

    # Create output variable (unbounded for maximization)
    output_var = model.addMVar(1, lb=-GRB.INFINITY, name="y")

    # Add the sparse neural network as predictor constraints
    # Use network.sparse as the trained model
    pred_constr = gml.add_predictor_constr(
        model,
        network.sparse,  # Use the sparse (pruned) model
        input_vars,
        output_var
    )
    pred_constr.print_stats()

    # Set objective: maximize the output
    model.setObjective(output_var[0], GRB.MAXIMIZE)

    # Define callback function to evaluate new solutions on dense model
    def callback(model, where):
        """Callback to evaluate each new MIP solution on the dense model"""
        if where == GRB.Callback.MIPSOL:
            # Get the new solution
            x_sol = model.cbGetSolution(input_vars)

            # Evaluate on both sparse and dense models (forward_sparse evaluates dense internally)
            x_tensor = torch.tensor(x_sol, dtype=torch.float32).reshape(1, -1)
            network.forward_sparse(x_tensor)  # This updates network.original_max automatically

    # Optimize with callback
    model.optimize(callback)

    # Extract results - only record the last (final) maximum value
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            # Get the optimal input and output from Gurobi
            optimal_x = input_vars.X
            optimal_y = output_var.X[0]  # This is the sparse model output from Gurobi

            # Evaluate on both models (forward_sparse evaluates dense internally and updates original_max)
            x_tensor = torch.tensor(optimal_x, dtype=torch.float32).reshape(1, -1)
            network.forward_sparse(x_tensor)

            # Record the sparse model maximum (Gurobi's optimized value)
            stats['max_'] = optimal_y
            stats['solve_time'] = time.time() - start_time
            stats['sol_count'] = model.SolCount

    else:
        # No solution found
        stats['max_'] = network.original_max if network.original_max > float('-inf') else 0.0
        stats['solve_time'] = time.time() - start_time
        stats['sol_count'] = 0

    # Update original_max tracking
    stats['original_max'] = network.original_max
    stats['original_max_time_elapsed'] = network.original_max_time

    return stats
