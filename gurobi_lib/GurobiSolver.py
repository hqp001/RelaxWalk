import time

import torch
import numpy as np

import gurobipy as gp
from .torch2gurobi import add_predictor_constr, check_correct_formulation

def dense_evaluation_callback(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        model._dense_eval_count += 1

        x_sol = model.cbGetSolution(model._input_vars)
        x_tensor = torch.tensor(x_sol, dtype=torch.float32)

        with torch.no_grad():
            dense_output = model._dense_network(x_tensor)
            dense_max = dense_output.max().item()

            if dense_max > model._original_max:
                model._original_max = dense_max
                model._original_x_max = x_tensor.clone()
                model._original_max_time = time.time() - model._start_time

def lp_relaxation_callback(model, where):
    """
    Callback to capture the root node LP relaxation solution and terminate immediately.
    Only captures input values, not neuron states.
    """
    if where == gp.GRB.Callback.MIPNODE:
        # Check node count - root node is node 0
        nodecount = model.cbGet(gp.GRB.Callback.MIPNODE_NODCNT)

        if nodecount > 0:
            # Root node completed, terminate now
            model.terminate()
            return

        # We're at root node, check if LP relaxation is optimal
        status = model.cbGet(gp.GRB.Callback.MIPNODE_STATUS)
        if status == gp.GRB.OPTIMAL:
            # Capture LP relaxation input values
            model._relaxed_input = model.cbGetNodeRel(model._input_vars)

def solve_lp_relaxation(network, time_limit, seed):
    """
    Solve LP relaxation independently to get the relaxed input solution.

    Uses callback approach from util.py to capture root node LP relaxation.

    Args:
        network: Network object with in_size attribute and dense model
        time_limit: Time limit for solver (default: 60 seconds)
        seed: Random seed (default: 42)

    Returns:
        tuple: (input_solution, solve_time)
            - input_solution (numpy.ndarray): Input values from LP relaxation
            - solve_time (float): Time spent solving in seconds
    """
    start_time = time.time()

    # Create independent Gurobi model
    model = gp.Model("lp_relaxation_solver")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Seed', seed)

    # Create input variables (bounded between 0 and 1)
    input_vars = model.addMVar((1, network.in_size), lb=0.0, ub=1.0, name="x")

    # Create output variable
    output_var = model.addMVar((1, 1), lb=-gp.GRB.INFINITY, name="y")

    # Add predictor constraints using dense model with standard add_relu_constr
    add_predictor_constr(model, network.dense, network.dense, input_vars, output_var)

    # Set objective to maximize output
    model.setObjective(output_var[0], gp.GRB.MAXIMIZE)

    # Update the model to ensure variables are fully created and accessible
    model.update()

    # Get the list of input variables for the callback
    # tolist() returns nested list for 2D MVar, so we need to flatten it
    input_var_nested = input_vars.tolist()
    input_var_list = [var for row in input_var_nested for var in row]

    # Store input variables on model for callback access
    model._input_vars = input_var_list
    model._relaxed_input = None

    # Optimize with callback to capture root node LP relaxation
    model.optimize(lp_relaxation_callback)

    # Check if we captured the relaxation solution
    if model._relaxed_input is None:
        raise RuntimeError("LP relaxation failed to capture solution")

    solve_time = time.time() - start_time

    return np.array(model._relaxed_input).reshape(1, -1), solve_time

def get_warm_start_from_relaxation(network, relaxed_input, time_limit=60, seed=42):
    """
    Fix relaxed input in sparse model and solve to get integer-feasible solution for warm start.

    Args:
        network: Network object with sparse model
        relaxed_input: Input values from LP relaxation (numpy array)
        time_limit: Time limit for solver (default: 60 seconds)
        seed: Random seed (default: 42)

    Returns:
        dict: Variable values for warm start {'input_vars': array, 'binary_vars': list of arrays}
    """
    # Create independent Gurobi model
    model = gp.Model("warm_start_solver")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Seed', seed)

    # Create input variables
    input_vars = model.addMVar((1, network.in_size), lb=0.0, ub=1.0, name="x")

    # Create output variable
    output_var = model.addMVar((1, 1), lb=-gp.GRB.INFINITY, name="y")

    # Add predictor constraints using sparse model
    add_predictor_constr(model, network.sparse, network.sparse, input_vars, output_var)

    # Fix input to relaxed values (like check_correct_formulation does)
    input_constr = model.addConstr(input_vars == relaxed_input)

    # Set objective to maximize output
    model.setObjective(output_var[0], gp.GRB.MAXIMIZE)

    # Optimize to get integer solution
    model.optimize()

    # Check if we got a solution
    if model.status != gp.GRB.OPTIMAL or model.SolCount == 0:
        raise RuntimeError(f"Warm start solver failed with status {model.status}")

    # Extract all variable values
    warm_start = {
        'input_vars': input_vars.X,
        'binary_vars': [binary_layer.X for binary_layer in model._binary]
    }
    return warm_start

def remove_region_callback(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        model._dense_eval_count += 1

        x_sol = model.cbGetSolution(model._input_vars)
        x_tensor = torch.tensor(x_sol, dtype=torch.float32)

        with torch.no_grad():
            dense_output = model._dense_network(x_tensor)
            dense_max = dense_output.max().item()

            if dense_max > model._original_max:
                model._original_max = dense_max
                model._original_x_max = x_tensor.clone()
                model._original_max_time = time.time() - model._start_time

        neuron_sol = [model.cbGetSolution(var) for var in model._binary]

        added_constr = 0

        for i in range(len(neuron_sol)):
            neuron_val = np.where(neuron_sol[i] >= 0.5, -1, 1).reshape(1, -1)
            added_constr += (model._binary[i].reshape(1, -1) @ neuron_val.T).item()
            added_constr += np.sum(neuron_val == -1)

        model.cbLazy(added_constr >= 1)

        print("Removed region")
