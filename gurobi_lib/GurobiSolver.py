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
            print(f"DENSE OUTPUT: {dense_max}")
            if dense_max > model._original_max:
                model._original_max = dense_max
                model._original_x_max = x_tensor.clone()
                model._original_max_time = time.time() - model._start_time

def lp_relaxation_callback(model, where):
    if where != gp.GRB.Callback.MIPNODE:
        return

    status = model.cbGet(gp.GRB.Callback.MIPNODE_STATUS)

    if status in (gp.GRB.OPTIMAL, gp.GRB.INTEGER):
        # LP relaxation values for your input vars
        model._relaxed_input = model.cbGetNodeRel(model._input_vars)

        # Compute dense output for the relaxed input
        x_tensor = torch.tensor(model._relaxed_input, dtype=torch.float32)
        with torch.no_grad():
            dense_output = model._dense_network(x_tensor)
            model._dense_output = dense_output.max().item()
            print(f"LP RELAXATION DENSE OUTPUT: {model._dense_output}")

        model.terminate()

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
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Seed', seed)
    #model.Params.NodeLimit = 0   # stop after root node is processed

    # Create input variables (bounded between 0 and 1)
    input_vars = model.addMVar((1, network.in_size), lb=0.0, ub=1.0, name="x")

    # Create output variable
    output_var = model.addMVar((1, 1), lb=-gp.GRB.INFINITY, name="y")

    # Add predictor constraints using dense model with skip ReLU constraints (LP relaxation)
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
    model._dense_output = None

    # Optimize with callback to capture root node LP relaxation
    model.optimize(lp_relaxation_callback)

    # Check if we captured the relaxation solution
    if model._relaxed_input is None:
        raise RuntimeError("LP relaxation failed to capture solution")

    # Check if we captured the dense output
    assert model._dense_output is not None, "LP relaxation failed to capture dense output"

    solve_time = time.time() - start_time

    print(f"LP Relaxation completed - Dense output: {model._dense_output}, Solve time: {solve_time:.4f}s")

    return np.array(model._relaxed_input).reshape(1, -1), solve_time, model._dense_output

def get_warm_start_from_relaxation(network, relaxed_input, time_limit, seed):
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

    warm_start = {
        'input_vars': input_vars.X,
        'binary_vars': [binary_layer.X for binary_layer in model._binary]
    }
    print(f"WARM START OUTPUT: {output_var.X}")
    return warm_start

def remove_region_callback(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        model._dense_eval_count += 1

        x_sol = model.cbGetSolution(model._input_vars)
        x_tensor = torch.tensor(x_sol, dtype=torch.float32)

        with torch.no_grad():
            dense_output = model._dense_network(x_tensor)
            dense_max = dense_output.max().item()
            print(f"DENSE OUTPUT: {dense_max}")
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
