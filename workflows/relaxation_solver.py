import gurobipy as gb
from gurobipy import GRB
import numpy as np
from data_types import Network
from typing import Optional, Tuple, List, Any
from logger_config import get_logger, TimingContext, timer


def gbdict2lst(dic, layer_dims):
    """Convert Gurobi dictionary to list format."""
    lst = []
    count = 0
    if isinstance(layer_dims, list) == False:
        for i in range(layer_dims):
            lst.append(dic[i].x)
        return lst
    else:
        for dim in layer_dims:
            temp = []
            for i in range(dim):
                temp.append(dic[count][i].x)
            count += 1
            lst.append(temp)
        return lst


def gbdict2lst_z(dic, layer_dims):
    """Convert Gurobi Z dictionary to list format."""
    lst = []
    count = 0
    if isinstance(layer_dims, list) == False:
        for i in range(layer_dims):
            lst.append(dic[i])
        return lst
    else:
        for dim in layer_dims:
            temp = []
            for i in range(dim):
                temp.append(dic[count][i])
            count += 1
            lst.append(temp)
        return lst


@timer("linear relaxation solve")
def solve_linear_relaxation(network: Network, time_limit: float) -> Tuple[Optional[List[float]], Optional[List[List[float]]]]:
    """
    Solve linear relaxation of neural network optimization problem.

    Args:
        network: Neural network instance
        time_limit: Time limit for optimization

    Returns:
        Tuple of (input_values, fractional_activations) or (None, None) if infeasible
    """
    logger = get_logger("relaxation_solver")
    logger.debug("Starting linear relaxation",
                network_size=f"{network.in_size}→{network.layer_dims}",
                time_limit=time_limit)

    w = network.get_weight_matrix()
    b = network.get_bias_matrix()
    layer_dims = network.layer_dims
    in_size = network.in_size

    with TimingContext(logger, "Gurobi model setup"):
        model = gb.Model()
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output
        global x_vals, z_vals
        x_vals = []
        z_vals = []

    def get_relaxation(model, where):
        if where == GRB.Callback.MIPNODE:
            global x_vals, z_vals
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if status == GRB.INTEGER:
                x_vals = model.cbGetSolution(x)
                z_vals = []
                for i in range(len(layer_dims)):
                    z_vals.append(model.cbGetNodeRel(z[i]))
                z_vals = model.cbGetSolution(z)
                model.terminate()
            if status == GRB.OPTIMAL:
                x_vals = model.cbGetNodeRel(x)
                z_vals = []
                for i in range(len(layer_dims)):
                    z_vals.append(model.cbGetNodeRel(z[i]))
                model.terminate()

    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    x = model.addVars(in_size, ub=1, lb=0, name="input")
    z = {}
    h = {}
    g = {}

    count = 0
    for dim in layer_dims:
        z[count] = model.addVars(dim, vtype=GRB.BINARY, name="neuron_layer_" + str(count))
        h[count] = model.addVars(dim)
        g[count] = model.addVars(dim, lb=-GRB.INFINITY)
        count += 1

    # First layer constraints
    for i in range(layer_dims[0]):
        model.addConstr(gb.quicksum(w[0][i][j] * x[j] for j in range(in_size)) + b[0][i] == g[0][i])
        model.addConstr((z[0][i] == 0) >> (h[0][i] == 0))
        model.addConstr((z[0][i] == 1) >> (h[0][i] == g[0][i]))
        model.addConstr((z[0][i] == 0) >> (g[0][i] <= 0))

    # Subsequent layer constraints
    count = 1
    for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
        for i in range(output_dim):
            model.addConstr(
                gb.quicksum(w[count][i][j] * h[count - 1][j] for j in range(input_dim)) + b[count][i] == g[count][i])
            model.addConstr((z[count][i] == 0) >> (h[count][i] == 0))
            model.addConstr((z[count][i] == 1) >> (h[count][i] == g[count][i]))
            model.addConstr((z[count][i] == 0) >> (g[count][i] <= 0))
        count += 1

    model.setObjective(g[count - 1][0], GRB.MAXIMIZE)

    with TimingContext(logger, "Gurobi optimization"):
        model.optimize()

    logger.debug("Optimization completed",
                status=model.status,
                obj_value=model.objVal if model.status == GRB.OPTIMAL else None)

    if model.status == GRB.TIME_LIMIT:  # Time limit reached
        logger.warning("Linear relaxation hit time limit")
        return None, None

    if model.status != GRB.OPTIMAL:
        logger.warning("Linear relaxation not optimal", status=model.status)
        return None, None

    # Get solution directly from model variables
    x_vals = [x[i].x for i in range(in_size)]
    z_vals_list = []
    for layer_idx in range(len(layer_dims)):
        layer_z = [z[layer_idx][i].x for i in range(layer_dims[layer_idx])]
        z_vals_list.append(layer_z)

    logger.debug("Linear relaxation solved successfully",
                objective_value=model.objVal,
                input_size=len(x_vals))

    return x_vals, z_vals_list


def solve_relaxation_with_restrictions(network: Network, activation_pattern: List[List[int]],
                                     restrictions: List[List[int]]) -> Tuple[Optional[List[float]], Optional[List[List[float]]]]:
    """
    Solve linear relaxation with neuron activation restrictions.

    Args:
        network: Neural network instance
        activation_pattern: Current binary activation pattern
        restrictions: List of [layer_idx, neuron_idx] restrictions

    Returns:
        Tuple of (input_values, fractional_activations) or (None, None) if infeasible
    """
    w = network.get_weight_matrix()
    b = network.get_bias_matrix()
    layer_dims = network.layer_dims
    in_size = network.in_size

    model = gb.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 100)

    x = model.addVars(in_size, ub=1, lb=0, name="input")
    z = {}
    h = {}
    g = {}

    count = 0
    for dim in layer_dims:
        z[count] = model.addVars(dim, vtype=GRB.CONTINUOUS, ub=1, lb=0, name="neuron_layer_" + str(count))
        h[count] = model.addVars(dim)
        g[count] = model.addVars(dim, lb=-GRB.INFINITY)
        count += 1

    # Add activation pattern constraints
    for layer_idx, layer_acts in enumerate(activation_pattern):
        for neuron_idx, activation in enumerate(layer_acts):
            model.addConstr(z[layer_idx][neuron_idx] == activation)

    # Add restriction constraints
    for layer_idx, neuron_idx in restrictions:
        model.addConstr(z[layer_idx][neuron_idx] == 1 - activation_pattern[layer_idx][neuron_idx])

    # First layer constraints
    for i in range(layer_dims[0]):
        model.addConstr(gb.quicksum(w[0][i][j] * x[j] for j in range(in_size)) + b[0][i] == g[0][i])
        model.addConstr((z[0][i] == 0) >> (h[0][i] == 0))
        model.addConstr((z[0][i] == 1) >> (h[0][i] == g[0][i]))

    # Subsequent layer constraints
    count = 1
    for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
        for i in range(output_dim):
            model.addConstr(
                gb.quicksum(w[count][i][j] * h[count - 1][j] for j in range(input_dim)) + b[count][i] == g[count][i])
            model.addConstr((z[count][i] == 0) >> (h[count][i] == 0))
            model.addConstr((z[count][i] == 1) >> (h[count][i] == g[count][i]))
        count += 1

    model.setObjective(g[count - 1][0], GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.INFEASIBLE or model.status == GRB.TIME_LIMIT:
        return None, None

    if model.status != GRB.OPTIMAL:
        return None, None

    x_vals = gbdict2lst(x, in_size)
    z_vals = gbdict2lst(z, layer_dims)
    return x_vals, z_vals