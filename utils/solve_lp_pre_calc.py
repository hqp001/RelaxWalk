# REFACTORED: Extracted from util.py - solves LP with pre-calculated coefficients using Gurobi
import gurobipy as gb
from gurobipy import GRB
from .calculate_coeffs_constants import calculate_coeffs_constants
from .gbdict2lst import gbdict2lst

def solve_lp_pre_calc(modelnn, activation_pattern):
    """
    Solve linear program with pre-calculated coefficients for given activation pattern.

    Args:
        modelnn: Neural network model
        activation_pattern: Binary activation pattern for each layer

    Returns:
        max_lp: Maximum objective value found
        x_new: Input values that achieve maximum
    """
    w = modelnn.get_weight_matrix()
    b = modelnn.get_bias_matrix()
    layer_dims = modelnn.layer_dims
    in_size = modelnn.in_size

    coeffs, constants = calculate_coeffs_constants(w, b, activation_pattern)

    model = gb.Model()
    model.setParam('OutputFlag', 0)
    x = model.addVars(in_size, ub=1, lb=0, name="input")
    g = {}
    count = 0
    for dim in layer_dims:
        g[count] = model.addVars(dim, lb=-GRB.INFINITY)
        count += 1

    # print(len(coeffs), len(coeffs[len(w) - 1]))

    constraints_count = 0

    count = 0

    for output_dim in layer_dims[:-1]:
        # if count == len(w) - 1:
        #     print("here")
        #     break
        # print(output_dim, count)
        for i in range(output_dim):
            model.addConstr(
                g[count][i] == gb.quicksum(coeffs[count][i][j] * x[j] for j in range(in_size)) + constants[count][
                    i])
            if activation_pattern[count][i] == 1:
                # model.addConstr(
                #     gb.quicksum(coeffs[count][i][j] * x[j] for j in range(in_size)) + constants[count][i] >= 0)
                model.addConstr(g[count][i] >= 0)
            else:
                # model.addConstr(
                #     gb.quicksum(coeffs[count][i][j] * x[j] for j in range(in_size)) + constants[count][i] <= 0)
                model.addConstr(g[count][i] <= 0)
            constraints_count += 1
        count += 1

    # print("constraints count: ", constraints_count)

    model.setObjective(
        gb.quicksum(coeffs[len(w) - 1][0][j] * x[j] for j in range(in_size)) + constants[len(w) - 1][0],
        GRB.MAXIMIZE)
    # model.setParam('Method', 1)
    model.optimize()

    # print("status: ", model.status)
    # model.computeIIS()
    # model.write("model.ilp")

    max_lp = model.getAttr('ObjVal')
    x_new = gbdict2lst(x, len(x))
    return max_lp, x_new