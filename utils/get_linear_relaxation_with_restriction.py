# REFACTORED: Extracted from util.py - solves linear relaxation with neuron restrictions using Gurobi
import gurobipy as gb
from gurobipy import GRB
import sys
from .gbdict2lst_z import gbdict2lst_z

# KEPT: Global variables for Gurobi callback (unavoidable with Gurobi's API)
x_vals = []
z_vals = []

def get_linear_relaxation_with_restriction(modelnn, activation_pattern, restriction):
    """
    Solve linear relaxation with restrictions on specific neurons.

    Args:
        modelnn: Neural network model
        activation_pattern: Current activation pattern to base restrictions on
        restriction: List of [layer, neuron] pairs to restrict

    Returns:
        x_vals: Input values from restricted relaxation solution
        z_vals: Activation values from restricted relaxation solution
    """
    w = modelnn.get_weight_matrix()
    b = modelnn.get_bias_matrix()
    layer_dims = modelnn.layer_dims
    in_size = modelnn.in_size

    model = gb.Model()
    global x_vals
    global z_vals
    x_vals = []
    z_vals = []

    def get_relaxation(model, where):
        if where == GRB.Callback.MIPNODE:
            # print('---- In Callback ----')
            global x_vals
            global z_vals
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            # print(f'Call back status: {status}')
            if status == GRB.INTEGER:
                # print('integer')
                x_vals = model.cbGetSolution(x)
                z_vals = []
                for i in range(len(layer_dims)):
                    z_vals.append(model.cbGetNodeRel(z[i]))
                z_vals = model.cbGetSolution(z)
                # print('--- Callback ends ---')
                model.terminate()
            if status == GRB.OPTIMAL:
                # print('optimal')
                x_vals = model.cbGetNodeRel(x)
                z_vals = []
                for i in range(len(layer_dims)):
                    # print(model.cbGetNodeRel(z[i]))
                    z_vals.append(model.cbGetNodeRel(z[i]))
                # print('--- Callback ends ---')
                model.terminate()

    model.setParam('OutputFlag', 0)
    x = model.addVars(in_size, ub=1, lb=0, name="input")
    z = {}
    h = {}
    g = {}
    hbar = {}

    count = 0
    for dim in layer_dims:
        z[count] = model.addVars(dim, vtype=GRB.BINARY, name="neuron_layer_" + str(count))
        h[count] = model.addVars(dim)
        hbar[count] = model.addVars(dim)
        g[count] = model.addVars(dim, lb=-GRB.INFINITY)
        count += 1

    for i in range(layer_dims[0]):
        model.addConstr(gb.quicksum(w[0][i][j] * x[j] for j in range(in_size)) + b[0][i] == g[0][i])
        model.addConstr((z[0][i] == 0) >> (h[0][i] == 0))
        model.addConstr((z[0][i] == 1) >> (h[0][i] == g[0][i]))
        model.addConstr((z[0][i] == 0) >> (g[0][i] <= 0))
        if [0, i] in restriction:
            # print(f'set restriction [0, {i}]')
            model.addConstr(z[0][i] == (1 - activation_pattern[0][i]))

    count = 1
    for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
        for i in range(output_dim):
            model.addConstr(
                gb.quicksum(w[count][i][j] * h[count - 1][j] for j in range(input_dim)) + b[count][i] == g[count][i])
            model.addConstr((z[count][i] == 0) >> (h[count][i] == 0))
            model.addConstr((z[count][i] == 1) >> (h[count][i] == g[count][i]))
            model.addConstr((z[count][i] == 0) >> (g[count][i] <= 0))
            if [count, i] in restriction:
                # print(f'set restriction [{count}, {i}]')
                model.addConstr(z[count][i] == (1 - activation_pattern[count][i]))
        count += 1

    model.setObjective(g[count - 1][0], GRB.MAXIMIZE)

    model.optimize(get_relaxation)

    status = model.status
    # print(status)
    # sys.stdout.flush()
    if status == GRB.Status.INF_OR_UNBD or status == GRB.Status.INFEASIBLE or len(z_vals) == 0:
        # print('model is infeasible')
        return None, None
    # print(z_vals)
    sys.stdout.flush()

    z_vals = gbdict2lst_z(z_vals, layer_dims)
    return x_vals, z_vals