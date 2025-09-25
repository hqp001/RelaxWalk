# REFACTORED: Extracted from util.py - solves linear relaxation with time limit using Gurobi
import gurobipy as gb
from gurobipy import GRB
from .gbdict2lst_z import gbdict2lst_z
from configs.log_config import get_logger

logger = get_logger(__name__)

# KEPT: Global variables for Gurobi callback (unavoidable with Gurobi's API)
x_vals = []
z_vals = []

def get_linear_relaxation(modelnn, time_limit):
    """
    Solve linear relaxation of neural network optimization with time limit.

    Args:
        modelnn: Neural network model
        time_limit: Maximum time for optimization

    Returns:
        x_vals: Input values from relaxation solution
        z_vals: Activation values from relaxation solution
    """
    logger.info(f"Starting linear relaxation with time_limit={time_limit}")

    try:
        w = modelnn.get_weight_matrix()
        b = modelnn.get_bias_matrix()
        layer_dims = modelnn.layer_dims
        in_size = modelnn.in_size
        logger.info(f"Network params: in_size={in_size}, layer_dims={layer_dims}")
    except Exception as e:
        logger.error(f"Failed to get network parameters: {e}")
        raise

    try:
        model = gb.Model()
        model.setParam('OutputFlag', 0)  # Move this here to avoid Gurobi output
        logger.debug("Created Gurobi model")
    except Exception as e:
        logger.error(f"Failed to create Gurobi model: {e}")
        raise

    global x_vals
    global z_vals
    x_vals = []
    z_vals = []

    def get_relaxation(model, where):
        if where == GRB.Callback.MIPNODE:
            global x_vals
            global z_vals
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
    # model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
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

    model.optimize(get_relaxation)

    # print(model.status)

    if model.status == 9:
        return None, None

    z_vals = gbdict2lst_z(z_vals, layer_dims)
    return x_vals, z_vals