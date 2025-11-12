import gurobipy as gb
from gurobipy import GRB
import numpy as np
from Network import Network
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch import relu, sigmoid, tanh, selu

import time

import matplotlib.pyplot as plt
import sys

def gbdict2lst(dic, layer_dims):
    lst = []
    count = 0
    if isinstance(layer_dims, list) == False:
        for i in range(layer_dims):
            # if dic[i].getAttr('X') is None:
            #     return None
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
    # print(layer_dims)
    # print(dic)
    lst = []
    count = 0
    if isinstance(layer_dims, list) == False:
        for i in range(layer_dims):
            # if dic[i].getAttr('X') is None:
            #     return None
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


#########################################################
def get_linear_relaxation(modelnn, time_limit):
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

    # Check if callback populated z_vals
    if not z_vals or len(z_vals) == 0:
        return None, None

    z_vals = gbdict2lst_z(z_vals, layer_dims)

    # Update original_max by passing x through the network
    if x_vals is not None and len(x_vals) > 0:
        modelnn.forward(torch.Tensor(list(x_vals.values())))

    return x_vals, z_vals
#########################################################


# Walking algorithm helper functions removed - not needed for MILP comparison


