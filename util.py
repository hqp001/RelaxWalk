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


def get_linear_relaxation_with_restriction(modelnn, activation_pattern, restriction):
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

    # Update original_max by passing x through the network
    if x_vals is not None and len(x_vals) > 0:
        modelnn.forward(torch.Tensor(list(x_vals.values())))

    return x_vals, z_vals


def update_x(modelnn, old_x, current_x, eps):
    in_size = modelnn.in_size
    x_new = []
    for i in range(in_size):
        if current_x[i] != old_x[i]:
            tmp = (current_x[i] - old_x[i]) * eps
            if current_x[i] + tmp >= 0 and current_x[i] + tmp <= 1:
                x_new.append(current_x[i] + tmp)
            else:
                x_new.append(current_x[i])
        else:
            x_new.append(current_x[i])
    return x_new

def calculate_coeffs_constants(weight, bias, ap):
    num_layers = len(weight)
    # print(num_layers)

    # Initialize dictionaries for coefficients, constants, pre-activate coefficients and constants.
    coeffs = {}
    constants = {}
    pre_coeffs = {}
    pre_constants = {}

    coeffs[-1] = np.identity(weight[0].shape[1])
    constants[-1] = np.zeros(weight[0].shape[1])
    pre_coeffs[-1] = coeffs[-1].copy()
    pre_constants[-1] = constants[-1].copy()

    for l in range(num_layers):
        layer_size = len(ap[l])
        prev_layer_size = coeffs[l - 1].shape[1]

        coeffs[l] = np.zeros((layer_size, prev_layer_size))
        constants[l] = np.zeros(layer_size)
        pre_coeffs[l] = np.zeros((layer_size, prev_layer_size))
        pre_constants[l] = np.zeros(layer_size)

        for j in range(layer_size):
            # for i in range(prev_layer_size):
            #     pre_coeffs[l][j, i] += np.dot(weight[l][j, :], coeffs[l - 1][:, i])
            pre_coeffs[l][j, :] = weight[l][j, :] @ coeffs[l - 1]
            pre_constants[l][j] += np.dot(weight[l][j, :], constants[l - 1]) + bias[l][j]

            if ap[l][j] == 0:
                continue

            coeffs[l][j] = pre_coeffs[l][j]
            constants[l][j] = pre_constants[l][j]

    return pre_coeffs, pre_constants


def solve_lp_pre_calc(modelnn, activation_pattern):
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

    # Check if optimization was successful
    if model.status != GRB.OPTIMAL:
        # Return None if optimization failed (infeasible, unbounded, etc.)
        return None, None

    max_lp = model.getAttr('ObjVal')
    x_new = gbdict2lst(x, len(x))

    # Update original_max by passing x through the network
    if x_new is not None and len(x_new) > 0:
        modelnn.forward(torch.Tensor(x_new))

    return max_lp, x_new

def get_binary_activations(model_nn, x):
    if isinstance(x, list):
        x = torch.Tensor(x)
    elif isinstance(x, dict):
        x = torch.Tensor(list(x.values()))
    else:
        x = torch.Tensor(x)
    model_nn.forward(x)
    binary_activations = {key: (value > 0).float() for key, value in model_nn.neurons.items()}
    activation_list = []
    for key, tensor in binary_activations.items():
        activation_list.append(tensor.numpy().tolist()[0])
    return activation_list


def get_prob_list_with_bias(int_z, frac_z, bias):
    prob_list = []
    for zi, zf in zip(int_z, frac_z):
        diff = abs(zi - zf)
        if len(prob_list) == 0:
            prob_list.append(diff + bias)
        else:
            prob_list.append(diff + prob_list[-1] + bias)
    return prob_list


def get_index_from_prob_list(prob_list, num):
    for i in range(len(prob_list)):
        if num <= prob_list[i]:
            return i


