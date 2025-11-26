import torch
import torch.fx
import torch.nn as nn
import numpy as np
import gurobipy as gp
import time

def add_predictor_constr(gurobi_model, sparse_model, dense_model, x, y):

    gurobi_model._binary = []
    gurobi_model._input_vars = x
    gurobi_model._sparse_network = sparse_model
    gurobi_model._dense_network = dense_model
    gurobi_model._original_max = float('-inf')
    gurobi_model._original_x_max = None
    gurobi_model._original_max_time = None
    gurobi_model._start_time = time.time()

    if isinstance(sparse_model, nn.Sequential):
        sparse_model = torch.fx.symbolic_trace(sparse_model)

    gurobi_nodes = {}

    sparse_model.graph.print_tabular()

    # Select which ReLU constraint function to use
    relu_constr_func = add_relu_constr

    for node in sparse_model.graph.nodes:

        if node.op == 'placeholder':

            gurobi_nodes[node.name] = x

        elif node.op == 'call_module':

            module_called = getattr(sparse_model, node.target)

            if isinstance(module_called, torch.nn.Linear):

                assert len(node.args) == 1

                input_node = gurobi_nodes[node.args[0].name]

                output_node = add_linear_constr(gurobi_model, input_node, module_called, name=node.name)

                gurobi_nodes[node.name] = output_node

            elif isinstance(module_called, torch.nn.ReLU):

                assert len(node.args) == 1

                input_node = gurobi_nodes[node.args[0].name]

                output_node = relu_constr_func(gurobi_model, input_node, name=node.name)

                gurobi_nodes[node.name] = output_node

            elif isinstance(module_called, torch.nn.Conv2d):

                assert len(node.args) == 1

                input_node = gurobi_nodes[node.args[0].name]

                output_node = add_conv2d_constr(gurobi_model, input_node, module_called, name=node.name)

                gurobi_nodes[node.name] = output_node

            elif isinstance(module_called, torch.nn.Flatten):

                assert len(node.args) == 1

                input_node = gurobi_nodes[node.args[0].name]

                output_node = add_flatten_constr(gurobi_model, input_node, module_called, name=node.name)

                gurobi_nodes[node.name] = output_node

            elif isinstance(module_called, onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation):

                assert(len(node.args) == 2)
                assert module_called.math_op_function == torch.add

                lhs = gurobi_nodes[node.args[0].name]
                rhs = gurobi_nodes[node.args[1].name]

                assert(lhs.shape == rhs.shape)

                output_shape = lhs.shape

                output_node = gurobi_model.addMVar(output_shape, lb=-gp.GRB.INFINITY, name=node.name)

                gurobi_model.addConstr(output_node == lhs + rhs)

                gurobi_nodes[node.name] = output_node

            elif isinstance(module_called, onnx2torch.node_converters.reshape.OnnxReshape):

                assert(len(node.args) == 2)

                tensor_to_reshape = gurobi_nodes[node.args[0].name]
                shape = gurobi_nodes[node.args[1].name]

                gurobi_nodes[node.name] = tensor_to_reshape.reshape(shape)


            else:

                raise NotImplementedError(f"Not supported module {type(module_called)}")

        elif node.op == 'call_function':

            function_called = getattr(sparse_model, node.target)

            raise NotImplementedError(f"Not supported, pls implement by yourself")

        elif node.op == 'get_attr':

            target = node.target
            target_atoms = target.split('.')
            attr_itr = sparse_model
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)

            assert(isinstance(attr_itr, torch.Tensor))

            output_layer = attr_itr.detach().numpy()

            gurobi_nodes[node.name] = output_layer

        elif node.op == "output":

            assert len(node.args) == 1

            input_node = gurobi_nodes[node.args[0].name]

            if y.shape != input_node.shape:
                raise ValueError(f"Wrong shape for output array. Expected {input_node.shape}, got {y.shape}")

            output_shape = y.shape

            y = y.reshape(-1)
            input_node = input_node.reshape(-1)

            for i in range(y.size):
                gurobi_model.addConstr(y[i] == input_node[i])

            y = y.reshape(output_shape)
            input_node = input_node.reshape(output_shape)

            gurobi_nodes[node.name] = y

        else:
            raise ValueError(f"Unexpected node opcode for {node.op}")

    return y

def add_conv2d_constr(gurobi_model, input_layer, conv_layer, name = "conv2d"):

    # Assert batch size is 1
    assert input_layer.shape[0] == 1
    assert conv_layer.padding_mode == "zeros"

    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation

    weight = conv_layer.weight.detach().numpy()
    bias = conv_layer.bias.detach().numpy()

    dummy_output = conv_layer(torch.ones(input_layer.shape))

    in_size = (input_layer.shape[2], input_layer.shape[3])
    out_size = (dummy_output.shape[2], dummy_output.shape[3])

    unfolded_tensor = torch.nn.Unfold(kernel_size = conv_layer.kernel_size, padding = conv_layer.padding, stride = conv_layer.stride, dilation = conv_layer.dilation)

    def index_tensor_to_MVar(index_tensor, gurobi_var, zero_var):

        var_list = []

        gurobi_var = gurobi_var.tolist()

        for i in range(index_tensor.size):
            if index_tensor[i] == 0:
                var_list.append(zero_var)
            else:
                var_list.append(gurobi_var[index_tensor[i] - 1])

        return var_list

    def create_var_image(image, unfolded_tensor, zero_var):
        feature_map = []

        index_tensor = torch.arange(1, in_size[0] * in_size[1] + 1, dtype=torch.float32)

        unf = unfolded_tensor(index_tensor.view(1, in_size[0], in_size[1])).transpose(0, 1).int().numpy()

        for i in range(unf.shape[0]):

            gurobi_patch = index_tensor_to_MVar(unf[i], image.reshape(-1), zero_var)

            feature_map.append(gurobi_patch)

        return gp.MVar.fromlist(feature_map)

    zero_var = gurobi_model.addVar(lb=0, ub=0, name="zero")

    output_shape = (1, out_channels, out_size[0], out_size[1])

    output_layer = gurobi_model.addMVar(output_shape, lb=-gp.GRB.INFINITY, name=name)

    output_image_shape = (1, out_channels, out_size[0] * out_size[1])
    output_images = gurobi_model.addMVar(output_image_shape, lb=0, ub=0, name="zero_MVar")
    output_images = output_images + (np.ones(output_image_shape) * bias.reshape(1, out_channels, 1))

    for in_channel in range(in_channels):

        unfolded_image = create_var_image(input_layer[0][in_channel], unfolded_tensor, zero_var)

        for out_channel in range(out_channels):

            kernel = weight[out_channel][in_channel].reshape(-1)
            feature_image = unfolded_image @ kernel
            output_images[0][out_channel] += feature_image

    output_layer = output_layer.reshape(output_image_shape)
    gurobi_model.addConstr(output_layer == output_images)

    output_layer = output_layer.reshape(output_shape)

    return output_layer

def add_linear_constr(gurobi_model, input_layer, linear_layer, name="linear"):

    assert input_layer.shape[0] == 1
    assert len(input_layer.shape) == 2

    input_size = linear_layer.in_features
    output_size = linear_layer.out_features

    weight = linear_layer.weight.detach().numpy()
    bias = linear_layer.bias.detach().numpy()

    assert(input_size == input_layer.size and output_size == bias.size)

    output_layer = gurobi_model.addMVar((1, output_size), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=name)

    gurobi_model.addConstr(output_layer == ( input_layer @ weight.T ) + bias)

    return output_layer

def add_relu_constr2(gurobi_model, input_layer, name):

    assert input_layer.shape[0] == 1

    input_shape = input_layer.shape
    output_layer = gurobi_model.addMVar(input_shape, name=name)

    input_layer = input_layer.reshape(-1)
    output_layer = output_layer.reshape(-1)

    for i in range(input_layer.size):
       gurobi_model.addConstr(output_layer[i] == gp.max_([input_layer[i]], constant=0.0))

    input_layer = input_layer.reshape(input_shape)
    output_layer = output_layer.reshape(input_shape)

    return output_layer

def add_relu_constr(gurobi_model, input_layer, name):

    input_size = input_layer.shape

    neuron_layer = gurobi_model.addMVar(input_size, vtype=gp.GRB.BINARY, name = f"neuron_{name}")

    gurobi_model._binary.append(neuron_layer)

    output_layer = gurobi_model.addMVar(input_size, name=name)
    for j in range(input_size[1]):
        gurobi_model.addConstr( (neuron_layer[0][j] == 0) >> (input_layer[0][j] <= 0) )
        gurobi_model.addConstr( (neuron_layer[0][j] == 0) >> (output_layer[0][j] == 0) )
        gurobi_model.addConstr( (neuron_layer[0][j] == 1) >> (output_layer[0][j] == input_layer[0][j]) )
        gurobi_model.addConstr( output_layer[0][j] >= 0 )


    return output_layer

def add_relu_constr3(gurobi_model, input_layer, name):
    """
    ReLU constraints using 3 indicator constraints like util.py.

    Variables:
    - input_layer (g): input to ReLU
    - output_layer (h): output of ReLU
    - neuron_layer (z): binary variables

    Constraints (from util.py lines 107-109):
    - (z == 0) >> (h == 0)
    - (z == 1) >> (h == g)
    - (z == 0) >> (g <= 0)
    """
    # Assert input shape
    assert input_layer.shape[0] == 1

    input_shape = input_layer.shape

    # z: binary variables
    neuron_layer = gurobi_model.addMVar(input_shape, vtype=gp.GRB.BINARY, name=f"neuron_{name}")
    gurobi_model._binary.append(neuron_layer)

    # h: output layer (ReLU output) - same shape as input
    output_layer = gurobi_model.addMVar(input_shape, name=name)

    # Assert output has same shape as input
    assert output_layer.shape == input_layer.shape

    for j in range(input_shape[1]):
        # g is input_layer[0][j], h is output_layer[0][j], z is neuron_layer[0][j]
        # Three indicator constraints from util.py:
        gurobi_model.addConstr((neuron_layer[0][j] == 0) >> (output_layer[0][j] == 0))  # (z == 0) >> (h == 0)
        gurobi_model.addConstr((neuron_layer[0][j] == 1) >> (output_layer[0][j] == input_layer[0][j]))  # (z == 1) >> (h == g)
        gurobi_model.addConstr((neuron_layer[0][j] == 0) >> (input_layer[0][j] <= 0))  # (z == 0) >> (g <= 0)

    return output_layer

def add_relu_constr_skip(gurobi_model, input_layer, name):
    """
    Skip ReLU constraints - just pass through the input without any transformation.
    This effectively treats the ReLU layer as an identity function.

    Variables:
    - input_layer: input to ReLU
    - Returns the input_layer directly without adding any constraints
    """
    return input_layer

def add_flatten_constr(gurobi_model, input_layer, module, name):

    start_dim = module.start_dim
    end_dim = module.end_dim

    if end_dim < 0:
        end_dim = input_layer.ndim + end_dim

    flatten_shape = 1
    for dim in range(start_dim, end_dim + 1):
        flatten_shape *= input_layer.shape[dim]

    new_shape = input_layer.shape[:start_dim] + (flatten_shape,) + input_layer.shape[end_dim + 1:]

    return input_layer.reshape(new_shape)

# Assert if gurobi_model formulate correctly
def check_correct_formulation(gurobi_model, nn_model, x, y):

    gurobi_model.update()

    # Keep original model for evaluation (Sequential models work fine for forward pass)
    eval_model = nn_model

    input_shape = x.shape

    lower_bound = np.nan_to_num(x.lb.astype(np.float32), copy=True, nan=0.0, posinf=10, neginf=-10)
    upper_bound = np.nan_to_num(x.ub.astype(np.float32), copy=True, nan=0.0, posinf=10, neginf=10)
    lower_bound_tensor = torch.tensor(lower_bound, dtype=torch.float32)
    upper_bound_tensor = torch.tensor(upper_bound, dtype=torch.float32)

    random_input = torch.rand(*input_shape, dtype=torch.float32)
    random_input = lower_bound_tensor + (upper_bound_tensor - lower_bound_tensor) * random_input

    input_constr = gurobi_model.addConstr(x == random_input.numpy())

    current_objective = (gurobi_model.getObjective(), gurobi_model.ModelSense)

    gurobi_model.setObjective(y[0][0], gp.GRB.MAXIMIZE)
    gurobi_model.optimize()

    if gurobi_model.status != gp.GRB.OPTIMAL or gurobi_model.SolCount != 1:
        return False

    eval_model.eval()
    with torch.no_grad():
        output = eval_model(random_input).detach().numpy()

    gurobi_output = y.X

    print(gurobi_output)
    print(output)

    gurobi_model.remove(input_constr)
    gurobi_model.setObjective(current_objective[0], current_objective[1])
    gurobi_model.update()

    error = np.max(np.abs(output - gurobi_output))

    print("Diff: ", error)

    return error < 1e-5


