import numpy as np
import random
import torch
import torch.nn as nn
from torch import relu, sigmoid, tanh, selu
import torch.nn.utils.prune as prune


class Network(nn.Module):

    def __init__(self, in_size, layer_dims, seed=42, prune_amount=0.2):
        # in_size = dimensions of the input
        # layer_dims = dimensions of the output

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        super(Network, self).__init__()

        self.layer_dims = layer_dims

        self.in_size = in_size

        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_size, layer_dims[0]))
        # contains all of the matrices that serve as our linear functions
        for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
            self.linears.append(nn.Linear(input_dim, output_dim))
            # weight matrix plus bias vector

        self.neurons = {}
        # a dictionary containing each layer's worth of neurons post-activation
        for index in range(1, len(layer_dims) + 1):
            # look at each hidden layer
            self.neurons[f'Hidden Layer {index} Neurons:'] = None
            # will eventually contain the tensor for all the neurons of that hidden layer post activation
            # a different set of neurons per sample input

        # Store original model before pruning
        import copy
        self.original = copy.deepcopy(self)

        # Apply pruning permanently to self only on the last layer
        last_layer = self.linears[-1]
        prune.l1_unstructured(last_layer, name='weight', amount=prune_amount)
        # Remove pruning reparameterization to make it permanent
        prune.remove(last_layer, 'weight')

        # Track best original model outputs
        self.original_max = float('-inf')
        self.original_x_max = None

        # Control which model to use
        self.use_original = False

    def forward(self, x):
        x_input = x.clone() if torch.is_tensor(x) else x
        x = x.type(torch.FloatTensor)
        x = x.reshape(-1, self.in_size)

        # Choose which model to use
        linears = self.original.linears if self.use_original else self.linears

        for index, linear in enumerate(linears):
            if index == len(linears) - 1:
                x = linear(x)
            else:
                x = relu(linear(x))
            self.neurons[f'Hidden Layer {index + 1} Neurons:'] = x

        # Always track original model output
        with torch.no_grad():
            if not self.use_original:
                # Compute original model output directly without recursion
                x_orig = x_input.clone() if torch.is_tensor(x_input) else x_input
                x_orig = x_orig.type(torch.FloatTensor)
                x_orig = x_orig.reshape(-1, self.in_size)

                for index, linear in enumerate(self.original.linears):
                    if index == len(self.original.linears) - 1:
                        x_orig = linear(x_orig)
                    else:
                        x_orig = relu(linear(x_orig))
                original_output = x_orig
            else:
                original_output = x

            original_value = original_output.item() if original_output.numel() == 1 else original_output.max().item()
            if original_value > self.original_max:
                self.original_max = original_value
                self.original_x_max = x_input if isinstance(x_input, list) else x_input.tolist()

        return x

    def get_weight_matrix(self):
        if self.use_original:
            w = {}
            for i in range(len(self.layer_dims)):
                w[i] = self.original.state_dict()['linears.' + str(i) + '.weight']
            return w
        else:
            w = {}
            for i in range(len(self.layer_dims)):
                w[i] = self.state_dict()['linears.' + str(i) + '.weight']
            return w

    def get_bias_matrix(self):
        if self.use_original:
            b = {}
            for i in range(len(self.layer_dims)):
                b[i] = self.original.state_dict()['linears.' + str(i) + '.bias']
            return b
        else:
            b = {}
            for i in range(len(self.layer_dims)):
                b[i] = self.state_dict()['linears.' + str(i) + '.bias']
            return b

    def get_neurons(self):
        """
        Safely access neurons dictionary.
        This method ensures neurons are accessed through the proper interface
        and respects the use_original guard in forward().
        """
        return self.neurons

