import numpy as np
import random
import torch
import torch.nn as nn
from torch import relu, sigmoid, tanh, selu


class Network(nn.Module):

    def __init__(self, in_size, layer_dims, seed=42):
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

    def forward(self, x):
        # x - input tensor
        x = x.type(torch.FloatTensor)
        # convert into a float tensor
        x = x.reshape(-1, self.in_size)
        # turn the tensor into one tensor containing a bunch of inner tensors, each of dimension self.in_size

        for index, linear in enumerate(self.linears):
            # go through the network
            if index == len(self.linears) - 1:
                # we're on the last output
                if self.layer_dims[-1] == 1:
                    # binary output
                    # x = sigmoid(linear(x))
                    x = linear(x)
                else:
                    # multi-class output
                    x = linear(x)
                    # no activation
            # otherwise we use relu
            else:
                # not on the last activation
                x = relu(linear(x))
                # relu activation
            # now to keep track of the neurons
            self.neurons[f'Hidden Layer {index + 1} Neurons:'] = x

        return x
        # return the output

    def get_weight_matrix(self):
        w = {}
        for i in range(len(self.layer_dims)):
            w[i] = self.state_dict()['linears.' + str(i) + '.weight']
        return w

    def get_bias_matrix(self):
        b = {}
        for i in range(len(self.layer_dims)):
            b[i] = self.state_dict()['linears.' + str(i) + '.bias']
        return b