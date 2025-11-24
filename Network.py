import numpy as np
import random
import time
import copy
import torch
import torch.nn as nn
from torch import relu, sigmoid, tanh, selu
import torch.nn.utils.prune as prune


class Network(nn.Module):

    def __init__(self, in_size, layer_dims, seed=42, prune_amount=0.2, start_time=None):
        # in_size = dimensions of the input
        # layer_dims = dimensions of the output (including final output dimension)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Store start time for tracking updates
        self.start_time = start_time
        super(Network, self).__init__()
        self.layer_dims = layer_dims
        self.in_size = in_size
        self.prune_amount = prune_amount

        # Build dense network: input -> hidden layers -> output
        layers = []
        in_dim = in_size
        for dim in layer_dims[:-1]:  # All hidden layers with ReLU activation
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim
        # Output layer (last dimension in layer_dims, no activation)
        layers.append(nn.Linear(in_dim, layer_dims[-1]))

        self.dense = nn.Sequential(*layers)

        # Create independent copy for sparse model
        self.sparse = copy.deepcopy(self.dense)
        parameters_to_prune = [
            (m, "weight") for m in self.sparse if isinstance(m, torch.nn.Linear)
        ]

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.prune_amount,
        )

        for m, _ in parameters_to_prune:
            prune.remove(m, "weight")

        # Track best original model outputs
        self.original_max = float('-inf')
        self.original_x_max = None
        self.original_max_time = None  # Track when original_max was last updated

    def forward_sparse(self, x):
        with torch.no_grad():
            # Forward through sparse model
            sparse_output = self.sparse(x)

            # Also forward through dense model and record original_max
            dense_output = self.dense(x)
            dense_max = dense_output.max().item()

            if dense_max > self.original_max:
                self.original_max = dense_max
                self.original_x_max = x.clone()
                if self.start_time is not None:
                    self.original_max_time = time.time() - self.start_time

        return sparse_output

    def forward_dense(self, x):
        with torch.no_grad():
            # Forward through dense model
            dense_output = self.dense(x)

            # Record original_max
            dense_max = dense_output.max().item()

            if dense_max > self.original_max:
                self.original_max = dense_max
                self.original_x_max = x.clone()
                if self.start_time is not None:
                    self.original_max_time = time.time() - self.start_time

        return dense_output

