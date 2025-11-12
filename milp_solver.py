"""
MILP Solver for Pruned Model with Dense Evaluation
"""

import torch
import time
from util import get_linear_relaxation, gbdict2lst
from Network import Network


def solve(network, time_limit, seed):
    """
    Solve MILP on pruned model and evaluate solution on dense model.

    Args:
        network: Network object with both pruned and dense models
        time_limit: Time limit for Gurobi solver (seconds)
        seed: Random seed

    Returns:
        dict: Results with keys: method, model_size, parameters, seed, prune_amount,
              max_, first_max, time_count, start_count, valid_start_count,
              original_max, original_max_time_elapsed
    """

    # TODO: Implement
    pass
