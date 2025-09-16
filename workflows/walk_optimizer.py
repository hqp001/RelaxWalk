import time
import numpy as np
import torch
import gurobipy as gb
from gurobipy import GRB
from data_types import Network
from typing import List, Tuple, Optional
from .activation_processor import get_binary_activations
from logger_config import get_logger, TimingContext, timer


def update_position(network: Network, old_x: List[float], target_x: List[float], eps: float) -> List[float]:
    """
    Update position by taking a step toward target.

    Args:
        network: Neural network instance
        old_x: Current position
        target_x: Target position
        eps: Step size

    Returns:
        New position after taking step
    """
    in_size = network.in_size
    x_new = []
    for i in range(in_size):
        if target_x[i] != old_x[i]:
            step = (target_x[i] - old_x[i]) * eps
            new_val = old_x[i] + step
            if 0 <= new_val <= 1:
                x_new.append(new_val)
            else:
                x_new.append(old_x[i])
        else:
            x_new.append(old_x[i])
    return x_new


def calculate_coeffs_constants(weights, biases, activation_pattern):
    """Calculate coefficients and constants for LP formulation."""
    num_layers = len(weights)
    coeffs = []
    constants = []

    for layer_idx in range(num_layers):
        layer_coeffs = []
        layer_constants = []

        if layer_idx == 0:
            # First layer: directly connected to input
            for neuron_idx in range(len(weights[layer_idx])):
                neuron_coeffs = weights[layer_idx][neuron_idx][:]
                neuron_constant = biases[layer_idx][neuron_idx]
                layer_coeffs.append(neuron_coeffs)
                layer_constants.append(neuron_constant)
        else:
            # Subsequent layers: need to propagate through previous layers
            input_size = len(weights[0][0])
            for neuron_idx in range(len(weights[layer_idx])):
                neuron_coeffs = [0.0] * input_size
                neuron_constant = biases[layer_idx][neuron_idx]

                for prev_neuron_idx, weight in enumerate(weights[layer_idx][neuron_idx]):
                    if activation_pattern[layer_idx - 1][prev_neuron_idx] == 1:
                        # Add contribution from active previous layer neuron
                        for input_idx in range(input_size):
                            neuron_coeffs[input_idx] += weight * coeffs[layer_idx - 1][prev_neuron_idx][input_idx]
                        neuron_constant += weight * constants[layer_idx - 1][prev_neuron_idx]

                layer_coeffs.append(neuron_coeffs)
                layer_constants.append(neuron_constant)

        coeffs.append(layer_coeffs)
        constants.append(layer_constants)

    return coeffs, constants


def solve_lp_for_activation_pattern(network: Network, activation_pattern: List[List[int]]) -> Tuple[float, List[float]]:
    """
    Solve LP for given activation pattern using pre-calculated coefficients.

    Args:
        network: Neural network instance
        activation_pattern: Binary activation pattern

    Returns:
        Tuple of (objective_value, optimal_input)
    """
    weights = network.get_weight_matrix()
    biases = network.get_bias_matrix()
    layer_dims = network.layer_dims
    in_size = network.in_size

    coeffs, constants = calculate_coeffs_constants(weights, biases, activation_pattern)

    model = gb.Model()
    model.setParam('OutputFlag', 0)
    x = model.addVars(in_size, ub=1, lb=0, name="input")

    # Add constraints for each layer
    for layer_idx, layer_dim in enumerate(layer_dims[:-1]):
        for neuron_idx in range(layer_dim):
            linear_expr = gb.quicksum(coeffs[layer_idx][neuron_idx][j] * x[j] for j in range(in_size)) + constants[layer_idx][neuron_idx]

            if activation_pattern[layer_idx][neuron_idx] == 1:
                model.addConstr(linear_expr >= 0)
            else:
                model.addConstr(linear_expr <= 0)

    # Set objective (output layer)
    final_layer_idx = len(weights) - 1
    objective = gb.quicksum(coeffs[final_layer_idx][0][j] * x[j] for j in range(in_size)) + constants[final_layer_idx][0]
    model.setObjective(objective, GRB.MAXIMIZE)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_vals = [x[i].x for i in range(in_size)]
        return model.objVal, x_vals
    else:
        return -float('inf'), []


def gradient_walk(network: Network, start_point: List[float], eps: float, time_limit: float) -> Tuple[Optional[List[float]], float, int, float]:
    """
    Perform gradient-based walk from starting point.

    Args:
        network: Neural network instance
        start_point: Starting input point
        eps: Step size for walking
        time_limit: Maximum time for walking

    Returns:
        Tuple of (best_input, best_objective, step_count, time_elapsed)
    """
    logger = get_logger("walk_optimizer")
    start_time = time.time()
    best_obj = -float('inf')
    best_x = None
    step_count = 0
    current_x = start_point[:]

    logger.debug("Starting gradient walk", eps=eps, time_limit=time_limit)

    # Get initial activation pattern and solve LP
    with TimingContext(logger, "initial LP solve"):
        activation_pattern = get_binary_activations(network, current_x)
        obj_val, target_x = solve_lp_for_activation_pattern(network, activation_pattern)

    logger.debug("Initial LP solved", initial_objective=obj_val)

    while obj_val > best_obj:
        step_count += 1
        best_obj = obj_val
        best_x = target_x[:]

        # Check time limit
        if time.time() - start_time >= time_limit:
            logger.debug("Time limit reached in gradient walk", steps_completed=step_count)
            break

        # Take step toward target
        current_x = update_position(network, current_x, target_x, eps)

        # Get new activation pattern and solve LP
        activation_pattern = get_binary_activations(network, current_x)
        obj_val, target_x = solve_lp_for_activation_pattern(network, activation_pattern)

        # Log progress every 50 steps or on improvements
        if step_count % 50 == 0:
            logger.debug("Walk progress",
                        step=step_count,
                        current_obj=obj_val,
                        best_obj=best_obj,
                        time_elapsed=f"{time.time() - start_time:.2f}s")

    time_elapsed = time.time() - start_time
    logger.debug("Gradient walk completed",
                steps=step_count,
                final_objective=best_obj,
                total_time=f"{time_elapsed:.3f}s")
    return best_x, best_obj, step_count, time_elapsed


def random_flip_walk(network: Network, start_point: List[float], time_limit: float) -> Tuple[Optional[List[float]], float, int, float]:
    """
    Perform random activation flipping walk from starting point.

    Args:
        network: Neural network instance
        start_point: Starting input point
        time_limit: Maximum time for walking

    Returns:
        Tuple of (best_input, best_objective, step_count, time_elapsed)
    """
    start_time = time.time()
    best_obj = -float('inf')
    best_x = None
    step_count = 0

    # Get initial activation pattern
    activation_pattern = get_binary_activations(network, start_point)
    obj_val, target_x = solve_lp_for_activation_pattern(network, activation_pattern)

    while obj_val > best_obj:
        step_count += 1
        best_obj = obj_val
        best_x = target_x[:]

        # Check time limit
        if time.time() - start_time >= time_limit:
            break

        # Randomly flip some activations
        activation_pattern = random_flip_activations(activation_pattern)
        obj_val, target_x = solve_lp_for_activation_pattern(network, activation_pattern)

    time_elapsed = time.time() - start_time
    return best_x, best_obj, step_count, time_elapsed


def random_flip_activations(activation_pattern: List[List[int]]) -> List[List[int]]:
    """
    Randomly flip some activations in the pattern.

    Args:
        activation_pattern: Current activation pattern

    Returns:
        Modified activation pattern
    """
    new_pattern = [layer[:] for layer in activation_pattern]  # Deep copy

    for layer_idx, layer in enumerate(new_pattern):
        for neuron_idx in range(len(layer)):
            # Small probability of flipping each neuron
            if np.random.random() < 0.1:
                new_pattern[layer_idx][neuron_idx] = 1 - new_pattern[layer_idx][neuron_idx]

    return new_pattern