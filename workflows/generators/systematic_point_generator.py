import numpy as np
import time
from typing import List, Tuple, Optional, Generator
from data_types import Network
from workflows.activation_processor import (
    get_binary_activations,
    compute_probability_lists,
    get_index_from_prob_list
)
from workflows.relaxation_solver import solve_relaxation_with_restrictions


def generate_systematic_starting_points(
    network: Network,
    base_activations: List[List[int]],
    frac_activations: List[List[float]],
    bias: float,
    time_limit: float,
    start_time: float
) -> Generator[List[float], None, None]:
    """
    Generate starting points systematically using probability-based neuron selection.

    Args:
        network: Neural network instance
        base_activations: Base binary activation pattern from relaxation
        frac_activations: Fractional activation pattern from relaxation
        bias: Bias term for probability computation
        time_limit: Total time limit
        start_time: Algorithm start time

    Yields:
        Starting points for walks
    """
    # Compute probability lists for each layer
    prob_lists = compute_probability_lists(base_activations, frac_activations, bias)

    # Track visited activation patterns to avoid duplicates
    visited_patterns = set()
    visited_patterns.add(_activation_pattern_to_tuple(base_activations))

    layer_count = len(base_activations)

    # Generate points by systematically adding restrictions layer by layer
    for layer_idx in range(min(2, layer_count)):  # Only first 2 layers as in original
        yield from _generate_points_for_layer(
            network, base_activations, prob_lists[layer_idx],
            layer_idx, visited_patterns, time_limit, start_time
        )

        if time.time() - start_time >= time_limit:
            break


def _generate_points_for_layer(
    network: Network,
    base_activations: List[List[int]],
    prob_list: List[float],
    layer_idx: int,
    visited_patterns: set,
    time_limit: float,
    start_time: float
) -> Generator[List[float], None, None]:
    """
    Generate points for a specific layer using probability-based neuron selection.

    Args:
        network: Neural network instance
        base_activations: Base activation pattern
        prob_list: Probability list for neuron selection
        layer_idx: Layer index to add restrictions
        visited_patterns: Set of visited activation patterns
        time_limit: Total time limit
        start_time: Algorithm start time

    Yields:
        Starting points for walks
    """
    restrictions = []
    picked_neurons = []
    layer_size = len(base_activations[layer_idx])

    while (len(restrictions) < layer_size and
           len(picked_neurons) < layer_size and
           time.time() - start_time < time_limit):

        # Probabilistically pick a neuron
        neuron_idx = _pick_neuron_probabilistically(prob_list, picked_neurons)
        if neuron_idx is None:
            break

        picked_neurons.append(neuron_idx)
        restrictions.append([layer_idx, neuron_idx])

        # Generate new point with current restrictions
        x_vals, _ = solve_relaxation_with_restrictions(network, base_activations, restrictions)

        if x_vals is None:
            # Infeasible, remove last restriction and continue
            restrictions.pop()
            continue

        # Check if this gives a new activation pattern
        new_activations = get_binary_activations(network, x_vals)
        pattern_tuple = _activation_pattern_to_tuple(new_activations)

        if pattern_tuple not in visited_patterns:
            visited_patterns.add(pattern_tuple)
            yield x_vals


def _pick_neuron_probabilistically(prob_list: List[float], picked_neurons: List[int]) -> Optional[int]:
    """
    Pick a neuron index probabilistically, avoiding already picked neurons.

    Args:
        prob_list: Cumulative probability list
        picked_neurons: List of already picked neuron indices

    Returns:
        Selected neuron index or None if all picked
    """
    if len(picked_neurons) >= len(prob_list):
        return None

    max_attempts = len(prob_list) * 2  # Avoid infinite loops
    attempts = 0

    while attempts < max_attempts:
        random_val = np.random.uniform(0, prob_list[-1])
        neuron_idx = get_index_from_prob_list(prob_list, random_val)

        if neuron_idx not in picked_neurons:
            return neuron_idx

        attempts += 1

    # Fallback: find first unpicked neuron
    for i in range(len(prob_list)):
        if i not in picked_neurons:
            return i

    return None


def _activation_pattern_to_tuple(activation_pattern: List[List[int]]) -> tuple:
    """
    Convert activation pattern to hashable tuple for set operations.

    Args:
        activation_pattern: Activation pattern to convert

    Returns:
        Tuple representation of activation pattern
    """
    return tuple(tuple(layer) for layer in activation_pattern)


def create_layer_restrictions(
    prob_list: List[float],
    picked_neurons: List[int],
    layer_idx: int
) -> List[List[int]]:
    """
    Create restriction list for a specific layer.

    Args:
        prob_list: Probability list for the layer
        picked_neurons: List of picked neuron indices
        layer_idx: Layer index

    Returns:
        List of restrictions in format [layer_idx, neuron_idx]
    """
    restrictions = []
    for neuron_idx in picked_neurons:
        if neuron_idx < len(prob_list):
            restrictions.append([layer_idx, neuron_idx])
    return restrictions


def validate_restrictions(
    restrictions: List[List[int]],
    layer_dims: List[int]
) -> bool:
    """
    Validate that restrictions are within valid bounds.

    Args:
        restrictions: List of [layer_idx, neuron_idx] restrictions
        layer_dims: Dimensions of each layer

    Returns:
        True if all restrictions are valid
    """
    for layer_idx, neuron_idx in restrictions:
        if layer_idx >= len(layer_dims) or neuron_idx >= layer_dims[layer_idx]:
            return False
        if layer_idx < 0 or neuron_idx < 0:
            return False
    return True