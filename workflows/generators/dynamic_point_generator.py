import numpy as np
import time
from typing import List, Tuple, Optional, Generator
from data_types import Network
from workflows.activation_processor import (
    get_binary_activations,
    compute_activation_differences,
    round_activations
)
from workflows.relaxation_solver import solve_relaxation_with_restrictions


def generate_dynamic_starting_points(
    network: Network,
    base_point: List[float],
    base_frac_activations: List[List[float]],
    time_limit: float,
    start_time: float
) -> Generator[Tuple[List[float], List[List[float]]], None, None]:
    """
    Generate starting points dynamically by random activation flipping.

    Args:
        network: Neural network instance
        base_point: Base input point from relaxation
        base_frac_activations: Fractional activations from relaxation
        time_limit: Total time limit
        start_time: Algorithm start time

    Yields:
        Tuples of (starting_point, fractional_activations)
    """
    current_point = base_point[:]
    current_frac_acts = [layer[:] for layer in base_frac_activations]

    # Track visited activation patterns
    visited_patterns = set()

    while time.time() - start_time < time_limit:
        # Get binary activations for current point
        binary_acts = get_binary_activations(network, current_point)
        pattern_tuple = _activation_pattern_to_tuple(binary_acts)

        # Skip if we've seen this pattern before
        if pattern_tuple in visited_patterns:
            # Try to find a new point
            new_point, new_frac_acts = _find_new_random_point(
                network, current_point, current_frac_acts
            )
            if new_point is not None:
                current_point = new_point
                current_frac_acts = new_frac_acts
                continue
            else:
                break

        visited_patterns.add(pattern_tuple)
        yield current_point[:], [layer[:] for layer in current_frac_acts]

        # Generate next point by random flipping
        new_point, new_frac_acts = _find_new_random_point(
            network, current_point, current_frac_acts
        )

        if new_point is not None:
            current_point = new_point
            current_frac_acts = new_frac_acts
        else:
            # No more feasible points found
            break


def _find_new_random_point(
    network: Network,
    current_point: List[float],
    current_frac_acts: List[List[float]]
) -> Tuple[Optional[List[float]], Optional[List[List[float]]]]:
    """
    Find a new relaxation point by randomly flipping differing activations.

    Args:
        network: Neural network instance
        current_point: Current input point
        current_frac_acts: Current fractional activations

    Returns:
        Tuple of (new_point, new_fractional_activations) or (None, None)
    """
    # Get binary activations
    binary_acts = get_binary_activations(network, current_point)

    # Round fractional activations
    rounded_frac_acts = round_activations(current_frac_acts)

    # Find differences between binary and rounded fractional
    diff_indices = compute_activation_differences(binary_acts, rounded_frac_acts)

    if not diff_indices:
        return None, None

    # Randomly pick a differing activation to flip
    max_attempts = min(len(diff_indices), 10)  # Limit attempts to avoid infinite loops

    for attempt in range(max_attempts):
        pick_idx = np.random.randint(len(diff_indices))
        diff_constraint = [list(diff_indices[pick_idx])]

        # Try to find new relaxation point with this constraint
        new_point, new_frac_acts = solve_relaxation_with_restrictions(
            network, binary_acts, diff_constraint
        )

        if new_point is not None:
            return new_point, new_frac_acts

        # If failed, try with a sequence of constraints
        for i in range(1, min(len(diff_indices), 5)):
            constraint_sequence = [list(diff_indices[j]) for j in range(i)]
            new_point, new_frac_acts = solve_relaxation_with_restrictions(
                network, binary_acts, constraint_sequence
            )

            if new_point is not None:
                return new_point, new_frac_acts

    return None, None


def find_random_flip_point(
    network: Network,
    base_point: List[float],
    pick_index: int,
    diff_indices: List[Tuple[int, int]]
) -> Tuple[Optional[List[float]], Optional[List[List[float]]]]:
    """
    Find new relaxation point by flipping a specific differing activation.

    Args:
        network: Neural network instance
        base_point: Base input point
        pick_index: Index of difference to use for flipping
        diff_indices: List of (layer_idx, neuron_idx) differences

    Returns:
        Tuple of (new_point, new_fractional_activations) or (None, None)
    """
    if pick_index >= len(diff_indices):
        return None, None

    # Get current binary activations
    binary_acts = get_binary_activations(network, base_point)

    # Create constraint to flip the selected activation
    constraint = [list(diff_indices[pick_index])]

    # Try to solve with this constraint
    new_point, new_frac_acts = solve_relaxation_with_restrictions(
        network, binary_acts, constraint
    )

    if new_point is not None:
        return new_point, new_frac_acts

    # If failed, try progressive constraints
    for i in range(1, min(len(diff_indices), len(diff_indices))):
        constraint_sequence = [list(diff_indices[j]) for j in range(i - 1, i)]
        new_point, new_frac_acts = solve_relaxation_with_restrictions(
            network, binary_acts, constraint_sequence
        )

        if new_point is not None:
            return new_point, new_frac_acts

    return None, None


def _activation_pattern_to_tuple(activation_pattern: List[List[int]]) -> tuple:
    """
    Convert activation pattern to hashable tuple for set operations.

    Args:
        activation_pattern: Activation pattern to convert

    Returns:
        Tuple representation of activation pattern
    """
    return tuple(tuple(layer) for layer in activation_pattern)


def validate_difference_indices(
    diff_indices: List[Tuple[int, int]],
    layer_dims: List[int]
) -> bool:
    """
    Validate that difference indices are within valid bounds.

    Args:
        diff_indices: List of (layer_idx, neuron_idx) differences
        layer_dims: Dimensions of each layer

    Returns:
        True if all indices are valid
    """
    for layer_idx, neuron_idx in diff_indices:
        if layer_idx >= len(layer_dims) or neuron_idx >= layer_dims[layer_idx]:
            return False
        if layer_idx < 0 or neuron_idx < 0:
            return False
    return True


def compute_point_statistics(
    points: List[List[float]]
) -> dict:
    """
    Compute statistics about generated points.

    Args:
        points: List of generated points

    Returns:
        Dictionary with statistics
    """
    if not points:
        return {"count": 0, "mean_distance": 0, "diversity": 0}

    # Compute pairwise distances
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            distances.append(dist)

    mean_distance = np.mean(distances) if distances else 0
    diversity = len(set(tuple(p) for p in points)) / len(points)

    return {
        "count": len(points),
        "mean_distance": mean_distance,
        "diversity": diversity,
        "min_distance": np.min(distances) if distances else 0,
        "max_distance": np.max(distances) if distances else 0
    }