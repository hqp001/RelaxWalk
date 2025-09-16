import torch
import numpy as np
from data_types import Network
from typing import List, Tuple


def get_binary_activations(network: Network, x: List[float]) -> List[List[int]]:
    """
    Compute binary activation pattern for given input.

    Args:
        network: Neural network instance
        x: Input values

    Returns:
        List of binary activation patterns for each layer
    """
    if isinstance(x, list):
        x = torch.Tensor(x)
    elif isinstance(x, dict):
        x = torch.Tensor(list(x.values()))
    else:
        x = torch.Tensor(x)

    network.forward(x)
    binary_activations = {key: (value > 0).float() for key, value in network.neurons.items()}
    activation_list = []
    for key, tensor in binary_activations.items():
        activation_list.append(tensor.numpy().tolist()[0])
    return activation_list


def compute_probability_lists(binary_acts: List[List[int]], frac_acts: List[List[float]],
                            bias: float) -> List[List[float]]:
    """
    Compute probability lists for neuron selection with bias.

    Args:
        binary_acts: Binary activation patterns
        frac_acts: Fractional activation patterns
        bias: Bias term for probability computation

    Returns:
        List of cumulative probability lists for each layer
    """
    prob_lists = []
    for layer_binary, layer_frac in zip(binary_acts, frac_acts):
        prob_list = get_prob_list_with_bias(layer_binary, layer_frac, bias)
        prob_lists.append(prob_list)
    return prob_lists


def get_prob_list_with_bias(int_z: List[int], frac_z: List[float], bias: float) -> List[float]:
    """
    Create cumulative probability list for neuron selection.

    Args:
        int_z: Binary activations
        frac_z: Fractional activations
        bias: Bias term

    Returns:
        Cumulative probability list
    """
    prob_list = []
    for zi, zf in zip(int_z, frac_z):
        diff = abs(zi - zf)
        if len(prob_list) == 0:
            prob_list.append(diff + bias)
        else:
            prob_list.append(diff + prob_list[-1] + bias)
    return prob_list


def get_index_from_prob_list(prob_list: List[float], num: float) -> int:
    """
    Get neuron index from probability value.

    Args:
        prob_list: Cumulative probability list
        num: Random number for selection

    Returns:
        Selected neuron index
    """
    for i in range(len(prob_list)):
        if num <= prob_list[i]:
            return i
    return len(prob_list) - 1


def compute_activation_differences(binary_acts: List[List[int]],
                                 frac_acts: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Find indices where binary and fractional activations differ.

    Args:
        binary_acts: Binary activation patterns
        frac_acts: Fractional activation patterns (rounded)

    Returns:
        List of (layer_idx, neuron_idx) tuples where they differ
    """
    index_list = []
    for layer_idx, (layer_binary, layer_frac) in enumerate(zip(binary_acts, frac_acts)):
        for neuron_idx, (binary_val, frac_val) in enumerate(zip(layer_binary, layer_frac)):
            if abs(binary_val - frac_val) > 1e-7:
                index_list.append((layer_idx, neuron_idx))
    return index_list


def round_activations(frac_acts: List[List[float]]) -> List[List[int]]:
    """
    Round fractional activations to binary values.

    Args:
        frac_acts: Fractional activation patterns

    Returns:
        Rounded binary activation patterns
    """
    rounded_acts = []
    for layer_frac in frac_acts:
        rounded_acts.append(np.round(layer_frac).astype(int).tolist())
    return rounded_acts


def hamming_distance(acts1: List[List[int]], acts2: List[List[int]]) -> int:
    """
    Compute Hamming distance between two activation patterns.

    Args:
        acts1: First activation pattern
        acts2: Second activation pattern

    Returns:
        Hamming distance
    """
    distance = 0
    for layer1, layer2 in zip(acts1, acts2):
        for act1, act2 in zip(layer1, layer2):
            if abs(act1 - act2) > 1e-7:
                distance += 1
    return distance