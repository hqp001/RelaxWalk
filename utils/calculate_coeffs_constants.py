# REFACTORED: Extracted from util.py - calculates coefficients and constants for neural network layers
import numpy as np

def calculate_coeffs_constants(weight, bias, ap):
    """
    Calculate coefficients and constants for each layer based on activation patterns.

    Args:
        weight: Dictionary of weight matrices for each layer
        bias: Dictionary of bias vectors for each layer
        ap: Activation pattern for each layer

    Returns:
        pre_coeffs: Pre-activation coefficients for each layer
        pre_constants: Pre-activation constants for each layer
    """
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