# REFACTORED: Extracted from util.py - gets binary activation patterns from neural network
import torch

def get_binary_activations(model_nn, x):
    """
    Get binary activation patterns from a neural network given input x.

    Args:
        model_nn: Neural network model instance
        x: Input values (list, dict, or tensor)

    Returns:
        activation_list: List of binary activation patterns for each layer
    """
    if isinstance(x, list):
        x = torch.Tensor(x)
    elif isinstance(x, dict):
        x = torch.Tensor(list(x.values()))
    else:
        x = torch.Tensor(x)
    model_nn.forward(x)
    binary_activations = {key: (value > 0).float() for key, value in model_nn.neurons.items()}
    activation_list = []
    for key, tensor in binary_activations.items():
        activation_list.append(tensor.numpy().tolist()[0])
    return activation_list