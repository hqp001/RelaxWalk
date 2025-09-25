# REFACTORED: Using utils module for imports
from utils import get_binary_activations, get_linear_relaxation_with_restriction


def find_new_relaxation_point_random(model_nn, x, pick, dif_index):
    ap = get_binary_activations(model_nn, x)

    new_x, new_frac_z = get_linear_relaxation_with_restriction(model_nn, ap, [dif_index[pick]])
    i = 1
    while new_x is None and i <= len(dif_index):
        new_x, new_frac_z = get_linear_relaxation_with_restriction(model_nn, ap, dif_index[i - 1:i])
        i += 1
    return new_x, new_frac_z
