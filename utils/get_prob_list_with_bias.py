# REFACTORED: Extracted from util.py - creates probability list with bias
def get_prob_list_with_bias(int_z, frac_z, bias):
    """
    Create cumulative probability list with bias for neuron selection.

    Args:
        int_z: Integer activation pattern
        frac_z: Fractional activation pattern from relaxation
        bias: Bias value to add to each probability

    Returns:
        prob_list: Cumulative probability list for random selection
    """
    prob_list = []
    for zi, zf in zip(int_z, frac_z):
        diff = abs(zi - zf)
        if len(prob_list) == 0:
            prob_list.append(diff + bias)
        else:
            prob_list.append(diff + prob_list[-1] + bias)
    return prob_list