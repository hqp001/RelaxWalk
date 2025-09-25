# REFACTORED: Extracted from util.py - updates input x values with epsilon step
def update_x(modelnn, old_x, current_x, eps):
    """
    Update x values by taking an epsilon step from old_x towards current_x.

    Args:
        modelnn: Neural network model (used for input size)
        old_x: Previous x values
        current_x: Target x values
        eps: Step size parameter

    Returns:
        x_new: Updated x values after epsilon step
    """
    in_size = modelnn.in_size
    x_new = []
    for i in range(in_size):
        if current_x[i] != old_x[i]:
            tmp = (current_x[i] - old_x[i]) * eps
            if current_x[i] + tmp >= 0 and current_x[i] + tmp <= 1:
                x_new.append(current_x[i] + tmp)
            else:
                x_new.append(current_x[i])
        else:
            x_new.append(current_x[i])
    return x_new