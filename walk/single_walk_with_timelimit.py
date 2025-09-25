# REFACTORED: Extracted from walk.py - performs single optimization walk with time limit
import time
import torch
from utils import get_binary_activations, solve_lp_pre_calc, update_x
from configs.log_config import get_logger

logger = get_logger(__name__)

def single_walk_with_timelimit(model_nn, x, eps, timelimit, original_model=None):
    """
    Perform a single optimization walk with time limit.

    Args:
        model_nn: Neural network model
        x: Starting input values
        eps: Step size parameter
        timelimit: Maximum time for the walk

    Returns:
        x_max: Best input found
        max_: Best objective value found
        step_count: Number of steps taken
        time_consuming: Actual time consumed
    """
    start = time.time()
    max_ = -1000
    x_max = None
    step_count = 0

    logger.debug(f"Starting single walk with timelimit={timelimit:.3f}s, eps={eps}")

    try:
        ap = get_binary_activations(model_nn, x)
        max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
        if original_model is not None:
            original_model(torch.FloatTensor(x_new))
        logger.debug(f"Initial LP solve: max_lp={max_lp:.6f}")
    except Exception as e:
        logger.error(f"Failed initial LP solve in single walk: {e}")
        return None, max_, step_count, time.time() - start

    while max_lp > max_:
        step_count += 1
        max_ = max_lp
        x_max = x_new
        elapsed_time = time.time() - start
        remaining_time = timelimit - elapsed_time

        logger.debug(f"Single walk step {step_count}: max_lp={max_lp:.6f}, elapsed={elapsed_time:.3f}s, remaining={remaining_time:.3f}s")

        if elapsed_time >= timelimit:
            logger.debug(f"Time limit reached in single walk at step {step_count}, elapsed={elapsed_time:.3f}s")
            break

        try:
            x = update_x(model_nn, x, x_new, eps)
            ap = get_binary_activations(model_nn, x)
            max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
            if original_model is not None:
                original_model(torch.FloatTensor(x_new))
            logger.debug(f"Single walk step {step_count}: updated x, new max_lp={max_lp:.6f}")
        except Exception as e:
            logger.error(f"Error in single walk step {step_count}: {e}")
            break

    time_consuming = time.time() - start
    logger.debug(f"Single walk completed: {step_count} steps, final_max={max_:.6f}, time_used={time_consuming:.3f}s")

    return x_max, max_, step_count, time_consuming