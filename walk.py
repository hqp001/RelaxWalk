from util import (
    get_binary_activations,
    solve_lp_pre_calc,
    update_x,
)
import time

def single_walk_with_timelimit(model_nn, x, eps, timelimit):
    # print(x)
    start = time.time()
    max_ = -1000
    x_max = None
    step_count = 0
    ap = get_binary_activations(model_nn, x)
    max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
    while max_lp is not None and max_lp > max_:
        # print(max_lp)
        step_count += 1
        max_ = max_lp
        x_max = x_new
        now = time.time()
        if now - start >= timelimit:
            break
        x = update_x(model_nn, x, x_new, eps)
        ap = get_binary_activations(model_nn, x)
        max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
    time_consuming = time.time() - start
    return x_max, max_, step_count, time_consuming
