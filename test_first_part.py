from util import (
    get_binary_activations,
    get_linear_relaxation,
    get_linear_relaxation_with_restriction,
    get_prob_list_with_bias,
    get_index_from_prob_list,
    solve_lp_pre_calc,
    update_x
)
from Network import Network
import numpy as np
import time

# Setup
input_size = 10
layer_num = 2
layer_size = 50
seed = 42
prune_amount = 0.5
bias = 0.05
timelimit = 60
eps = 0.001

layer_dims = layer_num * [layer_size] + [1]

model_nn = Network(in_size=input_size, layer_dims=layer_dims, seed=seed, prune_amount=prune_amount)
model_nn.use_original = False

start = time.time()
max_ = -1000
x_max = None

np.random.seed(seed)

# Use original model for initial solution search
x, frac_z = get_linear_relaxation(model_nn, timelimit)

int_z = get_binary_activations(model_nn, x)
prob_list = []
for i in range(layer_num):
    prob_list_layer = get_prob_list_with_bias(int_z[i], frac_z[i], bias)
    prob_list.append(prob_list_layer)

# First walk for relaxation - use original model
step_count = 0
ap = get_binary_activations(model_nn, x)
max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
while max_lp is not None and max_lp > max_:
    step_count += 1
    max_ = max_lp
    x_max = x_new
    if time.time() - start > timelimit:
        break
    x = update_x(model_nn, x, x_new, eps)
    ap = get_binary_activations(model_nn, x)
    max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
    if max_lp is None:
        break

first_max = max_
record_ap = [int_z]
ap = int_z

print(f"First max: {first_max}")
print(f"Steps in first walk: {step_count}")
print("\nTest complete!")
