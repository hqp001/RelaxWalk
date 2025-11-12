"""
Simple test for the first half of relaxation_walk_deep algorithm.
Tests: initial relaxation, binary activations, and first walk.
"""

from util import get_linear_relaxation, get_binary_activations, solve_lp_pre_calc, update_x
from Network import Network
import torch
import time

# Simple test parameters
input_size = 10
layer_num = 2
layer_size = 50
random_seed = 42
walk_eps = 0.001
prune_amount = 0.8
timelimit = 30

print("Creating network...")
layer_dims = layer_num * [layer_size] + [1]
model_nn = Network(in_size=input_size, layer_dims=layer_dims, seed=random_seed, prune_amount=prune_amount)
model_nn.use_original = False
print(f"Network: input={input_size}, layers={layer_dims}, prune={prune_amount}. Use prune model")
print()
print("TEST 1: get_linear_relaxation with ORIGINAL model")
x, frac_z = get_linear_relaxation(model_nn, timelimit)
print("x", x)
print("frac_z", frac_z)

print("TEST 2: get_binary_activations")
int_z = get_binary_activations(model_nn, x)
print("int_z", int_z)
print("TEST 3: First walk (original model)")
start = time.time()
max_ = -1000
x_max = None
eps = walk_eps

ap = get_binary_activations(model_nn, x)
max_lp, x_new = solve_lp_pre_calc(model_nn, int_z)
print("ap", ap)
print("x_new", x_new)

print(f"Initial LP solution: {max_lp}")

step_count = 0
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

print(f"SUCCESS: Walk completed in {step_count} steps")
print(f"Final max: {max_:.6f}")
print()
