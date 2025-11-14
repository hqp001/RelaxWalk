import gurobipy as gp
from gurobi_ml.torch import add_sequential_constr
import torch.nn as nn

# --- Step 1. Define a tiny network (2 hidden ReLU neurons) ---
net = nn.Sequential(
    nn.Linear(2, 2, bias=False),
    nn.ReLU(),
    nn.Linear(2, 1, bias=False)
)

# --- Step 2. Build optimization model ---
m = gp.Model()
x = m.addMVar(shape=2, lb=-gp.GRB.INFINITY, name="x")
y = m.addMVar(shape=1, lb=-gp.GRB.INFINITY, name="y")

# Add the neural-net constraints
nn_constr = add_sequential_constr(m, net, x, y, formulation="MILP")

# Example objective: maximize output
m.setObjective(y[0], gp.GRB.MAXIMIZE)
m.optimize()

# --- Step 3. Inspect binaries per layer ---
print("\n=== Layer-wise variable inspection ===")
for li, layer in enumerate(nn_constr.layers):
    print(f"\nLayer {li}: {layer.__class__.__name__}")
    for v in layer.vars:
        print(f"  {v.VarName:20s}  type={v.VType}  value={v.X}")

# --- Step 4. Collect all binary vars in sequential order ---
relu_bins = [v for layer in nn_constr.layers for v in layer.vars if v.VType == gp.GRB.BINARY]
print("\n=== Activation binaries (sequential order) ===")
for i, v in enumerate(relu_bins):
    print(f"Binary #{i}: {v.VarName}  value={v.X}")

