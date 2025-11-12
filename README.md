# MILP Neural Network Optimization: Pruned vs Dense Model Comparison

## Overview
This repository implements an algorithm to solve Mixed Integer Linear Programming (MILP) problems formulated for neural network optimization using Gurobi. The key approach is:
1. Solve the MILP on a **pruned model** (faster/easier optimization)
2. Evaluate the solution on the **dense model** (original network)
3. Compare solution quality and computational cost

## Algorithm
```
Input: Pruned neural network + Dense neural network
1. Formulate MILP on pruned model using Gurobi
2. Solve MILP to find optimal input x that maximizes pruned_model(x)
3. Evaluate x on dense model: original_max = dense_model(x)
4. Record: time_limit, max_ (pruned), original_max (dense), model_size, prune_amount, seed
```

## Repository Structure
```
RelaxWalk/
├── Network.py                  # Neural network with pruning support
├── util.py                     # Gurobi MILP formulation functions
├── io_csv.py                   # CSV output utilities
├── milp_solver.py              # Core: solve() function
├── run_single_experiment.py    # Run one experiment
├── run_all_experiments.py      # Run all experiments
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── archive/                    # Old walking algorithm code
└── results/                    # Experiment results (CSV files)
```

## Running Experiments

### Single Experiment
```bash
python run_single_experiment.py
```
Edit the `__main__` section to configure parameters.

### All Experiments
```bash
python run_all_experiments.py
```
This runs the full experiment grid:
- Input sizes: [100, 1000]
- Hidden layers: [3, 4, 5]
- Layer sizes: [100, 500]
- Pruning amounts: [0.0, 0.3, 0.5, 0.8]
- Seeds: [50, 51, 52, 53, 54]
- Time limit: 600 seconds

Results are saved to `results/MILP_comparison.csv`

## Output Format
CSV columns match the previous format for easy comparison:
```
method, model_size, parameters, seed, prune_amount, max_, first_max,
time_count, start_count, valid_start_count, original_max, original_max_time_elapsed
```

Key metrics:
- `max_`: Objective value on pruned model from MILP
- `original_max`: Evaluation of solution on dense model (KEY METRIC)
- `time_count`: Actual solve time

## Dependencies
- Python 3.x
- PyTorch
- Gurobi (with valid license)
- NumPy
- Pandas

Install with:
```bash
pip install -r requirements.txt
```

## Previous Work
The original "Relax-and-Walk" algorithm code has been archived in the `archive/` folder.

**Original Paper**: Optimization Over Trained Neural Networks Taking a Relaxing Walk
**Authors**: Jiatai Tong, Junyang Cai, Thiago Serra
**Link**: https://arxiv.org/abs/2401.03451

## Contact
For questions about the code, please contact the repository maintainer.
