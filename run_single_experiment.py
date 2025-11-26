"""
Run a single MILP experiment
"""

import time
from Network import Network
from milp_solver import solve
from io_csv import store_data


def run_experiment(input_size, layer_num, layer_size, seed, prune_amount, time_limit, output_file):
    """
    Run a single experiment with given configuration.

    Args:
        input_size: Input dimension
        layer_num: Number of hidden layers
        layer_size: Size of each hidden layer
        seed: Random seed
        prune_amount: Pruning percentage [0.0, 1.0]
        time_limit: Time limit for Gurobi (seconds)
        output_file: CSV file path to store results

    Returns:
        dict: Experiment results
    """

    # Create layer dimensions list: [layer_size, layer_size, ..., 1]
    # layer_num hidden layers of layer_size, then output layer of size 1
    layer_dims = [layer_size] * layer_num + [1]

    # Initialize Network with start time tracking
    start_time = time.time()
    network = Network(
        in_size=input_size,
        layer_dims=layer_dims,
        seed=seed,
        prune_amount=prune_amount,
        start_time=start_time
    )

    # Solve MILP
    results = solve(network, time_limit, seed)

    # Store results to CSV
    store_data(results, output_file)

    return results


if __name__ == "__main__":
    # Example usage
    result = run_experiment(
        input_size=1000,
        layer_num=2,
        layer_size=1000,
        seed=51,
        prune_amount=0.5,
        time_limit=600,
        output_file="results/test.csv"
    )
    print(f"Result: {result}")
