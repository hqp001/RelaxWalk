"""
Run a single MILP experiment
"""

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

    # TODO: Implement
    # 1. Create Network with given architecture
    # 2. Call solve() from milp_solver
    # 3. Store results using store_data()
    # 4. Return results

    pass


if __name__ == "__main__":
    # Example usage
    result = run_experiment(
        input_size=100,
        layer_num=3,
        layer_size=100,
        seed=50,
        prune_amount=0.5,
        time_limit=600,
        output_file="results/test.csv"
    )
    print(f"Result: {result}")
