"""
Run a single MILP experiment with custom architectures based on SLURM_ARRAY_TASK_ID
Each task writes to its own CSV file to avoid conflicts.
Configuration is read from environment variables set by run_slurm_custom_architectures.sh
"""

import sys
import os
import time
from Network import Network
from milp_solver import solve
from io_csv import store_data


# Define custom architectures
# Format: (architecture_name, input_size, layer_dims)
ARCHITECTURES = [
    ("100k_10k_1k_100_10_1", 100000, [10000, 1000, 100, 10, 1]),
    ("100k_1k_1k_100_10_1", 100000, [1000, 1000, 100, 10, 1]),
    ("10k_1k_100_10_1", 10000, [1000, 100, 10, 1]),
    ("10k_1k_100_100_1", 10000, [1000, 100, 100, 1]),
    ("1k_100_10_1", 1000, [100, 10, 1]),
    ("100_10_1", 100, [10, 1]),
    ("100_100_100_10k_1", 100, [100, 100, 10000, 1]),
    ("784_64_64_10_1", 784, [64, 64, 10, 1]),
    ("784_64_64_2_1", 784, [64, 64, 2, 1]),
    ("784_32_32_2_1", 784, [32, 32, 2, 1]),
]

# Experiment configuration - all parameters defined here
seed_list = [50, 51]  # Two seeds per architecture
prune_amount_list = [0.0, 0.3, 0.5, 0.8, 0.9, 0.95]  # 6 pruning rates
time_limit = 600  # 10 minutes per experiment (seconds)

# Base output directory
output_dir = "results"


def get_experiment_config(task_id):
    """
    Map SLURM_ARRAY_TASK_ID to specific experiment configuration.

    Total experiments = num_architectures × num_seeds × num_prune_amounts

    Args:
        task_id: SLURM_ARRAY_TASK_ID

    Returns:
        tuple: (arch_name, input_size, layer_dims, seed, prune_amount)
    """

    configs = []
    for arch_name, input_size, layer_dims in ARCHITECTURES:
        for seed in seed_list:
            for prune_amount in prune_amount_list:
                configs.append((arch_name, input_size, layer_dims, seed, prune_amount))

    if task_id >= len(configs):
        raise ValueError(f"Task ID {task_id} exceeds number of configurations {len(configs)}")

    return configs[task_id]


def main():
    """
    Run experiment based on SLURM_ARRAY_TASK_ID.
    Each task writes to a separate CSV file: results/MILP_custom_arch_task_{task_id}.csv
    """

    if len(sys.argv) < 2:
        print("Usage: python run_slurm_custom_architectures.py <SLURM_ARRAY_TASK_ID>")
        sys.exit(1)

    task_id = int(sys.argv[1])

    # Each task writes to its own file to avoid conflicts
    output_file = os.path.join(output_dir, f"MILP_custom_arch_task_{task_id}.csv")

    # Get configuration for this task
    arch_name, input_size, layer_dims, seed, prune_amount = get_experiment_config(task_id)

    print("=" * 80)
    print(f"SLURM Task ID: {task_id}")
    print(f"Architecture: {arch_name}")
    print(f"  Network: {input_size} -> {' -> '.join(map(str, layer_dims))}")
    print(f"  Seed: {seed}, Prune: {prune_amount}")
    print(f"Output file: {output_file}")
    print("=" * 80)

    try:
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
        result = solve(network, time_limit, seed)

        # Add architecture info to results
        result['architecture'] = arch_name
        # Set layer_num and layer_size to 0 since we use custom layer_dims now
        result['layer_num'] = 0
        result['layer_size'] = 0
        # Log the SLURM array task ID
        result['slurm_array_id'] = task_id

        # Store results to CSV
        store_data(result, output_file)

        print(f"\nResults:")
        print(f"  max_: {result['max_']:.4f}")
        print(f"  original_max: {result['original_max']:.4f}")
        print(f"  solve_time: {result['solve_time']:.2f}s")
        print("=" * 80)
        print("SUCCESS")

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        print("FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
