"""
Run a single MILP experiment based on SLURM_ARRAY_TASK_ID
Each task writes to its own CSV file to avoid conflicts.
Configuration is read from environment variables set by run_slurm_experiments.sh
"""

import sys
import os
from run_single_experiment import run_experiment


# Read config from environment variables - NO DEFAULTS, raise error if missing
try:
    seed_list = [int(x) for x in os.environ['SEED_LIST'].split(',')]
    input_size_list = [int(x) for x in os.environ['INPUT_SIZE_LIST'].split(',')]
    layer_num_list = [int(x) for x in os.environ['LAYER_NUM_LIST'].split(',')]
    layer_size_list = [int(x) for x in os.environ['LAYER_SIZE_LIST'].split(',')]
    prune_amount_list = [float(x) for x in os.environ['PRUNE_AMOUNT_LIST'].split(',')]
    time_limit = int(os.environ['TIME_LIMIT'])
except KeyError as e:
    print(f"ERROR: Missing required environment variable: {e}")
    print("Required variables: SEED_LIST, INPUT_SIZE_LIST, LAYER_NUM_LIST, LAYER_SIZE_LIST, PRUNE_AMOUNT_LIST, TIME_LIMIT")
    sys.exit(1)

# Base output directory
output_dir = "results"


def get_experiment_config(task_id):
    """
    Map SLURM_ARRAY_TASK_ID to specific experiment configuration.

    Total experiments = 3 input_sizes × 1 layer_num × 1 layer_size × 1 seed × 3 prune_amounts = 9

    Args:
        task_id: SLURM_ARRAY_TASK_ID (0-8)

    Returns:
        tuple: (input_size, layer_num, layer_size, seed, prune_amount)
    """

    configs = []
    for input_size in input_size_list:
        for layer_num in layer_num_list:
            for layer_size in layer_size_list:
                for seed in seed_list:
                    for prune_amount in prune_amount_list:
                        configs.append((input_size, layer_num, layer_size, seed, prune_amount))

    if task_id >= len(configs):
        raise ValueError(f"Task ID {task_id} exceeds number of configurations {len(configs)}")

    return configs[task_id]


def main():
    """
    Run experiment based on SLURM_ARRAY_TASK_ID.
    Each task writes to a separate CSV file: results/MILP_comparison_task_{task_id}.csv
    """

    if len(sys.argv) < 2:
        print("Usage: python run_slurm_experiment.py <SLURM_ARRAY_TASK_ID>")
        sys.exit(1)

    task_id = int(sys.argv[1])

    # Each task writes to its own file to avoid conflicts
    output_file = os.path.join(output_dir, f"MILP_comparison_task_{task_id}.csv")

    # Get configuration for this task
    input_size, layer_num, layer_size, seed, prune_amount = get_experiment_config(task_id)

    print("=" * 80)
    print(f"SLURM Task ID: {task_id}")
    print(f"Configuration: input={input_size}, layers={layer_num}x{layer_size}, "
          f"seed={seed}, prune={prune_amount}")
    print(f"Output file: {output_file}")
    print("=" * 80)

    try:
        result = run_experiment(
            input_size=input_size,
            layer_num=layer_num,
            layer_size=layer_size,
            seed=seed,
            prune_amount=prune_amount,
            time_limit=time_limit,
            output_file=output_file
        )

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
