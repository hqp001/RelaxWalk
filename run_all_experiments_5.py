"""
Run all MILP comparison experiments
"""

from run_single_experiment import run_experiment


# Experiment configuration (matching old testscript_exp.py)
# seed_list = [50]
# input_size_list = [1000, 10000, 100000]
# layer_num_list = [2, 3]
# layer_size_list = [100, 1000, 10000]
# prune_amount_list = [0.0, 0.5, 0.9, 0.95, 0.99, 0.995]
# time_limit = 600

seed_list = [51]
input_size_list = [100000]
layer_num_list = [2]
layer_size_list = [100, 1000, 10000]
prune_amount_list = [0.0, 0.5, 0.9, 0.95, 0.99, 0.995]
time_limit = 600

output_file = "results/MILP_comparison_5.csv"


def main():
    """
    Run all experiment configurations.
    """

    total_experiments = (len(input_size_list) * len(layer_num_list) *
                        len(layer_size_list) * len(seed_list) * len(prune_amount_list))
    current = 0

    for input_size in input_size_list:
        for layer_num in layer_num_list:
            for layer_size in layer_size_list:
                for seed in seed_list:
                    for prune_amount in prune_amount_list:
                        current += 1
                        try:
                            print(f"[{current}/{total_experiments}] Running: input={input_size}, "
                                  f"layers={layer_num}x{layer_size}, seed={seed}, prune={prune_amount}")

                            result = run_experiment(
                                input_size=input_size,
                                layer_num=layer_num,
                                layer_size=layer_size,
                                seed=seed,
                                prune_amount=prune_amount,
                                time_limit=time_limit,
                                output_file=output_file
                            )

                            print(f"  max_={result['max_']:.4f}, original_max={result['original_max']:.4f}, "
                                  f"time={result['solve_time']:.2f}s")
                            print("-" * 80)

                        except Exception as e:
                            print(f"ERROR in experiment: {type(e).__name__}: {str(e)}")
                            print("Skipping and continuing...")
                            print("-" * 80)


if __name__ == "__main__":
    main()
