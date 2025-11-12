"""
Run all MILP comparison experiments
"""

from run_single_experiment import run_experiment


# Experiment configuration (matching old testscript_exp.py)
seed_list = [50, 51, 52, 53, 54]
input_size_list = [100, 1000]
layer_num_list = [3, 4, 5]
layer_size_list = [100, 500]
prune_amount_list = [0.0, 0.3, 0.5, 0.8]
time_limit = 600

output_file = "results/MILP_comparison.csv"


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
                                  f"time={result['time_count']:.2f}s")
                            print("-" * 80)

                        except Exception as e:
                            print(f"ERROR in experiment: {type(e).__name__}: {str(e)}")
                            print("Skipping and continuing...")
                            print("-" * 80)


if __name__ == "__main__":
    main()
