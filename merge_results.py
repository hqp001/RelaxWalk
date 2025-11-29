"""
Merge individual task CSV files into a single combined file.
Run this after all SLURM jobs complete.
"""

import pandas as pd
import os
import glob


def merge_results(results_dir="results", output_file="results/MILP_comparison_combined.csv"):
    """
    Merge all MILP_comparison_task_*.csv files into a single file.
    Files are sorted by task ID to preserve experiment order.

    Args:
        results_dir: Directory containing the task CSV files
        output_file: Path to the combined output file
    """

    # Find all task result files
    pattern = os.path.join(results_dir, "MILP_comparison_task_*.csv")
    task_files = glob.glob(pattern)

    if not task_files:
        print(f"No task files found matching pattern: {pattern}")
        return

    # Sort by task ID (numerical order) to preserve experiment order
    # Extract task ID from filename like "MILP_comparison_task_5.csv" -> 5
    def get_task_id(filepath):
        basename = os.path.basename(filepath)
        # Extract number from "MILP_comparison_task_X.csv"
        task_id_str = basename.replace("MILP_comparison_task_", "").replace(".csv", "")
        return int(task_id_str)

    task_files = sorted(task_files, key=get_task_id)

    print(f"Found {len(task_files)} task result files:")
    for f in task_files:
        print(f"  - {f}")

    # Read and concatenate all CSV files
    dfs = []
    for file in task_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Read {len(df)} rows from {file}")
        except Exception as e:
            print(f"ERROR reading {file}: {e}")

    if not dfs:
        print("No data to merge!")
        return

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by experiment parameters to match the nested loop order:
    # seed -> input_size -> layer_num -> layer_size -> prune_amount
    # This matches the natural experiment iteration order
    sort_columns = ['seed', 'input_size', 'layer_num', 'layer_size', 'prune_amount']

    # Only sort by columns that exist in the dataframe
    existing_sort_columns = [col for col in sort_columns if col in combined_df.columns]

    if existing_sort_columns:
        combined_df = combined_df.sort_values(by=existing_sort_columns).reset_index(drop=True)
        print(f"\nSorted by: {existing_sort_columns}")

    # Save to output file
    combined_df.to_csv(output_file, index=False)

    print(f"\nSuccessfully merged {len(combined_df)} rows into {output_file}")
    print(f"\nSummary:")
    print(f"  Total experiments: {len(combined_df)}")
    print(f"  Columns: {list(combined_df.columns)}")


if __name__ == "__main__":
    merge_results()
