import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from pandas.core import base

print(matplotlib.__version__)

SMALLER_FONT = 18
SMALL_FONT = 24
MEDIUM_FONT = 28
LARGE_FONT = 30
DOT_SIZE = 80

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(baseline, data):
    """
    Prepare baseline (dense) and MILP-pruned data so they can be merged
    using model_size and seed.
    """

    # --- Split dense and pruned ---
    BAD_VALUES = [0.0, 0.999, 0.995]
    pruned = data[~data['prune_amount'].isin(BAD_VALUES)].copy()
    dense  = baseline.copy()

    # --- Pruning rate ---
    pruned['Pruning_Rate'] = pruned['prune_amount']
    dense['Pruning_Rate']  = 0.0

    # --- Normalize model_size arrays/lists into tuple-string keys ---
    def normalize_model_size(x):
        # If the CSV stores "[64, 64]" as a string, convert it
        if isinstance(x, str):
            x = eval(x)
        return str(tuple(x))  # "(64, 64)"

    def get_depth_size(x):
        # If the CSV stores "[64, 64]" as a string, convert it
        if isinstance(x, str):
            x = eval(x)
        return tuple(x)[1]  # "(64, 64)"

    # Convert to consistent tuple form
    pruned['model_size_tuple'] = pruned['model_size'].apply(normalize_model_size)
    dense['model_size_tuple']  = dense['model_size'].apply(normalize_model_size)

    pruned["Input Size"] = pruned["model_size_tuple"].apply(get_depth_size)

    # --- Build combined model_key ---
    # Example: "(64,64)_7"
    pruned['model_key'] = pruned.apply(
        lambda row: f"{row.model_size_tuple}_{row.seed}", axis=1
    )
    dense['model_key'] = dense.apply(
        lambda row: f"{row.model_size_tuple}_{row.seed}", axis=1
    )


    return pruned, dense

def merge_data(pruned, dense):
    """Merge pruned and dense datasets on common columns."""
    baseline_lookup = (
        dense.set_index("model_key")["original_max"].to_dict()
    )

    # Now for each pruned row, fetch baseline value
    pruned["baseline"] = pruned["model_key"].map(baseline_lookup)

    # Optional: warn if some model_keys had no match
    missing = pruned[pruned["baseline"].isna()]["model_key"].unique()
    if len(missing) > 0:
        print("⚠️ Warning: No baseline found for these model_keys:")
        print(missing)

    return pruned

def plot_scatterplots(merged, column_factor='Input Size'):
    """Generate scatter plots comparing pruned and dense models."""
    unique_rates = sorted(merged['Pruning_Rate'].dropna().unique())
    # unique_input_size = sorted(merged['Input size'].dropna().unique())
    # unique_number_of_layers = sorted(merged['Number of Layers'].dropna().unique())
    # unique_layer_size = sorted(merged['Layers Size'].dropna().unique())
    unique_col_values = sorted(merged[column_factor].dropna().unique())
    print(unique_col_values)


    row_values = 'Pruning_Rate'
    unique_rows = sorted(merged[row_values].dropna().unique())

    print(unique_col_values)

    fig, axes = plt.subplots(
            len(unique_rows),
            len(unique_col_values),
            figsize=(7 * len(unique_col_values),
                6 * len(unique_rows)),
            sharex=False, sharey=False, constrained_layout=True)


    #fig.tight_layout(pad=3,w_pad=10)

    fig.set_constrained_layout_pads(wspace=0.1,hspace=0.01)  # Increase horizontal padding

    if len(unique_rows) == 1 and len(unique_col_values) == 1:
        axes = np.array([[axes]])
    elif len(unique_rows) == 1:
        axes = np.array([axes])
    elif len(unique_col_values) == 1:
        axes = np.array([[ax] for ax in axes])


    #fig.suptitle("Not Finetuned", fontsize=LARGE_FONT, fontweight='bold')


    for i, rate in enumerate(unique_rows):
        for j in range(len(unique_col_values)):
            col_val = unique_col_values[j]
            ax = axes[i, j]

            subset = merged[(merged[row_values] == rate) & (merged[column_factor] == col_val)]
            #subset = merged[(merged[row_values] == rate) & (merged[column_factor] == col_val)]

            if subset.empty:
                ax.set_visible(False)
                continue

            dense_time, pruned_time = subset['baseline'], subset['original_max']

            clean_dense  = dense_time.replace(-np.inf, np.nan)
            clean_pruned = pruned_time.replace(-np.inf, np.nan)
            xmin = min(clean_dense.min(), clean_pruned.min())
            xmax = max(clean_dense.max(), clean_pruned.max())
            padding = 0.2 * (xmax - xmin)   # 10% padding

            xmin = xmin - padding
            xmax = xmax + padding

            ymin = xmin   # keep square axes for diagonal correctness
            ymax = xmax

            mask_square = (dense_time == -np.inf) | (pruned_time == -np.inf)
            mask_circle = ~mask_square
            dense_time  = dense_time.replace(-np.inf, xmin)
            pruned_time = pruned_time.replace(-np.inf, xmin)

            if mask_circle.any():
                ax.scatter(dense_time[mask_circle], pruned_time[mask_circle], marker='o', edgecolor='k', alpha=0.8, s=DOT_SIZE, color='black')
            if mask_square.any():
                ax.scatter(dense_time[mask_square], pruned_time[mask_square], marker='s', edgecolor='k', alpha=0.8, s=DOT_SIZE, color='red')

            ax.figure.canvas.draw()

# Fetch true final limits

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # ax.set_xscale('log')
            # ax.set_yscale('log')
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            # Draw diagonal corner to corner
            ax.plot([xmin, xmax], [ymin, ymax], 'k--', linewidth=0.8)

            ax.tick_params(axis='y', which='major', labelsize=SMALLER_FONT)
            ax.tick_params(axis='x', which='major', labelsize=SMALLER_FONT)

            num_above, num_below = np.sum(pruned_time > dense_time), np.sum(pruned_time < dense_time)
            n_points = len(dense_time)
            if n_points > 0:
                ax.text(0.05, 0.93, f"{(num_above / n_points * 100):.1f}%", transform=ax.transAxes, fontsize=MEDIUM_FONT, verticalalignment='top', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                ax.text(0.96, 0.1, f"{(num_below / n_points * 100):.1f}%", transform=ax.transAxes, fontsize=MEDIUM_FONT, horizontalalignment='right', verticalalignment='bottom', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

            if i == 0:
                if i == 0 and j == 0:
                    ax.set_title(f"Layer Width: {col_val}", fontsize=LARGE_FONT, fontweight='bold')
                else:
                    ax.set_title(f"Layer Width: {col_val}", fontsize=LARGE_FONT, fontweight='bold')

            if j == 0:
                ax.text(-0.12, 0.5, f"Pruning Rate: {int(rate*100)}%", transform=ax.transAxes, fontsize=LARGE_FONT, rotation=90, verticalalignment='center', fontweight='bold')
            if i == len(unique_rows) - 1:
                ax.set_xlabel("Dense Maximum Value", fontsize=SMALL_FONT)

            ax.yaxis.set_label_position('right')
            ax.yaxis.set_ticks_position('right')
            if j == len(unique_col_values) - 1:
                ax.set_ylabel("Pruned Maximum Value", fontsize=SMALL_FONT, labelpad=15)


    output_path = os.path.join(os.getcwd(), f"MILP_plot_Layer_Width.pdf")
    plt.savefig(output_path, dpi=300, format='pdf')
    print(f"Plot saved as: {output_path}")
    #plt.show()

def main():
    """Main execution function."""
    #filepath = 'data_Nov25.csv'
    #filepath = 'fashion_data_Dec3.csv'
    baseline = pd.read_csv("baseline.csv")
    milp = pd.read_csv("MILP_prune.csv")
    pruned, dense = preprocess_data(baseline, milp)

    #print(baseline, milp)
    pruned = merge_data(pruned, dense)

    # data = load_data(filepath)
    # pruned_data, dense_data = preprocess_data(data)
    # merged_data = merge_data(pruned_data, dense_data)
    plot_scatterplots(pruned)

if __name__ == "__main__":
    main()

