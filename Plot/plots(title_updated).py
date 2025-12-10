import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

print(matplotlib.__version__)

SMALLER_FONT = 18
SMALL_FONT   = 24
MEDIUM_FONT  = 28
LARGE_FONT   = 30
DOT_SIZE     = 80

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(data):
    """Process data by extracting pruning rates and categorizing models."""
    pruned = data[data['Model Name'] != 'dense'].copy()
    dense  = data[data['Model Name'] == 'dense'].copy()

    pruned['Pruning_Rate'] = pruned['Model Name'].str.extract(r'([0-9]\.[0-9]+)').astype(float)
    pruned['Finetune']     = pruned['Model Name'].str.contains('nonfinetune').map({True: 'nonfinetune', False: 'finetune'})
    pruned['PruningType']  = pruned['Model Name'].str.extract(r'(unstructured|structured)')
    pruned['MagnitudePrune'] = pruned['Model Name'].apply(lambda x: 'Random Pruning' if x.startswith('random') else 'Magnitude Pruning')

    return pruned, dense

def merge_data(pruned, dense):
    """Merge pruned and dense datasets on common columns."""
    merge_cols = ['Data Seed', 'Training Seed', 'Input Size', 'Number of Layers', 'Layers Size']
    merged = pd.merge(pruned, dense, on=merge_cols, suffixes=('_pruned', '_dense'), how='inner')
    merged.fillna({'TimeFirstNegative_pruned': 3000, 'TimeFirstNegative_dense': 3000}, inplace=True)

    merged['Total Time_pruned'] = np.minimum(merged['Prune Time_pruned'] + merged['TimeFirstNegative_pruned'], 3000)
    merged['Total Time_dense']  = merged['TimeFirstNegative_dense']

    return merged

def plot_scatterplots(merged, column_factor='Finetune'):
    """Generate scatter plots comparing pruned and dense models."""
    unique_rates = sorted(merged['Pruning_Rate'].dropna().unique())
    unique_col_values = sorted(merged[column_factor].dropna().unique())

    # Add 'total_finetune' for the "including finetuning time" column
    unique_col_values.append('total_finetune')

    row_values = 'Pruning_Rate'
    unique_rows = sorted(merged[row_values].dropna().unique())

    # If column_factor == 'Finetune', we swap so 'finetune' is first
    if column_factor == 'Finetune':
        unique_col_values[0], unique_col_values[1] = unique_col_values[1], unique_col_values[0]

    fig, axes = plt.subplots(
        len(unique_rows),
        len(unique_col_values),
        figsize=(7 * len(unique_col_values), 6 * len(unique_rows)),
        sharex=False, sharey=False, constrained_layout=True
    )
    fig.set_constrained_layout_pads(wspace=0.1, hspace=0.01)

    if len(unique_rows) == 1 and len(unique_col_values) == 1:
        axes = np.array([[axes]])
    elif len(unique_rows) == 1:
        axes = np.array([axes])
    elif len(unique_col_values) == 1:
        axes = np.array([[ax] for ax in axes])

    colors = [(0, 0, 1), (0.5, 0, 0.5), (1, 0, 0), (1, 1, 0)]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    norm = plt.Normalize(vmin=merged['Accuracy_pruned'].min(), vmax=merged['Accuracy_pruned'].max())

    for i, rate in enumerate(unique_rows):
        for j, col_val in enumerate(unique_col_values):
            ax = axes[i, j]

            # Subset data
            if col_val == 'total_finetune':
                subset = merged[(merged[row_values] == rate) & (merged[column_factor] == 'finetune')]
            else:
                subset = merged[(merged[row_values] == rate) & (merged[column_factor] == col_val)]

            if subset.empty:
                ax.set_visible(False)
                continue

            # Decide which columns for time
            if col_val == 'total_finetune':
                dense_time  = subset['Total Time_dense']
                pruned_time = subset['Total Time_pruned']
                pruned_acc  = subset['Accuracy_pruned']
            else:
                dense_time  = subset['TimeFirstNegative_dense']
                pruned_time = subset['TimeFirstNegative_pruned']
                pruned_acc  = subset['Accuracy_pruned']

            mask_square = (dense_time == 3000) | (pruned_time == 3000)
            mask_circle = ~mask_square

            if mask_circle.any():
                ax.scatter(
                    dense_time[mask_circle], pruned_time[mask_circle],
                    c=pruned_acc[mask_circle], cmap=custom_cmap, norm=norm,
                    marker='o', edgecolor='k', alpha=0.8, s=DOT_SIZE
                )
            if mask_square.any():
                ax.scatter(
                    dense_time[mask_square], pruned_time[mask_square],
                    c=pruned_acc[mask_square], cmap=custom_cmap, norm=norm,
                    marker='s', edgecolor='k', alpha=0.8, s=DOT_SIZE
                )

            ax.set_xscale('log')
            ax.set_yscale('log')

            # Axis limits
            x_min, x_max = dense_time.min(), dense_time.max()
            y_min, y_max = pruned_time.min(), pruned_time.max()
            overall_min = min(x_min, y_min) * 0.9
            overall_max = max(x_max, y_max) * 1.1
            ax.set_xlim([overall_min, overall_max])
            ax.set_ylim([overall_min, overall_max])

            # Diagonal line
            ax.plot([overall_min, overall_max], [overall_min, overall_max], 'k--', linewidth=0.8)

            # Tick label sizes
            ax.tick_params(axis='x', which='major', labelsize=SMALLER_FONT)
            ax.tick_params(axis='y', which='major', labelsize=SMALLER_FONT)

            # Above/below diagonal stats
            num_above = np.sum(pruned_time > dense_time)
            num_below = np.sum(pruned_time < dense_time)
            n_points  = len(dense_time)
            if n_points > 0:
                ax.text(
                    0.05, 0.93, f"{(num_above / n_points * 100):.1f}%",
                    transform=ax.transAxes, fontsize=MEDIUM_FONT,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
                )
                ax.text(
                    0.96, 0.1, f"{(num_below / n_points * 100):.1f}%",
                    transform=ax.transAxes, fontsize=MEDIUM_FONT,
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
                )

            # *** We remove the top-row title for 'nonfinetune' so we can do a figure-level text. ***
            if i == 0:
                if col_val == 'finetune':
                    ax.set_title("Excluding Finetuning Time", fontsize=SMALLER_FONT, pad=8)
                elif col_val == 'total_finetune':
                    ax.set_title("Including Finetuning Time", fontsize=SMALLER_FONT, pad=8)
                # else: if col_val == 'nonfinetune', do NOT set_title here.

            # Y-axis label on the leftmost column
            if j == 0:
                ax.text(
                    -0.12, 0.5,
                    f"Pruning Rate: {int(rate*100)}%",
                    transform=ax.transAxes, fontsize=LARGE_FONT,
                    rotation=90, verticalalignment='center', fontweight='bold'
                )

            # Bottom row x-label
            if i == len(unique_rows) - 1:
                ax.set_xlabel("Dense Runtime", fontsize=SMALL_FONT)

            # Rightmost column y-label
            ax.yaxis.set_label_position('right')
            ax.yaxis.set_ticks_position('right')
            if j == len(unique_col_values) - 1:
                ax.set_ylabel("Pruned Runtime", fontsize=SMALL_FONT, labelpad=15)

    # Color bar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm),
        ax=axes,
        orientation='horizontal',
        fraction=0.02,
        pad=0.015
    )
    cbar.set_label("Accuracy of Pruned Models", fontsize=MEDIUM_FONT)

    #
    # =========== FIGURE-LEVEL HEADERS =============
    #
    import matplotlib.lines as mlines

    # 1) Big "Not Finetuned" label (left side)
    fig.text(
        0.18, 0.995,     # x=0.18 means left-ish; y=1.02 for top
        "Not Finetuned",
        ha='center', va='bottom',
        fontsize=LARGE_FONT,
        fontweight='bold'
    )

    # 2) Big "Finetuned" label (right side, covering columns 2 & 3)
    fig.text(
        0.65, 1,     # x=0.70 means more right; same y=1.02
        "Finetuned",
        ha='center', va='bottom',
        fontsize=LARGE_FONT,
        fontweight='bold'
    )
    line_ft = mlines.Line2D(
        [0.35, 0.95], [1.00, 1.00],
        transform=fig.transFigure,
        color='black',
        linewidth=2
    )
    fig.add_artist(line_ft)

    # Save
    output_path = os.path.join(os.getcwd(), f"fashion_scatterplot_table_{column_factor}.pdf")
    plt.savefig(output_path, dpi=300, format='pdf')
    print(f"Plot saved as: {output_path}")
    # plt.show()

def main():
    filepath = 'data_Nov25.csv'
    data = load_data(filepath)
    pruned_data, dense_data = preprocess_data(data)
    merged_data = merge_data(pruned_data, dense_data)
    plot_scatterplots(merged_data)

if __name__ == "__main__":
    main()
