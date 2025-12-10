import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

print(matplotlib.__version__)

SMALLER_FONT = 18
SMALL_FONT = 24
MEDIUM_FONT = 28
LARGE_FONT = 30
DOT_SIZE = 80

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(data):
    """Process data by extracting pruning rates and categorizing models."""
    pruned = data[data['Model Name'] != 'dense'].copy()
    dense = data[data['Model Name'] == 'dense'].copy()

    pruned['Pruning_Rate'] = pruned['Model Name'].str.extract(r'([0-9]\.[0-9]+)').astype(float)
    pruned['Finetune'] = pruned['Model Name'].str.contains('nonfinetune').map({True: 'nonfinetune', False: 'finetune'})
    pruned['PruningType'] = pruned['Model Name'].str.extract(r'(unstructured|structured)')
    pruned['MagnitudePrune'] = pruned['Model Name'].apply(lambda x: 'Random Pruning' if x.startswith('random') else 'Magnitude Pruning')

    return pruned, dense

def merge_data(pruned, dense):
    """Merge pruned and dense datasets on common columns."""
    merge_cols = ['Data Seed', 'Training Seed', 'Input Size', 'Number of Layers', 'Layers Size']
    merged = pd.merge(pruned, dense, on=merge_cols, suffixes=('_pruned', '_dense'), how='inner')
    merged.fillna({'TimeFirstNegative_pruned': 3000, 'TimeFirstNegative_dense': 3000}, inplace=True)

    merged['Total Time_pruned'] = np.minimum(merged['Prune Time_pruned'] + merged['TimeFirstNegative_pruned'], 3000)
    merged['Total Time_dense'] = merged['TimeFirstNegative_dense']

    return merged

def plot_scatterplots(merged, column_factor='Finetune'):
    """Generate scatter plots comparing pruned and dense models."""
    unique_rates = sorted(merged['Pruning_Rate'].dropna().unique())
    # unique_input_size = sorted(merged['Input size'].dropna().unique())
    # unique_number_of_layers = sorted(merged['Number of Layers'].dropna().unique())
    # unique_layer_size = sorted(merged['Layers Size'].dropna().unique())
    unique_col_values = sorted(merged[column_factor].dropna().unique())

    unique_col_values.append('total_finetune')

    row_values = 'Pruning_Rate'
    unique_rows = sorted(merged[row_values].dropna().unique())

    print(unique_col_values)
    if column_factor == 'Finetune':
        unique_col_values[0], unique_col_values[1] = unique_col_values[1], unique_col_values[0]

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

    colors = [(0, 0, 1), (0.5, 0, 0.5), (1, 0, 0), (1, 1, 0)]

    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    norm = plt.Normalize(vmin=merged['Accuracy_pruned'].min(), vmax=merged['Accuracy_pruned'].max())

    #fig.suptitle("Finetuned", fontsize=LARGE_FONT, fontweight='bold')


    for i, rate in enumerate(unique_rows):
        for j in range(len(unique_col_values)):
            col_val = unique_col_values[j]
            ax = axes[i, j]

            if col_val == 'total_finetune':
                subset = merged[(merged[row_values] == rate) & (merged[column_factor] == 'finetune')]
            else:
                #subset = merged[(merged[row_values] == rate) & (merged[column_factor] == col_val) & (merged['Finetune'] == 'finetune')]
                subset = merged[(merged[row_values] == rate) & (merged[column_factor] == col_val)]

            if subset.empty:
                ax.set_visible(False)
                continue

            if col_val == 'total_finetune':
                dense_time, pruned_time, pruned_acc = subset['Total Time_dense'], subset['Total Time_pruned'], subset['Accuracy_pruned']
            else:
                dense_time, pruned_time, pruned_acc = subset['TimeFirstNegative_dense'], subset['TimeFirstNegative_pruned'], subset['Accuracy_pruned']

            mask_square = (dense_time == 3000) | (pruned_time == 3000)
            mask_circle = ~mask_square

            if mask_circle.any():
                ax.scatter(dense_time[mask_circle], pruned_time[mask_circle], c=pruned_acc[mask_circle], cmap=custom_cmap, norm=norm, marker='o', edgecolor='k', alpha=0.8, s=DOT_SIZE)
            if mask_square.any():
                ax.scatter(dense_time[mask_square], pruned_time[mask_square], c=pruned_acc[mask_square], cmap=custom_cmap, norm=norm, marker='s', edgecolor='k', alpha=0.8, s=DOT_SIZE)

            ax.set_xscale('log')
            ax.set_xlim([min(dense_time.min(), pruned_time.min()) * 0.9, max(dense_time.max(), pruned_time.max()) * 1.1])
            ax.set_yscale('log')
            ax.set_ylim([min(dense_time.min(), pruned_time.min()) * 0.9, max(dense_time.max(), pruned_time.max()) * 1.1])
            ax.plot([min(dense_time.min(), pruned_time.min()) * 0.9, max(dense_time.max(), pruned_time.max()) * 1.1],
                    [min(dense_time.min(), pruned_time.min()) * 0.9, max(dense_time.max(), pruned_time.max()) * 1.1],
                    'k--', linewidth=0.8)

            ax.tick_params(axis='y', which='major', labelsize=SMALLER_FONT)
            ax.tick_params(axis='x', which='major', labelsize=SMALLER_FONT)

            num_above, num_below = np.sum(pruned_time > dense_time), np.sum(pruned_time < dense_time)
            n_points = len(dense_time)
            if n_points > 0:
                ax.text(0.05, 0.93, f"{(num_above / n_points * 100):.1f}%", transform=ax.transAxes, fontsize=MEDIUM_FONT, verticalalignment='top', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                ax.text(0.96, 0.1, f"{(num_below / n_points * 100):.1f}%", transform=ax.transAxes, fontsize=MEDIUM_FONT, horizontalalignment='right', verticalalignment='bottom', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

            # TITLE HEADER

            if i == 0:
                if col_val == 'nonfinetune':
                    ax.set_title(f"Not Finetuned", fontsize=LARGE_FONT, fontweight='bold')
                elif col_val == 'finetune':
                    ax.set_title(f"Finetuned \n(without Finetuning Time)", fontsize=LARGE_FONT, fontweight='bold')

                elif col_val == 'total_finetune':
                    ax.set_title(f"Finetuned \n(with Finetuning Time)", fontsize=LARGE_FONT, fontweight='bold')
                else:
                    if i == 0 and j == 0:
                        ax.set_title(f"{col_val.capitalize()}", fontsize=LARGE_FONT, fontweight='bold')
                    else:
                        ax.set_title(f"{col_val.capitalize()}", fontsize=LARGE_FONT, fontweight='bold')


            if j == 0:
                ax.text(-0.12, 0.5, f"Pruning Rate: {int(rate*100)}%", transform=ax.transAxes, fontsize=LARGE_FONT, rotation=90, verticalalignment='center', fontweight='bold')
            if i == len(unique_rows) - 1:
                ax.set_xlabel("Dense Runtime", fontsize=SMALL_FONT)

            ax.yaxis.set_label_position('right')
            ax.yaxis.set_ticks_position('right')
            if j == len(unique_col_values) - 1:
                ax.set_ylabel("Pruned Runtime", fontsize=SMALL_FONT, labelpad=15)


    """
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1).set_label('Pruned Accuracy')
    """


    # Adjust color bar position
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm),
        ax=axes,
        orientation='horizontal',
        fraction=0.02,
        pad=0.015  # Adjusted to bring color bar closer
    )
    cbar.set_label("Accuracy of Pruned Models", fontsize=MEDIUM_FONT)


    output_path = os.path.join(os.getcwd(), f"fashion_scatterplot_table_{column_factor}.pdf")
    plt.savefig(output_path, dpi=300, format='pdf')
    print(f"Plot saved as: {output_path}")
    #plt.show()

def main():
    """Main execution function."""
    filepath = 'data_Nov25.csv'
    #filepath = 'fashion_data_Dec3.csv'
    data = load_data(filepath)
    pruned_data, dense_data = preprocess_data(data)
    merged_data = merge_data(pruned_data, dense_data)
    plot_scatterplots(merged_data)

if __name__ == "__main__":
    main()

