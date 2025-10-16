import pandas as pd
import os

# Define column order in ONE place
COLUMNS = ['method', 'model_size', 'parameters', 'seed', 'prune_amount', 'x_max', 'max_',
           'first_max', 'time_count', 'start_count', 'valid_start_count', 'original_max', 'update_list']

def store_data(data_dict, filename):
    """
    Store data to CSV file.

    Args:
        data_dict: Dictionary containing the data for one row
        filename: Name of the CSV file to write to
    """
    # Create DataFrame with explicit column order
    df = pd.DataFrame([data_dict], columns=COLUMNS)

    write_header = not os.path.exists(filename)

    # store into csv file (in append mode)
    df.to_csv(filename, mode='a', header=write_header, index=False)
