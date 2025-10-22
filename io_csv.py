import pandas as pd
import os

# Define column order in ONE place
COLUMNS = ['method', 'model_size', 'parameters', 'seed', 'prune_amount', 'max_',
           'first_max', 'time_count', 'start_count', 'valid_start_count', 'original_max', 'original_max_time_elapsed']

def store_data(data_dict, filename):
    """
    Store data to CSV file.

    Args:
        data_dict: Dictionary containing the data for one row
        filename: Name of the CSV file to write to
    """
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(filename)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create DataFrame with explicit column order
    df = pd.DataFrame([data_dict], columns=COLUMNS)

    write_header = not os.path.exists(filename)

    # store into csv file (in append mode)
    df.to_csv(filename, mode='a', header=write_header, index=False)
