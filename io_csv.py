import pandas as pd
import os

def store_data(data, filename):
    # create DataFrame
    df = pd.DataFrame(data, columns=['method', 'model_size', 'parameters', 'seed', 'prune_amount', 'x_max', 'max_', 'first_max', 'time_count', 'start_count', 'valid_start_count', 'original_max', 'update_list'])

    write_header = not os.path.exists(filename)

    # store into csv file (in append mode)
    df.to_csv(filename, mode='a', header=write_header, index=False)
