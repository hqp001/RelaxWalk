#!/usr/bin/env python3

import sqlite3
import pandas as pd
import json

def export_experiments_to_csv():
    # Connect to the database
    conn = sqlite3.connect('./experiments_timelimit200.db')

    # Query to get all experiment data with joined configs and results
    query = """
    SELECT
        e.id,
        e.name,
        e.method,
        e.config_hash,
        e.created_at,
        nc.input_size,
        nc.layer_num,
        nc.layer_size,
        nc.pruned_density,
        ac.walk_eps,
        ac.pick_bias,
        ac.random_seed,
        ac.timelimit,
        r.x_max,
        r.max_value,
        r.first_max,
        r.time_count,
        r.start_count,
        r.valid_start_count,
        r.original_model_max,
        r.original_model_x_max
    FROM experiments e
    LEFT JOIN network_configs nc ON e.id = nc.experiment_id
    LEFT JOIN algorithm_configs ac ON e.id = ac.experiment_id
    LEFT JOIN results r ON e.id = r.experiment_id
    ORDER BY e.name, nc.pruned_density
    """

    # Execute query and get results
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Create a list to store all rows - each pruned density gets its own row
    all_rows = []

    for _, row in df.iterrows():
        row_data = {
            'experiment_name': row['name'],
            'experiment_id': row['id'],
            'method': row['method'],
            'created_at': row['created_at'],
            # Network config (input)
            'input_size': row['input_size'],
            'layer_num': row['layer_num'],
            'layer_size': row['layer_size'],
            'pruned_density': row['pruned_density'],
            # Algorithm config (input)
            'walk_eps': row['walk_eps'],
            'pick_bias': row['pick_bias'],
            'random_seed': row['random_seed'],
            'timelimit': row['timelimit'],
            # Results (output)
            'x_max': row['x_max'],
            'max_value': row['max_value'],
            'first_max': row['first_max'],
            'time_count': row['time_count'],
            'start_count': row['start_count'],
            'valid_start_count': row['valid_start_count'],
            'original_model_max': row['original_model_max'],
            'original_model_x_max': row['original_model_x_max']
        }
        all_rows.append(row_data)

    # Convert to DataFrame
    result_df = pd.DataFrame(all_rows)

    # Reorder columns for better readability
    column_order = [
        'experiment_name', 'experiment_id', 'method',
        'input_size', 'layer_num', 'layer_size', 'pruned_density',
        'walk_eps', 'pick_bias', 'random_seed', 'timelimit',
        'max_value', 'time_count', 'start_count',
        'valid_start_count', 'original_model_max'
    ]

    # Only include columns that exist
    available_columns = [col for col in column_order if col in result_df.columns]
    result_df = result_df[available_columns]

    # Save to CSV
    output_file = 'experiments_200.csv'
    result_df.to_csv(output_file, index=False)

    print(f"Exported {len(result_df)} experiment runs to {output_file}")
    print(f"Data sorted by experiment name and pruned density")
    print("\nSample of data:")
    print(result_df[['experiment_name', 'pruned_density', 'max_value']].head(10))

    return output_file

if __name__ == "__main__":
    export_experiments_to_csv()
