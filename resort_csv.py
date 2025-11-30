import csv
import ast

def extract_model_size_keys(model_size_str):
    """Extract sorting keys from model_size string"""
    # Parse the string representation of the list
    model_size = ast.literal_eval(model_size_str)

    input_size = model_size[0]  # First element
    num_layers = len(model_size)  # Number of elements in array
    layer_size = model_size[1] if len(model_size) > 1 else 0  # Second element

    return (input_size, num_layers, layer_size)

# Read the CSV file
with open('MILP_comparison_combined.csv', 'r') as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
    rows = list(reader)

# Sort the rows by input_size, num_layers, layer_size, seed, then prune_amount
sorted_rows = sorted(rows, key=lambda row: (
    *extract_model_size_keys(row['model_size']),  # Primary: input_size, num_layers, layer_size
    int(row['seed']),   # Fourth: group by seed
    float(row['prune_amount'])  # Fifth: order by increasing prune_amount
))

# Write the sorted data back to the CSV file
with open('MILP_comparison_combined.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(sorted_rows)

print("CSV file has been resorted successfully!")
print(f"Total rows: {len(sorted_rows)}")
