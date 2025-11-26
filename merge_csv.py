#!/usr/bin/env python3
"""
Script to append MILP_comparison_2.csv through MILP_comparison_6.csv
into MILP_comparison_1.csv
"""

import csv

# Open the main file in append mode
with open('MILP_comparison_1.csv', 'a', newline='') as outfile:
    writer = csv.writer(outfile)

    # Loop through files 2-6
    for i in range(2, 7):
        filename = f'MILP_comparison_{i}.csv'
        print(f"Appending {filename}...")

        with open(filename, 'r', newline='') as infile:
            reader = csv.reader(infile)
            # Skip the header row
            next(reader)
            # Append all data rows
            for row in reader:
                writer.writerow(row)

        print(f"  ✓ {filename} appended")

print("\n✓ All files merged into MILP_comparison_1.csv")
