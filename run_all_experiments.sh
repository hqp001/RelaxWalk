#!/bin/bash

echo "Generating config files..."

python ./generate_configs.py

echo "Starting mass experiments..."

for config in configs/mass_experiments/config_*.yaml; do
    echo "Running $config"
    python run_experiment.py "$config"
done

echo "All experiments completed!"
