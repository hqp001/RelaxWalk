#!/usr/bin/env python3
"""
YAML-based Main Interface for Relax-and-Walk Algorithms

This module provides a clean interface for running the Relaxation Walk (RW)
and Dynamic Walk (RWD) algorithms using YAML configuration files.

Usage:
    python main_remodeled.py <config.yaml>
    python main_remodeled.py --create-example <algorithm> <output_file>
    python main_remodeled.py --batch <batch_config.yaml>

Examples:
    python main_remodeled.py configs/rw_experiment.yaml
    python main_remodeled.py --create-example relaxation_walk example_rw.yaml
    python main_remodeled.py --batch configs/batch_experiments.yaml
"""

import sys
import os
from pathlib import Path
from typing import List
from config_manager import (
    load_yaml_config,
    validate_config,
    create_example_config,
    load_batch_config
)
from data_types import AlgorithmConfiguration
from algorithm_runners import run_algorithm, run_algorithm_with_yaml_config
from logger_config import get_logger, TimingContext


def run_single_experiment(config: AlgorithmConfiguration) -> None:
    """
    Run a single algorithm experiment.

    Args:
        config: Algorithm configuration
    """
    logger = get_logger("main")

    # Validate configuration
    errors = validate_config(config)
    if errors:
        logger.error("Configuration validation failed", error_count=len(errors))
        for error in errors:
            logger.error(f"Validation error: {error}")
        sys.exit(1)

    # Log experiment setup
    logger.info("Starting experiment",
                algorithm=config.algorithm,
                network_size=f"{config.network.input_size}→{config.network.layer_size}×{config.network.layer_num}→1",
                walk_eps=config.params.walk_eps,
                pick_bias=config.params.pick_bias,
                random_seed=config.experiment.random_seed,
                time_limit=config.experiment.time_limit)


    try:
        with TimingContext(logger, "experiment execution",
                          algorithm=config.algorithm,
                          seed=config.experiment.random_seed):
            result = run_algorithm_with_yaml_config(config)

        # Log results
        logger.info("Experiment completed successfully",
                   execution_time=f"{result.execution_time:.2f}s",
                   best_objective=result.best_objective,
                   first_objective=result.first_objective,
                   total_starts=result.start_count,
                   valid_starts=result.valid_start_count,
                   updates=len(result.update_history))

        print("✅ Algorithm completed successfully!")

    except Exception as e:
        logger.error("Experiment failed", error=str(e))
        print(f"❌ Error running algorithm: {e}")
        sys.exit(1)


def run_batch_experiments(batch_config_path: str) -> None:
    """
    Run multiple experiments from batch configuration.

    Args:
        batch_config_path: Path to batch configuration file
    """
    logger = get_logger("main")

    try:
        configs = load_batch_config(batch_config_path)
        logger.info("Starting batch experiments",
                   config_file=batch_config_path,
                   experiment_count=len(configs))


        results = []
        with TimingContext(logger, "batch execution", experiment_count=len(configs)):
            for i, config in enumerate(configs, 1):
                logger.info(f"Running batch experiment {i}/{len(configs)}",
                           algorithm=config.algorithm,
                           seed=config.experiment.random_seed)

                # Run experiment
                run_single_experiment(config)
                logger.info(f"Batch experiment {i} completed")

        logger.info("All batch experiments completed successfully", total_experiments=len(configs))
        print("🎉 All experiments completed successfully!")

    except Exception as e:
        logger.error("Batch experiments failed", error=str(e))
        print(f"❌ Error running batch experiments: {e}")
        sys.exit(1)


def create_example_configs() -> None:
    """Create example configuration files."""
    # Create examples directory
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    # Create RW example
    rw_path = examples_dir / "relaxation_walk.yaml"
    create_example_config("relaxation_walk", str(rw_path))
    print(f"Created example RW config: {rw_path}")

    # Create RWD example
    rwd_path = examples_dir / "dynamic_walk.yaml"
    create_example_config("dynamic_walk", str(rwd_path))
    print(f"Created example RWD config: {rwd_path}")

    # Create batch example
    batch_example = {
        'experiments': [
            {
                'algorithm': 'relaxation_walk',
                'network': {'input_size': 10, 'layer_num': 2, 'layer_size': 20},
                'params': {'walk_eps': 0.1, 'pick_bias': 0.01},
                'experiment': {'random_seed': 42, 'time_limit': 1800}
            },
            {
                'algorithm': 'dynamic_walk',
                'network': {'input_size': 10, 'layer_num': 2, 'layer_size': 20},
                'params': {'walk_eps': 0.1},
                'experiment': {'random_seed': 123, 'time_limit': 1800}
            }
        ]
    }

    import yaml
    batch_path = examples_dir / "batch_experiments.yaml"
    with open(batch_path, 'w') as f:
        yaml.dump(batch_example, f, default_flow_style=False, indent=2)
    print(f"Created example batch config: {batch_path}")


def print_usage():
    """Print usage information."""
    print(__doc__)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    arg = sys.argv[1]

    # Handle special commands
    if arg == "--create-examples":
        create_example_configs()
        return

    elif arg == "--create-example":
        if len(sys.argv) != 4:
            print("Usage: python main_remodeled.py --create-example <algorithm> <output_file>")
            sys.exit(1)

        algorithm = sys.argv[2]
        output_file = sys.argv[3]

        if algorithm not in ['relaxation_walk', 'dynamic_walk', 'rw', 'rwd']:
            print(f"Invalid algorithm: {algorithm}")
            print("Valid algorithms: relaxation_walk, dynamic_walk, rw, rwd")
            sys.exit(1)

        create_example_config(algorithm, output_file)
        print(f"Created example configuration: {output_file}")
        return

    elif arg == "--batch":
        if len(sys.argv) != 3:
            print("Usage: python main_remodeled.py --batch <batch_config.yaml>")
            sys.exit(1)

        batch_config_path = sys.argv[2]
        run_batch_experiments(batch_config_path)
        return

    elif arg.startswith("--"):
        print(f"Unknown option: {arg}")
        print_usage()
        sys.exit(1)

    # Regular config file
    config_path = arg

    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        config = load_yaml_config(config_path)
        run_single_experiment(config)

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()