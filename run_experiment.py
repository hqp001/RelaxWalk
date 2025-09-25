"""
Main experiment runner using the new YAML + SQLite I/O pipeline.
"""

import sys
import argparse
from pathlib import Path

from multi_layer_relaxation_walk import relaxation_walk_deep
from utils.io import load_config, store_experiment_result
from configs.log_config import get_logger

logger = get_logger(__name__)


def run_experiment(config_path: str, db_path: str = "experiments.db"):
    """
    Run experiment with configuration from YAML file and store results in SQLite.

    Args:
        config_path: Path to YAML configuration file
        db_path: Path to SQLite database file
    """
    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        config_loader = load_config(config_path)

        # Get parameters for relaxation walk
        params = config_loader.get_relaxation_walk_params()
        input_size, layer_num, layer_size, random_seed, walk_eps, pick_bias, timelimit, pruned_density = params

        logger.info(f"Running experiment: {config_loader.get_experiment_name()}")
        logger.info(f"Parameters: input_size={input_size}, layer_num={layer_num}, "
                   f"layer_size={layer_size}, seed={random_seed}, eps={walk_eps}, "
                   f"bias={pick_bias}, timelimit={timelimit}, pruned_density={pruned_density}")

        # Run the experiment
        result = relaxation_walk_deep(
            input_size=input_size,
            layer_num=layer_num,
            layer_size=layer_size,
            random_seed=random_seed,
            walk_eps=walk_eps,
            pick_bias=pick_bias,
            timelimit=timelimit,
            pruned_density=pruned_density
        )

        if result is None or result[0] is None:
            logger.warning("Experiment returned no valid results")
            return None

        # Store results in database
        logger.info(f"Storing results in database: {db_path}")
        x_max, max_value, first_max, time_count, start_count, valid_start_count, update_list, original_model = result
        result_id = store_experiment_result(config_loader, result[:-1], db_path, original_model)

        logger.info(f"Experiment completed successfully!")
        logger.info(f"Result ID: {result_id}")
        logger.info(f"Max value: {max_value}")
        logger.info(f"Time: {time_count:.3f}s")
        logger.info(f"Steps: {start_count}, Valid starts: {valid_start_count}")

        return result_id

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Run RelaxWalk experiment with YAML configuration")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--db", default="experiments.db", help="Path to SQLite database file")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        result_id = run_experiment(str(config_path), args.db)
        if result_id:
            print(f"Experiment completed successfully. Result ID: {result_id}")
        else:
            print("Experiment completed but returned no results.")
            sys.exit(1)
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()