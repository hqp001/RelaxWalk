import time
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from data_types import Network, AlgorithmResult, AlgorithmConfiguration
from workflows import solve_linear_relaxation, get_binary_activations, gradient_walk
from workflows.generators import generate_systematic_starting_points, generate_dynamic_starting_points
from logger_config import get_logger, TimingContext, timer


@timer("network creation")
def create_network(config: AlgorithmConfiguration) -> Network:
    """
    Create neural network from configuration.

    Args:
        config: Algorithm configuration

    Returns:
        Neural network instance
    """
    layer_dims = [config.network.layer_size] * config.network.layer_num + [1]
    return Network(in_size=config.network.input_size, layer_dims=layer_dims, seed=config.experiment.random_seed)


def run_relaxation_walk(config: AlgorithmConfiguration) -> AlgorithmResult:
    """
    Run Relaxation Walk (RW) algorithm.

    Args:
        config: Algorithm configuration

    Returns:
        Algorithm execution result
    """
    logger = get_logger("algorithm_runners")
    logger.info("Starting Relaxation Walk algorithm",
               input_size=config.network.input_size,
               layer_num=config.network.layer_num,
               layer_size=config.network.layer_size,
               walk_eps=config.params.walk_eps,
               pick_bias=config.params.pick_bias,
               seed=config.experiment.random_seed,
               time_limit=config.experiment.time_limit)

    # Setup
    np.random.seed(config.experiment.random_seed)
    network = create_network(config)
    start_time = time.time()

    # Solve initial linear relaxation
    with TimingContext(logger, "linear relaxation solution"):
        x_relaxation, frac_activations = solve_linear_relaxation(network, config.experiment.time_limit)

    if x_relaxation is None:
        logger.error("Linear relaxation failed")
        return _create_failed_result("RW", config, time.time() - start_time)

    logger.debug("Linear relaxation completed successfully")

    # Get initial binary activations
    with TimingContext(logger, "binary activation computation"):
        binary_activations = get_binary_activations(network, x_relaxation)

    # Perform initial relaxation walk
    logger.debug("Starting initial gradient walk")
    with TimingContext(logger, "initial gradient walk"):
        first_best_x, first_best_obj, _, _ = gradient_walk(
            network, x_relaxation, config.params.walk_eps,
            config.experiment.time_limit - (time.time() - start_time)
        )

    logger.info("Initial walk completed", first_objective=first_best_obj)

    # Initialize tracking variables
    global_best_obj = first_best_obj if first_best_obj is not None else -float('inf')
    global_best_x = first_best_x
    start_count = 1
    valid_start_count = 1
    update_history = []

    if first_best_obj is not None:
        update_history.append([
            first_best_obj, first_best_x, start_count, valid_start_count,
            time.time() - start_time
        ])

    # Generate additional starting points systematically
    logger.debug("Starting systematic point generation")
    point_generator = generate_systematic_starting_points(
        network, binary_activations, frac_activations,
        config.params.pick_bias, config.experiment.time_limit, start_time
    )

    improvements = 0
    for starting_point in point_generator:
        if time.time() - start_time >= config.experiment.time_limit:
            logger.debug("Time limit reached, stopping point generation")
            break

        start_count += 1
        valid_start_count += 1

        # Perform walk from this starting point
        time_remaining = config.experiment.time_limit - (time.time() - start_time)
        if time_remaining <= 0:
            break

        with TimingContext(logger, f"gradient walk {start_count}", start_num=start_count):
            best_x, best_obj, _, _ = gradient_walk(
                network, starting_point, config.params.walk_eps, time_remaining
            )

        # Update global best if improved
        if best_obj is not None and best_obj > global_best_obj:
            improvements += 1
            old_best = global_best_obj
            global_best_obj = best_obj
            global_best_x = best_x
            update_history.append([
                global_best_obj, global_best_x, start_count, valid_start_count,
                time.time() - start_time
            ])
            logger.info("Found better solution",
                       start_num=start_count,
                       old_objective=old_best,
                       new_objective=global_best_obj,
                       improvement=global_best_obj - old_best)

        # Log progress every 10 starts
        if start_count % 10 == 0:
            logger.debug("Progress update",
                        starts_completed=start_count,
                        time_elapsed=f"{time.time() - start_time:.1f}s",
                        current_best=global_best_obj,
                        improvements=improvements)

    # Create result
    execution_time = time.time() - start_time
    result = AlgorithmResult(
        algorithm_name="RW",
        best_input=global_best_x,
        best_objective=global_best_obj if global_best_obj != -float('inf') else None,
        first_objective=first_best_obj,
        execution_time=execution_time,
        start_count=start_count,
        valid_start_count=valid_start_count,
        update_history=update_history,
        network_config=[config.network.input_size] + [config.network.layer_size] * config.network.layer_num + [1],
        algorithm_params=[config.params.walk_eps, config.params.pick_bias]
    )

    logger.info("Relaxation Walk completed",
               execution_time=f"{execution_time:.2f}s",
               total_starts=start_count,
               improvements=improvements,
               final_objective=result.best_objective)

    # No longer store results here - handled by run_algorithm_with_yaml_config
    return result


def run_dynamic_walk(config: AlgorithmConfiguration) -> AlgorithmResult:
    """
    Run Relaxation Walk Dynamic (RWD) algorithm.

    Args:
        config: Algorithm configuration

    Returns:
        Algorithm execution result
    """
    logger = get_logger("algorithm_runners")
    logger.info("Starting Dynamic Walk algorithm",
               input_size=config.network.input_size,
               layer_num=config.network.layer_num,
               layer_size=config.network.layer_size,
               walk_eps=config.params.walk_eps,
               seed=config.experiment.random_seed,
               time_limit=config.experiment.time_limit)

    # Setup
    np.random.seed(config.experiment.random_seed)
    network = create_network(config)
    start_time = time.time()

    # Solve initial linear relaxation
    with TimingContext(logger, "linear relaxation solution"):
        x_relaxation, frac_activations = solve_linear_relaxation(network, config.experiment.time_limit)

    if x_relaxation is None:
        logger.error("Linear relaxation failed")
        return _create_failed_result("RWD", config, time.time() - start_time)

    logger.debug("Linear relaxation completed successfully")

    # Perform initial relaxation walk
    logger.debug("Starting initial gradient walk")
    with TimingContext(logger, "initial gradient walk"):
        first_best_x, first_best_obj, _, _ = gradient_walk(
            network, x_relaxation, config.params.walk_eps,
            config.experiment.time_limit - (time.time() - start_time)
        )

    logger.info("Initial walk completed", first_objective=first_best_obj)

    # Initialize tracking variables
    global_best_obj = first_best_obj if first_best_obj is not None else -float('inf')
    global_best_x = first_best_x
    start_count = 1
    valid_start_count = 1
    update_history = []

    if first_best_obj is not None:
        update_history.append([
            first_best_obj, first_best_x, start_count, valid_start_count,
            time.time() - start_time
        ])

    # Generate additional starting points dynamically
    logger.debug("Starting dynamic point generation")
    point_generator = generate_dynamic_starting_points(
        network, x_relaxation, frac_activations, config.experiment.time_limit, start_time
    )

    improvements = 0
    for starting_point, _ in point_generator:
        if time.time() - start_time >= config.experiment.time_limit:
            logger.debug("Time limit reached, stopping point generation")
            break

        start_count += 1
        valid_start_count += 1

        # Perform walk from this starting point
        time_remaining = config.experiment.time_limit - (time.time() - start_time)
        if time_remaining <= 0:
            break

        with TimingContext(logger, f"gradient walk {start_count}", start_num=start_count):
            best_x, best_obj, _, _ = gradient_walk(
                network, starting_point, config.params.walk_eps, time_remaining
            )

        # Update global best if improved
        if best_obj is not None and best_obj > global_best_obj:
            improvements += 1
            old_best = global_best_obj
            global_best_obj = best_obj
            global_best_x = best_x
            update_history.append([
                global_best_obj, global_best_x, start_count, valid_start_count,
                time.time() - start_time
            ])
            logger.info("Found better solution",
                       start_num=start_count,
                       old_objective=old_best,
                       new_objective=global_best_obj,
                       improvement=global_best_obj - old_best)

        # Log progress every 10 starts
        if start_count % 10 == 0:
            logger.debug("Progress update",
                        starts_completed=start_count,
                        time_elapsed=f"{time.time() - start_time:.1f}s",
                        current_best=global_best_obj,
                        improvements=improvements)

    # Create result
    execution_time = time.time() - start_time
    result = AlgorithmResult(
        algorithm_name="RWD",
        best_input=global_best_x,
        best_objective=global_best_obj if global_best_obj != -float('inf') else None,
        first_objective=first_best_obj,
        execution_time=execution_time,
        start_count=start_count,
        valid_start_count=valid_start_count,
        update_history=update_history,
        network_config=[config.network.input_size] + [config.network.layer_size] * config.network.layer_num + [1],
        algorithm_params=[config.params.walk_eps]
    )

    logger.info("Dynamic Walk completed",
               execution_time=f"{execution_time:.2f}s",
               total_starts=start_count,
               improvements=improvements,
               final_objective=result.best_objective)

    # No longer store results here - handled by run_algorithm_with_yaml_config
    return result


def _create_failed_result(algorithm_name: str, config: AlgorithmConfiguration, execution_time: float) -> AlgorithmResult:
    """
    Create result object for failed algorithm execution.

    Args:
        algorithm_name: Name of the algorithm
        config: Algorithm configuration
        execution_time: Time spent before failure

    Returns:
        Failed result object
    """
    params = [config.params.walk_eps, config.params.pick_bias] if algorithm_name == "RW" else [config.params.walk_eps]

    return AlgorithmResult(
        algorithm_name=algorithm_name,
        best_input=None,
        best_objective=None,
        first_objective=None,
        execution_time=execution_time,
        start_count=0,
        valid_start_count=0,
        update_history=[],
        network_config=[config.network.input_size] + [config.network.layer_size] * config.network.layer_num + [1],
        algorithm_params=params
    )


def _store_algorithm_result(result: AlgorithmResult, config: AlgorithmConfiguration) -> str:
    """
    Store algorithm result using dual logging system (SQLite + YAML).

    Args:
        result: Algorithm execution result
        config: Algorithm configuration

    Returns:
        Experiment ID
    """
    # Initialize database and exporter
    db = ExperimentDatabase()
    exporter = YAMLResultExporter()

    try:

        # Store to database (comprehensive storage)
        experiment_id = db.store_experiment(config, result)

        # Export to YAML (readable summary)
        yaml_path = exporter.export_experiment(config, result, experiment_id)

        logger = get_logger("algorithm_runners")
        logger.info("Results stored",
                   database="experiments.db",
                   experiment_id=experiment_id,
                   yaml_path=yaml_path)

        return experiment_id

    finally:
        db.close()


def run_algorithm_with_yaml_config(yaml_config: AlgorithmConfiguration) -> AlgorithmResult:
    """
    Run algorithm with YAML configuration (includes both storage formats).

    Args:
        yaml_config: YAML-based algorithm configuration

    Returns:
        Algorithm execution result
    """
    from workflows.storage import ExperimentDatabase, YAMLResultExporter

    # Run the algorithm
    if yaml_config.algorithm.lower() in ["relaxation_walk", "rw"]:
        result = run_relaxation_walk(yaml_config)
    elif yaml_config.algorithm.lower() in ["dynamic_walk", "rwd", "relaxation_walk_dynamic"]:
        result = run_dynamic_walk(yaml_config)
    else:
        raise ValueError(f"Unknown algorithm: {yaml_config.algorithm}")

    # Store results with YAML config (for better metadata)
    db = ExperimentDatabase()
    exporter = YAMLResultExporter()

    try:
        experiment_id = db.store_experiment(yaml_config, result)
        yaml_path = exporter.export_experiment(yaml_config, result, experiment_id)

        logger = get_logger("algorithm_runners")
        logger.info("Results stored",
                   database="experiments.db",
                   experiment_id=experiment_id,
                   yaml_path=yaml_path)

        return result
    finally:
        db.close()


def run_algorithm(algorithm_name: str, config: AlgorithmConfiguration) -> AlgorithmResult:
    """
    Run specified algorithm with given configuration (legacy interface).

    Args:
        algorithm_name: Name of algorithm to run ("relaxation_walk" or "dynamic_walk")
        config: Algorithm configuration

    Returns:
        Algorithm execution result

    Raises:
        ValueError: If algorithm name is not recognized
    """
    if algorithm_name.lower() in ["relaxation_walk", "rw"]:
        return run_relaxation_walk(config)
    elif algorithm_name.lower() in ["dynamic_walk", "rwd", "relaxation_walk_dynamic"]:
        return run_dynamic_walk(config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def run_experiment_batch(configs: List[Tuple[str, AlgorithmConfiguration]]) -> List[AlgorithmResult]:
    """
    Run a batch of experiments with different algorithms and configurations.

    Args:
        configs: List of (algorithm_name, config) tuples

    Returns:
        List of algorithm results
    """
    logger = get_logger("algorithm_runners")
    results = []
    for algorithm_name, config in configs:
        logger.info("Running batch experiment",
                   algorithm=algorithm_name,
                   seed=config.experiment.random_seed)
        result = run_algorithm(algorithm_name, config)
        results.append(result)
        logger.info("Batch experiment completed",
                   algorithm=algorithm_name,
                   execution_time=f"{result.execution_time:.2f}s",
                   best_objective=result.best_objective)

    return results