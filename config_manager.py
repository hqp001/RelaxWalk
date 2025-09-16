"""
Configuration manager for YAML-based algorithm configuration.

This module handles loading, validating, and managing YAML configuration files
for the Relax-and-Walk algorithms.
"""

import yaml
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from pathlib import Path
from data_types import (
    NetworkConfig,
    AlgorithmParams,
    ExperimentConfig,
    AlgorithmConfiguration
)


def load_yaml_config(config_path: str) -> AlgorithmConfiguration:
    """
    Load and parse YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Parsed algorithm configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If configuration is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file: {e}")

    return parse_config_dict(raw_config)


def parse_config_dict(config_dict: Dict[str, Any]) -> AlgorithmConfiguration:
    """
    Parse configuration dictionary into structured configuration.

    Args:
        config_dict: Raw configuration dictionary

    Returns:
        Parsed algorithm configuration

    Raises:
        ValueError: If configuration structure is invalid
    """
    try:
        # Parse network configuration
        network_dict = config_dict.get('network', {})
        network_config = NetworkConfig(
            input_size=network_dict['input_size'],
            layer_num=network_dict['layer_num'],
            layer_size=network_dict['layer_size']
        )

        # Parse algorithm parameters
        params_dict = config_dict.get('params', {})
        algorithm_params = AlgorithmParams(
            walk_eps=params_dict['walk_eps'],
            pick_bias=params_dict.get('pick_bias')
        )

        # Parse experiment configuration
        experiment_dict = config_dict.get('experiment', {})
        experiment_config = ExperimentConfig(
            random_seed=experiment_dict['random_seed'],
            time_limit=experiment_dict['time_limit'],
            output_dir=experiment_dict.get('output_dir', 'results')
        )

        # Get algorithm type
        algorithm = config_dict['algorithm']

        return AlgorithmConfiguration(
            algorithm=algorithm,
            network=network_config,
            params=algorithm_params,
            experiment=experiment_config
        )

    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}")
    except Exception as e:
        raise ValueError(f"Invalid configuration format: {e}")


def validate_config(config: AlgorithmConfiguration) -> List[str]:
    """
    Validate algorithm configuration.

    Args:
        config: Configuration to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Validate algorithm type
    valid_algorithms = ['relaxation_walk', 'dynamic_walk', 'rw', 'rwd']
    if config.algorithm not in valid_algorithms:
        errors.append(f"Invalid algorithm '{config.algorithm}'. Must be one of: {valid_algorithms}")

    # Validate network parameters
    if config.network.input_size <= 0:
        errors.append("input_size must be positive")

    if config.network.layer_num <= 0:
        errors.append("layer_num must be positive")

    if config.network.layer_size <= 0:
        errors.append("layer_size must be positive")

    # Validate algorithm parameters
    if not (0 < config.params.walk_eps <= 1):
        errors.append("walk_eps must be between 0 and 1")

    # Validate pick_bias for relaxation_walk
    if config.algorithm in ['relaxation_walk', 'rw']:
        if config.params.pick_bias is None:
            errors.append("pick_bias is required for relaxation_walk algorithm")
        elif config.params.pick_bias < 0:
            errors.append("pick_bias must be non-negative")

    # Validate experiment parameters
    if config.experiment.time_limit <= 0:
        errors.append("time_limit must be positive")

    return errors


def create_example_config(algorithm: str, output_path: str) -> None:
    """
    Create an example YAML configuration file.

    Args:
        algorithm: Algorithm type ('relaxation_walk' or 'dynamic_walk')
        output_path: Path where to save the example config
    """
    if algorithm in ['relaxation_walk', 'rw']:
        example_config = {
            'algorithm': 'relaxation_walk',
            'network': {
                'input_size': 10,
                'layer_num': 2,
                'layer_size': 20
            },
            'params': {
                'walk_eps': 0.1,
                'pick_bias': 0.01
            },
            'experiment': {
                'random_seed': 42,
                'time_limit': 3600,
                'output_dir': 'results'
            }
        }
    else:  # dynamic_walk
        example_config = {
            'algorithm': 'dynamic_walk',
            'network': {
                'input_size': 10,
                'layer_num': 2,
                'layer_size': 20
            },
            'params': {
                'walk_eps': 0.1
            },
            'experiment': {
                'random_seed': 42,
                'time_limit': 3600,
                'output_dir': 'results'
            }
        }

    with open(output_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, indent=2)


def config_to_yaml(config: AlgorithmConfiguration, output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save
        output_path: Path where to save the config
    """
    config_dict = asdict(config)

    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def load_batch_config(config_path: str) -> List[AlgorithmConfiguration]:
    """
    Load batch configuration with multiple experiments.

    Args:
        config_path: Path to batch YAML configuration file

    Returns:
        List of algorithm configurations
    """
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    if 'experiments' not in raw_config:
        raise ValueError("Batch configuration must have 'experiments' key")

    configs = []
    for exp_config in raw_config['experiments']:
        config = parse_config_dict(exp_config)
        configs.append(config)

    return configs