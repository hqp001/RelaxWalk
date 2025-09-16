"""
Configuration data types for algorithm execution.

This module contains data structures for algorithm configuration.
"""

from typing import Optional
from dataclasses import dataclass



@dataclass
class NetworkConfig:
    """Neural network configuration parameters."""
    input_size: int
    layer_num: int
    layer_size: int


@dataclass
class AlgorithmParams:
    """Algorithm-specific parameters."""
    walk_eps: float
    pick_bias: Optional[float] = None  # Only used for RW algorithm


@dataclass
class ExperimentConfig:
    """Experiment execution configuration."""
    random_seed: int
    time_limit: float
    output_dir: Optional[str] = "results"


@dataclass
class AlgorithmConfiguration:
    """Complete algorithm configuration."""
    algorithm: str  # "relaxation_walk" or "dynamic_walk"
    network: NetworkConfig
    params: AlgorithmParams
    experiment: ExperimentConfig

