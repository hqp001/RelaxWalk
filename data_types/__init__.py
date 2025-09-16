"""
Type definitions for the RelaxWalk project.

This module exports all type definitions used throughout the project.
"""

from .result_types import AlgorithmResult
from .config_types import (
    NetworkConfig,
    AlgorithmParams,
    ExperimentConfig,
    AlgorithmConfiguration
)
from .Network import Network

__all__ = [
    'AlgorithmResult',
    'NetworkConfig',
    'AlgorithmParams',
    'ExperimentConfig',
    'AlgorithmConfiguration',
    'Network'
]