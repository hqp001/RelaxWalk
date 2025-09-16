"""
Result data types for algorithm execution.

This module contains shared data structures to avoid circular imports.
"""

from typing import List, Optional, Any
from dataclasses import dataclass


@dataclass
class AlgorithmResult:
    """Result from algorithm execution."""
    algorithm_name: str
    best_input: Optional[List[float]]
    best_objective: Optional[float]
    first_objective: Optional[float]
    execution_time: float
    start_count: int
    valid_start_count: int
    update_history: List[List[Any]]
    network_config: List[int]
    algorithm_params: List[float]