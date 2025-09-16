"""
Point generation workflows.

This module provides different strategies for generating starting points.
"""

from .systematic_point_generator import generate_systematic_starting_points
from .dynamic_point_generator import generate_dynamic_starting_points

__all__ = [
    'generate_systematic_starting_points',
    'generate_dynamic_starting_points'
]