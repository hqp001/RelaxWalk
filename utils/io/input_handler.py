"""
YAML configuration input handler for RelaxWalk experiments.
"""

import yaml
import hashlib
import json
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Loads and validates YAML experiment configurations."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = None
        self._load_config()

    def _load_config(self):
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._validate_config()

    def _validate_config(self):
        """Basic validation of required configuration fields."""
        required_sections = ['experiment', 'network', 'algorithm']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate experiment section
        exp_required = ['name', 'method']
        for field in exp_required:
            if field not in self.config['experiment']:
                raise ValueError(f"Missing required field in experiment section: {field}")

        # Validate network section
        net_required = ['input_size', 'layer_num', 'layer_size']
        for field in net_required:
            if field not in self.config['network']:
                raise ValueError(f"Missing required field in network section: {field}")

        # Validate pruned_density if present
        if 'pruned_density' in self.config['network']:
            density = self.config['network']['pruned_density']
            if not isinstance(density, (int, float)) or density < 0.0 or density > 1.0:
                raise ValueError("pruned_density must be a number between 0.0 and 1.0")

        # Validate algorithm section
        alg_required = ['walk_eps', 'pick_bias', 'random_seed', 'timelimit']
        for field in alg_required:
            if field not in self.config['algorithm']:
                raise ValueError(f"Missing required field in algorithm section: {field}")

    def get_config_hash(self) -> str:
        """Generate a unique hash for the configuration."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def get_experiment_name(self) -> str:
        """Get experiment name."""
        return self.config['experiment']['name']

    def get_method(self) -> str:
        """Get experiment method."""
        return self.config['experiment']['method']

    def get_network_config(self) -> Dict[str, Any]:
        """Get network configuration parameters."""
        return self.config['network']

    def get_algorithm_config(self) -> Dict[str, Any]:
        """Get algorithm configuration parameters."""
        return self.config['algorithm']

    def get_relaxation_walk_params(self) -> tuple:
        """Get parameters in the format expected by relaxation_walk_deep function."""
        net_config = self.get_network_config()
        alg_config = self.get_algorithm_config()

        return (
            net_config['input_size'],
            net_config['layer_num'],
            net_config['layer_size'],
            alg_config['random_seed'],
            alg_config['walk_eps'],
            alg_config['pick_bias'],
            alg_config['timelimit'],
            net_config.get('pruned_density', 1.0)  # Default to no pruning
        )


def load_config(config_path: str) -> ConfigLoader:
    """Load configuration from YAML file."""
    return ConfigLoader(config_path)