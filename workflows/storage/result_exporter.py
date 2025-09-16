"""
YAML Result Exporter for Readable Experiment Outputs

This module creates clean, human-readable YAML files containing
core configuration and results for each experiment.
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from data_types import AlgorithmConfiguration, AlgorithmResult


class YAMLResultExporter:
    """Exports experiment results to clean YAML format."""

    def __init__(self, output_base_dir: str = "results"):
        """
        Initialize YAML exporter.

        Args:
            output_base_dir: Base directory for YAML outputs
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)

    def export_experiment(self,
                         config: AlgorithmConfiguration,
                         result: AlgorithmResult,
                         experiment_id: str) -> str:
        """
        Export experiment to YAML file.

        Args:
            config: Algorithm configuration
            result: Algorithm execution result
            experiment_id: Unique experiment identifier

        Returns:
            Path to created YAML file
        """
        # Create algorithm-specific subdirectory
        algorithm_dir = self.output_base_dir / config.algorithm
        algorithm_dir.mkdir(exist_ok=True)

        # Generate YAML data structure
        yaml_data = self._create_yaml_structure(config, result, experiment_id)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_id}_{timestamp}.yaml"
        output_path = algorithm_dir / filename

        # Write YAML file
        with open(output_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2, sort_keys=False)

        return str(output_path)

    def _create_yaml_structure(self,
                             config: AlgorithmConfiguration,
                             result: AlgorithmResult,
                             experiment_id: str) -> Dict[str, Any]:
        """
        Create structured YAML data.

        Args:
            config: Algorithm configuration
            result: Algorithm execution result
            experiment_id: Unique experiment identifier

        Returns:
            Dictionary ready for YAML export
        """
        # Basic experiment metadata
        yaml_data = {
            'experiment': {
                'id': experiment_id,
                'timestamp': datetime.now().isoformat(),
                'algorithm': config.algorithm,
                'status': 'completed' if result.best_objective is not None else 'failed'
            },

            # Configuration section
            'configuration': {
                'network': {
                    'input_size': config.network.input_size,
                    'layer_num': config.network.layer_num,
                    'layer_size': config.network.layer_size,
                    'architecture': f"{config.network.input_size}→{config.network.layer_size}×{config.network.layer_num}→1"
                },
                'algorithm_params': {
                    'walk_eps': config.params.walk_eps,
                },
                'experiment_params': {
                    'random_seed': config.experiment.random_seed,
                    'time_limit': config.experiment.time_limit
                }
            },

            # Core results
            'results': {
                'execution_time': result.execution_time,
                'best_objective': result.best_objective,
                'first_objective': result.first_objective,
                'improvement': None,
                'exploration': {
                    'total_starts': result.start_count,
                    'valid_starts': result.valid_start_count,
                    'efficiency': None
                }
            },

            # Performance metrics
            'performance': {
                'convergence_steps': len(result.update_history),
                'time_per_start': None,
                'starts_per_second': None
            }
        }

        # Add algorithm-specific parameters
        if config.params.pick_bias is not None:
            yaml_data['configuration']['algorithm_params']['pick_bias'] = config.params.pick_bias

        # Calculate derived metrics
        if result.best_objective is not None and result.first_objective is not None:
            improvement = result.best_objective - result.first_objective
            yaml_data['results']['improvement'] = improvement
            yaml_data['results']['improvement_percent'] = (improvement / abs(result.first_objective)) * 100 if result.first_objective != 0 else 0

        if result.valid_start_count > 0:
            yaml_data['results']['exploration']['efficiency'] = result.valid_start_count / result.start_count

        if result.execution_time > 0:
            yaml_data['performance']['time_per_start'] = result.execution_time / result.start_count if result.start_count > 0 else 0
            yaml_data['performance']['starts_per_second'] = result.start_count / result.execution_time

        # Add best input (first 10 values for readability)
        if result.best_input:
            if len(result.best_input) <= 10:
                yaml_data['results']['best_input'] = result.best_input
            else:
                yaml_data['results']['best_input'] = {
                    'preview': result.best_input[:10],
                    'total_length': len(result.best_input),
                    'note': 'Showing first 10 values. Full input stored in database.'
                }

        # Add convergence summary
        if result.update_history:
            convergence_summary = self._create_convergence_summary(result.update_history)
            yaml_data['convergence'] = convergence_summary

        # Add environment info
        yaml_data['environment'] = self._get_environment_info()

        return yaml_data

    def _create_convergence_summary(self, update_history: List[List[Any]]) -> Dict[str, Any]:
        """
        Create convergence summary from update history.

        Args:
            update_history: List of updates [objective, input, start_count, valid_start_count, timestamp]

        Returns:
            Convergence summary dictionary
        """
        if not update_history:
            return {}

        objectives = [update[0] for update in update_history]
        timestamps = [update[4] for update in update_history]

        summary = {
            'total_improvements': len(update_history),
            'initial_objective': objectives[0],
            'final_objective': objectives[-1],
            'best_objective': max(objectives),
            'convergence_timeline': []
        }

        # Add key convergence points
        for i, update in enumerate(update_history[:5]):  # First 5 improvements
            summary['convergence_timeline'].append({
                'step': i + 1,
                'objective': update[0],
                'time': update[4],
                'cumulative_starts': update[2]
            })

        # Add final point if more than 5 improvements
        if len(update_history) > 5:
            final = update_history[-1]
            summary['convergence_timeline'].append({
                'step': len(update_history),
                'objective': final[0],
                'time': final[4],
                'cumulative_starts': final[2],
                'note': 'Final improvement'
            })

        return summary

    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        import sys
        import socket
        import platform

        env_info = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': platform.system(),
            'hostname': socket.gethostname()
        }

        # Try to get package versions
        try:
            import torch
            env_info['pytorch_version'] = torch.__version__
        except ImportError:
            pass

        try:
            import gurobipy as gp
            env_info['gurobi_version'] = f"{gp.GRB.VERSION_MAJOR}.{gp.GRB.VERSION_MINOR}.{gp.GRB.VERSION_TECHNICAL}"
        except ImportError:
            pass

        try:
            import numpy as np
            env_info['numpy_version'] = np.__version__
        except ImportError:
            pass

        return env_info

    def export_summary_report(self, experiment_ids: List[str], output_file: str = None) -> str:
        """
        Create summary report for multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to include
            output_file: Output file path (optional)

        Returns:
            Path to created summary file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_base_dir / f"experiment_summary_{timestamp}.yaml")

        # This would require database integration to fetch experiment data
        # For now, placeholder structure
        summary_data = {
            'summary': {
                'generated_at': datetime.now().isoformat(),
                'total_experiments': len(experiment_ids),
                'experiment_ids': experiment_ids
            },
            'note': 'Detailed summary generation requires database integration'
        }

        with open(output_file, 'w') as f:
            yaml.dump(summary_data, f, default_flow_style=False, indent=2)

        return output_file

    def create_quick_summary(self, config: AlgorithmConfiguration, result: AlgorithmResult) -> Dict[str, Any]:
        """
        Create a quick summary dictionary (without file export).

        Args:
            config: Algorithm configuration
            result: Algorithm execution result

        Returns:
            Summary dictionary
        """
        return {
            'algorithm': config.algorithm,
            'network_size': f"{config.network.input_size}-{config.network.layer_size}x{config.network.layer_num}-1",
            'best_objective': result.best_objective,
            'execution_time': result.execution_time,
            'convergence_steps': len(result.update_history),
            'exploration_efficiency': result.valid_start_count / result.start_count if result.start_count > 0 else 0
        }