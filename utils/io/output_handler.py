"""
SQLite output handler for RelaxWalk experiment results.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Any, Dict
from pathlib import Path


class ExperimentDatabase:
    """Handles SQLite database operations for experiment results."""

    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Check if database file exists."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}. Please run: sqlite3 {self.db_path} < database_schema.sql")

    def store_experiment_config(self, name: str, method: str, config_hash: str,
                                network_config: Dict, algorithm_config: Dict) -> int:
        """Store experiment configuration and return experiment ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if configuration already exists
            cursor.execute("SELECT id FROM experiments WHERE config_hash = ?", (config_hash,))
            result = cursor.fetchone()

            if result:
                return result[0]  # Return existing experiment ID

            # Insert new experiment configuration
            cursor.execute("""
                INSERT INTO experiments (name, method, config_hash)
                VALUES (?, ?, ?)
            """, (name, method, config_hash))

            experiment_id = cursor.lastrowid

            # Insert network configuration
            cursor.execute("""
                INSERT INTO network_configs (experiment_id, input_size, layer_num, layer_size, pruned_density)
                VALUES (?, ?, ?, ?, ?)
            """, (experiment_id, network_config['input_size'], network_config['layer_num'],
                  network_config['layer_size'], network_config.get('pruned_density', 1.0)))

            # Insert algorithm configuration
            cursor.execute("""
                INSERT INTO algorithm_configs (experiment_id, walk_eps, pick_bias, random_seed, timelimit)
                VALUES (?, ?, ?, ?, ?)
            """, (experiment_id, algorithm_config['walk_eps'], algorithm_config['pick_bias'],
                  algorithm_config['random_seed'], algorithm_config['timelimit']))

            conn.commit()
            return experiment_id

    def store_result(self, experiment_id: int, x_max: Optional[List], max_value: Optional[float],
                     first_max: Optional[float], time_count: float, start_count: int,
                     valid_start_count: int, update_list: List, original_model=None) -> int:
        """Store experiment result and return result ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get original model results if available
            original_max = original_model._max if original_model else None
            original_x_max = json.dumps(original_model._x_max) if original_model and original_model._x_max else None

            cursor.execute("""
                INSERT INTO results (experiment_id, x_max, max_value, first_max, time_count,
                                   start_count, valid_start_count, original_model_max, original_model_x_max)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                json.dumps(x_max) if x_max else None,
                max_value,
                first_max,
                time_count,
                start_count,
                valid_start_count,
                original_max,
                original_x_max
            ))

            result_id = cursor.lastrowid

            # Store update_list entries in separate table
            for sequence_order, update_entry in enumerate(update_list):
                max_val, x_val, start_cnt, valid_start_cnt, timestamp = update_entry
                cursor.execute("""
                    INSERT INTO update_list (result_id, experiment_id, max_value, x_max,
                                           start_count, valid_start_count, timestamp, sequence_order)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result_id,
                    experiment_id,
                    max_val,
                    json.dumps(x_val) if x_val else None,
                    start_cnt,
                    valid_start_cnt,
                    timestamp,
                    sequence_order
                ))

            conn.commit()
            return result_id

    def get_experiment_results(self, experiment_id: int) -> List[Dict]:
        """Get all results for a specific experiment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get results with experiment info and configs
            cursor.execute("""
                SELECT r.*, e.name, e.method,
                       nc.input_size, nc.layer_num, nc.layer_size,
                       ac.walk_eps, ac.pick_bias, ac.random_seed, ac.timelimit
                FROM results r
                JOIN experiments e ON r.experiment_id = e.id
                JOIN network_configs nc ON e.id = nc.experiment_id
                JOIN algorithm_configs ac ON e.id = ac.experiment_id
                WHERE r.experiment_id = ?
                ORDER BY r.created_at DESC
            """, (experiment_id,))

            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            results = []
            for row in rows:
                result = dict(zip(columns, row))

                # Parse x_max JSON
                if result['x_max']:
                    result['x_max'] = json.loads(result['x_max'])

                # Get update_list for this result
                cursor.execute("""
                    SELECT max_value, x_max, start_count, valid_start_count, timestamp
                    FROM update_list
                    WHERE result_id = ?
                    ORDER BY sequence_order
                """, (result['id'],))

                update_rows = cursor.fetchall()
                result['update_list'] = []
                for update_row in update_rows:
                    max_val, x_val, start_cnt, valid_start_cnt, timestamp = update_row
                    x_val_parsed = json.loads(x_val) if x_val else None
                    result['update_list'].append([max_val, x_val_parsed, start_cnt, valid_start_cnt, timestamp])

                # Reconstruct config dictionaries for backward compatibility
                result['network_config'] = {
                    'input_size': result['input_size'],
                    'layer_num': result['layer_num'],
                    'layer_size': result['layer_size']
                }
                result['algorithm_config'] = {
                    'walk_eps': result['walk_eps'],
                    'pick_bias': result['pick_bias'],
                    'random_seed': result['random_seed'],
                    'timelimit': result['timelimit']
                }

                results.append(result)

            return results

    def get_all_experiments(self) -> List[Dict]:
        """Get all experiments with their latest results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT e.*, COUNT(r.id) as result_count,
                       nc.input_size, nc.layer_num, nc.layer_size,
                       ac.walk_eps, ac.pick_bias, ac.random_seed, ac.timelimit
                FROM experiments e
                LEFT JOIN results r ON e.id = r.experiment_id
                LEFT JOIN network_configs nc ON e.id = nc.experiment_id
                LEFT JOIN algorithm_configs ac ON e.id = ac.experiment_id
                GROUP BY e.id
                ORDER BY e.created_at DESC
            """)

            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            experiments = []
            for row in rows:
                exp = dict(zip(columns, row))

                # Reconstruct config dictionaries for backward compatibility
                exp['network_config'] = {
                    'input_size': exp['input_size'],
                    'layer_num': exp['layer_num'],
                    'layer_size': exp['layer_size']
                }
                exp['algorithm_config'] = {
                    'walk_eps': exp['walk_eps'],
                    'pick_bias': exp['pick_bias'],
                    'random_seed': exp['random_seed'],
                    'timelimit': exp['timelimit']
                }

                experiments.append(exp)

            return experiments


def store_experiment_result(config_loader, result_data: tuple, db_path: str = "experiments.db", original_model=None) -> int:
    """
    Store experiment result using configuration and result data.

    Args:
        config_loader: ConfigLoader instance with experiment configuration
        result_data: Tuple from relaxation_walk_deep function
                    (x_max, max_, first_max, time_count, start_count, valid_start_count, update_list)
        db_path: Path to SQLite database file

    Returns:
        Result ID from database
    """
    db = ExperimentDatabase(db_path)

    # Store experiment configuration
    experiment_id = db.store_experiment_config(
        name=config_loader.get_experiment_name(),
        method=config_loader.get_method(),
        config_hash=config_loader.get_config_hash(),
        network_config=config_loader.get_network_config(),
        algorithm_config=config_loader.get_algorithm_config()
    )

    # Unpack result data
    x_max, max_value, first_max, time_count, start_count, valid_start_count, update_list = result_data

    # Store result
    result_id = db.store_result(
        experiment_id=experiment_id,
        x_max=x_max,
        max_value=max_value,
        first_max=first_max,
        time_count=time_count,
        start_count=start_count,
        valid_start_count=valid_start_count,
        update_list=update_list,
        original_model=original_model
    )

    return result_id