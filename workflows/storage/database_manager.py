"""
SQLite Database Manager for Experiment Storage

This module handles all database operations for storing comprehensive
experiment data including configurations, results, and convergence history.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from data_types import AlgorithmConfiguration, AlgorithmResult


class ExperimentDatabase:
    """Database manager for experiment storage and retrieval."""

    def __init__(self, db_path: str = "experiments.db"):
        """
        Initialize database connection and create tables.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Main experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT UNIQUE NOT NULL,
                timestamp DATETIME NOT NULL,
                algorithm TEXT NOT NULL,

                -- Network configuration
                input_size INTEGER NOT NULL,
                layer_num INTEGER NOT NULL,
                layer_size INTEGER NOT NULL,

                -- Algorithm parameters
                walk_eps REAL NOT NULL,
                pick_bias REAL,

                -- Experiment settings
                random_seed INTEGER NOT NULL,
                time_limit REAL NOT NULL,
                output_dir TEXT,

                -- Results
                execution_time REAL,
                best_objective REAL,
                first_objective REAL,
                start_count INTEGER,
                valid_start_count INTEGER,

                -- Full configuration as JSON for reference
                config_json TEXT NOT NULL,

                -- Environment info
                python_version TEXT,
                gurobi_version TEXT,
                hostname TEXT,

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Convergence history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS convergence_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                objective REAL NOT NULL,
                start_count INTEGER NOT NULL,
                valid_start_count INTEGER NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
                UNIQUE(experiment_id, step_number)
            )
        """)

        # Best input values table (to handle variable-length arrays)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS best_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                input_index INTEGER NOT NULL,
                input_value REAL NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
                UNIQUE(experiment_id, input_index)
            )
        """)

        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_algorithm ON experiments (algorithm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON experiments (timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_best_objective ON experiments (best_objective)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_random_seed ON experiments (random_seed)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_network_size ON experiments (input_size, layer_num, layer_size)")

        self.conn.commit()

    def store_experiment(self, config: AlgorithmConfiguration, result: AlgorithmResult) -> str:
        """
        Store complete experiment data to database.

        Args:
            config: Algorithm configuration
            result: Algorithm execution result

        Returns:
            Unique experiment ID
        """
        experiment_id = f"{config.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        cursor = self.conn.cursor()

        # Get environment info
        import sys
        import socket
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Try to get Gurobi version
        gurobi_version = None
        try:
            import gurobipy as gp
            gurobi_version = f"{gp.GRB.VERSION_MAJOR}.{gp.GRB.VERSION_MINOR}.{gp.GRB.VERSION_TECHNICAL}"
        except:
            pass

        hostname = socket.gethostname()

        # Insert main experiment record
        cursor.execute("""
            INSERT INTO experiments (
                experiment_id, timestamp, algorithm,
                input_size, layer_num, layer_size,
                walk_eps, pick_bias,
                random_seed, time_limit, output_dir,
                execution_time, best_objective, first_objective,
                start_count, valid_start_count,
                config_json,
                python_version, gurobi_version, hostname
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            datetime.now().isoformat(),
            config.algorithm,
            config.network.input_size,
            config.network.layer_num,
            config.network.layer_size,
            config.params.walk_eps,
            config.params.pick_bias,
            config.experiment.random_seed,
            config.experiment.time_limit,
            config.experiment.output_dir,
            result.execution_time,
            result.best_objective,
            result.first_objective,
            result.start_count,
            result.valid_start_count,
            json.dumps({
                "algorithm": config.algorithm,
                "network": {
                    "input_size": config.network.input_size,
                    "layer_num": config.network.layer_num,
                    "layer_size": config.network.layer_size
                },
                "params": {
                    "walk_eps": config.params.walk_eps,
                    "pick_bias": config.params.pick_bias
                },
                "experiment": {
                    "random_seed": config.experiment.random_seed,
                    "time_limit": config.experiment.time_limit,
                    "output_dir": config.experiment.output_dir
                }
            }),
            python_version,
            gurobi_version,
            hostname
        ))

        # Store convergence history
        if result.update_history:
            for i, update in enumerate(result.update_history):
                # update format: [objective, input, start_count, valid_start_count, timestamp]
                cursor.execute("""
                    INSERT INTO convergence_steps (
                        experiment_id, step_number, timestamp, objective, start_count, valid_start_count
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id,
                    i + 1,
                    update[4],  # timestamp
                    update[0],  # objective
                    update[2],  # start_count
                    update[3]   # valid_start_count
                ))

        # Store best input values
        if result.best_input:
            for i, value in enumerate(result.best_input):
                cursor.execute("""
                    INSERT INTO best_inputs (experiment_id, input_index, input_value)
                    VALUES (?, ?, ?)
                """, (experiment_id, i, value))

        self.conn.commit()
        return experiment_id

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete experiment data by ID.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            Dictionary with experiment data or None if not found
        """
        cursor = self.conn.cursor()

        # Get main experiment data
        cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))
        exp_row = cursor.fetchone()

        if not exp_row:
            return None

        # Convert row to dictionary
        experiment = dict(exp_row)

        # Get convergence history
        cursor.execute("""
            SELECT * FROM convergence_steps
            WHERE experiment_id = ?
            ORDER BY step_number
        """, (experiment_id,))
        convergence_steps = [dict(row) for row in cursor.fetchall()]
        experiment['convergence_history'] = convergence_steps

        # Get best input values
        cursor.execute("""
            SELECT input_index, input_value FROM best_inputs
            WHERE experiment_id = ?
            ORDER BY input_index
        """, (experiment_id,))
        input_rows = cursor.fetchall()
        if input_rows:
            experiment['best_input'] = [row['input_value'] for row in input_rows]
        else:
            experiment['best_input'] = None

        # Parse config JSON
        if experiment['config_json']:
            experiment['config'] = json.loads(experiment['config_json'])

        return experiment

    def query_experiments(self,
                         algorithm: Optional[str] = None,
                         min_objective: Optional[float] = None,
                         max_objective: Optional[float] = None,
                         time_limit: Optional[float] = None,
                         random_seed: Optional[int] = None,
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query experiments with filters.

        Args:
            algorithm: Filter by algorithm name
            min_objective: Minimum best objective value
            max_objective: Maximum best objective value
            time_limit: Filter by time limit
            random_seed: Filter by random seed
            limit: Maximum number of results

        Returns:
            List of experiment dictionaries
        """
        cursor = self.conn.cursor()

        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if algorithm:
            query += " AND algorithm = ?"
            params.append(algorithm)

        if min_objective is not None:
            query += " AND best_objective >= ?"
            params.append(min_objective)

        if max_objective is not None:
            query += " AND best_objective <= ?"
            params.append(max_objective)

        if time_limit is not None:
            query += " AND time_limit = ?"
            params.append(time_limit)

        if random_seed is not None:
            query += " AND random_seed = ?"
            params.append(random_seed)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_algorithm_stats(self, algorithm: str) -> Dict[str, Any]:
        """
        Get summary statistics for an algorithm.

        Args:
            algorithm: Algorithm name

        Returns:
            Dictionary with statistics
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total_experiments,
                AVG(best_objective) as avg_best_objective,
                MAX(best_objective) as max_best_objective,
                MIN(best_objective) as min_best_objective,
                AVG(execution_time) as avg_execution_time,
                AVG(start_count) as avg_start_count,
                AVG(valid_start_count) as avg_valid_start_count
            FROM experiments
            WHERE algorithm = ?
        """, (algorithm,))

        return dict(cursor.fetchone())

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()