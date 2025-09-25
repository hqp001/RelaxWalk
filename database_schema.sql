-- Normalized database schema for RelaxWalk experiments
-- This replaces the current JSON-based storage with proper relational tables

-- Experiments table - stores basic experiment metadata
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    method TEXT NOT NULL,
    config_hash TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Network configurations table - normalized network parameters
CREATE TABLE network_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    input_size INTEGER NOT NULL,
    layer_num INTEGER NOT NULL,
    layer_size INTEGER NOT NULL,
    pruned_density REAL NOT NULL DEFAULT 1.0,
    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
);

-- Algorithm configurations table - normalized algorithm parameters
CREATE TABLE algorithm_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    walk_eps REAL NOT NULL,
    pick_bias REAL NOT NULL,
    random_seed INTEGER NOT NULL,
    timelimit INTEGER NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
);

-- Results table - stores main result data
CREATE TABLE results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    x_max TEXT, -- Still JSON since it's a variable-length array
    max_value REAL,
    first_max REAL,
    time_count REAL,
    start_count INTEGER,
    valid_start_count INTEGER,
    original_model_max REAL, -- Maximum value found by original unpruned model
    original_model_x_max TEXT, -- Input that achieved the maximum on original model (JSON)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
);

-- Update list table - separate table for tracking optimization progress
CREATE TABLE update_list (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    experiment_id INTEGER NOT NULL,
    max_value REAL NOT NULL,
    x_max TEXT, -- JSON array of the solution at this update
    start_count INTEGER NOT NULL,
    valid_start_count INTEGER NOT NULL,
    timestamp REAL NOT NULL, -- Time elapsed when this update occurred
    sequence_order INTEGER NOT NULL, -- Order of this update in the sequence
    FOREIGN KEY (result_id) REFERENCES results (id),
    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
);

-- Indexes for better query performance
CREATE INDEX idx_experiments_config_hash ON experiments (config_hash);
CREATE INDEX idx_results_experiment_id ON results (experiment_id);
CREATE INDEX idx_update_list_result_id ON update_list (result_id);
CREATE INDEX idx_update_list_experiment_id ON update_list (experiment_id);
CREATE INDEX idx_network_configs_experiment_id ON network_configs (experiment_id);
CREATE INDEX idx_algorithm_configs_experiment_id ON algorithm_configs (experiment_id);