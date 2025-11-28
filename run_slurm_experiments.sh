#!/bin/bash
#SBATCH -p short # partition (queue)
#SBATCH -N 1 # (leave at 1 unless using multi-node specific code)
#SBATCH -n 4 # number of cores
#SBATCH --mem=163840 # total memory
#SBATCH --job-name="milp_exp" # job name
#SBATCH -o ./log/slurm.%A_%a.stdout.txt # STDOUT (%A=job ID, %a=array task ID)
#SBATCH -e ./log/slurm.%A_%a.stderr.txt # STDERR
#SBATCH --mail-user=username@bucknell.edu # address to email
#SBATCH --mail-type=ALL # mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --array=0-NUM_EXPERIMENTS%10 # REPLACE NUM_EXPERIMENTS with total-1, max 10 running at once

module load gurobi-optimizer/

# Experiment configuration - FILL THESE IN
seed_list=(50)
input_size_list=(10 20 30)
layer_num_list=(2)
layer_size_list=(100)
prune_amount_list=(0.0 0.5 0.9)
time_limit=60

# Convert arrays to comma-separated strings and export as environment variables
export SEED_LIST=$(IFS=,; echo "${seed_list[*]}")
export INPUT_SIZE_LIST=$(IFS=,; echo "${input_size_list[*]}")
export LAYER_NUM_LIST=$(IFS=,; echo "${layer_num_list[*]}")
export LAYER_SIZE_LIST=$(IFS=,; echo "${layer_size_list[*]}")
export PRUNE_AMOUNT_LIST=$(IFS=,; echo "${prune_amount_list[*]}")
export TIME_LIMIT=$time_limit

python run_slurm_experiment.py $SLURM_ARRAY_TASK_ID
