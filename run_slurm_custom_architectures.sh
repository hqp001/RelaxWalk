#!/bin/bash
#SBATCH -p short # partition (queue)
#SBATCH -N 1 # (leave at 1 unless using multi-node specific code)
#SBATCH -n 4 # number of cores
#SBATCH --mem=163840 # total memory
#SBATCH --job-name="milp_custom_arch" # job name
#SBATCH -o ./log/slurm.%A_%a.stdout.txt # STDOUT (%A=job ID, %a=array task ID)
#SBATCH -e ./log/slurm.%A_%a.stderr.txt # STDERR
#SBATCH --mail-user=username@bucknell.edu # address to email
#SBATCH --mail-type=ALL # mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --array=0-119%10 # 10 architectures × 2 seeds × 6 prune_amounts = 120 experiments (0-119), max 10 running at once

module load gurobi-optimizer/

# All configuration parameters are defined in run_slurm_custom_architectures.py
# Total experiments = 10 architectures × 2 seeds × 6 prune_amounts = 120

python run_slurm_custom_architectures.py $SLURM_ARRAY_TASK_ID
