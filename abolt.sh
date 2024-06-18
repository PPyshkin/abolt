#!/bin/bash

#SBATCH -J abolt
#SBATCH -c 2
#SBATCH -n 1
#SBATCH -p long-fat
#SBATCH --mail-type=ALL
#SBATCH -t 3-12:00
#SBATCH --mem-per-cpu=30000

# Set Array Job Options
#SBATCH --output=out/%x__%a.out
#SBATCH --error=out/%x__%a.err

# Define number of subjobs
#SBATCH --array=1-11

echo "Running Task: ${SLURM_ARRAY_TASK_ID} on host $(hostname)"
python3 abolt.py ${SLURM_ARRAY_TASK_ID}
