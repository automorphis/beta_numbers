#!/usr/bin/env bash

#SBATCH --job-name=calcperronnums
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --array=1-10
#SBATCH --mail-user=lane.662@osu.edu
#SBATCH --mail-type=ALL
#SBATCH --error=/home/lane.662/calc_perron_nums_error.txt
#SBATCH --output=/home/lane.662/calc_perron_nums_output.txt

srun sage -python ~/beta_numbers/scripts/temp.py ~/perron_nums $SLURM_ARRAY_TASK_MAX $SLURM_ARRAY_TASK_ID
#srun sage -python ~/beta_numbers/scripts/calc_perron_nums.py ~/perron_nums $SLURM_ARRAY_TASK_MAX $SLURM_ARRAY_TASK_ID 2 5000 3 1000 4 1000 5 1000 6 500 7 500 8 500 9 500 10 100 11 100
