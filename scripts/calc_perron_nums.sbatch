#!/usr/bin/env bash

#SBATCH --job-name=CalcPerronNums
#SBATCH --time=08:00:00
#SBATCH --ntasks=24
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --error=~/perronnumserror.txt
#SBATCH --mail-user=lane.662@osu.edu
#SBATCH --mail-type=ALL

SAVEDIR = ~/perronnums
MAINFILE = ~/beta_numbers/scripts/calc_perron_nums.py
BLKSIZE = 10000
TIME=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)
sage -python $MAINFILE $SLURM_NTASKS $SAVEDIR $BLKSIZE $TIME 2 11 3 9 4 7 5 5 6 3
