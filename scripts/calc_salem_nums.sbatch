#!/usr/bin/env bash

#SBATCH --job-name=08CalcSalemNums
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mail-user=lane.662@osu.edu
#SBATCH --mail-type=ALL

DOSETUP=True
SAVEDIR=/fs/project/thompson.2455/lane.662/salemnums/deg${SLURM_JOB_NAME:0:2}
MAINFILE=/home/lane.662/beta_numbers/scripts/calc_salem_nums.py
BLKSIZE=10000
DPS=500
TIMEOUT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)
UPDATEPERIOD=900
UPDATETIMEOUT=240
SECPERBLOCKUPPERBOUND=240
sage -python $MAINFILE $DOSETUP $SLURM_NTASKS $SAVEDIR $BLKSIZE $DPS $TIMEOUT $UPDATEPERIOD $UPDATETIMEOUT $SECPERBLOCKUPPERBOUND ${SLURM_JOB_NAME:0:2} 4096
