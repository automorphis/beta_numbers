#!/usr/bin/env bash

#SBATCH --job-name=CalcPerronNums
#SBATCH --time=24:00:00
#SBATCH --ntasks=24
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --error=/home/lane.662/perronnumserror.txt
#SBATCH --mail-user=lane.662@osu.edu
#SBATCH --mail-type=ALL

DOSETUP=True
SAVEDIR=/fs/project/thompson.2455/lane.662/perronnums
MAINFILE=/home/lane.662/beta_numbers/scripts/calc_perron_nums.py
BLKSIZE=10000
DPS=500
TIMEOUT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)
UPDATEPERIOD=900
UPDATETIMEOUT=240
SECPERBLOCKUPPERBOUND=240
sage -python $MAINFILE $DOSETUP $SLURM_NTASKS $SAVEDIR $BLKSIZE $DPS $TIMEOUT $UPDATEPERIOD $UPDATETIMEOUT $SECPERBLOCKUPPERBOUND 2 4096 3 2048 4 1024 5 512 6 256 7 128 8 64 9 32 10 16 11 8 12 4
