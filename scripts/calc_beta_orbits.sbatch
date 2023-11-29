#!/usr/bin/env bash

#SBATCH --job-name=CalcBetaOrbits
#SBATCH --time=24:00:00
#SBATCH --ntasks=24
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --error=/home/lane.662/betaorbitserror.txt
#SBATCH --output=/home/lane.662/betaorbitsoutput.txt
#SBATCH --mail-user=lane.662@osu.edu
#SBATCH --mail-type=ALL

PERRONDIR=/fs/project/thompson.2455/lane.662/perronnums
BETADIR=/fs/project/thompson.2455/lane.662/betaorbits
DOSETUP=False
MAINFILE=~/beta_numbers/scripts/calc_beta_orbits.py
BLKSIZE=25000
MAXORBITLEN=1000000
MAXDPS=500
TIMEOUT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)
UPDATEPERIOD=900
UPDATETIMEOUT=240
SECPERBLOCKUPPERBOUND=240
sage -python $MAINFILE $SLURM_NTASKS $PERRONDIR $BETADIR $DOSETUP $BLKSIZE $MAXORBITLEN $MAXDPS     $TIMEOUT $UPDATEPERIOD $UPDATETIMEOUT $SECPERBLOCKUPPERBOUND 2 4096 3 2048 4 1024 5 512 6 256 7 128 8 64 9 32 10 16 11 8 12 4
