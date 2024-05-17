#!/usr/bin/env bash

#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home/lane.662/slurm-%j.out
#SBATCH --error=/home/lane.662/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mail-user=lane.662@osu.edu
#SBATCH --mail-type=ALL
#SBATCH --signal=TERM@300

PERRONDIR=/fs/project/thompson.2455/lane.662/salemnums/nums/deg${SLURM_JOB_NAME:0:2}
BETADIR=/fs/project/thompson.2455/lane.662/salemnums/orbits/deg${SLURM_JOB_NAME:0:2}
DOSETUP=True
MAINFILE=~/beta_numbers/scripts/calc_beta_orbits.py
BLKSIZE=25000
MAXORBITLEN=100000000
MAXDPS=500
TIMEOUT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)
UPDATEPERIOD=900
UPDATETIMEOUT=240
SECPERBLOCKUPPERBOUND=240
sage -python $MAINFILE $SLURM_CPUS_PER_TASK $PERRONDIR $BETADIR $DOSETUP $BLKSIZE $MAXORBITLEN $MAXDPS $TIMEOUT $UPDATEPERIOD $UPDATETIMEOUT $SECPERBLOCKUPPERBOUND
