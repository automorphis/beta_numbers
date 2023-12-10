#!/usr/bin/env bash

#SBATCH --job-name=CalcBetaOrbits
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --error=/home/lane.662/betaorbitserror.txt
#SBATCH --output=/home/lane.662/betaorbitsoutput.txt
#SBATCH --mail-user=lane.662@osu.edu
#SBATCH --mail-type=ALL
#SBATCH --signal=TERM@300

PERRONDIR=/fs/project/thompson.2455/lane.662/perronnums
BETADIR=/fs/project/thompson.2455/lane.662/betaorbits
DOSETUP=True
MAINFILE=~/beta_numbers/scripts/calc_beta_orbits.py
BLKSIZE=25000
MAXORBITLEN=1000000
MAXDPS=500
TIMEOUT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)
UPDATEPERIOD=900
UPDATETIMEOUT=240
SECPERBLOCKUPPERBOUND=240
srun sage -python $MAINFILE $SLURM_CPUS_PER_TASK $PERRONDIR $BETADIR $DOSETUP $BLKSIZE $MAXORBITLEN $MAXDPS $TIMEOUT $UPDATEPERIOD $UPDATETIMEOUT $SECPERBLOCKUPPERBOUND
