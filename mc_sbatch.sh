#!/bin/bash

#SBATCH -J Cosmic_A
#SBATCH -p day
#SBATCH -c 1
#SBATCH -t 24:00:00
#SBATCH --array 0-9

module purge

module load miniconda
conda activate Cosmic_Anisotropy

steps=$1
dataset=$2
runnumber=$SLURM_ARRAY_TASK_ID
cont=$3

python SingleMCMCRun.py $steps $runnumber $dataset $cont
