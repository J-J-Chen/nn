#!/bin/bash

SBATCH --job-name=$1
SBATCH -n 18 # Number of cores requested
SBATCH -N 1 # Ensure that all cores are on one machine
SBATCH -t 0-01:00 # Runtime in minutes
SBATCH -p kaxiras_gpu,kaxiras,shared # Partition to submit to
SBATCH --mem-per-cpu=4000 # Memory per node in MB (see also --mem-per-cpu)
SBATCH --mem=256000
SBATCH --open-mode=append # Append when writing files
SBATCH -o hostname_%j.out # Standard out goes to this file
SBATCH -e hostname_%j.err # Standard err goes to this filehostname
SBATCH --mail-type=all
SBATCH --mail-user=john_chen@college.harvard.edu

python $1
