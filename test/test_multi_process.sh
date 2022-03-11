#!/bin/bash

#SBATCH -c 2
#SBATCH --array=0-10
#SBATCH -o ./slurm_outputs/output.%j.out

python -u test_multi_process.py