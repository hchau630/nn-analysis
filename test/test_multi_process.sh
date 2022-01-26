#!/bin/bash

#SBATCH -c 2
#SBATCH --array=0-8
#SBATCH -o ./slurm_outputs/output.%j.out

ls /mnt/smb/locker/issa-locker/
python -u test_multi_process.py