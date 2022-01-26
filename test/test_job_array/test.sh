#!/bin/bash
#
#SBATCH --array=0-1
#SBATCH -c 1
#SBATCH -o ./slurm_outputs/output.%j.out

echo \$SLURM_ARRAY_TASK_ID 
echo "Sleeping..."
sleep 30
echo "Done"