#!/bin/bash

python $(dirname $0)/py_scripts/check_consistent.py # check that x.pkl files are consistent before calculating metrics.

ret=$?
if [ $ret -ne 0 ]
then
     echo "check_consistent.py exited abnormally. Exiting script."
     exit 1
fi

# arguments: 1) config file path. 2) PID of the save.sh job
readarray params < $1

MAX_N_JOBS=500 # maximum allowed number of jobs in job array is 1001. I chose 1000 because 1001 is ugly.
N_PARAMS=${#params[@]}
MAX_N_PARAMS_PER_JOB=$(($N_PARAMS/$MAX_N_JOBS+1))

if [[ $N_PARAMS -gt $MAX_N_JOBS ]]
then
    N_JOBS=$MAX_N_JOBS
else
    N_JOBS=$N_PARAMS
fi

echo "Number of params: $N_PARAMS"
echo "Number of jobs: $N_JOBS"
echo "Maximum number of params per job: $MAX_N_PARAMS_PER_JOB"

# The weird syntax below is called a here document, in case you want to google it
sbatch -p burst<<EOT
#!/bin/bash
#SBATCH --parsable
#SBATCH -c 2
#SBATCH --mem=8gb
#SBATCH --time=$((20*$MAX_N_PARAMS_PER_JOB)):00
#SBATCH --array=0-$(($N_JOBS-1))
#SBATCH --requeue
#SBATCH --depend=afterany:$2
#SBATCH -o $(dirname $0)/slurm_outputs/output.%A_%a.out

sig_handler() {
    echo "BATCH interrupted"
    wait # wait for all children, this is important!
}

trap 'sig_handler' SIGINT SIGTERM SIGCONT # By default, the job step below will not be able to handle SIGTERM signals gracefully. Thus the need to trap the SIGTERM signal and wait for all job steps to finish handling SIGTERM signal. See this: https://dhruveshp.com/blog/2021/signal-propagation-on-slurm/

srun $(dirname $0)/analyze_single.sh $1 $N_JOBS
EOT