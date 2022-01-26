#!/bin/bash
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
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH --time=$((20*$MAX_N_PARAMS_PER_JOB)):00
#SBATCH --gres=gpu:1
#SBATCH --array=0-$(($N_JOBS-1))
#SBATCH --requeue
#SBATCH -o $(dirname $0)/slurm_outputs/output.%A_%a.out

sig_handler() {
    echo "BATCH interrupted"
    wait # wait for all children, this is important!
}

trap 'sig_handler' SIGINT SIGTERM SIGCONT # By default, the job step below will not be able to handle SIGTERM signals gracefully. Thus the need to trap the SIGTERM signal and wait for all job steps to finish handling SIGTERM signal. See this: https://dhruveshp.com/blog/2021/signal-propagation-on-slurm/

srun $(dirname $0)/save_single.sh $1 $N_JOBS
EOT