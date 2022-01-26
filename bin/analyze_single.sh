#!/bin/bash
readarray params < $1

N_PARAMS=${#params[@]}
N_JOBS=$2

echo "Number of params: $N_PARAMS"
echo "Number of jobs: $2"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

terminated=false

sig_handler() { 
  echo "Caught SIGINT/SIGTERM/SIGCONT signal!" 
  kill -TERM "$child" 2>/dev/null
  echo "Killed child process"
  terminated=true
}

trap 'sig_handler' SIGINT SIGTERM SIGCONT

PARAM_ID=$SLURM_ARRAY_TASK_ID

while [ $PARAM_ID -lt $N_PARAMS ]
do
    echo "Starting main_analyze.py for param index $PARAM_ID..."
    echo "${params[$PARAM_ID]}"
    python -u $(dirname $0)/py_scripts/main_analyze.py ${params[$PARAM_ID]} &

    child=$! 
    wait "$child"
    
    ret=$?
    if [ $ret -ne 0 ]
    then
         echo "Error in main_analyze.py. Exiting script."
         exit 1
    fi
    
    if [ "$terminated" = true ] # reference: https://stackoverflow.com/questions/2953646/how-can-i-declare-and-use-boolean-variables-in-a-shell-script?rq=1
    then
        echo "Job terminated, exiting while loop early"
        break
    fi
    
    PARAM_ID=$(($PARAM_ID+$N_JOBS))
done

# All the extra stuff is needed to pass SIGTERM signal to the python process. The python process has a SIGTERM handler that performs cleanup when SIGTERM is signaled. See this: https://unix.stackexchange.com/questions/146756/forward-sigterm-to-child-in-bash. Also, python -u is required for python print statements to be written to the slurm output file continuously.