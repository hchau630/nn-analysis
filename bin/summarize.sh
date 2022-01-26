#!/bin/bash

while getopts ":h" opt; do
    case ${opt} in
        h )
          echo "Usage:"
          echo "    summarize.sh -h                          Display this help message."
          echo "    summarize.sh [PID]                       Display summary of the job with PID."
          echo "    summarize.sh [PID] -i [N]                Refresh summary every N seconds."
          exit 0
          ;;
        \? )
          echo "Invalid option: -$OPTARG" 1>&2
          exit 1
          ;;
  esac
done

PID=$1
shift

while getopts ":i:" opt; do
    case ${opt} in
        i )
          N_SECONDS=$OPTARG
          ;;
        \? )
          echo "Invalid option: -$OPTARG" 1>&2
          exit 1
          ;;
        : )
          echo "Invalid option: $OPTARG requires an argument" 1>&2
          exit 1
          ;;
  esac
done

summarize() {
    sacct -j $PID --format=JobID%20,NodeList,Elapsed,State,ExitCode -X
}

if [ -z ${N_SECONDS+x} ]
then 
    summarize
else 
    while :
    do
        summarize
        sleep $N_SECONDS
    done
fi
