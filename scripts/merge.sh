#!/bin/bash
JOBNAME="merge_dataset"

# slurm doesn't source .bashrc automatically
source ~/.bashrc
mamba activate harvest

cd /atlas2/u/jonxuxu/harvest-piles/src
GPUS=0
echo "Number of GPUs: "${GPUS}
WRAP="python merge.py"
LOG_FOLDER="/atlas2/u/jonxuxu/slurm_logs"
echo ${WRAP}
echo "Log Folder:"${LOG_FOLDER}
mkdir -p ${LOG_FOLDER}
sbatch --output=${LOG_FOLDER}/%j.out --error=${LOG_FOLDER}/%j.err \
    --nodes=1 --ntasks-per-node=1 --time=1-00:00:00 --mem=64G \
    --account=atlas --partition=atlas --cpus-per-task=2 \
    --gres=gpu:${GPUS} --job-name=${JOBNAME} --wrap="${WRAP}"