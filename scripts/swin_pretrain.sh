#!/bin/bash
JOBNAME="swin_pretrain"

# slurm doesn't source .bashrc automatically
source ~/.bashrc
mamba activate harvest
cd /atlas2/u/jonxuxu/harvest-piles/src

GPUS=1
MEM=64
VRAM=24

echo "Number of GPUs: "${GPUS}
WRAP="WANDB__SERVICE_WAIT=300 python swin_pretrain.py"
LOG_FOLDER="/atlas2/u/jonxuxu/slurm_logs"
echo ${WRAP}
echo "Log Folder:"${LOG_FOLDER}
mkdir -p ${LOG_FOLDER}
sbatch --output=${LOG_FOLDER}/%j.out --error=${LOG_FOLDER}/%j.err \
    --nodes=1 --ntasks-per-node=1 --time=1-00:00:00 --mem=${MEM}G \
    --account=atlas --partition=atlas --cpus-per-task=2 \
    --gres=gpu:${GPUS} --constraint=${VRAM}G --job-name=${JOBNAME} --wrap="${WRAP}"