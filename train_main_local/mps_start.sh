#!/bin/bash
# the following must be performed with root privilege
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export CUDA_MPS_PIPE_DIRECTORY=/ppo_300_war/distributed_rl_train_server/train_main_local/nvidia-mps
#export CUDA_MPS_LOG_DIRECTORY=/ppo_300_war/distributed_rl_train_server/train_main_local/nvidia-log 
# nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
# nvidia-smi -i 1 -c EXCLUSIVE_PROCESS
# nvidia-smi -i 2 -c EXCLUSIVE_PROCESS
# nvidia-smi -i 3 -c EXCLUSIVE_PROCESS
# nvidia-smi -i 4 -c EXCLUSIVE_PROCESS
# nvidia-smi -i 5 -c EXCLUSIVE_PROCESS
# nvidia-smi -i 6 -c EXCLUSIVE_PROCESS
# nvidia-smi -i 7 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d