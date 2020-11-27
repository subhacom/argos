#!/bin/bash

# On biowulf k20 does not work with tensorflow 1.4.0 as its minimum Compute Capability (CC) >= 3.7 and k20 has CC=3.5
# Issue report here: https://github.com/pytorch/pytorch/issues/31285

# k80 should work, but there were following errors 
#    RuntimeError: DataLoader worker (pid(s) 26976, 26977) exited unexpectedly
# Could be due to loading cuda modules after starting the python envidornment. After loading modules and then activating the environment worked, but produced the following error, suggesting not enough RAM is allocated:
#    THCudaCheck FAIL file=/pytorch/aten/src/THC/THCCachingHostAllocator.cpp line=278 error=1 : invalid argument
#    ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
# Worked after node was acquired with enough RAM
#    sinteractive --gres=gpu:k80:1 --mem=64g
# 
# For batch mode, run it like
# sbatch --partition=gpu --gres=gpu:v100x:4 -c56 --mem=64g --time=72:00:00 training_command.sh

module load CUDA/10.1
module load cuDNN/7.6.5/CUDA-10.1
module load gcc

. /data/rays3/conda/etc/profile.d/conda.sh

echo "$SHELL"

conda activate track
pushd /data/rays3/locust_tracking/argos
export PYTHONPATH=.:$PYTHONPATH
# Start training with resnet101_reducedfc.pth (550x550 images). A copy of this file must be present in the save_folder.
# This is for fresh start with resnet 101
python -m yolact.train --save_folder /data/rays3/locust_tracking/yolact_train --save_interval 1000 --keep_latest --config /data/rays3/locust_tracking/yolact_train/yolact_config.yaml

# Uncomment this and comment above to resume interrupted training
# python -m yolact.train --save_folder=/data/rays3/locust_tracking/yolact_train --save_interval=1000 --keep_latest --config /data/rays3/locust_tracking/yolact_train/yolact_config.yaml --resume=interrupt

# Uncomment this and comment above to resume training from specified weights file
# python -m yolact.train --save_folder=/data/rays3/locust_tracking/yolact_train --save_interval=1000 --keep_latest --config=/data/rays3/locust_tracking/yolact_train/yolact_config.yaml --resume=/data/rays3/locust_tracking/yolact_train/babylocust_weights_5555_50000.pth --batch_size=32

echo "Finished training"
popd

