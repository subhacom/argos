#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --partition=gpu

# These are the modules needed for tensorflow 2.1.0
module load CUDA/10.1
module load cuDNN/7.6.5/CUDA-10.1
module load gcc
# source /data/rays3/conda/etc/profile.d/conda.sh
# conda activate deeplearn
pushd /data/rays3/locust_tracking/yolact
# batch_size should be 8 * num_gpus - running on 4
python train.py --config=babylocust_config --resume=weights/babylocust_resnet101_25999_78000.pth --start_iter=-1 --batch_size=8 --save_interval=1000 --keep_latest 
echo "Finished training"
popd

