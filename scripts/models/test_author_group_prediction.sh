#!/bin/bash
#SBATCH --job-name=test_author_group_prediction
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=50g
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1

# specify CUDA device (only for LIT machines)
#export CUDA_VISIBLE_DEVICES=0
python test_author_group_prediction.py
