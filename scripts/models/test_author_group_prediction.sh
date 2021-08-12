#!/bin/bash
#SBATCH --job-name=test_author_group_prediction
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=50g
#SBATCH --time=47:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1

GROUP_CATEGORIES=("location_region" "expert_pct_bin" "relative_time_bin")
TEXT_VAR="question_post"
#TEXT_VAR="question"
#GROUP_CATEGORIES=("relative_time_bin")
# queue server
# offline model because no internet connection in cluster
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 
python test_author_group_prediction.py --group_categories "${GROUP_CATEGORIES[@]}" --out_dir ../../data/reddit_data/group_classification_model/ --text_var $TEXT_VAR

# regular server
# specify CUDA device (only for LIT machines)
#export CUDA_VISIBLE_DEVICES=3
#python test_author_group_prediction.py --group_categories "${GROUP_CATEGORIES[@]}" --out_dir ../../data/reddit_data/group_classification_model/
# dumb parallel code that doesn't work fml
#export LOCAL_RANK=1
#python -m torch.distributed.launch --nproc_per_node 2 --use_env test_author_group_prediction.py "${GROUP_CATEGORIES[@]}"
