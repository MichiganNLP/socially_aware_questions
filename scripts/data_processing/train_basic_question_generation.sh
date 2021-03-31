#!/bin/bash
#SBATCH --job-name=train_question_model
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=20g
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

## data
## CNN data
#TRAIN_DATA=../../data/CNN_articles/cnn/article_question_generation_train_data.pt
#VAL_DATA=../../data/CNN_articles/cnn/article_question_generation_val_data.pt
# long data
#TRAIN_DATA=../../data/CNN_articles/cnn/CNN_long_train_data.pt
#VAL_DATA=../../data/CNN_articles/cnn/CNN_long_val_data.pt
## NYT data
# normal data
#TRAIN_DATA=../../data/nyt_comments/NYT_train_data.pt
#VAL_DATA=../../data/nyt_comments/NYT_val_data.pt
# long data
#TRAIN_DATA=../../data/nyt_comments/author_type_NYT_long_input_train_data.pt
#VAL_DATA=../../data/nyt_comments/author_type_NYT_long_input_val_data.pt
# clean NE data
#TRAIN_DATA=../../data/nyt_comments/author_type_NE_overlap_NYT_long_input_train_data.pt
#VAL_DATA=../../data/nyt_comments/author_type_NE_overlap_NYT_long_input_val_data.pt
#TRAIN_DATA=../../data/nyt_comments/author_type_NE_overlap_NYT_long_input_train_data.pt
#VAL_DATA=../../data/nyt_comments/author_type_NE_overlap_NYT_long_input_val_data.pt
# clean NE data, no authors
#TRAIN_DATA=../../data/nyt_comments/no_author_data/NE_overlap_NYT_long_input_train_data.pt
#VAL_DATA=../../data/nyt_comments/no_author_data/NE_overlap_NYT_long_input_val_data.pt
# clean NE data, all (4) months
#TRAIN_DATA=../../data/nyt_comments/full_data/author_type_NE_overlap_NYT_full_long_input_train_data.pt
#VAL_DATA=../../data/nyt_comments/full_data/author_type_NE_overlap_NYT_full_long_input_val_data.pt
## reddit data
#TRAIN_DATA=../../data/reddit_data/advice_subreddit_train_data.pt
#VAL_DATA=../../data/reddit_data/advice_subreddit_val_data.pt
## reddit+author data
TRAIN_DATA=../../data/reddit_data/author_text_data/author_type_advice_subreddit_author_train_data.pt
VAL_DATA=../../data/reddit_data/author_text_data/author_type_advice_subreddit_author_val_data.pt
# regular model
#OUT_DIR=../../data/nyt_comments/
#OUT_DIR=../../data/CNN_articles/cnn/
# model with author information
#OUT_DIR=../../data/nyt_comments/author_data_model/
#OUT_DIR=../../data/nyt_comments/cnn_fine_tune/
# longformer model
#OUT_DIR=../../data/CNN_articles/cnn/longformer_model/
#OUT_DIR=../../data/nyt_comments/no_author_data/NE_overlap/longformer_model/
# debug model
#OUT_DIR=../../data/nyt_comments/debug_model/
# reddit model
OUT_DIR=../../data/reddit_data/author_text_data/
# regular transformer
MODEL_CACHE_DIR=../../data/model_cache/
# longformer FML
#MODEL_CACHE_DIR=../../data/longformer_cache/
#DEVICE='cpu' # debug with small data ;_;
MODEL_TYPE="bart"
#MODEL_TYPE='longformer'
# model with author information
#AUTHOR_DATA=../../data/nyt_comments/author_comment_social_data.tsv
# optional: pretrained model
#PRETRAINED_MODEL=../../data/CNN_articles/cnn/question_generation_model/checkpoint-120500/pytorch_model.bin
export CUDA_VISIBLE_DEVICES=1
SAMPLE_PCT=1.0
# regular model
python train_basic_question_generation.py $TRAIN_DATA $VAL_DATA $OUT_DIR --model_type $MODEL_TYPE --model_cache_dir $MODEL_CACHE_DIR --sample_pct $SAMPLE_PCT
# pretrained model
#python train_basic_question_generation.py $TRAIN_DATA $VAL_DATA $OUT_DIR --device $DEVICE --model_type $MODEL_TYPE --model_cache_dir $MODEL_CACHE_DIR --pretrained_model $PRETRAINED_MODEL
