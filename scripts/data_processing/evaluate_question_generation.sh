#!/bin/bash
#SBATCH --job-name=evaluate_question_generation
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=10g
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# evaluate question generation based on test data, repetition, and copying training data
## model
## CNN training
#MODEL_FILE=../../data/CNN_articles/cnn/question_generation_model/checkpoint-120500/pytorch_model.bin
#OUT_DIR=../../data/CNN_articles/cnn/
#OUT_DIR=../../data/CNN_articles/cnn/NYT_eval/
# long data
#MODEL_FILE=../../data/CNN_articles/cnn/longformer_model/question_generation_model/checkpoint-60000/pytorch_model.bin
#OUT_DIR=../../data/CNN_articles/cnn/NYT_eval/longformer_model/
#MODEL_FILE=../../data/CNN_articles/cnn/longformer_model/question_generation_model/checkpoint-60000/pytorch_model.bin
#OUT_DIR=../../data/CNN_articles/cnn/longformer_model/
# NYT training
# w/out author
#MODEL_FILE=../../data/nyt_comments/question_generation_model/checkpoint-96000/pytorch_model.bin
#OUT_DIR=../../data/nyt_comments/no_author_data/
# w/author
#MODEL_FILE=../../data/nyt_comments/author_data_model/question_generation_model/checkpoint-73500/pytorch_model.bin
# long data w/ author
#MODEL_FILE=../../data/nyt_comments/longformer_model/question_generation_model/checkpoint-35500/pytorch_model.bin
#OUT_DIR=../../data/nyt_comments/longformer_model/
# NE overlap
#MODEL_FILE=../../data/nyt_comments/NE_overlap/longformer_model/question_generation_model/checkpoint-27500/pytorch_model.bin
#OUT_DIR=../../data/nyt_comments/NE_overlap/
# NE overlap w/out author
#MODEL_FILE=../../data/nyt_comments/no_author_data/NE_overlap/longformer_model/question_generation_model/checkpoint-9500/pytorch_model.bin
#OUT_DIR=../../data/nyt_comments/no_author_data/NE_overlap/
## CNN+NYT model evaluation
#MODEL_FILE=../../data/nyt_comments/cnn_fine_tune/question_generation_model/checkpoint-141000/pytorch_model.bin
#OUT_DIR=../../data/nyt_comments/cnn_fine_tune/
## reddit comments
MODEL_FILE=../../data/reddit_data/text_only_model/question_generation_model/checkpoint-116500/pytorch_model.bin
OUT_DIR=../../data/reddit_data/text_only_model/
## model cache folder
# w/out author
MODEL_CACHE_DIR=../../data/model_cache/
# w/ author
#MODEL_CACHE_DIR=../../data/nyt_comments/author_data_model/model_cache/
# w/out author, long input
#MODEL_CACHE_DIR=../../data/longformer_cache/
MODEL_TYPE='bart'
#MODEL_TYPE='longformer'

## data
### CNN data
#TRAIN_DATA=../../data/CNN_articles/cnn/article_question_generation_train_data.pt
#TEST_DATA=../../data/CNN_articles/cnn/article_question_generation_val_data.pt
# long data
#TRAIN_DATA=../../data/CNN_articles/cnn/CNN_long_train_data.pt
#TEST_DATA=../../data/CNN_articles/cnn/CNN_long_val_data.pt
### NYT data
# w/out author
#TRAIN_DATA=../../data/nyt_comments/no_author_data/NYT_train_data.pt
#TEST_DATA=../../data/nyt_comments/no_author_data/NYT_val_data.pt
# w/ author
#TRAIN_DATA=../../data/nyt_comments/author_data_model/author_type_NYT_question_data_train_data.pt
#TEST_DATA=../../data/nyt_comments/author_data_model/author_type_NYT_question_data_val_data.pt
# long w/ author
#TRAIN_DATA=../../data/nyt_comments/author_type_NYT_long_input_train_data.pt
#TEST_DATA=../../data/nyt_comments/author_type_NYT_long_input_val_data.pt
# long w/out author
#TRAIN_DATA=../../data/nyt_comments/no_author_data/NYT_long_input_train_data.pt
#TEST_DATA=../../data/nyt_comments/no_author_data/NYT_long_input_val_data.pt
# NE overlap
#TRAIN_DATA=../../data/nyt_comments/author_type_NE_overlap_NYT_long_input_train_data.pt
#TEST_DATA=../../data/nyt_comments/author_type_NE_overlap_NYT_long_input_val_data.pt
# NE overlap w/out author
#TRAIN_DATA=../../data/nyt_comments/no_author_data/NE_overlap_NYT_long_input_train_data.pt
#TEST_DATA=../../data/nyt_comments/no_author_data/NE_overlap_NYT_long_input_val_data.pt
# reddit data
TRAIN_DATA=../../data/reddit_data/advice_subreddit_train_data.pt
TEST_DATA=../../data/reddit_data/advice_subreddit_val_data.pt
DATA_NAME=text_only_model
## optional: post metadata
POST_METADATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
#DEVICE_NAME='cuda:2'
export CUDA_VISIBLE_DEVICES="1"
# normal evaluation
#python evaluate_question_generation.py $MODEL_FILE $OUT_DIR $TRAIN_DATA $TEST_DATA --model_type $MODEL_TYPE --model_cache_dir $MODEL_CACHE_DIR --data_name $DATA_NAME
# evaluation with metadata
python evaluate_question_generation.py $MODEL_FILE $OUT_DIR $TRAIN_DATA $TEST_DATA --model_type $MODEL_TYPE --model_cache_dir $MODEL_CACHE_DIR --data_name $DATA_NAME --post_metadata $POST_METADATA