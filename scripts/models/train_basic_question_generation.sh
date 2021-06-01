#!/bin/bash
#SBATCH --job-name=train_question_model
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=20g
#SBATCH --time=32:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1

###SBATCH --gres=gpu:1

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
#TRAIN_DATA=../../data/reddit_data/combined_data_train_data.pt
#VAL_DATA=../../data/reddit_data/combined_data_test_data.pt
TRAIN_DATA=../../data/reddit_data/combined_data_train_train_data.pt
VAL_DATA=../../data/reddit_data/combined_data_train_val_data.pt
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
## NOTE: for author models modify data/model_cache/BART_author_model_config.json before starting
# reddit model
#OUT_DIR=../../data/reddit_data/text_only_model/
#MODEL_TYPE='bart'
#MODEL_CONFIG_FILE=../../data/model_cache/BART_config.json
# reddit author model
# author token
#OUT_DIR=../../data/reddit_data/author_text_data/
#MODEL_TYPE="bart_author"
#MODEL_CONFIG_FILE=../../data/model_cache/BART_author_token_model_config.json
# author attention
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/
#MODEL_TYPE="bart_author_attention"
#MODEL_CONFIG_FILE=../../data/model_cache/BART_author_token_model_config.json
# author embed
# subreddit
#OUT_DIR=../../data/reddit_data/author_text_data/author_subreddit_embed_data/
#MODEL_TYPE="bart_author_embeds"
#MODEL_CONFIG_FILE=../../data/model_cache/BART_author_subreddit_embed_model_config.json
# text
#OUT_DIR=../../data/reddit_data/author_text_data/author_text_embed_data/
#MODEL_TYPE="bart_author_embeds"
#MODEL_CONFIG_FILE=../../data/model_cache/BART_author_text_embed_model_config.json
# author (decoder) embed
#OUT_DIR=../../data/reddit_data/author_text_data/author_decoder_embed_data/
# author attention
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/
OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight=0.1/
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight=0.9/
MODEL_TYPE="bart_author_attention"
MODEL_CONFIG_FILE=../../data/model_cache/BART_author_attention_model_config.json
# regular transformer
MODEL_CACHE_DIR=../../data/model_cache/
# longformer FML
#MODEL_CACHE_DIR=../../data/longformer_cache/
# optional: pretrained model
#PRETRAINED_MODEL=../../data/CNN_articles/cnn/question_generation_model/checkpoint-120500/pytorch_model.bin
# optional: multiple GPUs
N_GPU=1
#N_GPU=2
# use device for single-GPU processes
#export CUDA_VISIBLE_DEVICES=0
# regular model
python train_basic_question_generation.py $TRAIN_DATA $VAL_DATA $OUT_DIR --model_type $MODEL_TYPE --model_cache_dir $MODEL_CACHE_DIR --model_config_file $MODEL_CONFIG_FILE --n_gpu $N_GPU
#PID=$!
#MAX_MEMORY=50000000000 # 50G
#prlimit --pid $PID --as=$MAX_MEMORY
# pretrained model
#python train_basic_question_generation.py $TRAIN_DATA $VAL_DATA $OUT_DIR --device $DEVICE --model_type $MODEL_TYPE --model_cache_dir $MODEL_CACHE_DIR --pretrained_model $PRETRAINED_MODEL
