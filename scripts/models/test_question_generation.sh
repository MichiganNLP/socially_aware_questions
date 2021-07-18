#!/bin/bash
#SBATCH --job-name=test_question_model
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=30g
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# test question generation
# data params
# no training
#OUT_DIR=../../data/reddit_data/no_train_model/
# CNN QA data
#TEST_DATA=../../data/reddit_data/advice_subreddit_val_data.pt
#MODEL_FILE=../../data/CNN_articles/cnn/question_generation_model/checkpoint-120500/pytorch_model.bin
#OUT_DIR=../../data/CNN_articles/cnn/
# regular training reddit data
TRAIN_DATA=../../data/reddit_data/combined_data_train_data.pt
TEST_DATA=../../data/reddit_data/combined_data_test_data.pt
# mini train/val data for parameter tuning
#TRAIN_DATA=../../data/reddit_data/combined_data_train_train_data.pt
#TEST_DATA=../../data/reddit_data/combined_data_train_val_data.pt
# data w/ valid author data
#TRAIN_DATA=../../data/reddit_data/combined_data_valid_authors_train_data.pt
#TEST_DATA=../../data/reddit_data/combined_data_valid_authors_test_data.pt
## models
# CNN text only
#MODEL_FILE=../../data/CNN_articles/cnn/question_generation_model/checkpoint-120500/pytorch_model.bin
#MODEL_TYPE='bart'
#OUT_DIR=../../data/CNN_articles/cnn/
# text only
#MODEL_FILE=../../data/reddit_data/text_only_model/question_generation_model/checkpoint-273500/pytorch_model.bin
#MODEL_TYPE='bart'
#OUT_DIR=../../data/CNN_articles/cnn/
# text only
MODEL_FILE=../../data/reddit_data/text_only_model/question_generation_model/checkpoint-273500/pytorch_model.bin
MODEL_TYPE='bart'
#OUT_DIR=../../data/reddit_data/text_only_model/ # generate
OUT_DIR=../../data/reddit_data/text_only_model/generate_classify_output/ # generate and classify
# text+author token
#MODEL_FILE=../../data/reddit_data/author_text_data/question_generation_model/checkpoint-273500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/
# text+author token: fine-tuning author data
#MODEL_FILE=../../data/reddit_data/author_text_data/text_only_fine_tune/question_generation_model/checkpoint-213500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/text_only_fine_tune/no_author_data/
#MODEL_TYPE='bart_author_token' # test w/ author data
#MODEL_TYPE='bart' # test w/out author data
# reddit + author group attention
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/question_generation_model/checkpoint-273500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/
# reddit + author group attention: hyperparameter tuning
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight=0.1/question_generation_model/checkpoint-229000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight=0.1/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight=0.5/question_generation_model/checkpoint-229000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight=0.5/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight=0.9/question_generation_model/checkpoint-229000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight=0.9/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer=3/question_generation_model/checkpoint-229000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer=3/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer=5/question_generation_model/checkpoint-229000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer=5/
#MODEL_TYPE='bart_author_attention'
# reddit + subreddit embed (+ encoder)
#MODEL_FILE=../../data/reddit_data/author_text_data/author_subreddit_embed_data/question_generation_model/checkpoint-273500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_subreddit_embed_data/
#MODEL_TYPE='bart_author_embeds'
# reddit + subreddit embed (+ decoder)
#MODEL_FILE=../../data/reddit_data/author_text_data/author_subreddit_embed_decoder_data/question_generation_model/checkpoint-229000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_subreddit_embed_decoder_data/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_subreddit_embed_decoder_data/question_generation_model/checkpoint-229000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_subreddit_embed_decoder_data/
#MODEL_TYPE='bart_author_embeds'
# reddit + text embed (+ encoder)
#MODEL_FILE=../../data/reddit_data/author_text_data/author_text_embed_data/question_generation_model/checkpoint-273500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_text_embed_data/
# reddit + text embed (+ decoder)
#MODEL_FILE=../../data/reddit_data/author_text_data/author_text_embed_decoder_data/question_generation_model/checkpoint-229000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_text_embed_decoder_data/
#MODEL_TYPE='bart_author_embeds'

# metadata to test sub-sets of data
POST_METADATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
# model params
MODEL_CACHE_DIR=../../data/model_cache/
# model generation params
#GENERATION_PARAMS=../../data/model_cache/beam_search_generation_params.json
GENERATION_PARAMS=../../data/model_cache/sample_generation_params.json

# set GPU
#export CUDA_VISIBLE_DEVICES=3

# no model (i.e. zero-shot)
#(python test_question_generation.py $TEST_DATA --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR --post_metadata $POST_METADATA)&
# trained model
#python test_question_generation.py $TEST_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR
# trained model + test on post subsets (e.g. community)
#python test_question_generation.py $TEST_DATA --train_data $TRAIN_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR --post_metadata $POST_METADATA --generation_params $GENERATION_PARAMS
# generate + classify
(python test_question_generation.py $TEST_DATA --train_data $TRAIN_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR --post_metadata $POST_METADATA --generation_params $GENERATION_PARAMS --generate_classify)&
PID=$!
MAX_MEMORY=60000000000 # 50G
prlimit --pid $PID --as=$MAX_MEMORY