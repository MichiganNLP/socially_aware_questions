#!/bin/bash
#SBATCH --job-name=test_question_model
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=40g
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=mihalcea0

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
# load full data, then subtract training data
#TEST_DATA=FULL
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
MODEL_FILE=../../data/reddit_data/text_only_model/question_generation_model/checkpoint-129000/pytorch_model.bin
MODEL_TYPE='bart'
OUT_DIR=../../data/reddit_data/text_only_model/ # generate
#OUT_DIR=../../data/reddit_data/text_only_model/generate_classify_output/ # generate and classify
# text+author token
#MODEL_FILE=../../data/reddit_data/author_text_data/question_generation_model/checkpoint-129000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/
# text+author token: fine-tuning author data
#MODEL_FILE=../../data/reddit_data/author_text_data/text_only_fine_tune/question_generation_model/checkpoint-213500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/text_only_fine_tune/no_author_data/
#MODEL_TYPE='bart_author_token' # test w/ author data
#MODEL_TYPE='bart' # test w/out author data
# author group attention
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/question_generation_model/checkpoint-275500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/
#MODEL_TYPE='bart_author_attention'
## START hyperparameter tuning
# author group attention: hyperparameters
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_location\=encoder_config\=attnconcat/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_location\=encoder_config\=attnconcat/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_location\=encoder_config\=attnconcat/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_location\=encoder_config\=attnconcat/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=encoder_config\=attnconcat/question_generation_model/checkpoint-129000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=encoder_config\=attnconcat/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_location\=decoder_config\=attnconcat/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_location\=decoder_config\=attnconcat/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_location\=decoder_config\=attnconcat/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_location\=decoder_config\=attnconcat/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=decoder_config\=attnconcat/question_generation_model/checkpoint-129000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=decoder_config\=attnconcat/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.1_location\=encoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.1_location\=encoder/
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.1_location\=encoder/full_data/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.5_location\=encoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.5_location\=encoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.9_location\=encoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.9_location\=encoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_weight\=0.1_location\=encoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_weight\=0.1_location\=encoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_weight\=0.5_location\=encoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_weight\=0.5_location\=encoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_weight\=0.9_location\=encoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_weight\=0.9_location\=encoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_weight\=0.1_location\=encoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_weight\=0.1_location\=encoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_weight\=0.5_location\=encoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_weight\=0.5_location\=encoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_weight\=0.9_location\=encoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_weight\=0.9_location\=encoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.1_location\=decoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.1_location\=decoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.5_location\=decoder/question_generation_model/checkpoint-94500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.5_location\=decoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.9_location\=decoder/question_generation_model/checkpoint-129000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.9_location\=decoder/
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.9_location\=decoder/full_test_data/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_weight\=0.1_location\=decoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_weight\=0.1_location\=decoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_weight\=0.5_location\=decoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_weight\=0.5_location\=decoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_weight\=0.9_location\=decoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=3_weight\=0.9_location\=decoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_weight\=0.1_location\=decoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_weight\=0.1_location\=decoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_weight\=0.5_location\=decoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_weight\=0.5_location\=decoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_weight\=0.9_location\=decoder/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_weight\=0.9_location\=decoder/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.1_location\=encoder_config\=attnprob/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.1_location\=encoder_config\=attnprob/
## END hyperparameter tuning
# author attention: fine-tuning
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=decoder_finetune\=AUTHORS/question_generation_model/checkpoint-56500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=decoder_finetune\=AUTHORS/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=encoder_config\=attnconcat_sortdata\=readergroup/question_generation_model/checkpoint-129000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=encoder_config\=attnconcat_sortdata\=readergroup/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.9_location\=decoder_sortdata\=readergroup/question_generation_model/checkpoint-129000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.9_location\=decoder_sortdata\=readergroup/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=encoder_config\=attnconcat_freezeweights\=general/question_generation_model/checkpoint-76000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=encoder_config\=attnconcat_freezeweights\=general/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.9_location\=decoder_freezeweights\=general/question_generation_model/checkpoint-76000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=1_weight\=0.9_location\=decoder_freezeweights\=general/
# author attention: LM head
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight\=0.1_location\=lm_head/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight\=0.1_location\=lm_head/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight\=0.5_location\=lm_head/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight\=0.5_location\=lm_head/
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight\=0.9_location\=lm_head/question_generation_model/checkpoint-97000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/author_attention_weight\=0.9_location\=lm_head/
#MODEL_TYPE='bart_author_attention'
# reddit + subreddit embed (+ encoder)
#MODEL_FILE=../../data/reddit_data/author_text_data/author_subreddit_embed_data/question_generation_model/checkpoint-129000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_subreddit_embed_data/
#MODEL_TYPE='bart_author_embeds'
# reddit + subreddit embed (+ decoder)
#MODEL_FILE=../../data/reddit_data/author_text_data/author_subreddit_embed_decoder_data/question_generation_model/checkpoint-275500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_subreddit_embed_decoder_data/
#MODEL_TYPE='bart_author_embeds'
# reddit + text embed (+ encoder)
#MODEL_FILE=../../data/reddit_data/author_text_data/author_text_embed_data/question_generation_model/checkpoint-129000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_text_embed_data/
#MODEL_TYPE='bart_author_embeds'
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
# extra post sub-group (e.g. divisive posts) to test on
# divisive posts
#POST_SUBGROUP_FILE=../../data/reddit_data/paired_question_low_sim_simpct=25_data.gz
#POST_SUBGROUP_FILE=../../data/reddit_data/paired_question_low_sim_simpct=10_data.gz
#POST_SUBGROUP_FILE=../../data/reddit_data/paired_question_low_sim_simpct=5_data.gz
POST_SUBGROUP_FILE=../../data/reddit_data/paired_question_low_sim_simpct=10_data_simtype=word_embed.gz
#POST_SUBGROUP_FILE=../../data/reddit_data/paired_question_low_sim_simpct=5_data_simtype=word_embed.gz
# reader-specific posts
#POST_SUBGROUP_FILE=../../data/reddit_data/group_classification_model/question_post_data/reader_group_cutoff_pct=95.gz

## queue server
python test_question_generation.py $TEST_DATA --train_data $TRAIN_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR --post_metadata $POST_METADATA --generation_params $GENERATION_PARAMS --post_subgroup_file $POST_SUBGROUP_FILE

## regular server
# set GPU
#export CUDA_VISIBLE_DEVICES=3
# no model (i.e. zero-shot)
#(python test_question_generation.py $TEST_DATA --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR --post_metadata $POST_METADATA)&
# trained model
#python test_question_generation.py $TEST_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR
# regular generation
#(python test_question_generation.py $TEST_DATA --train_data $TRAIN_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR --post_metadata $POST_METADATA --generation_params $GENERATION_PARAMS --post_subgroup_file $POST_SUBGROUP_FILE)&
#PID=$!
#MAX_MEMORY=70000000000 # 70G
#prlimit --pid $PID --as=$MAX_MEMORY
