#!/bin/bash
#SBATCH --job-name=compare_model_performance
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=10g
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=mihalcea0

TEST_DATA=../../data/reddit_data/combined_data_test_data.pt
#MODEL_OUTPUT_FILES=(../../data/reddit_data/text_only_model/test_data_sample_top_p\=0.9_temperature\=1.0_output_text.gz ../../data/reddit_data/author_text_data/test_data_sample_top_p\=0.9_temperature\=1.0_output_text.gz ../../data/reddit_data/author_text_data/author_attention_data/test_data_sample_top_p\=0.9_temperature\=1.0_output_text.gz ../../data/reddit_data/author_text_data/author_subreddit_embed_data/test_data_sample_top_p\=0.9_temperature\=1.0_output_text.gz ../../data/reddit_data/author_text_data/author_text_embed_data/test_data_sample_top_p\=0.9_temperature\=1.0_output_text.gz)
MODEL_OUTPUT_FILES=(../../data/reddit_data/text_only_model/test_data_output_text.gz ../../data/reddit_data/author_text_data/test_data_output_text.gz)
#MODEL_NAMES=("text" "reader_token" "reader_attention" "reader_subreddit_embed" "reader_text_embed")
MODEL_NAMES=("text" "reader_token")
# extra post sub-group (e.g. divisive posts) to test on
POST_SUBGROUP_FILE=../../data/reddit_data/paired_question_low_sim_simpct=25_data.gz
POST_SUBGROUP_NAME=diff_questions

OUT_DIR=../../data/reddit_data/
python compare_model_performance.py $TEST_DATA --model_output_files "${MODEL_OUTPUT_FILES[@]}" --model_names "${MODEL_NAMES[@]}" --out_dir $OUT_DIR --post_subgroup_file $POST_SUBGROUP_FILE --post_subgroup_name $POST_SUBGROUP_NAME
