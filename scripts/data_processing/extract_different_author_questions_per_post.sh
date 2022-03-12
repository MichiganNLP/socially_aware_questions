#!/bin/bash
#SBATCH --job-name=extract_different_author_questions_per_post
#SBATCH --mail-user=ianbstew@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=1:00:00
#SBATCH --account=mihalcea0
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=80g

OUT_DIR=../../data/reddit_data/
TRAIN_DATA_FILE=../../data/reddit_data/combined_data_train_data.pt
#MAX_SIM_PCT=25
#MAX_SIM_PCT=10
MAX_SIM_PCT=5
#SIM_TYPE='sentence_embed'
SIM_TYPE='word_embed'
python extract_different_author_questions_per_post.py $OUT_DIR --filter_data_file $TRAIN_DATA_FILE --remove_data --max_sim_pct $MAX_SIM_PCT --sim_type $SIM_TYPE