#!/bin/bash
#SBATCH --job-name=clean_data
#SBATCH --mail-user=ianbstew@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=1:00:00
#SBATCH --account=mihalcea1
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=1000m
echo "starting to clean"

## NYT data
#DATA_DIR=../../data/NYT_scrape/
#DATA_NAME=NYT_long_input
#COMMENT_DIR=../../data/nyt_comments/
#COMMENT_MONTH_YEAR_PAIRS=('April_2018')
#OUT_DIR=../../data/nyt_comments/
#AUTHOR_DATA=../../data/nyt_comments/author_comment_social_data.tsv
## CNN data
DATA_FILE=../../data/CNN_articles/cnn/article_question_data.tsv
DATA_NAME=CNN_long
OUT_DIR=../../data/CNN_articles/cnn/
#MODEL_TYPE=bart
MODEL_TYPE=longformer
SAMPLE_PCT=1.0
# sampling for long data
#SAMPLE_PCT=0.25
# NYT
#python clean_data_for_generation.py $OUT_DIR --data_dir $DATA_DIR --data_name $DATA_NAME --comment_dir $COMMENT_DIR --comment_month_year_pairs $COMMENT_MONTH_YEAR_PAIRS --author_data $AUTHOR_DATA --model_type $MODEL_TYPE --sample_pct $SAMPLE_PCT
# CNN
python clean_data_for_generation.py $OUT_DIR --data_file $DATA_FILE --data_name $DATA_NAME --model_type $MODEL_TYPE --sample_pct $SAMPLE_PCT
