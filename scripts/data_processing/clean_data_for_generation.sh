DATA_DIR=../../data/NYT_scrape/
DATA_NAME=NYT
COMMENT_DIR=../../data/nyt_comments/
COMMENT_MONTH_YEAR_PAIRS=('April_2018')
OUT_DIR=../../data/nyt_comments/
python clean_data_for_generation.py $DATA_DIR $OUT_DIR --data_name $DATA_NAME --comment_dir $COMMENT_DIR --comment_month_year_pairs $COMMENT_MONTH_YEAR_PAIRS