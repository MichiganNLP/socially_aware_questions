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
# NYT
#python clean_data_for_generation.py $DATA_DIR $OUT_DIR --data_name $DATA_NAME --comment_dir $COMMENT_DIR --comment_month_year_pairs $COMMENT_MONTH_YEAR_PAIRS --author_data $AUTHOR_DATA --model_type $MODEL_TYPE
# CNN
python clean_data_for_generation.py $OUT_DIR --data_file $DATA_FILE --data_name $DATA_NAME --model_type $MODEL_TYPE