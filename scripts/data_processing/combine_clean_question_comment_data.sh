DATA_DIR=../../data/reddit_data/
DATA_NAME=advice_subreddit
POST_DATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
VALID_QUESTION_MODEL=../../data/reddit_data/valid_question_detection_model.pkl
#python combine_clean_question_comment_data.py $DATA_DIR --post_data $POST_DATA --valid_question_model $VALID_QUESTION_MODEL
MAX_MEMORY=55000000000 # 55G
python combine_clean_question_comment_data.py $DATA_DIR --post_data $POST_DATA --valid_question_model $VALID_QUESTION_MODEL &
PID=$!
prlimit --pid $PID --as=$MAX_MEMORY
