DATA_DIR=../../data/reddit_data/
POST_DATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
# no filter
FILTER_OVERLAP=False
DATA_NAME=advice_subreddit_no_filter
# filter by question content and overlap
#VALID_QUESTION_MODEL=../../data/reddit_data/valid_question_detection_model.pkl
#FILTER_OVERLAP=True
#DATA_NAME=advice_subreddit
#python combine_clean_question_comment_data.py $DATA_DIR --post_data $POST_DATA --valid_question_model $VALID_QUESTION_MODEL
MAX_MEMORY=55000000000 # 55G
(python combine_clean_question_comment_data.py $DATA_DIR --post_data $POST_DATA --valid_question_model $VALID_QUESTION_MODEL --filter_overlap $FILTER_OVERLAP) &
PID=$!
prlimit --pid $PID --as=$MAX_MEMORY
