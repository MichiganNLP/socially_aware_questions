DATA_DIR=../../data/reddit_data/
POST_DATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
# no filter
#DATA_NAME=advice_subreddit_no_filter
# filter
DATA_NAME=advice_subreddit_filter
# filter by question content and overlap
VALID_QUESTION_MODEL=../../data/reddit_data/annotation_data/round_2_annotation_sample/test_data/RandomForestClassifier.pkl
#FILTER_OVERLAP=1
#DATA_NAME=advice_subreddit
MAX_MEMORY=55000000000 # 55G
#(python combine_clean_question_comment_data.py $DATA_DIR --data_name $DATA_NAME --post_data $POST_DATA) &
(python combine_clean_question_comment_data.py $DATA_DIR --data_name $DATA_NAME --post_data $POST_DATA --filter_overlap --valid_question_model $VALID_QUESTION_MODEL)&
PID=$!
prlimit --pid $PID --as=$MAX_MEMORY
