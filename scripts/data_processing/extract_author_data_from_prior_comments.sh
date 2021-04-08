# NOTE: only run where we have stored author data (lit100.eecs)
AUTHOR_DATA_DIR=../../data/reddit_data/author_data/
## TODO: replace with non-missing data??
QUESTION_DATA=../../data/reddit_data/subreddit_combined_comment_question_data.gz
POST_DATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
MAX_MEMORY=10000000000 # 10G
(python extract_author_data_from_prior_comments.py $AUTHOR_DATA_DIR $QUESTION_DATA $POST_DATA)&
PID=$!
prlimit --pid $PID --as $MAX_MEMORY