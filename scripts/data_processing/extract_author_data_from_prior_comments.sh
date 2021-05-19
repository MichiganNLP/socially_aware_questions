# NOTE: only run where we have stored author data (lit100.eecs)
AUTHOR_DATA_DIR=../../data/reddit_data/author_data/
## TODO: replace with non-missing data??
QUESTION_DATA=../../data/reddit_data/subreddit_combined_comment_question_data.gz
POST_DATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
# optional: subreddit/text embeddings
#EMBEDDING_DATA=../../data/reddit_data/author_data/author_date_subreddits.gz
EMBEDDING_DATA_FILES=(../../data/reddit_data/author_data/author_date_embeddings_type=text.gz ../../data/reddit_data/author_data/author_date_embeddings_type=subreddit.gz)
#ulimit -n 10000 # limit number of open files?
MAX_MEMORY=10000000000 # 10G
(python extract_author_data_from_prior_comments.py $AUTHOR_DATA_DIR $QUESTION_DATA $POST_DATA --author_embeddings_data "${EMBEDDING_DATA_FILES[@]}")&
PID=$!
echo $PID
prlimit --pid $PID --as=$MAX_MEMORY