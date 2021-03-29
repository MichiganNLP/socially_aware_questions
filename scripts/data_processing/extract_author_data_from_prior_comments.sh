# NOTE: only run where we have stored author data (lit100.eecs)
AUTHOR_DATA_DIR=../../data/reddit_data/author_data/
QUESTION_DATA=../../data/reddit_data/subreddit_combined_comment_question_data.gz
POST_DATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
python extract_author_data_from_prior_comments.py $AUTHOR_DATA_DIR $QUESTION_DATA $POST_DATA