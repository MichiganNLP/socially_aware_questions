#COMMENT_DATA=../../data/reddit_data/subreddit_combined_comment_question_data.gz
#COMMENT_DATA=../../data/reddit_data/advice_subreddit_question_data.gz # questions after filtering...yikes
COMMENT_DATA=../../data/reddit_data/advice_subreddit_filter_comment_question_data.gz
SAMPLE_AUTHORS_PER_GROUP=15000
SAMPLE_POSTS_PER_AUTHOR=100
OUT_DIR=../../data/reddit_data/author_data/
python collect_prior_posts_from_question_authors.py $COMMENT_DATA --sample_authors_per_group $SAMPLE_AUTHORS_PER_GROUP --sample_posts_per_author $SAMPLE_POSTS_PER_AUTHOR --out_dir $OUT_DIR