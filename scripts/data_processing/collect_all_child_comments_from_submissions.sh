SUBMISSION_DATA=../..data/reddit_data/advice_subreddit_submissions_2018-01_2019-12.gz
OUT_DIR=../../data/reddit_data/
DATA_DIR=/local2/lbiester/pushshift/comments/
COMMENT_DATES="2018,7,2019,12"
python collect_all_child_comments_from_submissions.py $SUBMISSION_DATA --out_dir $OUT_DIR --data_dir $DATA_DIR --comment_dates $COMMENT_DATES