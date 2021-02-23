POST_DIR=/local2/lbiester/pushshift/submissions/ # post directory on lit100
OUT_DIR=../../data/reddit_data/
POST_DATES="2019,1,2019,1"
python collect_subreddit_cross_posting_graph.py $POST_DIR --out_dir $OUT_DIR --post_dates $POST_DATES