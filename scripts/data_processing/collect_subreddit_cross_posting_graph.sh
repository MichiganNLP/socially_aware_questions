POST_DIR=/local2/lbiester/pushshift/submissions/ # post directory on lit100
OUT_DIR=../../data/reddit_data/
# serial
#POST_DATES="2019,1,2019,1"
#python collect_subreddit_cross_posting_graph.py $POST_DIR --out_dir $OUT_DIR --post_dates $POST_DATES
# parallel
COMMENT_YEAR=2019
COMMENT_START_MONTH=1
COMMENT_END_MONTH=1
## generate all combos lol
COMMENT_YEAR_MONTH_PAIRS=()
for COMMENT_MONTH in `seq $COMMENT_START_MONTH $COMMENT_END_MONTH`;
do
  COMMENT_YEAR_MONTH_PAIRS+=($COMMENT_YEAR,"$COMMENT_MONTH",$COMMENT_YEAR,"$COMMENT_MONTH")
done
JOBS=4
parallel --jobs $JOBS python collect_subreddit_cross_posting_graph.py $POST_DIR --out_dir $OUT_DIR --post_dates {} ::: "${COMMENT_YEAR_MONTH_PAIRS[@]}"