#QUESTION_DATA=../../data/reddit_data/subreddit_combined_valid_question_data.gz
# no-filter questions for annotation
QUESTION_DATA=../../data/reddit_data/advice_subreddit_no_filter_comment_question_data.gz
POST_DATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
SAMPLE_QUESTIONS_PER_GROUP=115
ANNOTATORS=3
ANNOTATORS_PER_QUESTION=2
MAX_POST_LEN=500
#OUT_DIR=../../data/reddit_data/annotation_samples/
OUT_DIR=../../data/reddit_data/annotation_samples/round_2_annotation_sample/test_data/
python get_sample_questions_for_annotation.py $QUESTION_DATA $POST_DATA --sample_questions_per_group $SAMPLE_QUESTIONS_PER_GROUP --out_dir $OUT_DIR --annotators $ANNOTATORS --annotators_per_question $ANNOTATORS_PER_QUESTION --max_post_len $MAX_POST_LEN