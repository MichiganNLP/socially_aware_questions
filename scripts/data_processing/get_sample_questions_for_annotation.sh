#QUESTION_DATA=../../data/reddit_data/subreddit_combined_valid_question_data.gz
# no-filter questions for annotation
QUESTION_DATA=../../data/reddit_data/advice_subreddit_no_filter_comment_question_data.gz
POST_DATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
SAMPLE_QUESTIONS_PER_GROUP=100
ANNOTATORS=3
ANNOTATORS_PER_QUESTION=2
#OUT_DIR=../../data/reddit_data/annotation_data/
OUT_DIR=../../data/reddit_data/annotation_data/round_2_annotations/
python get_sample_questions_for_annotation.py $QUESTION_DATA $POST_DATA --sample_questions_per_group $SAMPLE_QUESTIONS_PER_GROUP --out_dir $OUT_DIR --annotators $ANNOTATORS --annotators_per_question $ANNOTATORS_PER_QUESTION