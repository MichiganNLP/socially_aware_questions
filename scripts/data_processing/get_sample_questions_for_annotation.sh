QUESTION_DATA=../../data/reddit_data/subreddit_combined_valid_question_data.gz
POST_DATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
SAMPLE_QUESTIONS_PER_GROUP=5
OUT_DIR=../../data/reddit_data/annotation_data/
python get_sample_questions_for_annotation.py $QUESTION_DATA $POST_DATA --sample_questions_per_group $SAMPLE_QUESTIONS_PER_GROUP --out_dir $OUT_DIR