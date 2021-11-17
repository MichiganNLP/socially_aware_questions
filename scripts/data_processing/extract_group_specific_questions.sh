TEST_DATA=../../data/reddit_data/combined_data_test_data.pt
OUT_DIR=../../data/reddit_data/group_classification_model/question_post_data/
POST_DATA_FILE=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
MODEL_DIR=../../data/reddit_data/group_classification_model/question_post_data/
PROB_CUTOFF_PCT=90
## set GPU
export CUDA_VISIBLE_DEVICES=3
(python extract_group_specific_questions.py $TEST_DATA $OUT_DIR --post_data_file $POST_DATA_FILE --model_dir $MODEL_DIR --prob_cutoff_pct $PROB_CUTOFF_PCT)&
PID=$!
MAX_MEMORY=50000000000 # 50G
prlimit --pid $PID --as=$MAX_MEMORY