# test question generation

# data params
# reddit data
TEST_DATA=../../data/reddit_data/advice_subreddit_val_data.pt
# regular training
#MODEL_FILE=../../data/reddit_data/text_only_model/question_generation_model/checkpoint-171000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/text_only_model/
# reddit+author data
#TEST_DATA=../../data/reddit_data/author_text_data/author_type_advice_subreddit_author_val_data.pt
# author training
#MODEL_FILE=../../data/reddit_data/author_text_data/question_generation_model/checkpoint-189000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/
# no training
OUT_DIR=../../data/reddit_data/no_train_model/
# CNN QA data
#MODEL_FILE=../../data/CNN_articles/cnn/question_generation_model/checkpoint-120500/pytorch_model.bin
#OUT_DIR=../../data/CNN_articles/cnn/
## metadata to test sub-sets of data
POST_METADATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
# model params
MODEL_CACHE_DIR=../../data/model_cache/
MODEL_TYPE='bart'
# set GPU
export CUDA_VISIBLE_DEVICES=1

# no model (i.e. zero-shot)
python test_question_generation.py $TEST_DATA --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR
# trained model
#python test_question_generation.py $TEST_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR
# trained model; testing on post subsets
#python test_question_generation.py $TEST_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR --post_metadata $POST_METADATA