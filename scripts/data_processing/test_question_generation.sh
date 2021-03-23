# test question generation

# data params
# reddit data
# regular training
MODEL_FILE=../../data/reddit_data/text_only_model/question_generation_model/checkpoint-116500/pytorch_model.bin
TEST_DATA=../../data/reddit_data/advice_subreddit_val_data.pt

# model params
MODEL_CACHE_DIR=../../data/model_cache/
MODEL_TYPE='bart'
# set GPU
export CUDA_VISIBLE_DEVICES=0

# no model (i.e. zero-shot)
#python test_question_generation.py $TEST_DATA --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE
# trained model
python test_question_generation.py $TEST_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE