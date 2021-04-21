# test question generation

# data params
# regular training reddit data
#TEST_DATA=../../data/reddit_data/advice_subreddit_val_data.pt
#MODEL_FILE=../../data/reddit_data/text_only_model/question_generation_model/checkpoint-171000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/text_only_model/
# reddit+author token data
#TEST_DATA='../../data/reddit_data/author_text_data/author_type_advice_subreddit_author_data=tokens_val_data.pt'
#MODEL_FILE=../../data/reddit_data/author_text_data/question_generation_model/checkpoint-215000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/
# reddit+author embed data
#TEST_DATA='../../data/reddit_data/author_text_data/author_embed_data/author_type_advice_subreddit_author_data=embeds_val_data.pt'
#MODEL_FILE=../../data/reddit_data/author_text_data/author_embed_data/question_generation_model/checkpoint-170500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_embed_data/
# no training
#OUT_DIR=../../data/reddit_data/no_train_model/
# CNN QA data
#TEST_DATA=../../data/reddit_data/advice_subreddit_val_data.pt
#MODEL_FILE=../../data/CNN_articles/cnn/question_generation_model/checkpoint-120500/pytorch_model.bin
#OUT_DIR=../../data/CNN_articles/cnn/
## metadata to test sub-sets of data
POST_METADATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
# model params
MODEL_CACHE_DIR=../../data/model_cache/
# regular model
#MODEL_TYPE='bart'
# embed model
MODEL_TYPE='bart_author'
# set GPU
export CUDA_VISIBLE_DEVICES=1

# no model (i.e. zero-shot)
#python test_question_generation.py $TEST_DATA --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR
# trained model
#python test_question_generation.py $TEST_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR
# trained model; testing on post subsets (e.g. community)
(python test_question_generation.py $TEST_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR --post_metadata $POST_METADATA)&
PID=$!
MAX_MEMORY=50000000000
prlimit --pid $PID --as=$MAX_MEMORY
