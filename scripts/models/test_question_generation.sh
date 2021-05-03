# test question generation
# data params
# no training
#OUT_DIR=../../data/reddit_data/no_train_model/
# CNN QA data
#TEST_DATA=../../data/reddit_data/advice_subreddit_val_data.pt
#MODEL_FILE=../../data/CNN_articles/cnn/question_generation_model/checkpoint-120500/pytorch_model.bin
#OUT_DIR=../../data/CNN_articles/cnn/
# regular training reddit data
TRAIN_DATA=../../data/reddit_data/combined_data_train_data.pt
TEST_DATA=../../data/reddit_data/combined_data_val_data.pt
## models
# text only
MODEL_FILE=../../data/reddit_data/text_only_model/question_generation_model/checkpoint-198500/pytorch_model.bin
MODEL_TYPE='bart'
OUT_DIR=../../data/reddit_data/text_only_model/
# reddit+author token
#MODEL_FILE=../../data/reddit_data/author_text_data/question_generation_model/checkpoint-215000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/
#MODEL_TYPE='bart_author_embeds_token'
# reddit + author embed (+ encoder)
#MODEL_FILE=../../data/reddit_data/author_text_data/author_embed_data/question_generation_model/checkpoint-170500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_embed_data/
#MODEL_TYPE='bart_author_embeds'
# reddit + author embed (+ decoder)
#MODEL_FILE=../../data/reddit_data/author_text_data/author_decoder_embed_data/question_generation_model/checkpoint-170500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_decoder_embed_data
#MODEL_TYPE='bart_author_embeds'
# reddit + text embed (+ encoder)
#MODEL_FILE=../../data/reddit_data/author_text_data/author_text_embed_data/question_generation_model/checkpoint-188500/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_text_embed_data/
#MODEL_TYPE='bart_author_embeds'
# reddit + author group attention
#MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/question_generation_model/checkpoint-215000/pytorch_model.bin
#OUT_DIR=../../data/reddit_data/author_text_data/author_attention_data/
#MODEL_TYPE='bart_author_embeds_attention'
# metadata to test sub-sets of data
POST_METADATA=../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
# model params
MODEL_CACHE_DIR=../../data/model_cache/
# author group attention model

# set GPU
export CUDA_VISIBLE_DEVICES=1

# no model (i.e. zero-shot)
#(python test_question_generation.py $TEST_DATA --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR --post_metadata $POST_METADATA)&
# trained model
#python test_question_generation.py $TEST_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR
# trained model + test on post subsets (e.g. community)
(python test_question_generation.py $TEST_DATA --train_data $TRAIN_DATA --model_file $MODEL_FILE --model_cache_dir $MODEL_CACHE_DIR --model_type $MODEL_TYPE --out_dir $OUT_DIR --post_metadata $POST_METADATA)&
PID=$!
MAX_MEMORY=60000000000 # 50G
prlimit --pid $PID --as=$MAX_MEMORY