# evaluate question generation based on test data, repetition, and copying training data
## CNN evaluation
#MODEL_FILE=../../data/CNN_articles/cnn/question_generation_model/checkpoint-120500/pytorch_model.bin
#OUT_DIR=../../data/CNN_articles/cnn/
#OUT_DIR=../../data/CNN_articles/cnn/NYT_eval/
## NYT evaluation
# w/out author
MODEL_FILE=../../data/nyt_comments/question_generation_model/checkpoint-96000/pytorch_model.bin
OUT_DIR=../../data/nyt_comments/no_author_data/
# w/author
#MODEL_FILE=../../data/nyt_comments/author_data_model/question_generation_model/checkpoint-73500/pytorch_model.bin
## CNN+NYT model evaluation
#MODEL_FILE=../../data/nyt_comments/cnn_fine_tune/question_generation_model/checkpoint-141000/pytorch_model.bin
#OUT_DIR=../../data/nyt_comments/cnn_fine_tune/
### CNN data
#TRAIN_DATA=../../data/CNN_articles/cnn/article_question_generation_train_data.pt
#TEST_DATA=../../data/CNN_articles/cnn/article_question_generation_val_data.pt
### NYT data
# w/out author
TRAIN_DATA=../../data/nyt_comments/no_author_data/NYT_train_data.pt
TEST_DATA=../../data/nyt_comments/no_author_data/NYT_val_data.pt
# w/ author
#TRAIN_DATA=../../data/nyt_comments/author_data_model/author_type_NYT_question_data_train_data.pt
#TEST_DATA=../../data/nyt_comments/author_data_model/author_type_NYT_question_data_val_data.pt
# w/out author
MODEL_CACHE_DIR=../../data/CNN_articles/cnn/model_cache/
# w/ author
#MODEL_CACHE_DIR=../../data/nyt_comments/author_data_model/model_cache/
MODEL_TYPE='bart'
DEVICE_NAME='cuda:2'
python evaluate_question_generation.py $MODEL_FILE $OUT_DIR $TRAIN_DATA $TEST_DATA --model_type $MODEL_TYPE --device_name $DEVICE_NAME --model_cache_dir $MODEL_CACHE_DIR