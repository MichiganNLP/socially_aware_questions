## CNN data
TRAIN_DATA=../../data/CNN_articles/cnn/CNN_long_train_data.pt
VAL_DATA=../../data/CNN_articles/cnn/CNN_long_val_data.pt
## NYT data
#TRAIN_DATA=../../data/nyt_comments/NYT_train_data.pt
#VAL_DATA=../../data/nyt_comments/NYT_val_data.pt
# regular model
#OUT_DIR=../../data/nyt_comments/
# model with author information
#OUT_DIR=../../data/nyt_comments/author_data_model/
#OUT_DIR=../../data/nyt_comments/cnn_fine_tune/
# longformer model
OUT_DIR=../../data/CNN_articles/cnn/longformer_model/
# debug model
#OUT_DIR=../../data/nyt_comments/debug_model/
# regular transformer
#MODEL_CACHE_DIR=../../data/nyt_comments/author_data_model/model_cache/
# longformer FML
MODEL_CACHE_DIR=../../data/longformer_cache/
DEVICE="cuda:2"
#DEVICE='cpu' # debug with small data ;_;
#MODEL_TYPE="bart"
MODEL_TYPE='longformer'
# model with author information
#AUTHOR_DATA=../../data/nyt_comments/author_comment_social_data.tsv
# optional: pretrained model
#PRETRAINED_MODEL=../../data/CNN_articles/cnn/question_generation_model/checkpoint-120500/pytorch_model.bin
export CUDA_VISIBLE_DEVICES=2
# regular model
python train_basic_question_generation.py $TRAIN_DATA $VAL_DATA $OUT_DIR --device $DEVICE --model_type $MODEL_TYPE --model_cache_dir $MODEL_CACHE_DIR
#python train_basic_question_generation.py $TRAIN_DATA $VAL_DATA $OUT_DIR --device $DEVICE --model_type $MODEL_TYPE --model_cache_dir $MODEL_CACHE_DIR --pretrained_model $PRETRAINED_MODEL