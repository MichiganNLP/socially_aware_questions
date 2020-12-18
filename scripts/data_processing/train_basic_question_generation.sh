## CNN data
#TRAIN_DATA=../../data/CNN_articles/cnn/article_question_data.tsv
#OUT_DIR=../../data/CNN_articles/cnn/
## NYT data
TRAIN_DATA=../../data/nyt_comments/NYT_question_data.tsv
# regular model
#OUT_DIR=../../data/nyt_comments/
# model with author information
OUT_DIR=../../data/nyt_comments/author_data_model/
DEVICE="cuda:0"
#DEVICE='cpu' # debug evaluation output on CPU with small data ;_;
MODEL_TYPE="bart"
# model with author information
AUTHOR_DATA=../../data/nyt_comments/author_comment_social_data.tsv
# sample data for faster training
#SAMPLE_PCT=1.0
SAMPLE_PCT=0.25
# regular model
#python train_basic_question_generation.py $TRAIN_DATA $OUT_DIR --device $DEVICE --model_type $MODEL_TYPE
python train_basic_question_generation.py $TRAIN_DATA $OUT_DIR --device $DEVICE --model_type $MODEL_TYPE --author_data $AUTHOR_DATA --sample_pct $SAMPLE_PCT