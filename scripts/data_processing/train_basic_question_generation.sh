## CNN data
#TRAIN_DATA=../../data/CNN_articles/cnn/article_question_data.tsv
#OUT_DIR=../../data/CNN_articles/cnn/
## NYT data
TRAIN_DATA=../../data/nyt_comments/NYT_question_data.tsv
OUT_DIR=../../data/nyt_comments/
DEVICE="cuda:0"
MODEL_TYPE="bart"
python train_basic_question_generation.py $TRAIN_DATA $OUT_DIR --device $DEVICE --model_type $MODEL_TYPE