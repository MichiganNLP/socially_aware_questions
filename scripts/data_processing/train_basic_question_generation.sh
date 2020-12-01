TRAIN_DATA=../../data/CNN_articles/cnn/article_question_data.tsv
OUT_DIR=../../data/CNN_articles/cnn/
DEVICE="cuda:0"
MODEL_TYPE="bart"
python train_basic_question_generation.py $TRAIN_DATA $OUT_DIR --device $DEVICE --model_type $MODEL_TYPE