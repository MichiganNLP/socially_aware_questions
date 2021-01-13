# evaluate question generation based on test data, repetition, and copying training data
## NYT evaluation
MODEL_FILE=../../data/nyt_comments/author_data_model/question_generation_model/checkpoint-73500/pytorch_model.bin
OUT_DIR=../../data/nyt_comments/author_data_model/
TRAIN_DATA=../../data/nyt_comments/author_data_model/author_type_NYT_question_data_train_data.pt
TEST_DATA=../../data/nyt_comments/author_data_model/author_type_NYT_question_data_val_data.pt
MODEL_TYPE='bart'
DEVICE_NAME='cuda:2'
python evaluate_question_generation.py $MODEL_FILE $OUT_DIR $TRAIN_DATA $TEST_DATA --model_type $MODEL_TYPE --device_name $DEVICE_NAME