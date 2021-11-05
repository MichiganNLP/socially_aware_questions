TEST_DATA_FILE=../../data/reddit_data/combined_data_test_data.pt
TEXT_MODEL_DATA_FILE=../../data/reddit_data/text_only_model/test_data_output_text.gz
TEXT_MODEL_FILE=../../data/reddit_data/text_only_model/question_generation_model/checkpoint-129000/pytorch_model.bin
# reader-attention
#READER_MODEL_DATA_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=encoder_config\=attnconcat/test_data_output_text.gz
#READER_MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=encoder_config\=attnconcat/question_generation_model/checkpoint-129000/pytorch_model.bin
#READER_MODEL_TYPE='bart_author_attention'
# reader-token
READER_MODEL_DATA_FILES=(../../data/reddit_data/author_text_data/question_generation_model/test_data_output_text.gz ../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=encoder_config\=attnconcat/test_data_output_text.gz)
READER_MODEL_FILES=(../../data/reddit_data/author_text_data/question_generation_model/checkpoint-129000/pytorch_model.bin ../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=encoder_config\=attnconcat/question_generation_model/checkpoint-129000/pytorch_model.bin)
READER_MODEL_TYPES=('bart_author_token' 'bart_author_attention')
OUT_DIR=../../data/reddit_data/annotation_data/generated_text_evaluation/compare_model_output_round_2/
FILTER_DATA_FILE=../../data/reddit_data/paired_question_low_sim_simpct\=10_data.gz
#FILTER_DATA_FILE=../../data/reddit_data/paired_question_low_sim_simpct\=25_data.gz
export CUDA_VISIBLE_DEVICES=0
python get_sample_questions_for_annotation_qualtrics.py $TEST_DATA_FILE $TEXT_MODEL_DATA_FILE $OUT_DIR --filter_data_file $FILTER_DATA_FILE --text_model_file $TEXT_MODEL_FILE --reader_model_data_files "${READER_MODEL_DATA_FILES[@]}" --reader_model_files "${READER_MODEL_FILES[@]}" --reader_model_types "${READER_MODEL_TYPES[@]}"
