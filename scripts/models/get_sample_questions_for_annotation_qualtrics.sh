TEST_DATA_FILE=../../data/reddit_data/combined_data_test_data.pt
TEXT_MODEL_DATA_FILE=../../data/reddit_data/text_only_model/test_data_output_text.gz
TEXT_MODEL_FILE=../../data/reddit_data/text_only_model/question_generation_model/checkpoint-129000/pytorch_model.bin
READER_MODEL_DATA_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=encoder_config\=attnconcat/test_data_output_text.gz
READER_MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/author_attention_layer\=5_location\=encoder_config\=attnconcat/question_generation_model/checkpoint-129000/pytorch_model.bin
OUT_DIR=../../data/reddit_data/annotation_data/generated_text_evaluation/compare_model_output_round_2/
FILTER_DATA_FILE=../../data/reddit_data/paired_question_low_sim_simpct\=25_data.gz
python get_sample_questions_for_annotation_qualtrics.py $TEST_DATA_FILE $TEXT_MODEL_DATA_FILE $READER_MODEL_DATA_FILE $READER_MODEL_FILE $OUT_DIR --filter_data_file $FILTER_DATA_FILE