TEST_DATA_FILE=../../data/reddit_data/combined_data_test_data.pt
TEXT_MODEL_DATA_FILE=../../data/reddit_data/text_only_model/test_data_output_text.gz
READER_MODEL_DATA_FILE=../../data/reddit_data/author_text_data/author_attention_data/test_data_output_text.gz
READER_MODEL_FILE=../../data/reddit_data/author_text_data/author_attention_data/question_generation_model/checkpoint-275500/pytorch_model.bin
OUT_DIR=../../data/reddit_data/annotation_data/generated_text_evaluation/compare_model_output/
FILTER_DATA_FILE=../../data/reddit_data/paired_question_low_sim_simpct\=10_data.gz
python get_sample_questions_for_annotation_qualtrics.py $TEST_DATA_FILE $TEXT_MODEL_DATA_FILE $READER_MODEL_DATA_FILE $READER_MODEL_FILE $OUT_DIR --filter_data_file $FILTER_DATA_FILE