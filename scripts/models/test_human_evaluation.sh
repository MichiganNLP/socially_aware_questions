SURVEY_DATA_FILE="data/reddit_data/annotation_data/generated_text_evaluation/compare_model_output_round_2/reader_question_generation_prolific_March\ 11\,\ 2022_12.30.tsv"
ANNOTATION_DATA_DIR="data/reddit_data/annotation_data/generated_text_evaluation/compare_model_output_round_2/"
START_DATE="2021-11-06"

python test_human_evaluation.py $SURVEY_DATA_FILE $ANNOTATION_DATA_DIR --start_date $START_DATE