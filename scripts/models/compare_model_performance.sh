TEST_DATA=../../data/reddit_data/combined_data_test_data.pt
MODEL_OUTPUT_FILES=(../../data/reddit_data/text_only_model/test_data_sample_top_p\=0.9_temperature\=1.0_output_text.gz ../../data/reddit_data/author_text_data/test_data_sample_top_p\=0.9_temperature\=1.0_output_text.gz ../../data/reddit_data/author_text_data/author_attention_data/test_data_sample_top_p\=0.9_temperature\=1.0_output_text.gz ../../data/reddit_data/author_text_data/author_subreddit_embed_data/test_data_sample_top_p\=0.9_temperature\=1.0_output_text.gz ../../data/reddit_data/author_text_data/author_text_embed_data/test_data_sample_top_p\=0.9_temperature\=1.0_output_text.gz)
MODEL_NAMES=("text" "reader_token" "reader_attention", "reader_subreddit_embed", "reader_text_embed")
OUT_DIR=../../data/reddit_data/

python compare_model_performance.py $TEST_DATA --model_output_files "${MODEL_OUTPUT_FILES[@]}" --model_names "${MODEL_NAMES[@]}" --out_dir $OUT_DIR