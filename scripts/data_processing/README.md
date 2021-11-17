# Data processing
This directory contains code for collecting and processing data for the task of question generation.

## Code execution order

Text data:
- Collect subreddit submissions:
    - bash collect_submissions_from_subreddits.sh
- Collect all comments on submissions:
    - bash collect_all_child_comments_from_submissions.sh
- Collect data for valid-question annotation:
    - bash get_sample_questions_for_annotation.sh
- After annotation, train (simple!) model to detect clarification questions:
    - bash process_annotated_data.sh
- Filter questions for validity:
    - bash combine_clean_question_comment_data.sh

Author data
- Collect prior comments from authors:
    - bash collect_prior_posts_from_question_authors.sh
- Compute author embeddings (from text, subreddits):
    - bash generate_author_embeddings.sh
- Extract data from prior comments (location, expertise, relative time of question):
    - bash extract_author_data_from_prior_comments.sh

All data:
- Combine and clean text and author data for generation:
    - bash clean_data_for_generation.sh
- Extract highly different questions for evaluation:
    - bash extract_different_author_questions_per_posh.sh

After training group-classification models (models/test_author_group_prediction.py)
- Extract questions that are specific to different reader groups:
    - bash extract_group_specific_questions.sh

Analysis:
- Data summary
    - compute_data_summary_statistics.ipynb
- Example questions per reader group
    - compare_reader_group_questions.ipynb#Get-example-questions-from-different-groups
- Comparing word counts per reader group
    - compare_reader_group_questions.ipynb#Identify-word/phrase-differences-across-groups
- Inspect valid-question data annotation
    - look_for_valid_questions_in_comments.ipynb#Test-data-from-annotators:-round-2
- Test validity of reader groups (e.g. is 75th percentile "good enough" for expertise?)
    - test_reader_group_validity.ipynb