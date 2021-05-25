# Data processing
This directory contains code for collecting and processing data for the task of question generation.

## Code execution order

Text data:
- Collect subreddit submissions:
    - bash collect_submissions_from_subreddits.sh
- Collect all comments on submissions:
    - bash collect_all_child_comments_from_submissions.sh
- Collect data for annotation (valid questions):
    - bash get_sample_questions_for_annotation.sh
- After annotation, train (simple!) model to detect clarification questions:
    - bash process_annotated_data.sh
- Filter questions for validity:
    - bash combine_clean_question_comment_data.sh

Author data
- Collect prior comments from authors:
    - bash collect_prior_posts_from_question_authors.sh
- Compute author embeddings (e.g. from text, subreddits):
    - bash generate_author_embeddings.sh
- Extract data from prior comments (location, expertise, relative time of question):
    - bash extract_author_data_from_prior_comments.sh

All data:
- Combine and clean text and author data for generation:
    - bash clean_data_for_generation.sh
