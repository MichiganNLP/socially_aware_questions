# Models

This directory contains code for prediction and generation.

## Code execution order

- Predict reader groups from questions + posts
    - bash test_author_group_prediction.sh
- Train generation model
    - bash train_basic_question_generation.sh
- Compare generation performance between different models
    - bash compare_model_performance.sh
- Sample model output for quality annotation
    - bash get_sample_questions_for_annotation_qualtrics.sh

Notebooks:
- Compare errors made by different models
    - error_analysis.ipynb
- Test question generation for different reader groups
    - test_model_response_to_reader_input.ipynb