"""
Test human evaluation data.
"""
import os
from argparse import ArgumentParser
from datetime import datetime
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_helpers import clean_survey_data, join_with_ground_truth_data, plot_quality_data
import pandas as pd
import krippendorff

def main():
    parser = ArgumentParser()
    parser.add_argument('survey_data_file')
    parser.add_argument('annotation_data_dir')
    parser.add_argument('--start_date', default=None)
    args = vars(parser.parse_args())
    survey_data_file = args['survey_data_file']
    annotation_data_dir = args['annotation_data_dir']
    start_date = args['start_date']

    ## load data
    survey_data = pd.read_csv(survey_data_file,
                              sep='\t', index_col=False, encoding='utf-16', skiprows=[1, 2],
                              converters={'RecordedDate': lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')})
    survey_data.rename(columns={'Q537': 'PROLIFIC_PID'}, inplace=True)
    # remove invalid data
    if(start_date is not None):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        survey_data = survey_data[(survey_data.loc[:, 'RecordedDate'] >= start_date)]
    # clean data
    flat_survey_data  = clean_survey_data(survey_data)
    quality_annotation_data, group_annotation_data = join_with_ground_truth_data(flat_survey_data, annotation_data_dir)

    ## get overall scores
    # quality
    model_var = 'question_text_type'
    rating_type_var = 'Q_rating_type'
    rating_val_var = 'annotation_num'
    overall_annotation_scores = quality_annotation_data.groupby([model_var, rating_type_var]).apply(lambda x: x.loc[:, rating_val_var].mean()).reset_index().pivot(index=model_var, columns=rating_type_var)
    # per-subreddit
    group_var = 'subreddit'
    per_subreddit_scores = get_scores_per_group(model_var, quality_annotation_data, rating_type_var, rating_val_var, group_var)
    # per social group
    group_var = 'group_category'
    per_social_group_scores = get_scores_per_group(model_var, quality_annotation_data, rating_type_var, rating_val_var, group_var)
    overall_annotation_scores.to_csv(os.path.join(annotation_data_dir, 'quality_scores.tsv'), sep='\t', index=False)
    per_subreddit_scores.to_csv(os.path.join(annotation_data_dir, 'quality_scores_subreddit.tsv'), sep='\t', index=False)
    per_social_group_scores.to_csv(os.path.join(annotation_data_dir, 'quality_scores_group.tsv'), sep='\t', index=False)

    ## get agreement


def get_scores_per_group(model_var, data, rating_type_var, rating_val_var, group_var):
    per_group_score_data = []
    for group_i, data_i in data.groupby(group_var):
        subreddit_score_data_i = data_i.groupby([model_var, rating_type_var]).apply(
            lambda x: x.loc[:, rating_val_var].mean()).reset_index().pivot(index=model_var, columns=rating_type_var)
        per_group_score_data.append(subreddit_score_data_i)
    per_group_score_data = pd.concat(per_group_score_data, axis=0)
    return per_group_score_data

if __name__ == '__main__':
    main()