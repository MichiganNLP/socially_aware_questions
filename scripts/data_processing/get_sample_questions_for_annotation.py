"""
Get sample questions for annotation:
(1) relevance; (2) asking for more information.
"""
import os
from argparse import ArgumentParser
import numpy as np
np.random.seed(123)
import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('question_data')
    parser.add_argument('--sample_questions_per_group', type=int, default=200)
    parser.add_argument('--out_dir', default='../../data/reddit_data/')
    args = vars(parser.parse_args())
    question_data_file = args['question_data']
    sample_questions_per_group = args['sample_questions_per_group']
    out_dir = args['out_dir']

    ## load data
    question_data = pd.read_csv(question_data_file, sep='\t', compression='gzip', index_col=False)
    # remove edited posts
    question_data = question_data[question_data.loc[:, 'parent_edited'].apply(lambda x: type(x) is bool and not x)]

    ## sample data
    sample_data = []
    data_cols = ['parent_title', 'parent_text', 'question']
    for group_i, data_i in question_data.groupby('subreddit'):
        data_i = data_i.loc[:, data_cols]
        sample_data_i = data_i.loc[np.random.choice(data_i.index, sample_questions_per_group, replace=False), :]
        sample_data.append(sample_data_i)
    sample_data = pd.concat(sample_data, axis=0)
    # add annotation columns
    sample_data = sample_data.assign(**{
        'question_is_relevant': -1,
        'question_asks_for_more_info' : -1,
    })

    ## write to file
    data_name = question_data_file.replace('.gz', '_annotation_sample.tsv')
    out_file = os.path.join(out_dir, data_name)
    sample_data.to_csv(out_file, sep='\t', index=False)

if __name__ == '__main__':
    main()