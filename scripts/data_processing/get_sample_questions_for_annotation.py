"""
Get sample questions for annotation:
(1) relevance; (2) asking for more information.
"""
import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from functools import reduce

import numpy as np
np.random.seed(123)
import pandas as pd
from data_helpers import load_zipped_json_data

def main():
    parser = ArgumentParser()
    parser.add_argument('question_data')
    parser.add_argument('post_data')
    parser.add_argument('--sample_questions_per_group', type=int, default=200)
    parser.add_argument('--annotators', type=int, default=1)
    parser.add_argument('--annotators_per_question', type=int, default=2)
    parser.add_argument('--out_dir', default='../../data/reddit_data/')
    args = vars(parser.parse_args())
    question_data_file = args['question_data']
    post_data_file = args['post_data']
    sample_questions_per_group = args['sample_questions_per_group']
    annotators = args['annotators']
    annotators_per_question = args['annotators_per_question']
    out_dir = args['out_dir']
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    ## load data
    question_data = pd.read_csv(question_data_file, sep='\t', compression='gzip', index_col=False)
    # remove edited posts
   # question_data = question_data[question_data.loc[:, 'edited'].apply(lambda x: type(x) is bool and not x)]
    # fix parent ID
    question_data = question_data.assign(**{'parent_id' : question_data.loc[:, 'parent_id'].apply(lambda x: x.split('_')[-1])})
    # print(question_data.loc[:, 'parent_id'].iloc[:10])
    # post_data = pd.read_csv(post_data_file, sep='\t', compression='gzip', index_col=False, usecols=['id', 'selftext', 'title', 'edited'])
    post_data = load_zipped_json_data(post_data_file)
    # remove edited posts
    post_data = post_data[post_data.loc[:, 'edited'].apply(lambda x: type(x) is bool and not x)]
    # print(post_data.loc[:, 'id'].iloc[:10])
    # fix columns
    post_data.rename(columns={'id':'parent_id', 'selftext':'post_text', 'title':'post_title'}, inplace=True)
    # fix post text
    return_matcher = re.compile('[\n\r]')
    post_data = post_data.assign(**{'post_text': post_data.loc[:, 'post_text'].apply(lambda x: return_matcher.sub('', x))})
    # combine data!!
    question_post_data = pd.merge(question_data, post_data.loc[:, ['parent_id', 'post_text', 'post_title']], on='parent_id')
    # print(f'post data cols {question_post_data.columns}')

    ## sample data
    sample_data = []
    data_cols = ['post_title', 'post_text', 'question', 'parent_id']
    # tmp debugging
    print(f'subreddit counts = {question_post_data.loc[:, "subreddit"].value_counts()}')
    for group_i, data_i in question_post_data.groupby('subreddit'):
        print(f'group = {group_i}, data = {data_i.shape[0]}')
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
    data_name = os.path.basename(question_data_file).replace('.gz', '_annotation_sample.tsv')
    out_file = os.path.join(out_dir, data_name)
    sample_data.to_csv(out_file, sep='\t', index=False)

    ## optional: split by annotator
    ## split by annotator
    if (annotators > 1):
        ## assign same number of subreddit posts per annotator
        annotator_data = defaultdict(list)
        annotator_idx = list(range(annotators))
        for subreddit_i, data_i in sample_data.groupby('subreddit'):
            questions_per_annotator_i = int(data_i.shape[0] / annotators * annotators_per_question)
            data_idx_j = list(reduce(lambda x,y: x+y, [data_i.index.tolist(),]*annotators_per_question))
            # ex. [0,66], [66,123], [123:200]
            for annotator_j, annotator_id_j in enumerate(annotator_idx):
                annotator_data[annotator_id_j] += data_i.loc[data_idx_j[(annotator_j*questions_per_annotator_i):((annotator_j+1)*questions_per_annotator_i)]]
            ## rotate annotator IDs to distribute data evenly
            annotator_idx = annotator_idx[1:] + [annotator_idx[0]]
        ## save each annotator data file separately
        for annotator_id_i, annotator_data_i in annotator_data.items():
            annotator_data_i = pd.concat(annotator_data_i, axis=1)
            annotator_data_file_i = os.path.join(out_dir, data_name.replace('.tsv', f'_{annotator_id_i}.tsv'))
            annotator_data_i.to_csv(annotator_data_file_i, sep='\t', index=True)

if __name__ == '__main__':
    main()
