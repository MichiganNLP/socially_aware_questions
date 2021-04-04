"""
Get sample questions for annotation:
(1) relevance; (2) asking for more information.
"""
import os
import re
rom argparse import ArgumentParser
import numpy as np
np.random.seed(123)
import pandas as pd
from data_helpers import load_zipped_json_data

def main():
    parser = ArgumentParser()
    parser.add_argument('question_data')
    parser.add_argument('post_data')
    parser.add_argument('--sample_questions_per_group', type=int, default=200)
    parser.add_argument('--out_dir', default='../../data/reddit_data/')
    args = vars(parser.parse_args())
    question_data_file = args['question_data']
    post_data_file = args['post_data']
    sample_questions_per_group = args['sample_questions_per_group']
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

if __name__ == '__main__':
    main()
