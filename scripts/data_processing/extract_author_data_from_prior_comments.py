"""
Extract author-specific data from prior
comments to provide to generation model:

- "expert" vs. "novice" (degree of prior posting in group)
- "early" vs. "late" (time of response relative to original post time)
"""
import os
import re
from argparse import ArgumentParser
from datetime import datetime

import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('author_data_dir')
    parser.add_argument('question_data') # need question data for time of author questions => get all comments before specified time(s)
    parser.add_argument('post_data')
    args = vars(parser.parse_args())
    author_data_dir = args['author_data_dir']
    question_data_file = args['question_data']
    post_data_file = args['post_data']

    ## load existing data
    post_data = pd.read_csv(post_data_file, sep='\t', index_col=False, compression='gzip', usecols=['id', 'created_utc'])
    question_data = pd.read_csv(question_data_file, sep='\t', index_col=False, compression='gzip', usecols=['author', 'created_utc', 'parent_id'])
    # get date info
    post_data = post_data.assign(**{'parent_date' : post_data.loc[:, 'created_utc'].apply(lambda x: datetime.fromtimestamp(x))})
    question_data = question_data.assign(**{'date': question_data.loc[:, 'created_utc'].apply(lambda x: datetime.fromtimestamp(x))})
    # add to question data
    post_data.rename(columns={'id' : 'parent_id'}, inplace=True)
    question_data = question_data.assign(**{'parent_id' : question_data.loc[:, 'parent_id'].apply(lambda x: x.split('_')[-1])})
    question_data = pd.merge(question_data, post_data.loc[:, ['parent_id', 'parent_date']], on='parent_id')

    ## iterate over all author data
    ## extract (1) expertise (2) relative time
    author_data = []
    author_data_cols = ['author', 'date', 'expert_pct', 'relative_time']
    author_file_matcher = re.compile('.+_comments.gz')
    author_data_files = list(filter(lambda x: author_file_matcher.match(x) is not None, os.listdir(author_data_dir)))
    # tmp debugging
    author_data_files = author_data_files[:1000]
    for author_file_i in author_data_files:
        author_i = author_file_i.replace('_comments.gz', '')
        author_comment_file_i = os.path.join(author_data_dir, author_file_i)
        author_comment_data_i = pd.read_csv(author_comment_file_i, sep='\t', compression='gzip', usecols=['author', 'subreddit', 'created_utc'])
        question_data_i = question_data[question_data.loc[:, 'author']==author_i].drop_duplicates(['author', 'parent_id'])
        for idx_j, data_j in question_data_i.iterrows():
            # expertise
            date_j = data_j.loc['date']
            subreddit_j = data_j.loc['subreddit']
            author_prior_comment_data_j = author_comment_data_i[author_comment_data_i.loc[:, 'created_utc'] <= date_j]
            relevant_prior_comment_data_j = author_prior_comment_data_j[author_prior_comment_data_j.loc[:, 'subreddit']==subreddit_j]
            expertise_pct_j = relevant_prior_comment_data_j.shape[0] / author_prior_comment_data_j.shape[0]
            # relative time
            post_date_j = data_j.loc['parent_date']
            relative_time_j = (post_date_j - date_j).seconds
            combined_author_data_j = pd.Series([author_i, date_j, expertise_pct_j, relative_time_j])
            author_data.append(combined_author_data_j)
    author_data = pd.DataFrame(author_data, columns=author_data_cols)

    ## TODO: convert to categorical with percentiles etc.

    ## save
    author_data_out_file = os.path.join(author_data_dir, 'author_prior_comment_data.gz')
    author_data.to_csv(author_data_out_file, sep='\t', compression='gzip', index=False)

if __name__ == '__main__':
    main()