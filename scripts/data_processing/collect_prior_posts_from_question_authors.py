"""
Collect prior posts from question authors.
1. Sample M=1000 (larger?) authors, dates from each subreddit's comments.
2. For each author: collect N=100 posts prior to comment date.
"""
import os
import re
from argparse import ArgumentParser
from math import ceil

import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
from data_helpers import load_reddit_api, load_all_author_data, write_flush_data

np.random.seed(123)

def sample_authors_to_collect(comment_data, sample_authors_per_group):
    sample_authors = set()
    sample_author_start_dates = []
    for group_i, data_i in comment_data.groupby('subreddit'):
        authors_i = list(set(data_i.loc[:, 'author'].unique()) - sample_authors)
        sample_authors_per_group_i = min(sample_authors_per_group, len(authors_i))
        sample_authors_i = np.random.choice(authors_i, sample_authors_per_group_i, replace=False)
        data_i = data_i[data_i.loc[:, 'author'].isin(sample_authors_i)]
        # get earliest post date
        author_min_post_dates_i = data_i.groupby('author').apply(lambda x: x.loc[:, 'post_date'].min()).reset_index().rename(columns={0: 'post_date'})
        sample_authors.update(sample_authors_i)
        sample_author_start_dates.append(author_min_post_dates_i)
    sample_author_start_dates = pd.concat(sample_author_start_dates, axis=0)
    print(f'{sample_author_start_dates.shape[0]} author/start date combos')
    return sample_author_start_dates


def collect_prior_comments(out_dir, reddit_auth_data_file, rewrite_author_files, sample_author_start_dates, sample_posts_per_author):
    reddit_api, pushshift_reddit_api = load_reddit_api(reddit_auth_data_file)
    data_cols = ['author', 'subreddit', 'body', 'created_utc', 'edited', 'id', 'author_flair_text', 'parent_id', 'reply_delay']
    for idx_i, data_i in tqdm(sample_author_start_dates.iterrows()):
        author_i = data_i.loc['author']
        min_date_i = data_i.loc['post_date']
        out_file_i = os.path.join(out_dir, f'{author_i}_comments.gz')
        if (not os.path.exists(out_file_i) or rewrite_author_files):
            # tmp debugging
            # print(f'about to search for N={sample_posts_per_author} comments from author={author_i}')
            comments_i = pushshift_reddit_api.search_comments(
                author=author_i,
                until=min_date_i,
                limit=sample_posts_per_author,
            )
            # convert to useful format
            comment_data_i = list(map(lambda x: x.__dict__, comments_i))
            # tmp debugging
            # print(f'comment data {comment_data_i}')
            comment_data_i = list(map(lambda x: pd.Series({data_col: x[data_col] for data_col in data_cols}), comment_data_i))
            ## write to file
            if (len(comment_data_i) > 0):
                comment_data_i = pd.DataFrame(comment_data_i)
                comment_data_i.to_csv(out_file_i, sep='\t', compression='gzip', index=False)

def main():
    parser = ArgumentParser()
    parser.add_argument('comment_data')
    parser.add_argument('--sample_authors_per_group', type=int, default=1000)
    parser.add_argument('--sample_posts_per_author', type=int, default=100)
    parser.add_argument('--reddit_auth_data', default='../../data/auth_data/reddit_auth.csv')
    parser.add_argument('--out_dir', default='../../data/reddit_data/author_data/')
    parser.add_argument('--rewrite_author_files', action='store_true', default=False)
    args = vars(parser.parse_args())
    comment_data_file = args['comment_data']
    sample_authors_per_group = args['sample_authors_per_group']
    sample_posts_per_author = args['sample_posts_per_author']
    reddit_auth_data_file = args['reddit_auth_data']
    out_dir = args['out_dir']
    rewrite_author_files = args['rewrite_author_files']
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    comment_data = pd.read_csv(comment_data_file, sep='\t', compression='gzip', index_col=False, usecols=['author', 'subreddit', 'created_utc'])
    # fix date values
    comment_data = comment_data[comment_data.loc[:, 'created_utc'].apply(lambda x: type(x) is int)]
    comment_data = comment_data.assign(**{'post_date' : comment_data.loc[:, 'created_utc'].apply(lambda x: datetime.fromtimestamp(int(x)))})

    ## choose authors
    sample_author_start_dates = sample_authors_to_collect(comment_data, sample_authors_per_group)

    ## collect prior posts
    ## TODO: save to database for easier retrieval
    # print(f'mining author comments')
    # collect_prior_comments(out_dir, reddit_auth_data_file, rewrite_author_files, sample_author_start_dates, sample_posts_per_author)

    ## TODO: collect parent post data ("slow" vs. "fast" response) => parent_id | author | created_utc | post_title
    # get parent IDs from non-edited comments
    author_comment_data = load_all_author_data(out_dir, usecols=['created_utc', 'parent_id', 'author', 'edited'])
    # print(author_comment_data.loc[:, 'edited'].value_counts())
    author_comment_data = author_comment_data[~author_comment_data.loc[:, 'edited']]
    # fix parent IDs
    author_comment_data = author_comment_data.assign(**{
        'parent_id' : author_comment_data.loc[:, 'parent_id'].apply(lambda x: x.split('_')[1] if '_' in x else x)
    })
    comment_parent_post_ids = author_comment_data.loc[:, 'parent_id'].unique()
    reddit_api, pushshift_reddit_api = load_reddit_api(reddit_auth_data_file)
    parent_post_data_file = os.path.join(out_dir, f'comment_parent_post_data.gz')
    old_parent_post_data = []
    if(os.path.exists(parent_post_data_file)):
        old_parent_post_data = pd.read_csv(parent_post_data_file, sep='\t', compression='gzip', index_col=False)
        comment_parent_post_ids = list(filter(lambda x: x not in old_parent_post_data.loc[:, 'id'].unique(), comment_parent_post_ids))
    chunk_size = 500
    chunk_count = int(ceil(len(comment_parent_post_ids) / chunk_size))
    parent_post_data = []
    comment_parent_cols = ['author', 'edited', 'created_utc', 'subreddit', 'title', 'id']
    write_ctr = 10
    print(f'mining {len(comment_parent_post_ids)} parent post IDs')
    for i in tqdm(range(chunk_count)):
        post_ids_i = comment_parent_post_ids[(i*chunk_size):((i+1)*chunk_size)]
        posts_i = pushshift_reddit_api.search_submissions(ids=post_ids_i, fields=comment_parent_cols, metadata=True, limit=chunk_size)
        post_data_i = pd.DataFrame(list(map(lambda x: [x.__dict__.get(y) for y in comment_parent_cols], posts_i)), columns=comment_parent_cols)
        # tmp debugging
        # print(f'post data {post_data_i}')
        # fix "author" and "subreddit" fields
        post_data_i = post_data_i.assign(**{'author' : post_data_i.loc[:, 'author'].apply(lambda x: x.name if x is not None else x)})
        post_data_i = post_data_i.assign(**{'subreddit' : post_data_i.loc[:, 'subreddit'].apply(lambda x: x.name if x is not None else x)})
        # post_data_i = list(map(lambda x: [x.__dict__.get(y) for y in comment_parent_cols], posts_i))
        if(len(post_data_i) > 0):
        # tmp debugging
        # print(f'post data sample = {post_data_i.head(10)}')
        #     post_data_i = post_data_i.loc[:, comment_parent_cols]
            parent_post_data.append(post_data_i)
        #     parent_post_data.extend(post_data_i)
            # tmp debugging
            # print(f'parent post data has shape {pd.DataFrame(parent_post_data[:10]).shape}')
        if(i % write_ctr == 0 and len(parent_post_data) > 0):
            parent_post_data = write_flush_data(comment_parent_cols, parent_post_data_file, parent_post_data)
    # remove duplicate parent posts, etc.
    if(len(parent_post_data) > 0):
        write_flush_data(comment_parent_cols, parent_post_data_file, parent_post_data)
        parent_post_data = pd.read_csv(parent_post_data_file, sep='\t', index_col=False, compression='gzip')
        parent_post_data.drop_duplicates('id', inplace=True)
        # write out
        parent_post_data.to_csv(parent_post_data_file, sep='\t', index=False, compression='gzip')
    ## add parent post time to comment data; rewrite comment data
    parent_post_data = pd.read_csv(parent_post_data_file, sep='\t', index_col=False, compression='gzip')
    parent_post_data.rename(columns={'id' : 'parent_id', 'created_utc' : 'parent_created_utc'}, inplace=True)
    author_file_matcher = re.compile('.+_comments.gz')
    author_files = list(filter(lambda x: author_file_matcher.match(x) is not None, os.listdir(out_dir)))
    author_files = list(map(lambda x: os.path.join(out_dir, x), author_files))
    print(f'adding post data to author files')
    for author_file_i in tqdm(author_files):
        try:
            author_data_i = pd.read_csv(author_file_i, sep='\t', index_col=False, compression='gzip')
            author_data_i = author_data_i.assign(**{'parent_id' : author_data_i.loc[:, 'parent_id'].apply(lambda x: x.split('_')[1])})
            parent_post_data_i = parent_post_data[parent_post_data.loc[:, 'parent_id'].isin(author_data_i.loc[:, 'parent_id'].unique())]
            if(parent_post_data_i.shape[0] > 0):
                # print(f'found parent post data for author data = {os.path.basename(author_file_i)}')
                # print(f'parent post data size = {parent_post_data_i.shape[0]}')
                # get rid of old parent post data
                if('parent_created_utc' in author_data_i.columns):
                    author_data_i.drop('parent_created_utc', axis=1, inplace=True)
                author_data_i = pd.merge(author_data_i, parent_post_data.loc[:, ['parent_id', 'parent_created_utc']], on='parent_id', how='left')
                author_data_i = author_data_i.assign(**{
                    'reply_delay' : author_data_i.loc[:, 'created_utc'] - author_data_i.loc[:, 'parent_created_utc']
                })
                # print(f'updated author data with reply_delay = {author_data_i.loc[:, "reply_delay"]}')
                author_data_i.to_csv(author_file_i, sep='\t', compression='gzip', index=False)
                # break
        except Exception as e:
            print(f'error with author data {os.path.basename(author_file_i)}: {e}')

if __name__ == '__main__':
    main()