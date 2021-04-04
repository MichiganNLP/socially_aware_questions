"""
Collect prior posts from question authors.
1. Sample M=1000 (larger?) authors, dates from each subreddit's comments.
2. For each author: collect N=100 posts prior to comment date.
"""
import os
from argparse import ArgumentParser
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
from data_helpers import load_reddit_api

np.random.seed(123)

def main():
    parser = ArgumentParser()
    parser.add_argument('comment_data')
    parser.add_argument('--sample_authors_per_group', type=int, default=1000)
    parser.add_argument('--sample_posts_per_author', type=int, default=100)
    parser.add_argument('--reddit_auth_data', default='../../data/auth_data/reddit_auth.csv')
    parser.add_argument('--out_dir', default='../../data/reddit_data/author_data/')
    args = vars(parser.parse_args())
    comment_data_file = args['comment_data']
    sample_authors_per_group = args['sample_authors_per_group']
    sample_posts_per_author = args['sample_posts_per_author']
    reddit_auth_data_file = args['reddit_auth_data']
    out_dir = args['out_dir']
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    comment_data = pd.read_csv(comment_data_file, sep='\t', compression='gzip', index_col=False, usecols=['author', 'subreddit', 'created_utc'])
    # fix date values
    comment_data = comment_data[comment_data.loc[:, 'created_utc'].apply(lambda x: type(x) is int)]
    comment_data = comment_data.assign(**{'post_date' : comment_data.loc[:, 'created_utc'].apply(lambda x: datetime.fromtimestamp(int(x)))})

    ## choose authors
    sample_authors = set()
    sample_author_start_dates = []
    for group_i, data_i in comment_data.groupby('subreddit'):
        authors_i = list(set(data_i.loc[:, 'author'].unique()) - sample_authors)
        sample_authors_i = np.random.choice(authors_i, sample_authors_per_group, replace=False)
        data_i = data_i[data_i.loc[:, 'author'].isin(sample_authors_i)]
        # get earliest post date
        author_min_post_dates_i = data_i.groupby('author').apply(lambda x: x.loc[:, 'post_date'].min()).reset_index().rename(columns={0:'post_date'})
        sample_authors.update(sample_authors_i)
        sample_author_start_dates.append(author_min_post_dates_i)
    sample_author_start_dates = pd.concat(sample_author_start_dates, axis=0)
    print(f'{sample_author_start_dates.shape[0]} author/start date combos')

    ## collect prior posts
    ## TODO: collect from files on disk rather than API queries? annoying to set up but less download/rate-limiting errors
    ## TODO: save to database for easier retrieval
    reddit_api, pushshift_reddit_api = load_reddit_api(reddit_auth_data_file)
    data_cols = ['author', 'subreddit', 'body', 'created_utc', 'edited', 'id', 'author_flair_text']
    for idx_i, data_i in tqdm(sample_author_start_dates.iterrows()):
        author_i = data_i.loc['author']
        min_date_i = data_i.loc['post_date'].timestamp()
        out_file_i = os.path.join(out_dir, f'{author_i}_comments.gz')
        if(not os.path.exists(out_file_i)):
            comments_i = pushshift_reddit_api.search_comments(
                author=author_i,
                end=min_date_i,
                limit=sample_posts_per_author,
            )
            # convert to useful format
            comment_data_i = list(map(lambda x: x.__dict__, comments_i))
            comment_data_i = list(map(lambda x: pd.Series({data_col : x[data_col] for data_col in data_cols}), comment_data_i))
            ## write to file
            if(len(comment_data_i) > 0):
                comment_data_i = pd.DataFrame(comment_data_i)
                comment_data_i.to_csv(out_file_i, sep='\t', compression='gzip', index=False)

if __name__ == '__main__':
    main()