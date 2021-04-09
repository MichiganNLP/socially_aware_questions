"""
Generate author embeddings for later use
in text generation.
"""
import os
import re
from argparse import ArgumentParser
from datetime import datetime
from math import ceil
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

def generate_embeddings(data, min_author_count=20, dim=100):
    ## remove low-count subreddits
    subreddit_counts = data.groupby('subreddit').apply(lambda x: x.loc[:, 'author'].nunique())
    high_freq_subreddits = set(subreddit_counts[subreddit_counts >= min_author_count].index.tolist())
    data = data[data.loc[:, 'subreddit'].isin(high_freq_subreddits)]
    ## generate subreddit-author matrix
    subreddit_author_lists = data.groupby('subreddit').apply(lambda x: x.loc[:, 'author'].values)
    min_author_count = 100
    valid_subreddit_author_lists = subreddit_author_lists[subreddit_author_lists.apply(lambda x: len(x) >= min_author_count)]
    valid_subreddit_author_str = valid_subreddit_author_lists.apply(lambda x: ' '.join(x))
    cv = CountVectorizer(min_df=0., max_df=1., tokenizer=lambda x: x.split(' '))
    subreddit_author_dtm = cv.fit_transform(valid_subreddit_author_str)
    ## compute NPMI
    subreddit_probs = subreddit_author_dtm.sum(axis=1) / subreddit_author_dtm.sum().sum()
    author_probs = subreddit_author_dtm.sum(axis=0) / subreddit_author_dtm.sum().sum()
    subreddit_author_probs = subreddit_author_dtm / subreddit_author_dtm.sum().sum()
    subreddit_author_joint_probs = subreddit_probs * author_probs
    subreddit_author_npmi = -np.log(subreddit_author_probs / subreddit_author_joint_probs)
    # fix inf errors
    subreddit_author_npmi[np.isinf(subreddit_author_npmi)] = 0
    # SVD
    svd = TruncatedSVD(n_components=dim)
    subreddit_npmi_embeds = svd.fit_transform(subreddit_author_npmi)
    ## add subreddit info
    subreddit_npmi_embeds = pd.DataFrame(subreddit_npmi_embeds, index=valid_subreddit_author_lists.index)
    return subreddit_npmi_embeds

def collect_author_data(data_dir):
    author_file_matcher = re.compile('[0-9A-Za-z_]+_comments.gz')
    author_data_files = list(filter(lambda x: author_file_matcher.match(x) is not None, os.listdir(data_dir)))
    author_data_files = list(map(lambda x: os.path.join(data_dir, x), author_data_files))
    # tmp debugging
    # author_data_files = author_data_files[:5000]

    author_data = []
    for author_data_file_i in tqdm(author_data_files):
        try:
            data_i = pd.read_csv(author_data_file_i, sep='\t',
                                 compression='gzip', index_col=False)
            author_data.append(data_i)
        except Exception as e:
            # print(f'bad file {author_data_file_i}')
            pass
    author_data = pd.concat(author_data, axis=0)
    ## remove bad data
    author_data = author_data[(author_data.loc[:, 'author'].apply(lambda x: type(x) is str)) &
                              (author_data.loc[:, 'subreddit'].apply(lambda x: type(x) is str))]
    # get data with actual dates
    num_matcher = re.compile('\d+\.\d+')
    author_data = author_data[
        author_data.loc[:, 'created_utc'].apply(
            lambda x: num_matcher.match(str(x)) is not None)]
    author_data = author_data.assign(**{
        'date': author_data.loc[:, 'created_utc'].apply(
            lambda x: datetime.fromtimestamp(int(float(x))))
    })
    author_data = author_data.assign(**{
        'date_day': author_data.loc[:, 'date'].apply(
            lambda x: datetime(year=x.year, month=x.month, day=x.day))
    })
    return author_data

def main():
    parser = ArgumentParser()
    parser.add_argument('author_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--dim', type=int, default=100)
    args = vars(parser.parse_args())
    author_dir = args['author_dir']
    out_dir = args['out_dir']
    dim = args['dim']
    author_data = collect_author_data(author_dir)

    ## bin data by time period
    min_year = 2018
    max_year = 2019
    month_gap = 6
    author_time_periods = []
    start_date = datetime(year=min_year, month=1, day=1)
    time_period_chunks = int(ceil((max_year - min_year + 1)*12 / month_gap))
    for chunk_i in range(time_period_chunks):
        # date_i = start_date + timedelta(months=chunk_i*month_gap)
        date_i = start_date + relativedelta(months=chunk_i*month_gap)
        author_time_periods.append(date_i.timestamp())
    author_data = author_data.assign(**{
        'date_day_bin': np.digitize(
            author_data.loc[:, 'date_day'].apply(
                lambda x: x.timestamp()), bins=author_time_periods)
    })
    date_fmt = '%Y-%m-%d'
    author_data = author_data.assign(**{
        'date_day_bin': author_data.loc[:, 'date_day_bin'].apply(
            lambda x: datetime.strftime(
                datetime.fromtimestamp(author_time_periods[x]),
                date_fmt) if x < len(author_time_periods) else -1)
    })
    author_data = author_data[author_data.loc[:, 'date_day_bin'] != -1]
    print(author_data.loc[:, 'date_day_bin'].value_counts())

    ## get embeddings for each date bin
    date_bin_embeddings = []
    for date_i, data_i in author_data.groupby('date_day_bin'):
        embeds_i = generate_embeddings(data_i, dim=dim)
        date_bin_embeddings.append([date_i, embeds_i])

    ## write embeddings to file
    for date_i, embeds_i in date_bin_embeddings:
        out_file_i = os.path.join(out_dir, f'subreddit_embeddings_{date_i}.gz')
        embeds_i.to_csv(out_file_i, sep='\t', compression='gzip', index=True)

    ## TODO: generate mean author embeddings for each time period lol
    # format: author | time_period_start | subreddits
    author_agg_subreddit_data = []
    date_bin_embedding_lookup = dict(date_bin_embeddings)
    for (author_i, date_bin_i), data_i in author_data.groupby(['author', 'date_day_bin']):
        embeds_i = date_bin_embedding_lookup[date_bin_i]
        subreddits_i = list(filter(lambda x: x in embeds_i.index, data_i.loc[:, 'subreddit'].values))
        if(len(subreddits_i) > 0):
            subreddit_embed_i = embeds_i.loc[subreddits_i, :].mean(axis=0)
            author_agg_subreddit_data.append([author_i, date_bin_i, subreddit_embed_i.values.tolist()])
    author_agg_subreddit_data = pd.DataFrame(author_agg_subreddit_data, columns=['author', 'date_bin', 'subreddit_embed'])
    author_agg_subreddit_out_file = os.path.join(out_dir, 'author_date_subreddits.gz')
    author_agg_subreddit_data.to_csv(author_agg_subreddit_out_file, sep='\t', index=False, compression='gzip')

if __name__ == '__main__':
    main()