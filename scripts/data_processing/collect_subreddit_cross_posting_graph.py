"""
Collect subreddit cross-posting graph based on the bipartite network
of authors posting to subreddits.
"""
import json
import os
from argparse import ArgumentParser
from itertools import product
from data_helpers import FileReader
from collections import defaultdict
import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('post_dir') # reddit post/comment directory
    parser.add_argument('--out_dir', default='../../data/reddit_data/')
    parser.add_argument('--post_dates', default='2018,1,2019,12')
    args = vars(parser.parse_args())
    post_dir = args['post_dir']
    out_dir = args['out_dir']
    post_dates = list(map(int, args['post_dates'].split(',')))
    start_year, start_month, end_year, end_month = post_dates
    min_month = 1
    max_month = 12
    year_month_pairs = []
    for year_i in range(start_year, end_year+1):
        if(year_i == start_year):
            start_month_i = start_month
        else:
            start_month_i = min_month
        if(year_i == end_year):
            end_month_i = end_month
        else:
            end_month_i = max_month
        year_month_pairs.extend(list(product([year_i], list(range(start_month_i, end_month_i+1)))))

    ## iterate over posts
    post_files = os.listdir(post_dir)
    invalid_text = set(['[removed]', '[deleted]', ''])
    invalid_authors = set(['[removed]', 'AutoModerator'])
    ## TODO: filter on min edge count to save space?
    for year_i, month_i in year_month_pairs:
        subreddit_author_counts = defaultdict(int)
        post_file_i = list(filter(lambda x: 'RS_%d-%.2d'%(year_i, month_i) in x, post_files))[0]
        post_file_i = os.path.join(post_dir, post_file_i)
        file_reader = FileReader(post_file_i)
        for i, line_i in enumerate(file_reader):
            data_i = json.loads(line_i)
            if((data_i['selftext'] not in invalid_text) and
               (data_i['author'] not in invalid_authors)):
                edge_i = (data_i['subreddit'], data_i['author'])
                subreddit_author_counts[edge_i] += 1
            if(i % 1000000 == 0):
                print(f'processed {i} posts')
            # tmp debugging
            # if (len(subreddit_author_counts) >= 10000):
            #     break
        ## TODO: project from subreddit-author to subreddit-subreddit network
        subreddit_author_count_data_i = pd.Series(subreddit_author_counts).reset_index(name='count').rename(columns={'level_0':'subreddit', 'level_1':'author'})
        subreddit_subreddit_author_sets_i = defaultdict(set)

        for author_j, data_j in subreddit_author_count_data_i.groupby('author'):
            # get combinations of all subreddits fml
            subreddits_j = data_j.loc[:, 'subreddit'].sort_values().values # sort values to ensure same order
            # subreddit_combos_j = [(subreddits_j[k], subreddits_j[l]) for k in range(len(subreddits_j)) for l in range(k+1, len(subreddits_j))]
            for k, subreddit_k in enumerate(subreddits_j):
                count_k = data_j[data_j.loc[:, 'subreddit']==subreddit_k].loc[:, 'count'].iloc[0] #
                for l in range((k+1), len(subreddits_j)):
                    subreddit_l = subreddits_j[l]
                    # count_l = data_j[data_j.loc[:, 'subreddit']==subreddit_l].loc[:, 'count'].iloc[0]
                    # # compute average as count => count high-overlap
                    # count_k_l = (count_k + count_l)*0.5
                    # subreddit_subreddit_counts_i[(subreddit_k, subreddit_l)] += count_k_l
                    subreddit_subreddit_author_sets_i[(subreddit_k, subreddit_l)].add(author_j)
        subreddit_subreddit_counts_i = pd.Series({k : len(v) for k,v in subreddit_subreddit_author_sets_i.items()}).reset_index(name='count').rename(columns={'level_0':'subreddit_i', 'level_1':'subreddit_j'})
        # tmp debugging
        # print(f'subreddit count data\n{subreddit_subreddit_counts_i.head(10)}')
        ## TODO: normalize for total author counts??
        ## save to file!!
        file_name_i = 'comment_%d-%.2d_cross_posting.gz'%(year_i, month_i)
        subreddit_subreddit_count_data_file_i = os.path.join(out_dir, file_name_i)
        subreddit_subreddit_counts_i.to_csv(subreddit_subreddit_count_data_file_i, sep='\t', compression='gzip')

if __name__ == '__main__':
    main()