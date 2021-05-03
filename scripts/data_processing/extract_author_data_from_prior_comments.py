"""
Extract author-specific data from prior
comments to provide to generation model:

- "expert" vs. "novice" (degree of prior posting in group)
- "early" vs. "late" (time of response relative to original post time)
"""
import gzip
import os
import re
from argparse import ArgumentParser
from ast import literal_eval
from datetime import datetime
from nltk import PunktSentenceTokenizer
from tqdm import tqdm
from data_helpers import extract_age, full_location_pipeline
import numpy as np
import pandas as pd
import stanza
from data_helpers import assign_date_bin

def main():
    parser = ArgumentParser()
    parser.add_argument('author_data_dir')
    parser.add_argument('question_data') # need question data for time of author questions => get all comments before specified time(s)
    parser.add_argument('post_data')
    parser.add_argument('--author_embeddings_data', nargs='+', default=None)
    args = vars(parser.parse_args())
    author_data_dir = args['author_data_dir']
    question_data_file = args['question_data']
    post_data_file = args['post_data']

    ## load existing data
    # post_data = pd.DataFrame(load_zipped_json_data(post_data_file)).loc[:, ['id', 'created_utc']]
    post_data = pd.read_csv(post_data_file, sep='\t', compression='gzip', index_col=False, usecols=['id', 'created_utc'])
    question_data = pd.read_csv(question_data_file, sep='\t', index_col=False, compression='gzip', usecols=['id', 'parent_id', 'created_utc', 'author', 'subreddit'])
    # remove null vals
    post_data.dropna(inplace=True)
    # question_data.dropna(inplace=True, subset=['created_utc', 'parent_id'])
    post_data = post_data[post_data.loc[:, 'created_utc'].apply(lambda x: type(x) is int)]
    question_data = question_data[question_data.loc[:, 'created_utc'].apply(lambda x: type(x) is int)]
    # print(f'{question_data.shape[0]} question data')
    # get date info
    post_data = post_data.assign(**{'parent_date' : post_data.loc[:, 'created_utc'].apply(lambda x: datetime.fromtimestamp(x))})
    question_data = question_data.assign(**{'date': question_data.loc[:, 'created_utc'].apply(lambda x: datetime.fromtimestamp(float(x)))})
    # add to question data
    post_data.rename(columns={'id' : 'parent_id'}, inplace=True)
    question_data = question_data.assign(**{'parent_id' : question_data.loc[:, 'parent_id'].apply(lambda x: x.split('_')[-1])})
    question_data = pd.merge(question_data, post_data.loc[:, ['parent_id', 'parent_date']], on='parent_id')
    # round to day
    question_data = question_data.assign(**{'date_day': question_data.loc[:, 'date'].apply(lambda x: datetime(year=x.year, month=x.month, day=x.day))})
    # tmp debugging
    # print(f'{question_data.shape[0]} data after merge')
    question_authors = set(question_data.loc[:, 'author'].unique())

    ## get data
    ## iterate over all author data
    ## collect dynamic data
    ## extract (1) expertise (2) relative time
    dynamic_author_data_cols = ['author', 'date_day', 'subreddit', 'expert_pct', 'relative_time']
    author_file_matcher = re.compile('.+_comments.gz')
    author_data_files = list(filter(lambda x: author_file_matcher.match(x) is not None, os.listdir(author_data_dir)))
    # tmp debugging
    # author_data_files = author_data_files[:1000]
    author_dynamic_data_file = os.path.join(author_data_dir, 'author_prior_comment_data.gz')
    # get existing authors; do not mine them!
    existing_dynamic_authors = set()
    if(os.path.exists(author_dynamic_data_file)):
        existing_author_data = pd.read_csv(author_dynamic_data_file, sep='\t', compression='gzip', index_col=False)
        existing_dynamic_authors = set(existing_author_data.loc[:, 'author'].unique())
    # if(not os.path.exists(author_data_out_file)):
    dynamic_author_data = []
    # with gzip.open(author_data_out_file, 'wt') as author_data_out:
    #     author_data_col_str = "\t".join(author_data_cols)
    #     author_data_out.write(author_data_col_str + '\n')
    # filter for existing authors
    new_author_data_files = list(filter(lambda x: x.replace('_comments.gz', '') not in existing_dynamic_authors, author_data_files))
    # filter for question-asking authors
    new_author_data_files = list(filter(lambda x: x.replace('_comments.gz', '') in question_authors, new_author_data_files))
    # author_ctr = 0
    # tmp debugging
    # print(f'for dynamic data, we have {len(new_author_data_files)} authors')
    for i, author_file_i in enumerate(tqdm(new_author_data_files)):
        author_i = author_file_i.replace('_comments.gz', '')
        author_comment_file_i = os.path.join(author_data_dir, author_file_i)
        if(author_i not in existing_dynamic_authors):
            try:
                author_comment_data_i = pd.read_csv(author_comment_file_i, sep='\t', compression='gzip', usecols=['author', 'subreddit', 'created_utc'])
                author_comment_data_i = author_comment_data_i.assign(**{'date' : author_comment_data_i.loc[:, 'created_utc'].apply(lambda x: datetime.fromtimestamp((x)))})
                question_data_i = question_data[question_data.loc[:, 'author']==author_i].drop_duplicates(['author', 'subreddit', 'date_day'])
                # dynamic data
                for idx_j, data_j in question_data_i.iterrows():
                    # expertise
                    date_day_j = data_j.loc['date_day']
                    date_j = data_j.loc['date']
                    subreddit_j = data_j.loc['subreddit']
                    author_prior_comment_data_j = author_comment_data_i[author_comment_data_i.loc[:, 'date'].apply(lambda x: x <= date_day_j)]
                    if(author_prior_comment_data_j.shape[0] > 0):
                        relevant_prior_comment_data_j = author_prior_comment_data_j[author_prior_comment_data_j.loc[:, 'subreddit']==subreddit_j]
                        expertise_pct_j = relevant_prior_comment_data_j.shape[0] / author_prior_comment_data_j.shape[0]
                    else:
                        expertise_pct_j = 0.
                    # relative time
                    post_date_j = data_j.loc['parent_date']
                    relative_time_j = (post_date_j - date_j).seconds
                    combined_author_data_j = [author_i, date_day_j, subreddit_j, expertise_pct_j, relative_time_j]
                    dynamic_author_data.append(combined_author_data_j)
                    # combined_author_data_str_j = '\t'.join(list(map(str, combined_author_data_j)))
                    # author_data_out.write(combined_author_data_str_j + '\n')
                    # author_ctr += 1
                    # write data to file periodically
                    if (len(dynamic_author_data) % 1000 == 0):
                        dynamic_author_data = write_flush_data(dynamic_author_data_cols, author_dynamic_data_file, dynamic_author_data)
            except Exception as e:
                print(f'failed to read file {author_comment_file_i} because error {e}')
    # write remaining data to file
    if(len(dynamic_author_data) > 0):
        # tmp debugging
        print(f'final dynamic author data sample = {dynamic_author_data[0]}')
        write_flush_data(dynamic_author_data_cols, author_dynamic_data_file, dynamic_author_data)
    # author_data = pd.DataFrame(author_data, columns=author_data_cols)

    ## collect static data: location, age
    author_static_data_file = os.path.join(author_data_dir, 'author_static_prior_comment_data.gz')
    author_static_data_cols = ['author', 'age', 'location']
    existing_static_authors = set()
    if(os.path.exists(author_static_data_file)):
        existing_static_author_data = pd.read_csv(author_static_data_file, sep='\t', compression='gzip', index_col=False)
        existing_static_authors = set(existing_static_author_data.loc[:, 'author'].unique())
    # if(not os.path.exists(author_static_data_file)):
    nlp_pipeline = stanza.Pipeline(lang='en', processors='tokenize,ner',
                                   use_gpu=False)
    location_matcher = re.compile('(?<=i\'m from )[a-z0-9\, ]+|(?<=i am from )[a-z0-9\, ]+|(?<=i live in )[a-z0-9\, ]+')
    sent_tokenizer = PunktSentenceTokenizer()
    # with gzip.open(author_static_data_file, 'wt') as author_static_data_out:
    #     author_data_col_str = "\t".join(['author'] + author_static_data_cols)
    #     author_static_data_out.write(author_data_col_str + '\n')
    static_author_data = []
    # filter for existing authors
    new_author_data_files = list(filter(lambda x: x.replace('_comments.gz', '') not in existing_static_authors, author_data_files))
    # filter for question-asking authors
    new_author_data_files = list(filter(lambda x: x.replace('_comments.gz', '') in question_authors, new_author_data_files))
    # tmp debugging
    # new_author_data_files = new_author_data_files[:200]
    for i, author_file_i in enumerate(tqdm(new_author_data_files)):
        author_i = author_file_i.replace('_comments.gz', '')
        author_comment_file_i = os.path.join(author_data_dir, author_file_i)
        if(author_i not in existing_static_authors):
            try:
                author_comment_data_i = pd.read_csv(author_comment_file_i, sep='\t', compression='gzip', usecols=['author', 'body',])
                text_i = author_comment_data_i.loc[:, 'body'].values
                # age
                age_i = extract_age(text_i)
                # location
                loc_i = full_location_pipeline(text_i, location_matcher, sent_tokenizer, nlp_pipeline)
                static_author_data.append([author_i, age_i, loc_i])
                # author_static_data_out.write('\t'.join([author_i, str(age_i), loc_i]) + '\n')
                if (len(static_author_data) % 100 == 0):
                    print(f'writing static author data to file ')
                    static_author_data = write_flush_data(author_static_data_cols, author_static_data_file, static_author_data)
            except Exception as e:
                print(
                    f'failed to read file {author_comment_file_i} because error {e}')
    # handle remaining data
    if(len(static_author_data) > 0):
        write_flush_data(author_static_data_cols, author_static_data_file, static_author_data)
    # import sys
    # sys.exit(0)

    ## reload dynamic data, convert to categorical with percentiles, etc.
    combined_author_data = pd.read_csv(author_dynamic_data_file, sep='\t', index_col=False, compression='gzip')
    # remove duplicates
    combined_author_data.drop_duplicates(['author', 'date_day', 'subreddit'], inplace=True)
    category_cutoff_pct_vals = [95, 50]
    category_vars = ['expert_pct', 'relative_time']
    for category_var_i, category_cutoff_pct_i in zip(category_vars, category_cutoff_pct_vals):
        bin_var_i = f'{category_var_i}_bin'
        bin_vals = [np.percentile(combined_author_data.loc[:, category_var_i].values, category_cutoff_pct_i)]
        combined_author_data = combined_author_data.assign(**{
            bin_var_i : np.digitize(combined_author_data.loc[:, category_var_i], bins=bin_vals)
        })
    ## reload static data, combine w/ dynamic data, add regions to locations
    author_static_data = pd.read_csv(author_static_data_file, sep='\t', compression='gzip', index_col=False)
    combined_author_data = pd.merge(combined_author_data, author_static_data, on='author')
    location_region_lookup = {'us' : 'US'}
    location_region_lookup.update({x : 'NONUS' for x in combined_author_data.loc[:, 'location'].unique() if x not in {'UNK', 'us'}})
    location_region_lookup.update({'UNK' : 'UNK'})
    combined_author_data = combined_author_data.assign(**{
        'location_region' : combined_author_data.loc[:, 'location'].apply(location_region_lookup.get)
    })

    ## optional: add author embeddings
    author_embeddings_data_files = args.get('author_embeddings_data')
    if(author_embeddings_data_files is not None):
        combined_author_data = combined_author_data.assign(**{'date_day': combined_author_data.loc[:, 'date_day'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))})
        for author_embeddings_data_file in author_embeddings_data_files:
            ## TODO: shift date bin to avoid reading the future??
            author_embeddings_data = pd.read_csv(author_embeddings_data_file, sep='\t', compression='gzip', index_col=False)
            embed_var = list(filter(lambda x: x.endswith('_embed'), author_embeddings_data.columns))[0]
            author_embeddings_data = author_embeddings_data.assign(**{embed_var : author_embeddings_data.loc[:, embed_var].apply(lambda x: literal_eval(x))})
            author_embeddings_data = author_embeddings_data.assign(**{'date_day_bin' : author_embeddings_data.loc[:, 'date_day_bin'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))})
            ## join via date
            embedding_date_bins = author_embeddings_data.loc[:, 'date_day_bin'].apply(lambda x: x.timestamp()).unique()
            combined_author_data = combined_author_data.assign(**{
                'date_day_bin' : combined_author_data.loc[:, 'date_day'].apply(lambda x: assign_date_bin(x.timestamp(), embedding_date_bins))
            })
            combined_author_data = pd.merge(combined_author_data, author_embeddings_data.loc[:, ['author', 'date_day_bin', embed_var]], on=['author', 'date_day_bin'], how='left')

    # save to single file
    combined_author_data_file = os.path.join(author_data_dir, 'combined_author_prior_comment_data.gz')
    combined_author_data.to_csv(combined_author_data_file, sep='\t', compression='gzip', index=False)

def write_flush_data(data_cols, out_file, new_data):
    """
    Combine old data with new data, and clean new data list.
    :param data_cols: 
    :param out_file: 
    :param new_data: 
    :return: 
    """
    try:
        new_data_df = pd.DataFrame(new_data, columns=data_cols)
        if (os.path.exists(out_file)):
            old_data_df = pd.read_csv(out_file, sep='\t', compression='gzip', index_col=False)
            old_data_df = pd.concat([old_data_df, new_data_df], axis=0)
        else:
            old_data_df = new_data_df.copy()
        old_data_df.to_csv(out_file, sep='\t', compression='gzip', index=False)
    except Exception as e:
        print(f'bad new data example {new_data[0]}; needs data cols {data_cols}')
        print(f'could not combine old/new data because error {e}')
    new_data = []
    return new_data


if __name__ == '__main__':
    main()
