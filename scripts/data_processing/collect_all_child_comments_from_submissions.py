"""
Collect all child comments from submissions.
"""
import gzip
import json
import os
from argparse import ArgumentParser
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from data_helpers import FileReader, load_zipped_json_data
from tqdm import tqdm
import logging

def main():
    parser = ArgumentParser()
    parser.add_argument('submission_data') # ../..data/reddit_data/advice_subreddit_submissions_2018-01_2019-12.gz
    parser.add_argument('--out_dir', default='../../data/reddit_data/')
    parser.add_argument('--data_dir', default='/local2/lbiester/pushshift/comments/')
    parser.add_argument('--comment_dates', default='2018,1,2019,12')
    args = vars(parser.parse_args())
    submission_data_file = args['submission_data']
    out_dir = args['out_dir']
    data_dir = args['data_dir']
    comment_start_year, comment_start_month, comment_end_year, comment_end_month = list(map(int, args['comment_dates'].split(',')))
    # tmp debugging
    # print(f'start year = {comment_start_year}; start month = {comment_start_month}')
    # import sys
    # sys.exit()
    logging.basicConfig(filename=f'../../logs/collect_all_child_comments_from_submissions_{comment_start_year}-{comment_start_month}_{comment_end_year}-{comment_end_month}.txt',
                        filemode='w', format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.INFO)

    ## load submission data
    submission_data = pd.read_csv(submission_data_file, sep='\t', compression='gzip', index_col=False)
    # submission_data = pd.DataFrame([pd.Series(json.loads(x.strip())) for x in gzip.open(submission_data_file, 'rt')])
    parent_ids = submission_data.loc[:, 'id'].unique()

    ## collect comments
    comment_dir_files = os.listdir(data_dir)
    parent_id_author_lookup = dict(zip(submission_data.loc[:, 'id'].values, submission_data.loc[:, 'author'].values))
    print(f'{len(parent_ids)} unique parent IDs')
    valid_subreddits = set(submission_data.loc[:, 'subreddit'].unique())
    comment_data_vars = ['author', 'author_flair_text', 'author_fullname',
                         'body', 'created_utc', 'edited', 'id', 'parent_id',
                         'score', 'subreddit']
    invalid_text = set(['[removed]', '[deleted]', ''])
    invalid_authors = set(['[removed]', '[deleted]', 'AutoModerator'])
    word_tokenizer = WordPunctTokenizer()
    comment_years = list(range(comment_start_year, comment_end_year+1))
    min_month = 1
    max_month = 12
    min_comment_len = 10
    for comment_year_i in comment_years:
        start_month_i = min_month
        end_month_i = max_month
        if(comment_year_i == comment_start_year):
            start_month_i = comment_start_month
        elif(comment_year_i == comment_end_year):
            end_month_i = comment_end_month
        for comment_month_i in range(start_month_i, end_month_i+1):
            logging.info(f'processing {comment_year_i}-{comment_month_i}')
            comment_file_i = list(filter(lambda x: 'RC_%d-%.2d' % (comment_year_i, comment_month_i) in x, comment_dir_files))[0]
            comment_file_i = os.path.join(data_dir, comment_file_i)
            file_reader = FileReader(comment_file_i)
            # write out one file at a time => parallelize me Cap'n
            out_file_i = os.path.join(out_dir,
                                      'subreddit_comments_%d-%.2d.gz' % (
                                      comment_year_i, comment_month_i))
            with gzip.open(out_file_i, 'wt') as subreddit_comment_out:
                for j, line_j in enumerate(tqdm(file_reader)):
                    data_j = json.loads(line_j)
                    author_j = data_j['author']
                    parent_id_j = data_j['parent_id'].split('_')[-1]
                    if ((data_j['subreddit'] in valid_subreddits) and
                            (data_j['body'] not in invalid_text) and
                            (data_j['author'] not in invalid_authors) and
                            # parent ID condition: comment is direct child of known parent OR
                            # comment was written by parent author (possible answer to clarification Q)
                            (parent_id_j in parent_ids or
                             (parent_id_j in parent_id_author_lookup and author_j == [parent_id_j]))):
                        text_j = data_j['body']
                        text_tokens_j = word_tokenizer.tokenize(text_j)
                        # text length condition
                        if (len(text_tokens_j) >= min_comment_len):
                            # TODO: question condition
                            #                     questions_j =
                            filter_data_j = {v: data_j[v] for v in
                                             comment_data_vars if v in data_j}
                            subreddit_comment_out.write(
                                f'{json.dumps(filter_data_j)}\n')
                    if(j % 1000000 == 0):
                        logging.info(f'processed {j} lines')

if __name__ == '__main__':
    main()