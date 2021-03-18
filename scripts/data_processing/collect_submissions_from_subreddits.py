"""
Collect submissions sent to specified subreddits,
from pre-downloaded data.
"""
from argparse import ArgumentParser
import json
from nltk.tokenize import WordPunctTokenizer
import os
from tqdm import tqdm
import gzip
from data_helpers import FileReader

def main():
    parser = ArgumentParser()
    parser.add_argument('submission_dir') # /local2/lbiester/pushshift/submissions/
    parser.add_argument('--subreddits', nargs='+', type=set, default=['AmItheAsshole', 'legaladvice', 'pcmasterrace', 'Advice', 'personalfinance'])
    parser.add_argument('--out_dir', default='../../data/reddit_data/')
    parser.add_argument('--submission_dates', nargs='+', default=[2018,1,2019,12])
    args = vars(parser.parse_args())

    out_dir = args['out_dir']
    submission_dir = args['submission_dir']
    subreddits = args['subreddits']
    submission_dates = list(map(int, args['submission_dates']))
    submission_start_year, submission_start_month, submission_end_year, submission_end_month = submission_dates
    submission_dir_files = os.listdir(submission_dir)
    invalid_text = ['[removed]', '[deleted]', '']
    invalid_authors = ['[removed]', '[deleted]', 'AutoModerator']
    tokenizer = WordPunctTokenizer()
    min_submission_len = 20
    min_comment_count = 1
    submission_data_vars = ['author', 'author_flair_text', 'author_fullname', 'category', 'created_utc', 'edited', 'id', 'num_comments', 'score', 'selftext', 'subreddit', 'title']
    subreddit_submission_out_file = os.path.join(out_dir, 'subreddit_submissions_%d-%.2d_%d-%.2d.gz'%(submission_start_year, submission_start_month, submission_end_year, submission_end_month))
    submission_year_month_pairs = []
    min_month = 1
    max_month = 12
    for year_i in range(submission_start_year, submission_end_year+1):
        start_month_i = min_month
        end_month_i = max_month
        if(year_i == submission_start_year):
            start_month_i = submission_start_month
        elif(year_i == submission_end_year):
            end_month_i = submission_end_month
        for month_j in range(start_month_i, end_month_i+1):
            submission_year_month_pairs.append([year_i, month_j])
    with gzip.open(subreddit_submission_out_file, 'wt') as subreddit_submission_out:
        for submission_year_i, submission_month_i in submission_year_month_pairs:
            print(
                f'processing month={submission_month_i} year={submission_year_i}')
            submission_file_i = list(filter(lambda x: 'RS_%d-%.2d' % (
            submission_year_i, submission_month_i) in x, submission_dir_files))[
                0]
            submission_file_i = os.path.join(submission_dir, submission_file_i)
            file_reader = FileReader(submission_file_i)
            for j, line_j in enumerate(tqdm(file_reader)):
                data_j = json.loads(line_j)
                if ((data_j['subreddit'] in subreddits) and
                        (data_j['selftext'] not in invalid_text) and
                        (data_j['author'] not in invalid_authors) and
                        (data_j['num_comments'] >= min_comment_count)):
                    text_j = data_j['selftext']
                    text_tokens_j = tokenizer.tokenize(text_j)
                    if (len(text_tokens_j) >= min_submission_len):
                        filter_data_j = {v: data_j[v] for v in
                                         submission_data_vars if v in data_j}
                        subreddit_submission_out.write(
                            f'{json.dumps(filter_data_j)}\n')

if __name__ == '__main__':
    main()