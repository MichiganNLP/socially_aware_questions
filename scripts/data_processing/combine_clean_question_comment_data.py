"""
Combine all comments collected from subreddits,
extract questions.
"""
from argparse import ArgumentParser
import re
import os
from data_helpers import load_zipped_json_data, extract_questions_all_data
import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('data_dir')
    args = vars(parser.parse_args())
    data_dir = args['data_dir']

    ## get all comment files
    comment_file_matcher = re.compile('subreddit_comments_\d{4}-\d{2}.gz')
    comment_files = list(filter(lambda x: comment_file_matcher.match(x), os.listdir(data_dir)))
    comment_files = list(map(lambda x: os.path.join(data_dir, x), comment_files))
    ## load all data
    comment_data = pd.concat(list(map(lambda x: load_zipped_json_data(x), comment_files)), axis=0)
    # don't add submission data because of space (M submissions x N comments x O questions/comment = a lot)
    # submission_data =
    # remove comments without parents
    comment_data = comment_data[comment_data.loc[:, 'parent_id'].apply(lambda x: type(x) is not float)]

    ## extract questions
    min_question_len = 5
    comment_data = comment_data.assign(**{
        'questions' : extract_questions_all_data(comment_data.loc[:, 'body'], min_question_len=min_question_len)
    })
    # remove invalid questions: quotes, bots
    quote_matcher = re.compile('&gt;[^\n]+\n')
    comment_data = comment_data.assign(**{
        'questions': comment_data.loc[:, 'questions'].apply(
            lambda x: list(
                filter(lambda y: quote_matcher.search(y) is None, x)))
    })
    invalid_authors = ['LocationBot', 'AutoModerator']
    comment_data = comment_data[~comment_data.loc[:, 'author'].isin(invalid_authors)]
    out_file = os.path.join(data_dir, 'subreddit_combined_comment_question_data.gz')
    comment_data.to_csv(out_file, sep='\t', compression='gzip', index=False)

if __name__ == '__main__':
    main()