"""
Combine all comments collected from subreddits,
extract questions.
"""
import pickle
from argparse import ArgumentParser
import re
import os
from ast import literal_eval

from nltk import WordPunctTokenizer, PunktSentenceTokenizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from data_helpers import load_zipped_json_data, extract_questions_all_data, flatten_columns, tokenize_stem_text, compute_sent_word_overlap
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

def filter_comments_by_post_overlap(comment_data, post_data, overlap_score_range=[0.125, 0.5]):
    # restrict to valid posts
    # tmp debugging
    # print(f'comment IDs {comment_data.loc[:, "parent_id"].unique()[:10]}')
    # print(f'post IDs {post_data.loc[:, "parent_id"].unique()[:10]}')
    valid_post_ids = set(comment_data.loc[:, 'parent_id'].unique()) & set(post_data.loc[:, 'parent_id'].unique())
    print(f'{len(valid_post_ids)}/{post_data.loc[:, "parent_id"].nunique()} valid post IDs')
    # print(f'{len(valid_post_ids)} valid post IDs')
    # post_data = post_data[post_data.loc[:, 'parent_id'].isin(valid_post_ids)]
    # comment_data = comment_data[comment_data.loc[:, 'parent_id'].isin(valid_post_ids)]
    # print(f'post IDs {post_data.loc[:, "parent_id"].unique()[:10]}')
    # print(f'comment parent IDs {comment_data.loc[:, "parent_id"].unique()[:10]}')
    ## combine with questions
    post_cols = ['parent_id', 'parent_created', 'parent_sents', 'parent_title',
                 'parent_edited', 'parent_author']
    ## tokenize/stem before joining to save space? yeah sure
    word_tokenizer = WordPunctTokenizer()
    sent_tokenizer = PunktSentenceTokenizer()
    stemmer = PorterStemmer()
    post_data = post_data.assign(**{
        'parent_sents': post_data.loc[:, 'parent_text'].apply(
            lambda x: tokenize_stem_text(x, stemmer, word_tokenizer,
                                         sent_tokenizer)),
    })
    # comment_data = comment_data.assign(**{
    #     'question_sents': comment_data.loc[:,'question'].apply(lambda x: tokenize_stem_text(x, stemmer, word_tokenizer, sent_tokenizer))
    # })
    ## join data
    ## TODO: if memory overload, don't join just iterate by parent_id and combine later
    post_data = pd.merge(comment_data, post_data.loc[:, post_cols],
                         on='parent_id')
    print(f'{post_data.shape[0]}/{comment_data.shape[0]} comments retained after merge with posts')
    ## compute overlap
    print(f'compute post/question overlap')
    post_data = post_data.assign(**{
        'post_question_overlap': post_data.progress_apply(
            lambda x: compute_sent_word_overlap(x.loc['parent_sents'],
                                                [x.loc['question']]),
            axis=1)
    })
    post_data = post_data.assign(**{
        'post_question_overlap_score': post_data.loc[:,
                                       'post_question_overlap'].apply(
            lambda x: x[0]),
        'post_question_overlap_sent': post_data.loc[:,
                                      'post_question_overlap'].apply(
            lambda x: x[1][0]),
    })
    # restrict to overlap [0.05, 0.5] based on earlier tests
    # overlap_score_range = [0.05, 0.5]
    valid_overlap_comment_data = post_data[(post_data.loc[:,'post_question_overlap_score'] >= overlap_score_range[0]) &
                                           (post_data.loc[:, 'post_question_overlap_score'] < overlap_score_range[1])]
    # valid_comment_ids = valid_overlap_comment_data.loc[:, 'id']
    comment_data = pd.merge(
        comment_data,
        valid_overlap_comment_data.loc[:,
        ['post_question_overlap_score', 'id', 'question_id']],
        on=['id', 'question_id']
    )
    print(f'{valid_overlap_comment_data.shape[0]}/{post_data.shape[0]} questions retained after filtering for overlap')
    return comment_data

def filter_comments_by_valid_question_prob(comment_data, model_file):
    valid_question_model = pickle.load(open(model_file, 'rb'))
    # vocab_file = model_file.replace('.pkl', '_vocab.txt')
    vocab_file = os.path.join(os.path.dirname(model_file), 'model_vocab.txt')
    model_vocab = list(map(lambda x: x.strip(), open(vocab_file, 'r')))
    cv = CountVectorizer(vocabulary=model_vocab)
    flat_question_dtm = cv.fit_transform(comment_data.loc[:, 'question'].values)
    question_valid_probs = valid_question_model.predict_proba(flat_question_dtm)
    comment_data = comment_data.assign(
        **{'valid_question_prob': question_valid_probs[:, 1]})
    valid_prob_cutoff = 0.5
    valid_comment_data = comment_data[comment_data.loc[:, 'valid_question_prob'] > valid_prob_cutoff]
    print(f'{valid_comment_data.shape[0]}/{comment_data.shape[0]} comments retained after filtering for P(valid question)')
    return valid_comment_data

def main():
    parser = ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--data_name', default='advice_subreddit')
    parser.add_argument('--post_data', default=None) # ../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
    parser.add_argument('--valid_question_model', default=None) # ../../data/reddit_data/valid_question_detection_model.pkl
    parser.add_argument('--filter_overlap', dest='feature', action='store_true') # filter by post overlap, information questions
    args = vars(parser.parse_args())
    data_dir = args['data_dir']

    ## get all comment files
    comment_file_matcher = re.compile('subreddit_comments_\d{4}-\d{2}.gz')
    comment_files = list(filter(lambda x: comment_file_matcher.match(x), os.listdir(data_dir)))
    comment_files = list(map(lambda x: os.path.join(data_dir, x), comment_files))
    ## load all data
    comment_data = pd.concat(list(map(lambda x: load_zipped_json_data(x), comment_files)), axis=0)
    # fix parent IDs
    comment_data = comment_data.assign(**{
        'parent_id' : comment_data.loc[:, 'parent_id'].apply(lambda x: x.split('_')[-1])
    })
    # remove comments without parents
    comment_data = comment_data[comment_data.loc[:, 'parent_id'].apply(lambda x: type(x) is not float)]
    print(f'{comment_data.shape[0]} comments before filtering')
    # tmp debugging
    # comment_data = comment_data.iloc[:50000, :]

    ## extract questions
    print(f'extracting questions')
    min_question_len = 5
    comment_data = comment_data.assign(**{
        'questions' : extract_questions_all_data(comment_data.loc[:, 'body'], min_question_len=min_question_len)
    })
    # remove comment body to save space!!
    comment_data.drop('body', axis=1, inplace=True)
    # remove invalid questions: quotes, bots
    quote_matcher = re.compile('&gt;.+')
    comment_data = comment_data.assign(**{
        'questions': comment_data.loc[:, 'questions'].apply(
            lambda x: list(
                filter(lambda y: quote_matcher.search(y) is None, x)))
    })
    invalid_authors = ['LocationBot', 'AutoModerator']
    comment_data = comment_data[~comment_data.loc[:, 'author'].isin(invalid_authors)]
    ## remove bad comments
    # remove null questions
    comment_data = comment_data[~comment_data.loc[:, 'questions'].apply(lambda x: type(x) is float and np.isnan(x))]

    # flatten, add question ID for later comparisons
    flat_col = 'questions'
    comment_data = flatten_columns(comment_data, flat_col=flat_col)
    comment_data.rename(columns={'questions': 'question'}, inplace=True)
    comment_data = comment_data.assign(**{'question_id' : comment_data.loc[:, 'question'].apply(lambda x: hash(x))})
    # remove duplicates
    comment_data.drop_duplicates(['parent_id', 'question_id'], inplace=True)
    # tmp debugging: save to inspect parent ID overlap
    # comment_data.to_csv('tmp_comment_question_data_flat.gz', sep='\t', compression='gzip', index=False)
    # import sys
    # sys.exit(0)

    ## load post data
    post_data_cols = ['id', 'created_utc', 'selftext', 'title', 'edited', 'author']
    post_data = pd.read_csv(args['post_data'], sep='\t', compression='gzip', index_col=False, usecols=post_data_cols)
    post_data.rename(columns={'id': 'parent_id', 'created_utc': 'parent_created', 'selftext': 'parent_text', 'title': 'parent_title', 'edited': 'parent_edited', 'author': 'parent_author'}, inplace=True)
    # remove null posts
    post_data = post_data[~post_data.loc[:, 'parent_id'].apply(lambda x: type(x) is float and np.isnan(x))]
    post_data = post_data[~post_data.loc[:, 'parent_edited'].apply(lambda x: type(x) is float and np.isnan(x))]
    post_data = post_data[post_data.loc[:, 'parent_text'].apply(lambda x: type(x) is str)]
    # remove edited posts
    bool_matcher = re.compile('True|False')
    post_data = post_data[post_data.loc[:, 'parent_edited'].apply(lambda x: bool_matcher.match(str(x)) is not None and not literal_eval(bool_matcher.match(x).group(0)))]
    # remove comments written by post author
    comment_post_data = pd.merge(comment_data, post_data.loc[:, ['parent_id', 'parent_author']], on='parent_id', how='inner')
    comment_post_data = comment_post_data[comment_post_data.loc[:, 'author'] != comment_post_data.loc[:, 'parent_author']]
    valid_comment_ids = comment_post_data.loc[:, 'id'].unique()
    comment_data = comment_data[comment_data.loc[:, 'id'].isin(valid_comment_ids)]

    ## filter for post overlap
    # tmp debugging
    # print(f'filter overlap {filter_overlap}')
    print(f'{comment_data.shape[0]} questions before filtering')
    if('filter_overlap' in args):
        # post_data = load_zipped_json_data(args['post_data'])
        # if(args.get('post_data') is not None):
        print(f'computing post overlap')
        overlap_score_range = [0.125, 0.30]
        comment_data = filter_comments_by_post_overlap(comment_data, post_data, overlap_score_range=overlap_score_range)
    ## filter for valid clarification questions
    if(args.get('valid_question_model') is not None):
        model_file = args['valid_question_model']
        comment_data = filter_comments_by_valid_question_prob(comment_data, model_file)
    print(f'{comment_data.shape[0]} questions after filtering')

    ## write to file
    data_name = args['data_name']
    out_file = os.path.join(data_dir, f'{data_name}_comment_question_data.gz')
    comment_data.to_csv(out_file, sep='\t', compression='gzip', index=False)

if __name__ == '__main__':
    main()
