"""
Combine all comments collected from subreddits,
extract questions.
"""
import pickle
from argparse import ArgumentParser
import re
import os
from nltk import WordPunctTokenizer, PunktSentenceTokenizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from data_helpers import load_zipped_json_data, extract_questions_all_data, flatten_columns, tokenize_stem_text, compute_sent_word_overlap
import pandas as pd
import numpy as np

def filter_comments_by_post_overlap(comment_data, post_data_file):
    post_data = load_zipped_json_data(post_data_file)
   post_data.rename(columns={
        'id': 'parent_id', 'created_utc': 'parent_created',
        'selftext': 'parent_text', 'title': 'parent_title',
        'edited': 'parent_edited', 'author': 'parent_author'
    }, inplace=True)
   # remove edits
    post_data = post_data[post_data.loc[:, 'parent_edited'].apply(
        lambda x: type(x) is bool and not x)]
    ## combine with questions
    post_cols = ['parent_id', 'parent_created', 'parent_text', 'parent_title',
                 'parent_edited', 'parent_author']
   # print(f'comment data cols {comment_data.columns}')
    comment_post_data = pd.merge(comment_data, post_data.loc[:, post_cols],
                                 on='parent_id')
    print(f'{comment_post_data.shape[0]}/{comment_data.shape[0]} comments retained after merge with posts')
    # get sentences/tokens
    word_tokenizer = WordPunctTokenizer()
    sent_tokenizer = PunktSentenceTokenizer()
    stemmer = PorterStemmer()
    comment_post_data = comment_post_data.assign(**{
        'parent_sents': comment_post_data.loc[:, 'parent_text'].apply(
            lambda x: tokenize_stem_text(x, stemmer, word_tokenizer,
                                         sent_tokenizer)),
        'question_sents': comment_post_data.loc[:,
                          'question'].apply(
            lambda x: tokenize_stem_text(x, stemmer, word_tokenizer,
                                         sent_tokenizer))
    })
    ## compute overlap
    comment_post_data = comment_post_data.assign(**{
        'post_question_overlap': comment_post_data.apply(
            lambda x: compute_sent_word_overlap(x.loc['parent_sents'],
                                                x.loc['question_sents']),
            axis=1)
    })
    comment_post_data = comment_post_data.assign(**{
        'post_question_overlap_score': comment_post_data.loc[:,
                                       'post_question_overlap'].apply(
            lambda x: x[0]),
        'post_question_overlap_sent': comment_post_data.loc[:,
                                      'post_question_overlap'].apply(
            lambda x: x[1][0]),
    })
    # restrict to overlap [0.1, 0.5] based on earlier tests
    overlap_score_range = [0.1, 0.5]
    valid_overlap_comment_data = comment_post_data[(comment_post_data.loc[:,
                                                    'post_question_overlap_score'] >=
                                                    overlap_score_range[0]) &
                                                   (comment_post_data.loc[:,
                                                    'post_question_overlap_score'] <
                                                    overlap_score_range[1])]
    valid_comment_ids = valid_overlap_comment_data.loc[:, 'id']
    comment_data = pd.merge(
        comment_data,
        valid_overlap_comment_data.loc[:,
        ['post_question_overlap_score', 'id', 'question_id']],
        on=['id', 'question_id']
    )
    return comment_data

def filter_comments_by_valid_question_prob(comment_data, model_file):
   valid_question_model = pickle.load(open(model_file, 'rb'))
    vocab_file = model_file.replace('.pkl', '_vocab.txt')
    model_vocab = list(map(lambda x: x.strip(), open(vocab_file, 'r')))
    cv = CountVectorizer(vocabulary=model_vocab)
    flat_question_dtm = cv.fit_transform(comment_data.loc[:, 'question'].values)
    question_valid_probs = valid_question_model.predict_proba(flat_question_dtm)
    comment_data = comment_data.assign(
        **{'valid_question_prob': question_valid_probs[:, 1]})
    valid_prob_cutoff = 0.5
    comment_data = comment_data[
        comment_data.loc[:, 'valid_question_prob'] > valid_prob_cutoff]
    return comment_data

def main():
    parser = ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--data_name', default='advice_subreddit')
    parser.add_argument('--post_data', default=None) # ../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz
    parser.add_argument('--valid_question_model', default=None) # ../../data/reddit_data/valid_question_detection_model.pkl
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
    # don't add submission data because of space (M submissions x N comments x O questions/comment = a lot)
    # submission_data =
    # remove comments without parents
    comment_data = comment_data[comment_data.loc[:, 'parent_id'].apply(lambda x: type(x) is not float)]
    # tmp debugging
    # comment_data = comment_data.iloc[:10000, :]

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
    ## remove bad comments
    # remove comments with no parent
    comment_data = comment_data[comment_data.loc[:, 'parent_id'].apply(lambda x: type(x) is not float)]
    # remove null questions
    comment_data = comment_data[~comment_data.loc[:, 'questions'].apply(lambda x: type(x) is float and np.isnan(x))]
    # flatten, add question ID for later comparisons
    flat_cols = ['questions']
    comment_data = flatten_columns(comment_data, cols=flat_cols)
    comment_data.rename(columns={'questions': 'question'}, inplace=True)
    comment_data = comment_data.assign(**{'question_id' : comment_data.loc[:, 'question'].apply(lambda x: hash(x))})

    ## filter for post overlap
    if(args.get('post_data') is not None):
        comment_data = filter_comments_by_post_overlap(comment_data, args['post_data'])
    ## filter for valid clarification questions
    if(args.get('valid_question_model') is not None):
        model_file = args['valid_question_model']
        comment_data = filter_comments_by_valid_question_prob(comment_data, model_file)

    ## write to file
    data_name = args['data_name']
    out_file = os.path.join(data_dir, f'{data_name}_comment_question_data.gz')
    comment_data.to_csv(out_file, sep='\t', compression='gzip', index=False)

if __name__ == '__main__':
    main()
