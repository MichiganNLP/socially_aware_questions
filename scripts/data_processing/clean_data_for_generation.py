"""
Clean text data for question generation.
We expect the format:

article ID | article text | question text
"""
import os
import re
from argparse import ArgumentParser
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
import numpy as np
from data_helpers import prepare_question_data
from transformers import BartTokenizer, LongformerTokenizer
from datetime import datetime

def load_all_articles(data_dir, data_name):
    article_files = list(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))
    # tmp debugging for badly-formed files
    # for article_file in article_files:
    #     try:
    #         pd.read_csv(article_file, sep='\t', index_col=False)
    #     except Exception as e:
    #         print(article_file)
    article_data = pd.concat(list(map(lambda x: pd.read_csv(x, sep='\t', index_col=False), article_files)), axis=0)
    if('NYT' in data_name):
        article_id_matcher = re.compile('(?<=article_)[0-9a-zA-Z]+(?=\.tsv)')
        article_ids = list(map(lambda x: article_id_matcher.search(x).group(0), article_files))
        article_data = article_data.assign(**{
            'article_id' : article_ids
        })
        article_data.rename(columns={'text' : 'article_text'}, inplace=True)
        # remove null articles??
        article_data = article_data[~article_data.loc[:, 'article_text'].apply(lambda x: type(x) is float and np.isnan(x))]
        # clean text
        matcher_pairs = [(re.compile('<.+>'), ' '),
                         (re.compile('\.{2,}'), '.'),
                         (re.compile(' \- '), '-'),
                         (re.compile(' ([\.\?\!,\'\"]+)'), '\\1'),
                         (re.compile('([\']) (?=[a-z])'), '\\1'),  # fix contraction spacing
                         (re.compile('(\() '), '\\1'), (re.compile(' (\))'), '\\1'),  # parentheses spacing
                         (re.compile('[\n\r\t]'), ' '),  # return spaces
                         (re.compile('Credit\s?: .{8,100}$'), ''),  # remove author credit
                         ]
        word_tokenizer = WordPunctTokenizer()
        article_data = article_data.assign(**{
            'article_text': article_data.loc[:, 'article_text'].apply(
                lambda x: clean_text_body(x, word_tokenizer, matcher_pairs))
        })
    # remove null data
    article_data = article_data[article_data.loc[:, 'article_text'] != '']
    return article_data

def clean_text_body(txt, word_tokenizer, matcher_pairs):
    clean_txt = ' '.join(word_tokenizer.tokenize(txt))
    for matcher_i, sub_i in matcher_pairs:
        clean_txt = matcher_i.sub(sub_i, clean_txt)
    return clean_txt

def load_all_comment_questions(comment_dir, comment_month_years=[('April', '2018')]):
    comment_files = list(map(lambda x: f'Comments{x[0]}{x[1]}.csv', comment_month_years))
    comment_files = list(map(lambda x: os.path.join(comment_dir, x), comment_files))
    comment_data = []
    for comment_file_i in comment_files:
        comment_data_i = pd.read_csv(comment_file_i, sep=',', index_col=False, usecols=['articleID', 'createDate', 'commentBody', 'commentType', 'parentID', 'userLocation', 'userID', 'userDisplayName'])
        comment_data.append(comment_data_i)
    comment_data = pd.concat(comment_data, axis=0)
    # remove duplicates? OK
    comment_data.drop_duplicates(['articleID', 'commentBody'], inplace=True)
    # clean comment text
    # fix punctuation without spaces and HTML
#     html_matcher = re.compile('<.+>')
    matcher_pairs = [(re.compile('<.+>'), ' '),
                     (re.compile('\.{2,}'), '.'),
                     (re.compile(' \- '), '-'),
                     (re.compile(' ([\.\?\!,\'\"]+)'), '\\1'),
                     (re.compile('([\']) (?=[a-z])'), '\\1'), # fix contraction spacing
                     (re.compile('(\() '), '\\1'), (re.compile(' (\))'), '\\1'), # parentheses spacing
                     (re.compile('[\n\r\t]'), ' '), # return spaces
                     ]
    word_tokenizer = WordPunctTokenizer()
    comment_data = comment_data.assign(**{
        'commentBody' : comment_data.loc[:, 'commentBody'].apply(lambda x: clean_text_body(x, word_tokenizer, matcher_pairs))
    })
    ## clean article
    # find questions
    question_matcher = re.compile('\?$')
    comment_data = comment_data.assign(**{
        'comment_questions' : comment_data.loc[:, 'commentBody'].apply(lambda x: list(filter(lambda y: question_matcher.search(y) is not None, sent_tokenize(x))))
    })
    # convert to question data (one row/question)
    question_data = []
    # print(f'comment data cols {comment_data.columns}')
    question_cols = ['createDate', 'articleID', 'commentBody', 'commentType', 'parentID', 'userID', 'userLocation', 'userDisplayName']
    for idx_i, data_i in comment_data.iterrows():
        for question_j in data_i.loc['comment_questions']:
            question_data.append(pd.Series(data_i.loc[question_cols].append(pd.Series([question_j], index=['question']))))
    question_data = pd.concat(question_data, axis=1).transpose()
    question_data.rename(columns={'articleID':'article_id'}, inplace=True)
    ## clean short/malformed questions
    min_question_len = 6
    question_data = question_data.assign(**{
        'question_len' : question_data.loc[:, 'question'].apply(lambda x: len(word_tokenizer.tokenize(x)))
    })
    question_data = question_data[question_data.loc[:, 'question_len'] >= min_question_len]
    ## cleanup columns
    # NOTE: we need author ID and post date
    question_data = question_data.loc[:, ['article_id', 'commentBody', 'question', 'userID', 'createDate']]
    # print(f'sample question data {question_data.head(10)}')
    return question_data

def main():
    parser = ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--data_dir', default='../../data/')
    parser.add_argument('--data_file', default=None)
    parser.add_argument('--data_name', default='NYT')
    parser.add_argument('--comment_dir', default=None) # ../../data/nyt_comments/
    parser.add_argument('--comment_month_year_pairs', nargs='+', default=None) # 'April_2018'
    parser.add_argument('--sample_pct', type=float, default=1.0)
    parser.add_argument('--author_data', default=None)
    parser.add_argument('--model_type', default='bart')
    parser.add_argument('--NE_overlap', type=bool, default=False)
    args = vars(parser.parse_args())

    ## load raw data
    data_dir = args['data_dir']
    data_name = args['data_name']
    data_file = args['data_file']
    if(data_file is None):
        article_data = load_all_articles(data_dir, data_name)
    else:
        article_data = pd.read_csv(data_file, sep='\t', index_col=False)

    ## optional: get questions from comments
    if(args.get('comment_dir') is not None):
        comment_dir = args['comment_dir']
        comment_month_year_pairs = list(map(lambda x: x.split('_'), args['comment_month_year_pairs']))
        question_data = load_all_comment_questions(comment_dir, comment_month_year_pairs)
        article_data = pd.merge(article_data, question_data, on='article_id', how='inner')

    ## prepare data for training
    sample_pct = args['sample_pct']
    if (sample_pct < 1.0):
        N_sample = int(article_data.shape[0] * sample_pct)
        article_data_idx = np.random.choice(article_data.index, N_sample, replace=False)
        article_data = article_data.loc[article_data_idx, :]
    train_pct = 0.8
    author_data = args['author_data']
    if (author_data is not None):
        author_data = pd.read_csv(author_data, sep='\t', index_col=False)
        # fix date
        date_day_fmt = '%Y-%m-%d'
        author_data = author_data.assign(**{
            'date_day': author_data.loc[:, 'date_day'].apply(lambda x: datetime.strptime(x, date_day_fmt))
        })
    out_dir = args['out_dir']
    NE_overlap = args['NE_overlap']
    if (NE_overlap):
        data_name = f'NE_overlap_{data_name}'
    train_data_file = os.path.join(out_dir, f'{data_name}_train_data.pt')
    model_type = args['model_type']
    tokenizer_lookup = {
        'bart' : (BartTokenizer, 'facebook/bart-base',),
        'longformer' : (LongformerTokenizer, 'allenai/longformer-base-4096')
    }
    tokenizer_class, tokenizer_name = tokenizer_lookup[model_type]
    max_len_lookup = {
        'bart' : (1024, 64),
        'longformer' : (3072, 128), # 4096 => memory overload in training
        # 'longformer': (4096, 128),  # ONLY for big GPU server
    }
    max_source_length, max_target_length = max_len_lookup[model_type]
    if (not os.path.exists(train_data_file)):
        if(author_data is not None):
            data_name_base = f'author_type_{data_name}'
        else:
            data_name_base = data_name
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        prepare_question_data(article_data, out_dir, data_name_base,
                              tokenizer=tokenizer, train_pct=train_pct,
                              author_data=author_data,
                              max_source_length=max_source_length,
                              max_target_length=max_target_length,
                              article_question_NE_overlap=NE_overlap)
        # if we include author data: also generate "clean" no-author data for comparison
        if(author_data is not None):
            clean_out_dir = os.path.join(out_dir, 'no_author_data/')
            if(not os.path.exists(clean_out_dir)):
                os.mkdir(clean_out_dir)
            # need clean tokenizer
            tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
            prepare_question_data(article_data, clean_out_dir, data_name,
                                  tokenizer=tokenizer, train_pct=train_pct,
                                  author_data=None,
                                  max_source_length=max_source_length,
                                  max_target_length=max_target_length,
                                  article_question_NE_overlap=NE_overlap)
    ## save raw data to file
    out_file_name = os.path.join(out_dir, f'{data_name}_question_data.tsv')
    article_data.to_csv(out_file_name, sep='\t', index=False)

if __name__ == '__main__':
    main()