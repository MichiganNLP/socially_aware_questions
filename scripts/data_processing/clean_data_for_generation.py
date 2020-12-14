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

def load_all_articles(data_dir, data_name):
    article_files = list(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))
    article_data = pd.concat(list(map(lambda x: pd.read_csv(x, sep='\t', index_col=False), article_files)), axis=0)
    if(data_name == 'NYT'):
        article_id_matcher = re.compile('(?<=article_)[0-9a-zA-Z]+(?=\.tsv)')
        article_ids = list(map(lambda x: article_id_matcher.search(x).group(0), article_files))
        article_data = article_data.assign(**{
            'article_id' : article_ids
        })
        article_data.rename(columns={'text' : 'article_text'}, inplace=True)
        # remove null articles??
        article_data = article_data[~article_data.loc[:, 'article_text'].apply(lambda x: type(x) is float and np.isnan(x))]
        # clean text
        matcher_pairs = [(re.compile('<.+>'), ' <HTML> '),
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
        comment_data_i = pd.read_csv(comment_file_i, sep=',', index_col=False, usecols=['articleID', 'approveDate', 'commentBody', 'commentType', 'parentID', 'userLocation', 'userID', 'userDisplayName'])
        comment_data.append(comment_data_i)
    comment_data = pd.concat(comment_data, axis=0)
    # remove duplicates? OK
    comment_data.drop_duplicates(['articleID', 'commentBody'], inplace=True)
    # clean comment text
    # fix punctuation without spaces and HTML
#     html_matcher = re.compile('<.+>')
    matcher_pairs = [(re.compile('<.+>'), ' <HTML> '),
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
    # clean article

    # find questions
    question_matcher = re.compile('\?$')
    comment_data = comment_data.assign(**{
        'comment_questions' : comment_data.loc[:, 'commentBody'].apply(lambda x: list(filter(lambda y: question_matcher.search(y) is not None, sent_tokenize(x))))
    })
    # convert to question data (one row/question)
    question_data = []
    question_cols = ['approveDate', 'articleID', 'commentBody', 'commentType', 'parentID', 'userID', 'userLocation', 'userDisplayName']
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
    question_data = question_data.loc[:, ['article_id', 'commentBody', 'question']]
    return question_data

def main():
    parser = ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--data_name', default='NYT')
    parser.add_argument('--comment_dir', default=None) # ../../data/nyt_comments/
    parser.add_argument('--comment_month_year_pairs', nargs='+', default=None) # 'April_2018'
    args = vars(parser.parse_args())

    ## load raw data
    data_dir = args['data_dir']
    data_name = args['data_name']
    article_data = load_all_articles(data_dir, data_name)

    ## optional: load questions
    if(args.get('comment_dir') is not None):
        comment_dir = args['comment_dir']
        comment_month_year_pairs = list(map(lambda x: x.split('_'), args['comment_month_year_pairs']))
        question_data = load_all_comment_questions(comment_dir, comment_month_year_pairs)
        article_data = pd.merge(article_data, question_data, on='article_id')

    ## save to file
    out_dir = args['out_dir']
    out_file_name = os.path.join(out_dir, f'{data_name}_question_data.tsv')
    article_data.to_csv(out_file_name, sep='\t', index=False)

if __name__ == '__main__':
    main()