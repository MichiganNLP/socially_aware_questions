"""
Clean text data for question generation.
We expect the format:

article ID | article text | question text
"""
import os
import re
from argparse import ArgumentParser
from ast import literal_eval

import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
import numpy as np
from data_helpers import prepare_question_data
from transformers import BartTokenizer, LongformerTokenizer
import logging
np.random.seed(123)

def clean_str_array(x, space_matcher):
    return np.array(literal_eval(space_matcher.sub(',', x)))

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
    # tmp debugging
    print(f'pre-processing: loaded {comment_data.shape[0]} comments total from {len(comment_month_years)} month year pairs')
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

def read_clean_author_data(author_data):
    author_data = pd.read_csv(author_data, sep='\t', compression='gzip', index_col=False, parse_dates=['date_day', 'date_day_bin'])
    # fix date
    author_data = author_data.assign(**{
        'date_day_bin': author_data.loc[:, 'date_day_bin'].apply(lambda x: x.timestamp()).astype(float)
    })
    # read embeds
    embed_cols = list(filter(lambda x: x.endswith('embed'), author_data.columns))
    for embed_col in embed_cols:
        author_data = author_data.assign(**{embed_col: author_data.loc[:, embed_col].apply(lambda x: literal_eval(x) if type(x) is not float else x)})
    return author_data

def main():
    parser = ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--data_dir', default='../../data/')
    parser.add_argument('--data_file', default=None)
    parser.add_argument('--data_name', default='NYT')
    parser.add_argument('--comment_data', default=None)
    parser.add_argument('--comment_dir', default=None) # ../../data/nyt_comments/
    parser.add_argument('--comment_month_year_pairs', nargs='+', default=None) # 'April_2018'
    parser.add_argument('--sample_pct', type=float, default=1.0)
    parser.add_argument('--author_data', default=None)
    # parser.add_argument('--author_data_type', default=None) # {tokens, embeds}
    parser.add_argument('--model_type', default='bart')
    # parser.add_argument('--NE_overlap', type=bool, default=False)
    args = vars(parser.parse_args())
    out_dir = args['out_dir']
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    logging.basicConfig(filename=os.path.join(out_dir, 'clean_data_generation_log.txt'), filemode='w', format='%(asctime)-15s %(message)s', level=logging.DEBUG)
    ## load raw data
    data_dir = args['data_dir']
    data_name = args['data_name']
    data_file = args['data_file']
    if(data_file is None):
        article_data = load_all_articles(data_dir, data_name)
    else:
        article_data = pd.read_csv(data_file, sep='\t', compression='gzip', index_col=False)
        article_data.rename(columns={'id' : 'article_id', 'selftext' : 'article_text'}, inplace=True)
        article_data = article_data.loc[:, ['article_id', 'article_text', 'title', 'created_utc', 'subreddit']]
        # clean up time data (for optional merging with author data)
        article_data.dropna(subset=['created_utc'], inplace=True)
        article_data = article_data[article_data.loc[:, 'created_utc'].apply(lambda x: str(x).isdigit())]
        article_data = article_data.assign(**{'created_utc' : article_data.loc[:, 'created_utc'].astype(int)})

    ## get questions from comments
    if(args.get('comment_data') is not None):
        question_data = pd.read_csv(args['comment_data'], sep='\t', compression='gzip', index_col=False)
        # fix ID var
        question_data.rename(columns={'parent_id' : 'article_id'}, inplace=True)
        question_data = question_data.loc[:, ['article_id', 'id', 'question', 'author']]
        article_data = pd.merge(article_data, question_data, on='article_id', how='inner')
        # print(f'article data cols {article_data.columns}')
    elif(args.get('comment_dir') is not None):
        comment_dir = args['comment_dir']
        comment_month_year_pairs = list(map(lambda x: x.split('_'), args['comment_month_year_pairs']))
        # tmp debugging
        print(f'month year pairs {comment_month_year_pairs}')
        question_data = load_all_comment_questions(comment_dir, comment_month_year_pairs)
        # tmp debugging
        print(f'loaded {question_data} questions total from {len(comment_month_year_pairs)} month year pairs')
        article_data = pd.merge(article_data, question_data, on='article_id', how='inner')
    print(f'loaded article/question {article_data.shape[0]} data')

    ## prepare data for training
    sample_pct = args['sample_pct']
    if (sample_pct < 1.0):
        N_sample = int(article_data.shape[0] * sample_pct)
        article_data_idx = np.random.choice(article_data.index, N_sample, replace=False)
        # tmp debugging
        # tmp_out_file = 'tmp.txt'
        # with open(tmp_out_file, 'w') as tmp_out:
        #     tmp_out.write('\n'.join(article_data_idx))
        #     import sys
        #     sys.exit(0)
        article_data = article_data.loc[article_data_idx, :]
        # tmp debugging
        # tmp_out_file = 'tmp.txt'
        # with open(tmp_out_file, 'w') as tmp_out:
        #     tmp_out.write('\n'.join(article_data.loc[:, 'article_id'].values))
        #     import sys
        #     sys.exit(0)
    train_pct = 0.8
    author_data_file = args.get('author_data')
    # author_data_type = args.get('author_data_type')
    if (author_data_file is not None):
        author_data = read_clean_author_data(author_data_file)
        # date_day_fmt = '%Y-%m-%d %H:%M:%S'
        # date_day_fmt = '%Y-%m-%d'
        # author_data = author_data.assign(**{
        #     'date_day': author_data.loc[:, 'date_day'].apply(lambda x: datetime.strptime(x, date_day_fmt) if type(x) is str else x)
        # })
        # author_data = author_data[author_data.loc[:, 'date_day_bin'].apply(lambda x: type(x) is str)]
        # author_data = author_data.assign(**{
        #     'date_day_bin' : author_data.loc[:, 'date_day_bin'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        # })
        # author_data.loc[:, 'date_day_bin'] = author_data.loc[:, 'date_day_bin'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        # fix date bin type??
        # author_data = author_data.assign(**{
        #     'date_day_bin' : author_data.loc[:, 'date_day_bin'].apply(lambda x: x.to_pydatetime())
        # })
        # print(f'author data date bin sample ({author_data.loc[:, "date_day_bin"].iloc[0]}) has type {type(author_data.loc[:, "date_day_bin"].iloc[0])}')
        # fix UNKs
        author_data.replace('UNK', np.nan, inplace=True)
        # fix subreddit embeds
        # if ('subreddit_embed' in author_data.columns):
        #     space_matcher = re.compile('(?<=\d)[\n\s]+(?=[\d\-])')
        #     author_data = author_data.assign(**{
        #         'subreddit_embed' : author_data.loc[:, 'subreddit_embed'].apply(lambda x: clean_str_array(x, space_matcher) if type(x) is str else None)
        #     })
            # remove authors with null embeddings
            # author_data = author_data[author_data.loc[:, 'subreddit_embed'].apply(lambda x: x is not None)]
            # tmp debugging
            # print(f'sample embed {author_data.loc[:, "subreddit_embed"].iloc[0]}')
        # drop data without authors
        article_data = article_data[~article_data.loc[:, 'author'].apply(lambda x: type(x) is not str and np.isnan(x))]
        # tmp debugging
        # overlap_authors = set(article_data.loc[:, "author"].unique()) & set(author_data.loc[:, 'author'].unique())
        # print(f'{len(overlap_authors)} author overlap in articles/questions')
    # NE_overlap = args['NE_overlap']
    # if (NE_overlap):
    #     data_name = f'NE_overlap_{data_name}'
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
        # if(author_data is not None):
        #     data_name_base = f'author_type_{data_name}_data={author_data_type}'
        # else:
        #     data_name_base = data_name
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        prepare_question_data(article_data, out_dir, data_name,
                              tokenizer=tokenizer, train_pct=train_pct,
                              author_data=author_data,
                              # author_data_type=author_data_type,
                              max_source_length=max_source_length,
                              max_target_length=max_target_length,
                              NE_data_dir=out_dir)
    # if we include author data: also generate "clean" no-author data for comparison
    # if(author_data is not None):
    #     no_author_out_dir = os.path.join(out_dir, 'no_author_data/')
    #     if(not os.path.exists(no_author_out_dir)):
    #         os.mkdir(no_author_out_dir)
    #     no_author_train_data_file = os.path.join(no_author_out_dir, f'{data_name}_train_data.pt')
    #     if(not os.path.exists(no_author_train_data_file)):
    #         # need clean tokenizer
    #         tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
    #         prepare_question_data(article_data, no_author_out_dir, data_name,
    #                               tokenizer=tokenizer, train_pct=train_pct,
    #                               author_data=None,
    #                               max_source_length=max_source_length,
    #                               max_target_length=max_target_length,
    #                               article_question_NE_overlap=NE_overlap,
    #                               NE_data_dir=out_dir)
    ## save raw data to file
    out_file_name = os.path.join(out_dir, f'{data_name}_question_data.gz')
    article_data.to_csv(out_file_name, sep='\t', index=False, compression='gzip')



if __name__ == '__main__':
    main()