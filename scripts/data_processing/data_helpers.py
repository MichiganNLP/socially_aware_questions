"""
Data helper functions.
"""
import gzip
import json
import logging
import shutil
from itertools import product

import numpy as np
import pandas as pd
import re
import os

import requests
from rouge_score.rouge_scorer import RougeScorer
from transformers import BartTokenizer, LongformerTokenizer
import torch
from tqdm import tqdm
from stanza import Pipeline
from nltk.translate.bleu_score import sentence_bleu
from datetime import datetime
import nlp
from nltk.tokenize import WordPunctTokenizer, PunktSentenceTokenizer
from stop_words import get_stop_words
from gensim.corpora.dictionary import Dictionary
from time import sleep
import zstandard
import lzma
from praw import Reddit
from psaw import PushshiftAPI
# from datasets.arrow_dataset import Dataset
from nlp.arrow_dataset import Dataset

def assign_label_by_cutoff_pct(data, label_var='gender', min_pct=0.75):
    """
    Assign one label per value based on
    cutoff value; e.g. assign "MALE" to name
    if >=X% of name belongs to "MALE".

    :param data:
    :param label_var:
    :param min_pct:
    :return:
    """
    data.sort_values('count_pct', inplace=True, ascending=False)
    cutoff_data = data[data.loc[:, 'count_pct'] >= min_pct]
    label = 'UNK'
    if (cutoff_data.shape[0] > 0):
        label = cutoff_data.iloc[0, :].loc[label_var]
    return label

def load_name_gender_data(name_data_dir):
    """
    Load per-name gender data from
    directory of Social Security
    birth name records.

    :param name_data_dir:
    :return:
    """
    # let's get all names from likely periods of birth for comments, i.e. 1930-2000
    name_data_matcher = re.compile('yob19[3-9][0-9]')
    name_data_files = list(filter(lambda x: name_data_matcher.search(x), os.listdir(name_data_dir)))
    name_data_files = list(map(lambda x: os.path.join(name_data_dir, x), name_data_files))
    name_data = pd.concat(list(map(lambda x: pd.read_csv(x, sep=',', header=None, index_col=False), name_data_files)),
                          axis=0)
    name_data.columns = ['name', 'gender', 'count']
    # group by name, get raw count
    name_count_data = name_data.groupby(['name', 'gender']).apply(
        lambda x: x.loc[:, 'count'].sum()).reset_index().rename(columns={0: 'count'})
    # # get gender percent
    name_gender_data = name_count_data.groupby('name').apply(
        lambda x: x.assign(**{'count_pct': x.loc[:, 'count'] / x.loc[:, 'count'].sum()}).drop(['count', ],
                                                                                              axis=1)).reset_index(
        drop=True)

    min_gender_pct = 0.75
    name_gender_label_data = []
    for name_i, data_i in name_gender_data.groupby('name'):
        label_i = assign_label_by_cutoff_pct(data_i, label_var='gender', min_pct=min_gender_pct)
        name_gender_label_data.append([name_i, label_i])
    name_gender_label_data = pd.DataFrame(name_gender_label_data, columns=['name', 'gender'])
    # lowercase for consistency
    name_gender_label_data = name_gender_label_data.assign(
        **{'name': name_gender_label_data.loc[:, 'name'].apply(lambda x: x.lower())})
    return name_gender_label_data

def extract_name(text, camel_matcher):
    """
    Extract name from raw text. Assume either "first_name last_name" or "FirstnameLastname".

    :param text:
    :param camel_matcher:
    :return: name
    """
    name = text
    text_tokens = text.split(' ')
    if(len(text_tokens) > 0):
        name = text_tokens[0]
    elif(camel_matcher.search(text) is not None):
        name = camel_matcher.search(text).group(0)
    name = name.lower()
    return name

def clean_text_matchers(txt, word_tokenizer, matcher_pairs):
    """

    :param txt:
    :param word_tokenizer:
    :param matcher_pairs: pairs of regex and string to replace regex with
    :return:
    """
    clean_txt = ' '.join(word_tokenizer.tokenize(txt))
    for matcher_i, sub_i in matcher_pairs:
        clean_txt = matcher_i.sub(sub_i, clean_txt)
    return clean_txt

class DataProcessor:
    """
    Process data for conversion to matrix format.
    """

    def __init__(self, tokenizer, model_type="t5", max_source_length=512, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.hl_token = "<hl>"

        if model_type == "t5":
            self.sep_token = "<sep>"
        elif model_type == "bart":
            self.sep_token = "<sep>"
        else:
            self.sep_token = "[SEP]"

    def process(self, dataset):
        if self.model_type == "t5":
            dataset = dataset.map(self._add_eos_examples)

        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)

        return dataset

    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example

    def _add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example

    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True,
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True,
        )

        encodings = {
            'source_ids': source_encoding['input_ids'],
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
            'article_id' : example_batch['article_id'],
        }

        return encodings

## text generation

def generate_predictions(model, data, tokenizer, device_name='cuda:0',
                         generation_method='beam_search', num_beams=4,
                         temperature=1.0, top_p=1.0):
    """
    Generate predicted text from transformer model.

    :param model:
    :param data:
    :param device_name:
    :return:
    """
    max_decoding_length = 64
    length_penalty = 1
    device = torch.device(device_name)
    model.to(device)
    pred_text = []
    for batch_i in tqdm(data):
        source_i = batch_i['source_ids']
        attention_i = batch_i['attention_mask']
        # fix type in case of difference
        if(type(source_i) is list):
            source_i = torch.LongTensor(source_i)
        if(type(attention_i) is list):
            attention_i = torch.Tensor(attention_i)
        if(generation_method == 'beam_search'):
            output_i = model.generate(
                input_ids=source_i.to(device).reshape(1,-1),
                attention_mask=attention_i.to(device).reshape(1,-1),
                num_beams=num_beams,
                temperature=temperature,
                max_length=max_decoding_length,
                length_penalty=length_penalty,
                num_return_sequences=1,
            )
        elif(generation_method == 'sample'):
            output_i = model.generate(
                input_ids=source_i.to(device).reshape(1, -1),
                attention_mask=attention_i.to(device).reshape(1, -1),
                temperature=temperature,
                top_p=top_p,
                max_length=max_decoding_length,
                length_penalty=length_penalty,
                num_return_sequences=1,
                do_sample=True,
            )
        prediction = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_i]
        pred_text.extend(prediction)
    return pred_text

def cleanup_transformer_tokens(tokens, tokenizer, special_tokens, space_token):
    tokens = list(filter(lambda x: x not in special_tokens, tokens))
    ## TODO: why does convert_tokens_to_string fail with OOV chars?
    tokens = list(map(lambda x: x.replace(' ', space_token), tokens))
    # remove OOV chars
    vocab = tokenizer.get_vocab()
    tokens = list(filter(lambda x: x in vocab, tokens))
    token_txt = tokenizer.convert_tokens_to_string(tokens)
    # token_txt = ' '.join(tokens)
    return token_txt

def compare_pred_text_with_target(data, pred_text, tokenizer,
                                  max_txt_len=300, cutoff_idx=0, extra_data_vars=None):
    """
    Compare predicted text with target data.

    :param data:
    :param pred_text:
    :param tokenizer:
    :param max_txt_len:
    :return:
    """
    special_tokens = {'<pad>', '<s>', '</s>'}
    space_token = 'Ġ'
    for i, (batch_i, pred_text_i) in enumerate(zip(data, pred_text)):
        source_text_i = [tokenizer.decode(x, skip_special_tokens=False) for x in batch_i['source_ids']]
        target_text_i = [tokenizer.decode(x, skip_special_tokens=True) for x in batch_i['target_ids']] # retain special tokens for e.g. author identity
        # cleanup
        source_text_i = cleanup_transformer_tokens(source_text_i, tokenizer, special_tokens, space_token)
        target_text_i = cleanup_transformer_tokens(target_text_i, tokenizer, special_tokens, space_token)
        # source_text_i = list(filter(lambda x: x not in special_tokens, source_text_i))
        # target_text_i = list(filter(lambda x: x not in special_tokens, target_text_i))
        # source_text_i = list(map(lambda x: x.replace(' ', space_token), source_text_i))
        # target_text_i = list(map(lambda x: x.replace(' ', space_token), target_text_i))
        # source_text_i = tokenizer.convert_tokens_to_string(source_text_i)
        # target_text_i = tokenizer.convert_tokens_to_string(target_text_i)
        print('*~*~*~*~*~*')
        if(extra_data_vars is not None):
            for v in extra_data_vars:
                print(f'{v} = {data[v][i]}')
        print(f'source text = {source_text_i[:max_txt_len]}...')
        print(f'target text = {target_text_i}')
        print(f'pred text = {pred_text_i}')
        if(cutoff_idx  > 0 and i >= cutoff_idx):
            break

## model evaluation
def compute_text_rouge(txt_1, txt_2, scorer=None):
    if(scorer is None):
        scorer = RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(txt_1, txt_2)
    return score
def compute_text_bleu(txt_1, txt_2, weights):
    score = sentence_bleu([txt_1], txt_2, weights=weights)
    return score
def compute_max_sent_score(test_questions, gold_question, weights):
    test_question_text = list(map(lambda x: x['question'].lower(), test_questions))
    test_question_bleu_scores = np.array(list(map(lambda x: compute_text_bleu(x, gold_question, weights=weights), test_question_text)))
    max_score = np.max(test_question_bleu_scores)
    max_score_question = test_question_text[np.where(test_question_bleu_scores == max_score)[0][0]]
    return max_score, max_score_question

## date management

def round_date_to_day(time_stamp):
    raw_date = datetime.fromtimestamp(time_stamp)
    round_date = datetime(year=raw_date.year, month=raw_date.month, day=raw_date.day)
    return round_date

## named entities

def extract_all_named_entities(text, pipeline, valid_NE_types={'PERSON', 'GPE', 'ORG', 'EVENT'}):
    """
    Extract all named entities using stanza NER pipeline.

    :param text:
    :param pipeline:
    :param valid_NE_types:
    :return:
    """
    text_doc = pipeline(text)
    named_entities = []
    for text_sent_i in text_doc.sentences:
        NE_tokens_i = []
        for token_j in text_sent_i.tokens:
            token_j_NE = token_j.ner
            token_j_NE_type = token_j_NE.split('-')[-1]
            if(token_j_NE != 'O' and token_j_NE_type in valid_NE_types):
                token_j_text = token_j.text
                # "END" or "SINGLE" token => pop from queue
                if(token_j_NE.startswith('E') or token_j_NE.startswith('S')):
                    NE_tokens_i.append(token_j_text)
                    NE_final = '_'.join(NE_tokens_i)
                    named_entities.append(NE_final)
                    NE_tokens_i = []
                else:
                    NE_tokens_i.append(token_j_text)
    return named_entities

## question analysis

def extract_questions(text, word_tokenizer, sent_tokenizer, question_matcher, min_question_len=5):
    """
    Extract all questions from a span of text.

    :param text:
    :param word_tokenizer:
    :param sent_tokenizer:
    :param question_matcher:
    :param min_question_len:
    :return:
    """
    questions = []
    for sent_i in sent_tokenizer.tokenize(text):
        words_i = word_tokenizer.tokenize(sent_i)
        if(len(words_i) >= min_question_len and question_matcher.match(sent_i)):
            questions.append(sent_i)
    return questions

def extract_questions_all_data(data, min_question_len=5):
    """
    Extract all questions from data.

    :param data:
    :return:
    """
    word_tokenizer = WordPunctTokenizer()
    sent_tokenizer = PunktSentenceTokenizer()
    question_matcher = re.compile('.+\?$')
    questions = list(map(lambda x: extract_questions(x, word_tokenizer, sent_tokenizer, question_matcher, min_question_len=min_question_len), tqdm(data)))
    return questions

def prepare_question_data(data, out_dir, data_name, tokenizer,
                          data_vars=['article_text', 'question', 'article_id'],
                          author_data=None,
                          author_data_type='tokens', # {'tokens', 'embeddings'}
                          train_pct=0.8,
                          max_source_length=1024, max_target_length=64,
                          article_question_NE_overlap=False,
                          NE_data_dir=None):
    """
    Convert raw article/question pairs to source/target pairs
    in matrix format.

    :param data:
    :param out_dir:
    :param data_name:
    :param tokenizer:
    :param author_data:
    :param train_pct:
    :return:
    """
    # optional: add author data
    if (author_data is not None):
        # author_var = 'userID'
        author_var = 'author'
        date_var = 'date_day'
        community_var = 'subreddit'
        static_vars = ['location_region']
        dynamic_vars = ['expert_pct_bin', 'relative_time_bin']
        # fix date variable
        # logging.debug(f'data cols {data.columns}')
        data = data.assign(**{
            date_var : data.loc[:, 'created_utc'].apply(lambda x: round_date_to_day(x))
        })
        # print(f'pre-author-merge data shape={data.shape}')
        if(author_data_type == 'tokens'):
            dynamic_author_data = author_data[(~author_data.loc[:, 'date_day'].isna()) &
                                              (~author_data.loc[:, 'subreddit'].isna())]
            static_author_data = author_data.drop_duplicates(author_var, inplace=False)
            data = pd.merge(data, dynamic_author_data.loc[:, [author_var, date_var, community_var] + dynamic_vars], on=[author_var, date_var, community_var], how='left')
            data = pd.merge(data, static_author_data.loc[:, [author_var]+static_vars], on=author_var, how='left')
            # tmp debugging
            valid_author_attr_data = data.dropna(subset=dynamic_vars, axis=0, how='any')
            logging.info(f'{valid_author_attr_data.shape[0]}/{data.shape[0]} data with valid dynamic author attributes')
            valid_author_attr_data = data.dropna(subset=static_vars, axis=0, how='any')
            logging.info(f'{valid_author_attr_data.shape[0]}/{data.shape[0]} data with valid static author attributes')
            # remove null authors
            data = data[~data.loc[:, 'author'].isna()]
            # print(f'post-author-merge data shape 2={data.shape}')
            # need author vars for later: we'll add author tokens to input
            author_vars = static_vars + dynamic_vars
            data_vars.extend(author_vars)
        elif(author_data_type == 'embeds'):
            # tmp debugging
            logging.info(f'before combining author/post data: author data and post data have {len(set(author_data[author_data.loc[:, "subreddit_embed"].apply(lambda x: type(x) is not float and x is not None)].loc[:, "author"].unique()) & set(data.loc[:, "author"].unique()))} shared authors')
            ## add date bin var
            date_bins = author_data.loc[:, 'date_bin'].unique()
            # tmp debugging
            # print(f'date bins {date_bins}')
            data = data.assign(**{
                'date_bin' : data.loc[:, date_var].apply(lambda x: assign_date_bin(x.timestamp(), date_bins))
            })
            # remove data that can't be linked to valid dates
            data = data[data.loc[:, 'date_bin']!=-1]
            # fix date type
            data = data.assign(**{'date_bin' : data.loc[:, 'date_bin'].apply(lambda x: x.timestamp())})
            # print(f'data date bin sample {type(data.loc[:, "date_bin"].iloc[0])}')
            # print(f'author data date bin sample {type(author_data.loc[:, "date_bin"].iloc[0])}')
            # dynamic_author_data = author_data[
            #     (~author_data.loc[:, date_var].isna()) # &
            #     (~author_data.loc[:, community_var].isna()) &
            #     (~author_data.loc[:, 'subreddit_embed'].isna())
            #     ]
            # print(f'sample embed: {type(author_data[author_data.loc[:, "subreddit_embed"].apply(lambda x: type(x) is not float and x is not None)].loc[:, "subreddit_embed"].iloc[0])}')
            dynamic_author_data = author_data[author_data.loc[:, 'subreddit_embed'].apply(lambda x: type(x) is not float and x is not None)]
            # tmp debugging
            # print(f'author data has {dynamic_author_data.shape[0]} author-date embeddings; {dynamic_author_data.loc[:, "author"].nunique()} authors')
            # print(f'author data has sample authors {dynamic_author_data.loc[:, "author"].unique()[:50]}')
            print(f'author data and post data have {len(set(dynamic_author_data.loc[:, "author"].unique()) & set(data.loc[:, "author"].unique()))} shared authors')
            # merge with author-date pairs
            # data = pd.merge(data, dynamic_author_data.loc[:, ['author', 'date_bin', 'subreddit_embed']], on=['author', 'date_bin'], how='left')
            # tmp debugging
            # tmp debugging: merge with author regardless of date
            data = pd.merge(data, dynamic_author_data.loc[:, ['author', 'subreddit_embed']].drop_duplicates('author'), on='author', how='left')
            author_embed_var = 'author_embeds'
            data.rename(columns={'subreddit_embed' : author_embed_var}, inplace=True)
            data_vars.append(author_embed_var)
            # add null embed for all data with missing embed
            embed_dim = len(dynamic_author_data.loc[:, 'subreddit_embed'].iloc[0])
            print(f'{data[data.loc[:, author_embed_var].apply(lambda x: type(x) is not float and x is not None)].shape[0]}/{data.shape[0]} data with author embeds')
            # null_embed = [0,]*embed_dim
            # data.fillna(value={author_embed_var : null_embed}, inplace=True)
            generate_null_embed = lambda : np.random.randn(embed_dim)
            data = data.assign(**{
                author_embed_var : data.loc[:, author_embed_var].apply(lambda x: generate_null_embed() if type(x) is float or x is None else x)
            })
            # data_with_author_embeds = data[data.loc[:, author_embed_var].apply(lambda x: not all(np.array(x)==0))]

            # tmp debugging
            # import sys
            # sys.exit(0)
            # remove authors without embeds
            # data.dropna(axis=0, subset=[author_embed_var], inplace=True)
            # print(f'{data.shape[0]} data with author embeds')
            # print(f'author embed data sample {data.head()}')
    # change to clean source/target format
    clean_data = data.loc[:, data_vars].rename(
        columns={'article_text': 'source_text', 'question': 'target_text'})
    # deduplicate article/answer pairs
    clean_data.drop_duplicates(['source_text', 'target_text'], inplace=True)
    clean_data = clean_data[(clean_data.loc[:, 'source_text'].apply(lambda x: type(x) is str)) &
                            (clean_data.loc[:, 'target_text'].apply(lambda x: type(x) is str))]
    # clean up return chars
    return_char_matcher = re.compile('[\n\r]')
    clean_data = clean_data.assign(**{
        'source_text' : clean_data.loc[:, 'source_text'].apply(lambda x: return_char_matcher.sub('', x)),
        'target_text': clean_data.loc[:, 'target_text'].apply(lambda x: return_char_matcher.sub('', x)),
    })
    # tmp debugging
    # print('blah')
    # logging.debug(f'after deduplicating, data has {clean_data.shape[0]} questions')
    # logging.debug(clean_data.head())
    # shorten source/target to fit model
    ## add author var tokens at the end of each source text => helps decoding? TBD
    if (author_data is not None):
        ## add special tokens
        # author_tokens = [
        #     '<US_AUTHOR>', '<NONUS_AUTHOR>',  # location
        #     '<COMMENT_COUNT_0_AUTHOR>', '<COMMENT_COUNT_1_AUTHOR>',  # prior comment count
        #     '<COMMENT_LEN_0_AUTHOR>', '<COMMENT_LEN_1_AUTHOR>',  # prior comment length
        # ]
        if(author_data_type == 'tokens'):
            clean_data = add_author_tokens(author_vars, clean_data, max_source_length, tokenizer)
    # optional: filter questions that have >=1 NEs shared with article
    if(article_question_NE_overlap):
        clean_data = filter_data_NE_overlap(NE_data_dir, clean_data, data_name)

    ## split train/val data
    # split by articles! to avoid bleeding between train/test
    article_ids = list(sorted(clean_data.loc[:, 'article_id'].unique()))
    N_train = int(len(article_ids) * train_pct)
    train_article_ids = np.random.choice(article_ids, N_train, replace=False)
    val_article_ids = list(set(article_ids) - set(train_article_ids))
    # tmp debugging lol
    # with open('tmp_article_ids.txt', 'w') as tmp_out:
    #     print(f'writing {len(train_article_ids)} article IDs')
    #     tmp_out.write('\n'.join(map(str, train_article_ids)))
    #     import sys
    #     sys.exit(0)
    # print(f'{len(train_article_ids)} train articles')
    # print(f'{len(val_article_ids)} val articles')
    clean_data_train = clean_data[clean_data.loc[:, 'article_id'].isin(train_article_ids)]
    clean_data_val = clean_data[clean_data.loc[:, 'article_id'].isin(val_article_ids)]
    # print(f'{clean_data_train.shape[0]} train data')
    # print(f'{clean_data_val.shape[0]} val data')
    ## split train/val data by questions
    # N = clean_data.shape[0]
    # N_train = int(N * train_pct)
    # np.random.shuffle(clean_data.values)
    # clean_data_train = clean_data.iloc[:N_train, :]
    # clean_data_val = clean_data.iloc[N_train:, :]
    # clean_data_train_out_file = os.path.join(out_dir, f'{data_name}_train_data.csv')
    # clean_data_val_out_file = os.path.join(out_dir, f'{data_name}_val_data.csv')
    # # print(f'train data columns = {clean_data_train.columns}')
    # # tmp debugging
    # if(not os.path.exists(clean_data_train_out_file)):
    #     clean_data_train.to_csv(clean_data_train_out_file, sep=',', index=False)
    #     clean_data_val.to_csv(clean_data_val_out_file, sep=',', index=False)
    # # reload data into correct format lol
    # data_dir = os.path.dirname(clean_data_train_out_file)
    # # every time we load data, we have to re-download csv loader? yikes
    # train_data_set = nlp.load_dataset('csv', data_files=clean_data_train_out_file, data_dir=data_dir)
    # val_data_set = nlp.load_dataset('csv', data_files=clean_data_val_out_file, data_dir=data_dir)
    # # remove temporary directory...ugh
    # tmp_data_dir = os.path.join(data_dir, 'csv')
    # if(os.path.exists(tmp_data_dir)):
    #     shutil.rmtree(tmp_data_dir)
    # tmp debugging
    # print(f'train data sample = {clean_data_train.head()}')
    dataset_columns = ['source_text', 'target_text', 'article_id']
    if(author_data_type == 'embeds'):
        dataset_columns.append('author_embeds')
    train_data_set = convert_dataframe_to_data_set(clean_data_train, dataset_columns)
    val_data_set = convert_dataframe_to_data_set(clean_data_val, dataset_columns)
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # get max lengths
    # source_text_tokens = clean_data.loc[:, 'source_text'].apply(lambda x: tokenizer.tokenize(x))
    # target_text_tokens = clean_data.loc[:, 'target_text'].apply(lambda x: tokenizer.tokenize(x))
    #     max_source_length = max(source_text_tokens.apply(lambda x: len(x)))
    #     max_target_length = max(target_text_tokens.apply(lambda x: len(x)))
    data_processor = DataProcessor(tokenizer=tokenizer,
                                   model_type='bert',
                                   max_source_length=max_source_length,
                                   max_target_length=max_target_length)
    # print(f'{train_data_set} train data')
    # print(f'{len(val_data_set["source_text"])} val data')
    train_data = data_processor.process(train_data_set)
    val_data = data_processor.process(val_data_set)
    data_columns = ["source_ids", "target_ids", "attention_mask"]
    if(author_data_type == 'embeds'):
        data_columns.append('author_embeds')
    # columns = ["source_ids", "target_ids", "attention_mask", "source_text", "target_text"]
    train_data.set_format(type='torch', columns=data_columns)
    val_data.set_format(type='torch', columns=data_columns)
    #     logging.debug(f'train data {train_data}')
    train_data_out_file = os.path.join(out_dir, f'{data_name}_train_data.pt')
    val_data_out_file = os.path.join(out_dir, f'{data_name}_val_data.pt')
    torch.save(train_data, train_data_out_file)
    torch.save(val_data, val_data_out_file)
    # save tokenizer?? yes because we will need to post-process other data
    tokenizer_name_lookup = {
        BartTokenizer : 'BART',
        LongformerTokenizer : 'LongFormer'
    }
    tokenizer_name = tokenizer_name_lookup[type(tokenizer)]
    tokenizer_out_file = os.path.join(out_dir, f'{tokenizer_name}_tokenizer.pt')
    torch.save(tokenizer, tokenizer_out_file)


def convert_dataframe_to_data_set(data_frame, dataset_columns):
    data_dict = {
        k: data_frame.loc[:, k].values.tolist() for k in dataset_columns
    }
    data_set = Dataset.from_dict(data_dict)
    # vec_columns = ['author_embed']
    # vec_columns = list(filter(lambda x: x in dataset_columns, vec_columns))
    # data_set.set_format(type='torch', columns=vec_columns)
    # 'source_text', 'target_text'
    return data_set


def filter_data_NE_overlap(NE_data_dir, clean_data, data_name):
    # check for NE data!! don't want to do this multiple times
    # also: remove "author" from data name if necessary
    NE_data_name = data_name
    if ('author_type' in data_name):
        NE_data_name = data_name.replace('author_type', '')
    NE_question_data_file = os.path.join(NE_data_dir, f'{NE_data_name}_NE_tags.gz')
    if (not os.path.exists(NE_question_data_file)):
        tqdm.pandas()
        ## extract all NEs from questions, articles
        nlp_pipeline = Pipeline(lang='en', processors='tokenize,ner', use_gpu=True)
        valid_NE_types = {'PERSON', 'GPE', 'ORG', 'EVENT'}
        NE_article_data = clean_data.drop_duplicates('article_id', inplace=False).loc[:, ['source_text', 'article_id']]
        NE_article_data = NE_article_data.assign(**{
            'article_NEs': NE_article_data.loc[:, 'source_text'].progress_apply(lambda x: extract_all_named_entities(x, nlp_pipeline, valid_NE_types=valid_NE_types))
        })
        # NE_article_data = pd.merge(NE_article_data.loc[:, ['article_id', 'article_NEs']], clean_data, on='article_id', how='right')
        NE_question_data = clean_data.loc[:, ['article_id', 'target_text', 'source_text']]
        NE_question_data = NE_question_data.assign(**{
            'question_NEs': NE_question_data.loc[:, 'target_text'].progress_apply(lambda x: extract_all_named_entities(x, nlp_pipeline, valid_NE_types=valid_NE_types))
        })
        NE_question_data = pd.merge(NE_article_data.loc[:, ['article_id', 'article_NEs']], NE_question_data, on='article_id', how='right')
        NE_question_data = NE_question_data.assign(**{
            'article_question_NEs': NE_question_data.apply(lambda x: set(x.loc['article_NEs']) & set(x.loc['question_NEs']), axis=1)
        })
        NE_question_data = NE_question_data[NE_question_data.loc[:, 'article_question_NEs'].apply(lambda x: len(x)) > 0]
        NE_question_data.to_csv(NE_question_data_file, sep='\t', compression='gzip', index=False)
    else:
        NE_question_data = pd.read_csv(NE_question_data_file, sep='\t', compression='gzip', index_col=False)
    # tmp debugging
    logging.debug(f'{NE_question_data.shape[0]}/{clean_data.shape[0]} questions with NE overlap')
    clean_data = pd.merge(clean_data, NE_question_data.loc[:, ['source_text', 'target_text']],
                          on=['source_text', 'target_text'], how='inner')
    # clean_data.drop_duplicates(['source_text', 'target_text'], inplace=True)
    # tmp debugging
    logging.debug(f'{clean_data.shape[0]} clean questions/article pairs after filtering for NE overlap')
    return clean_data


def add_author_tokens(author_vars, clean_data, max_source_length, tokenizer):
    author_tokens = [
        '<US_AUTHOR>', '<NONUS_AUTHOR>',
        '<EXPERT_PCT_0_AUTHOR>', '<EXPERT_PCT_1_AUTHOR>',  # prior comment activity in subreddit
        '<RESPONSE_TIME_0_AUTHOR>', '<RESPONSE_TIME_1_AUTHOR>',  # question response time
    ]
    # for author_token in author_tokens:
    # tokenizer.add_special_tokens({'cls_token': author_token})
    tokenizer.add_tokens(author_tokens, special_tokens=True)
    ## add special tokens to all data
    author_location_token_lookup = {
        'US': '<US_AUTHOR>',
        'NONUS': '<NONUS_AUTHOR>',
    }
    author_var_template_lookup = {
        # 'prior_comment_count_bin' : '<COMMENT_COUNT_%d_AUTHOR>',
        # 'prior_comment_len_bin': '<COMMENT_LEN_%d_AUTHOR>',
        'expert_pct_bin': '<EXPERT_PCT_%d_AUTHOR>',
        'relative_time_bin': '<RESPONSE_TIME_%d_AUTHOR>',
    }
    # author_vars = ['location_region', 'prior_comment_count_bin', 'prior_comment_len_bin']
    author_txt_data = []
    source_text_var = 'source_text'
    pad_space = 1  # need to remove tokens from start and end to make space for pads
    # add author variable value to each source text
    no_author_data = clean_data[clean_data.isna().loc[:, author_vars].apply(lambda x: all(x), axis=1)]
    valid_author_data = clean_data.dropna(axis=0, subset=author_vars, how='all')
    for data_idx, data_i in valid_author_data.iterrows():
        # tokenize, fit to max length
        source_text_i = data_i.loc[source_text_var]
        source_text_tokens_i = tokenizer.tokenize(source_text_i)
        source_text_tokens_i = source_text_tokens_i[pad_space:(max_source_length - 1 - pad_space)]
        # filter to valid vars
        valid_author_data_i = data_i.loc[author_vars].dropna()
        valid_author_vars = valid_author_data_i.index
        for author_var in valid_author_vars:
            # if(not np.isnan(data_i.loc[author_var])):
            data_j = data_i.copy()
            source_text_tokens_j = list(source_text_tokens_i)
            author_val_i = data_i.loc[author_var]
            if (author_var == 'location_region'):
                author_token_val_i = author_location_token_lookup[author_val_i]
            else:
                # convert bin value to token e.g. "0" + "prior_comment_count_bin" = <COMMENT_COUNT_0_AUTHOR>
                author_token_val_i = author_var_template_lookup[author_var] % (author_val_i)
            source_text_tokens_j.append(author_token_val_i)
            # tmp debugging
            # if(len(source_text_tokens_j) > max_source_length):
            #     logging.debug(f'error: {len(source_text_tokens_j)} tokens generated in input')
            # else:
            #     logging.debug(f'correct: {len(source_text_tokens_j)} tokens generated in input')
            source_text_j = tokenizer.convert_tokens_to_string(source_text_tokens_j)
            data_j.loc[source_text_var] = source_text_j
            author_txt_data.append(data_j)
    author_txt_data = pd.concat(author_txt_data, axis=1).transpose()
    # recombine data without-author and with-author
    clean_data = pd.concat([no_author_data, author_txt_data], axis=0)
    # tmp debugging: check for author vars
    for author_token in author_tokens:
        for txt_i in clean_data.loc[:, 'source_text'].values:
            if (author_token in txt_i):
                logging.debug(f'found author token {author_token} in at least one doc')
                break
    # remove author data to avoid NAN bugs in later data reading
    clean_data = clean_data.loc[:, ['source_text', 'target_text', 'article_id']]
    return clean_data

def convert_ids_to_clean_str(token_ids, tokenizer):
    """
    Convert word token IDs to clean string for comparison.

    :param token_ids:
    :param tokenizer:
    :return:
    """
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
    token_str = tokenizer.convert_tokens_to_string(tokens)
    token_str = token_str.strip() # remove extra white space
    return token_str

## author identification

# age
AGE_MATCHER = re.compile('.*?(i am|i\'m) (a |an )?(\d+) (years|year|yrs|yr) old[^e].*?') # adapted from here https://github.com/cfwelch/compositional_demographic_embeddings/blob/master/compose/find_self_statements.py
NUM_MATCHER = re.compile('\d+')
def extract_age(text, age_matcher=None, age_err_cutoff=5):
    combined_text = ' '.join(text).lower()
    if(age_matcher is None):
        age_matcher = AGE_MATCHER
    age_match = age_matcher.findall(combined_text)
    age_match = [y for x in age_match for y in x if NUM_MATCHER.match(y)]
    approx_age = -1
    if(len(age_match) > 0):
        # tmp debugging
        # print(f'age match = {age_match}')
        # print('blah')
        ages = list(map(lambda x: int(NUM_MATCHER.search(x).group(0)), age_match))
        # if ages have larger STD than expected, ignore
        age_std = np.std(ages)
        if(age_std < age_err_cutoff):
            approx_age = int(np.mean(ages))
    return approx_age

# location
def extract_NE_locations(text, location_matcher, sent_tokenizer, ner_pipeline, valid_NE_types={'GPE'}):
    locations = []
    for text_i in text:
        sents = sent_tokenizer.tokenize(text_i)
        for sent_j in sents:
            location_match_j = location_matcher.findall(sent_j.lower())
            if(len(location_match_j) > 0):
                NE_j = extract_all_named_entities(sent_j, ner_pipeline, valid_NE_types=valid_NE_types)
                NE_j = list(map(lambda x: x.replace('_', ' ').lower(), NE_j))
                # look for overlaps
                valid_location_match_j = []
                for location_match_k in location_match_j:
                    NE_matches_k = list(filter(lambda x: x in location_match_k, NE_j))
                    valid_location_match_j.extend(NE_matches_k)
                if(len(valid_location_match_j) > 0):
                    locations.extend(valid_location_match_j)
    return locations
import geocoder
def estimate_locations(text_locations):
    ## TODO: keep full location for e.g. state comparison?
    location_estimates = list(map(lambda x: geocoder.osm(x, method='geocode'), text_locations))
    # get countries for valid estimates
    location_countries = []
    for location_estimate_i in location_estimates:
        if(location_estimate_i.geojson is not None and len(location_estimate_i.geojson['features']) > 0):
            location_country = location_estimate_i.geojson['features'][0]['properties']['country_code']
            location_countries.append(location_country)
    return location_countries
def estimate_country(locations, location_pct_cutoff=0.5):
    location_country_est = 'UNK'
    country_counts = pd.Series(locations).value_counts() / len(locations)
    country_counts.sort_values(inplace=True, ascending=False)
    # get max country
    max_country_count = country_counts.iloc[0]
    if(max_country_count >= location_pct_cutoff):
        location_country_est = country_counts.index[0]
    return location_country_est
def full_location_pipeline(text, location_matcher,
                           sent_tokenizer,
                           ner_pipeline,
                           valid_NE_types={'GPE'},
                           location_pct_cutoff=0.5):
    locations = extract_NE_locations(text, location_matcher, sent_tokenizer, ner_pipeline, valid_NE_types=valid_NE_types)
    location_country_est = 'UNK'
    if(len(locations) > 0):
        location_countries = estimate_locations(locations)
        location_country_est = estimate_country(location_countries, location_pct_cutoff=location_pct_cutoff)
    return location_country_est

## topic modeling
PUNCT = list(',.?!;:"\'-’')
def convert_docs_to_corpus(docs, doc_dict=None):
    """
    Convert raw text to corpus for LDA.

    :param docs:
    :param doc_dict:
    :return:
    """
    word_tokenizer = WordPunctTokenizer()
    doc_tokens = list(map(lambda x: word_tokenizer.tokenize(x.lower()), docs))
    # remove stop words
    en_stops = set(get_stop_words('en') + PUNCT)
    doc_tokens = list(map(lambda x: list(filter(lambda y: y not in en_stops, x)), doc_tokens))
    if(doc_dict is None):
        doc_dict = Dictionary(doc_tokens)
    doc_corpus = list(map(lambda x: doc_dict.doc2bow(x), doc_tokens))
    return doc_dict, doc_corpus
def convert_docs_to_topics(docs, doc_dict, model):
    """
    Convert documents to topic distribution.

    :param docs:
    :param doc_dict:
    :param model:
    :return:
    """
    doc_dict, doc_corpus = convert_docs_to_corpus(docs, doc_dict=doc_dict)
    # get topics
    doc_topics = list(map(lambda x: pd.Series(list(zip(*x))[1], index=list(zip(*x))[0]),
                          model.get_document_topics(doc_corpus, minimum_probability=0.)))
    doc_topics = pd.concat(doc_topics, axis=1).transpose()
    return doc_topics

## twitter API mess
def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers
SLEEP_TIME=300 # sleep time between requests = 5 min
def connect_to_endpoint(search_url, headers, params):
    """
    Connect to

    :param url:
    :param headers:
    :param params:
    :return:
    """
    success = False
    while(not success):
        response = requests.request("GET", search_url, headers=headers, params=params)
    #     print(response.status_code)
        # rate-limit => sleep to recover
        if(response.status_code == 429):
            print(f'sleeping for {SLEEP_TIME} because rate limit error {response}')
            sleep(SLEEP_TIME)
        elif(response.status_code != 200):
            raise Exception(response.status_code, response.text)
        else:
            success = True
    return response.json()

def collect_all_tweets(search_url, headers, query_params, verbose=False, max_tweets=0):
    """
    Collect all tweets based on search query.

    :param search_url:
    :param headers:
    :param query_params:
    :return:
    """
    combined_tweets = []
    has_next_page = True
    max_tweets_reached = False
    ctr = 0
    while(has_next_page and not max_tweets_reached):
        json_response = connect_to_endpoint(search_url, headers, query_params)
        if('data' in json_response):
            response_data = json_response['data']
            # if(verbose):
            #     print(f'response data = {response_data}')
            response_data = pd.DataFrame(response_data)
            # optional: add user information
            if('includes' in json_response):
                if('users' in json_response['includes']):
                    user_data = pd.DataFrame(json_response['includes']['users'])
                    user_data.rename(columns={'id' : 'author_id'}, inplace=True)
                    response_data = pd.merge(response_data, user_data, on='author_id')
            combined_tweets.append(response_data)
        has_next_page = ('meta' in json_response) and ('next_token' in json_response['meta'])
        if(has_next_page):
            query_params['next_token'] = json_response['meta']['next_token']
        ctr += 1
        if(verbose and ctr % 10 == 0):
            print(f'collected {len(combined_tweets)} total')
        # end early if we hit max tweets
        if(max_tweets > 0):
            max_tweets_reached = len(combined_tweets) >= max_tweets
    if(len(combined_tweets) > 0):
        combined_tweets = pd.concat(combined_tweets, axis=0)
    return combined_tweets

class Zreader:
    def __init__(self, file, chunk_size=16384):
        '''Init method'''
        self.fh = open(file,'rb')
        self.chunk_size = chunk_size
        self.dctx = zstandard.ZstdDecompressor()
        self.reader = self.dctx.stream_reader(self.fh)
        self.buffer = ''
    def readlines(self):
        '''Generator method that creates an iterator for each line of JSON'''
        while True:
            chunk = self.reader.read(self.chunk_size).decode()
            if not chunk:
                break
            lines = (self.buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line
            self.buffer = lines[-1]
class FileReader:
    def __init__(self, file_name):
        self.file_name = file_name
        if(file_name.endswith('.xz')):
            self.file_iter = lzma.open(file_name, mode='rt')
        elif(file_name.endswith('.zst')):
            self.file_iter = Zreader(file_name).readlines()
    def __iter__(self):
        return self.file_iter.__iter__()

## load json data
def load_zipped_json_data(data_file):
    data = []
    try:
        for l_i in gzip.open(data_file, 'rt'):
            data_i = json.loads(l_i.strip())
            data.append(data_i)
    except Exception as e:
        print(f'ending data collection early because error {e}')
    data = pd.DataFrame(data)
    return data

## text overlap
def tokenize_stem_text(text, stemmer, word_tokenizer, sent_tokenizer):
    text_sents = sent_tokenizer.tokenize(text)
    # tokenize and stem
    text_sent_tokens = list(map(lambda x:list(map(lambda y:stemmer.stem(y),word_tokenizer.tokenize(x))),text_sents))
    return text_sent_tokens
def compute_word_overlap(text_1, text_2):
    # Jaccard similarity
    word_overlap = set(text_1) & set(text_2)
    word_union = set(text_1) | set(text_2)
    word_overlap_sim = len(word_overlap) / len(word_union)
    return word_overlap_sim
def compute_sent_word_overlap(text_1, text_2):
    # compute word overlap for all pairs of sentences
    # then get max score
    sent_pairs = list(product(text_1, text_2))
    sent_word_overlap_scores = np.array([compute_word_overlap(sent_i, sent_j) for sent_i, sent_j in sent_pairs])
    max_word_overlap_score = max(sent_word_overlap_scores)
    max_word_overlap_sent_pair = sent_pairs[np.argmax(sent_word_overlap_scores)]
    return max_word_overlap_score, max_word_overlap_sent_pair

## text cleaning

def remove_edit_data(text):
    # remove edit data based on structure
    # "EDIT( #): ...\n"
    edit_span = re.search('^edit( [0-9]+)?:[^\n]+$|\nedit( [0-9]+)?:[^\n]+', text.lower())
    if(edit_span is not None):
        span_start, span_end = edit_span.span()
        text = text[:span_start] + text[span_end:]
    return text

## Reddit API
def load_reddit_api(reddit_auth_file):
    reddit_auth = pd.read_csv(reddit_auth_file, sep=',', index_col=False).iloc[
                  0, :]
    client_id = reddit_auth.loc['client_id']
    client_secret = reddit_auth.loc['client_secret']
    user_agent = 'user_agent_123'
    reddit_api = Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )
    pushshift_reddit_api = PushshiftAPI(reddit_api)
    return reddit_api, pushshift_reddit_api

def flatten_columns(df, flat_col):
    """Flattens multiple columns in a data frame, cannot specify all columns!"""
    flat_data = []
    for idx_i, data_i in tqdm(df.iterrows()):
        flat_col_vals = data_i.loc[flat_col]
        for val_j in flat_col_vals:
            data_j = data_i.copy()
            data_j.drop(flat_col, inplace=True)
            data_j = data_j.append(pd.Series({flat_col : val_j}))
            flat_data.append(data_j)
    flat_data = pd.concat(flat_data, axis=1).transpose()
    # NOTE: this approach mixed up the order of comments w.r.t. posts;
    # made it hard to re-connect with parent submissions
    # flattened_cols = {}
    # for col in cols:
    #     if(join_col is None):
    #         flattened_cols[col] = pd.DataFrame([(index, value) for (index, values) in tqdm(df.loc[:, col].iteritems()) for value in values],
    #                                            columns=['index', col]).set_index('index')
    #     else:
    #         flattened_cols[col] = pd.DataFrame([(index, value, values[1]) for (index, values) in tqdm(df.loc[:, [col, join_col]].iteritems()) for value in values[0]], columns=['index', col, join_col]).set_index('index')
    # flattened_df = df.drop(cols, axis=1)
    # for col in cols:
    #     if(join_col is None):
    #         flattened_df = flattened_df.join(flattened_cols[col])
    #     else:
    #         flattened_df = flattened_df.join(flattened_cols[col], on=join_col)
    # # remove null vals??
    # for col in cols:
    #     flattened_df = flattened_df[~flattened_df.loc[:, col].apply(lambda x: type(x) is float and np.isnan(x))]
    return flat_data


def assign_date_bin(date, date_bins):
    diffs = date - date_bins
    valid_diffs = diffs[diffs > 0]
    if (len(valid_diffs) > 0):
        min_diff = min(valid_diffs)
        min_diff_idx = np.where(diffs == min_diff)[0][0]
        date_bin = date_bins[min_diff_idx]
        date_bin = datetime.fromtimestamp(date_bin)
    else:
        date_bin = -1
    return date_bin