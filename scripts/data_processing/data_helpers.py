"""
Data helper functions.
"""
import numpy as np
import pandas as pd
import re
import os
from transformers.training_args import TrainingArguments
from dataclasses import field
from typing import Optional
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from datetime import datetime
import nlp

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
        }

        return encodings

class DataArguments(TrainingArguments):
    train_file_path: str = field(
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: str = field(
        metadata={"help": "Path for cached valid dataset"},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path for data files"},
    )
    task: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which task 'qa', 'qg', 'e2e_qg', 'ans_ext', 'multi'. 'multi' means 'qa', 'qg', 'ans_ext' tasks"},
    )
    qg_format: Optional[str] = field(
        default='prepend_qg_format',
        metadata={"help": "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )
    n_gpu: Optional[int] = field(
        default=1,
    )

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
    # tokens = list(map(lambda x: x.    replace(' ', space_token), tokens))
    # token_txt = tokenizer.convert_tokens_to_string(tokens)
    token_txt = ' '.join(tokens)
    return token_txt

def compare_pred_text_with_target(data, pred_text, tokenizer, max_txt_len=300, cutoff_idx=0):
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
        print(f'source text = {source_text_i[:max_txt_len]}...')
        print(f'target text = {target_text_i}')
        print(f'pred text = {pred_text_i}')
        if(cutoff_idx  > 0 and i >= cutoff_idx):
            break

## model evaluation

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


def prepare_question_data(data, out_dir, data_name, tokenizer, author_data=None, train_pct=0.8):
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
    data_vars = ['article_text', 'question', 'article_id']
    # optional: add author data
    if (author_data is not None):
        author_var = 'userID'
        date_var = 'date_day'
        # fix date variable
        # print(f'data cols {data.columns}')
        data = data.assign(**{
            date_var : data.loc[:, 'createDate'].apply(lambda x: round_date_to_day(x))
        })
        data = pd.merge(data, author_data, on=[author_var, date_var])
        # add author identity to end of source
        author_vars = ['location_region', 'prior_comment_count_bin', 'prior_comment_len_bin']
        data_vars.extend(author_vars)
    # change to clean source/target format
    clean_data = data.loc[:, data_vars].rename(
        columns={'article_text': 'source_text', 'question': 'target_text'})
    # print(clean_data.head())
    # shorten source/target to fit model
    ## TODO: increase max input length!! without breaking memory lol
    max_source_length = 1024
    max_target_length = 64
    ## add author var tokens at the end of each source text => helps decoding? TBD
    if (author_data is not None):
        ## add special tokens
        author_tokens = [
            '<US_AUTHOR>', '<NONUS_AUTHOR>',  # location
            '<COMMENT_COUNT_0_AUTHOR>', '<COMMENT_COUNT_1_AUTHOR>',  # prior comment count
            '<COMMENT_LEN_0_AUTHOR>', '<COMMENT_LEN_1_AUTHOR>',  # prior comment length
        ]
        # for author_token in author_tokens:
            # tokenizer.add_special_tokens({'cls_token': author_token})
        tokenizer.add_tokens(author_tokens, special_tokens=True)
        ## add special tokens to all data
        author_location_token_lookup = {
            'US' : '<US_AUTHOR>',
            'non_US': '<NONUS_AUTHOR>',
        }
        author_var_template_lookup = {
            'prior_comment_count_bin' : '<COMMENT_COUNT_%d_AUTHOR>',
            'prior_comment_len_bin': '<COMMENT_LEN_%d_AUTHOR>',
        }
        author_vars = ['location_region', 'prior_comment_count_bin', 'prior_comment_len_bin']
        author_txt_data = []
        source_text_var = 'source_text'
        pad_space = 1 # need to remove tokens from start and end to make space for pads
        # add author variable value to each source text
        for data_idx, data_i in clean_data.iterrows():
            # tokenize, fit to max length
            source_text_i = data_i.loc[source_text_var]
            source_text_tokens_i = tokenizer.tokenize(source_text_i)
            source_text_tokens_i = source_text_tokens_i[pad_space:(max_source_length-1-pad_space)]
            for author_var in author_vars:
                data_j = data_i.copy()
                source_text_tokens_j = list(source_text_tokens_i)
                author_val_i = data_i.loc[author_var]
                if(author_var == 'location_region'):
                    author_token_val_i = author_location_token_lookup[author_val_i]
                else:
                    # convert bin value to token e.g. "0" + "prior_comment_count_bin" = <COMMENT_COUNT_0_AUTHOR>
                    author_token_val_i = author_var_template_lookup[author_var]%(author_val_i)
                source_text_tokens_j.append(author_token_val_i)
                # tmp debugging
                # if(len(source_text_tokens_j) > max_source_length):
                #     print(f'error: {len(source_text_tokens_j)} tokens generated in input')
                # else:
                #     print(f'correct: {len(source_text_tokens_j)} tokens generated in input')
                source_text_j = tokenizer.convert_tokens_to_string(source_text_tokens_j)
                data_j.loc[source_text_var] = source_text_j
                author_txt_data.append(data_j)
        clean_data = pd.concat(author_txt_data, axis=1).transpose()
    # deduplicate article/answer pairs
    clean_data.drop_duplicates(['source_text', 'target_text'], inplace=True)
    ## split train/val data
    # split by articles! to avoid bleeding between train/test
    article_ids = list(clean_data.loc[:, 'article_id'].unique())
    N_train = int(len(article_ids) * train_pct)
    np.random.shuffle(article_ids)
    train_article_ids = article_ids[:N_train]
    val_article_ids = article_ids[N_train:]
    clean_data_train = clean_data[clean_data.loc[:, 'article_id'].isin(train_article_ids)]
    clean_data_val = clean_data[clean_data.loc[:, 'article_id'].isin(val_article_ids)]
    ## split train/val data by questions
    # N = clean_data.shape[0]
    # N_train = int(N * train_pct)
    # np.random.shuffle(clean_data.values)
    # clean_data_train = clean_data.iloc[:N_train, :]
    # clean_data_val = clean_data.iloc[N_train:, :]
    clean_data_train_out_file = os.path.join(out_dir, f'{data_name}_train_data.csv')
    clean_data_val_out_file = os.path.join(out_dir, f'{data_name}_val_data.csv')
    clean_data_train.to_csv(clean_data_train_out_file, sep=',', index=False)
    clean_data_val.to_csv(clean_data_val_out_file, sep=',', index=False)
    # reload data into correct format lol
    train_data_set = nlp.load_dataset('csv', data_files=clean_data_train_out_file)
    val_data_set = nlp.load_dataset('csv', data_files=clean_data_val_out_file)
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
    train_data = data_processor.process(train_data_set)
    val_data = data_processor.process(val_data_set)
    columns = ["source_ids", "target_ids", "attention_mask"]
    # columns = ["source_ids", "target_ids", "attention_mask", "source_text", "target_text"]
    train_data.set_format(type='torch', columns=columns)
    val_data.set_format(type='torch', columns=columns)
    #     print(f'train data {train_data}')
    train_data_out_file = os.path.join(out_dir, f'{data_name}_train_data.pt')
    val_data_out_file = os.path.join(out_dir, f'{data_name}_val_data.pt')
    torch.save(train_data, train_data_out_file)
    torch.save(val_data, val_data_out_file)
    # save tokenizer?? yes because we will need to post-process other data
    tokenizer_out_file = os.path.join(out_dir, 'BART_tokenizer.pt')
    torch.save(tokenizer, tokenizer_out_file)

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