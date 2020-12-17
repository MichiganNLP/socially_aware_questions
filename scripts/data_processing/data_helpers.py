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

def generate_predictions(model, data, tokenizer, device_name='cuda:0', generation_method='beam_search', num_beams=4, temperature=1.0, top_p=1.0):
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
            )
        prediction = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_i]
        pred_text.extend(prediction)
    return pred_text

def compare_pred_text_with_target(data, pred_text, tokenizer):
    """
    Compare predicted text with target data.

    :param data:
    :param pred_text:
    :param tokenizer:
    :return:
    """
    special_tokens = set(['<pad>', '<s>', '</s>'])
    for i, (batch_i, pred_text_i) in enumerate(zip(data, pred_text)):
        source_text_i = [tokenizer.decode(x, skip_special_tokens=True) for x in batch_i['source_ids']]
        target_text_i = [tokenizer.decode(x, skip_special_tokens=True) for x in batch_i['target_ids']]
        # cleanup
        source_text_i = ' '.join(list(filter(lambda x: x not in special_tokens, source_text_i)))
        target_text_i = ' '.join(list(filter(lambda x: x not in special_tokens, target_text_i)))
        print('*~*~*~*~*~*')
        print(f'source text = {source_text_i[:300]}...')
        print(f'target text = {target_text_i}')
        print(f'pred text = {pred_text_i}')

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
