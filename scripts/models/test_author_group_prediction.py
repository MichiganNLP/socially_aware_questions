"""
Test how easily we can predict whether a question
was written by a member of group 1 or 2.
"""
import os

import pandas as pd
import numpy as np
np.random.seed(123)
def sample_by_subreddit_author_group(data, group_var):
    subreddit_group_counts = data.loc[:, ['subreddit', group_var]].value_counts()
    min_group_count = subreddit_group_counts.min()
    sample_data = []
    for (subreddit_i, group_var_i), data_i in data.groupby(['subreddit', group_var]):
        sample_idx_i = np.random.choice(data_i.index, min_group_count, replace=False)
        sample_data.append(data_i.loc[sample_idx_i, :])
    sample_data = pd.concat(sample_data, axis=0)
    return sample_data

def load_sample_data():
    question_data = pd.read_csv(
        '../../data/reddit_data/advice_subreddit_filter_comment_question_data.gz',
        sep='\t', compression='gzip', index_col=False)
    ## load author data
    author_data = pd.read_csv(
        '../../data/reddit_data/author_data/combined_author_prior_comment_data.gz',
        sep='\t', compression='gzip', index_col=False)
    ## add date info
    from datetime import datetime
    question_data = question_data.assign(**{
        'date': question_data.loc[:, 'created_utc'].apply(
            lambda x: datetime.fromtimestamp(x))
    })
    question_data = question_data.assign(**{
        'date_day': pd.Series(question_data.loc[:, 'date'].apply(
            lambda x: datetime(year=x.year, month=x.month, day=x.day)).values,
                              dtype='object')
    })
    clean_author_data = author_data.dropna(subset=['date_day'], inplace=False)
    clean_author_data = clean_author_data.assign(**{
        'date_day': pd.Series(clean_author_data.loc[:, 'date_day'].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d')).values, dtype='object')
    })
    dynamic_author_vars = ['expert_pct_bin', 'relative_time_bin']
    static_author_vars = ['location_region']
    question_author_data = question_data.copy()
    question_author_data = pd.merge(
        question_author_data,
        clean_author_data.dropna(subset=dynamic_author_vars, how='all').loc[:,
        ['date_day', 'author', ] + dynamic_author_vars],
        on=['date_day', 'author'], how='left',
    )
    question_author_data = pd.merge(
        question_author_data,
        clean_author_data.dropna(subset=static_author_vars, how='all').loc[:,
        ['author', ] + static_author_vars],
        on='author', how='left',
    )
    # drop null rows
    question_author_data.dropna(subset=dynamic_author_vars + static_author_vars,
                                how='all', inplace=True)
    for dynamic_author_var in dynamic_author_vars:
        question_author_data.drop_duplicates(
            ['author', 'date_day', dynamic_author_var], inplace=True)
    ## load post data
    post_data = pd.read_csv(
        '../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz',
        sep='\t', compression='gzip', index_col=False,
        usecols=['id', 'selftext', 'title'])
    post_data.rename(
        columns={'id': 'parent_id', 'selftext': 'post', 'title': 'post_title'},
        inplace=True)
    # sample data
    group_vars = ['location_region', 'expert_pct_bin', 'relative_time_bin']
    sample_question_data = []
    for group_var in group_vars:
        # drop UNK vals FML
        if (group_var == 'location_region'):
            data_to_sample = question_author_data[
                question_author_data.loc[:, group_var] != 'UNK']
        else:
            data_to_sample = question_author_data.copy()
        sample_question_data_i = sample_by_subreddit_author_group(
            data_to_sample, group_var)
        # reformat to prevent overlap!!
        # text | author group
        sample_question_data_i = sample_question_data_i.loc[:,
                                 [group_var, 'question', 'subreddit',
                                  'parent_id']]
        sample_question_data_i.rename(columns={group_var: 'author_group'},
                                      inplace=True)
        sample_question_data_i = sample_question_data_i.assign(**{
            'author_group': sample_question_data_i.loc[:, 'author_group'].apply(
                lambda x: f'{group_var}={x}'),
            'group_category': group_var,
        })
        sample_question_data.append(sample_question_data_i)
    sample_question_data = pd.concat(sample_question_data, axis=0)
    # tmp debugging
    # print(f'sample question data has label distribution = {sample_question_data.loc[:, "author_group"].value_counts()}')
    sample_post_question_data = pd.merge(post_data, sample_question_data,
                                         on='parent_id', how='inner')
    return sample_post_question_data

import torch
from torch.utils.data import Dataset
class BasicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # item = {k: torch.Tensor(v[idx]) for k, v in self.encodings.items()}
        item = {
            'input_ids' : torch.LongTensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.Tensor(self.encodings['attention_mask'][idx]),
        }
        item["labels"] = torch.LongTensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

## tokenize data
def combine_post_question_text(data, tokenizer, max_length=1024):
    post_tokens = tokenizer.tokenize(data.loc['post'])
    question_tokens = tokenizer.tokenize(data.loc['question'])
    combined_tokens = post_tokens + ['<QUESTION>'] + question_tokens
    if(len(combined_tokens) > max_length):
        combined_tokens = combined_tokens[-max_length:]
    combined_text = tokenizer.convert_tokens_to_string(combined_tokens)
    return combined_text


import torch

torch.manual_seed(123)
import numpy as np

np.random.seed(123)
from transformers import TrainingArguments, Trainer, \
    BartForSequenceClassification
from sklearn.metrics import f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    # print(f'final preds = {[x.shape for x in pred.predictions]}')
    preds = np.argmax(pred.predictions[0], axis=-1)
    score = f1_score(labels, preds)
    metrics = {'F1': score}
    return metrics

def train_test_transformer_model(data, tokenizer,
                                 text_var='post_question',
                                 pred_var='group_category',
                                 train_pct=0.8, max_length=1024):
    # tmp debugging
    print(f'data label dist = {data.loc[:, pred_var].value_counts()}')
    # get train/test data
    np.random.shuffle(data.values)
    N = data.shape[0]
    N_train = int(train_pct * N)
    train_data = data.iloc[:N_train, :]
    test_data = data.iloc[-N_train:, :]
    # get train/test encoding
    train_encodings = tokenizer(train_data.loc[:, text_var].values.tolist(),
                                truncation=True, padding=True,
                                max_length=max_length)
    test_encodings = tokenizer(test_data.loc[:, text_var].values.tolist(),
                               truncation=True, padding=True,
                               max_length=max_length)
    # get train, test datasets
    train_dataset = BasicDataset(train_encodings,
                                 train_data.loc[:, pred_var].values)
    test_dataset = BasicDataset(test_encodings,
                                test_data.loc[:, pred_var].values)
    # get model
    num_labels = data.loc[:, pred_var].nunique()
    model_name = 'facebook/bart-base'
    model = BartForSequenceClassification.from_pretrained(model_name,
                                                          cache_dir='../../data/model_cache/',
                                                          num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))
    # set up training regime
    out_dir = f'../../data/reddit_data/group_classification_model/group={pred_var}/'
    training_args = TrainingArguments(
        output_dir=out_dir,
        # output directory
        num_train_epochs=5,  # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        load_best_model_at_end=True,
        # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=1000,  # log & save weights each logging_steps
        evaluation_strategy="epoch",  # evaluate each `epoch`
        save_total_limit=1,
        eval_accumulation_steps=100,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # evaluate
    test_output = trainer.evaluate()
    test_output = pd.Series(test_output)
    ## save to file!!
    test_output_file = os.path.join(out_dir, f'{pred_var}_prediction_results.csv')
    test_output.to_csv(test_output_file)

def main():
    ## load question data
    post_question_data = load_sample_data()
    # tmp debugging
    # post_question_data = post_question_data.iloc[:1000, :]
    ## set up model etc.
    from transformers import BartTokenizer, BartForSequenceClassification
    model_name = 'facebook/bart-base'
    #tokenizer = BartTokenizer.from_pretrained(model_name,
    #                                          cache_dir='../../data/model_cache/')
    #tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir='../../data/model_cache/')
    tokenizer = torch.load('../../data/model_cache/BART_tokenizer.pt')
    # add special token for combining post + question
    tokenizer.add_special_tokens({'cls_token': '<QUESTION>'})
    max_length = 1024
    post_question_data = post_question_data.assign(**{
        'post_question': post_question_data.apply(
            lambda x: combine_post_question_text(x, tokenizer,
                                                 max_length=max_length), axis=1)
    })
    text_var = 'post_question'
    for group_var_i, data_i in post_question_data.groupby(
            'group_category'):
        out_dir_i = f'../../data/reddit_data/group_classification_model/group={group_var_i}/'
        test_output_file_i = os.path.join(out_dir_i, f'{group_var_i}_prediction_results.csv')
        if(not os.path.exists(test_output_file_i)):
            print(f'testing var = {group_var_i}')
            group_vals_i = data_i.loc[:, 'author_group'].unique()
            data_i = data_i.assign(**{
                group_var_i: (data_i.loc[:, 'author_group'] == group_vals_i[
                    0]).astype(int)
            })
            # print(f'var has dist = {data_i.loc[:, group_var_i].value_counts()}')
            train_test_transformer_model(data_i, tokenizer,
                                     text_var=text_var,
                                     pred_var=group_var_i, train_pct=0.8)

if __name__ == '__main__':
    main()
