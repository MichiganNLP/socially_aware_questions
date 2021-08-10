"""
Test how easily we can predict whether a question
was written by a member of group 1 or 2.
"""
import os
import pickle
import random
import re
import time
from argparse import ArgumentParser
from ast import literal_eval
from datetime import datetime, timedelta
from itertools import cycle
from math import ceil

import pandas as pd
import numpy as np
from accelerate import Accelerator 
from torch.utils.data import DataLoader, TensorDataset, random_split, \
    RandomSampler, SequentialSampler
from tqdm import tqdm
tqdm.pandas()
from model_helpers import BasicDataset, select_from_dataset
from datasets import load_metric
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
torch.manual_seed(123)
np.random.seed(123)
def sample_by_subreddit_author_group(data, group_var, sample_size=0):
    subreddit_group_counts = data.loc[:, ['subreddit', group_var]].value_counts()
    if(sample_size == 0):
        sample_size = subreddit_group_counts.min()
    sample_data = []
    for (subreddit_i, group_var_i), data_i in data.groupby(['subreddit', group_var]):
        N_i = len(data_i)
        # replace samples if sample > data, i.e. over-sampling
        sample_idx_i = np.random.choice(data_i.index, sample_size, replace=(N_i < sample_size))
        sample_data.append(data_i.loc[sample_idx_i, :])
    sample_data = pd.concat(sample_data, axis=0)
    return sample_data

def load_sample_data(sample_type='all', sample_size=0):
    question_data = pd.read_csv('../../data/reddit_data/advice_subreddit_filter_comment_question_data.gz', sep='\t', compression='gzip', index_col=False)
    ## load author data
    author_data = pd.read_csv('../../data/reddit_data/author_data/combined_author_prior_comment_data.gz', sep='\t', compression='gzip', index_col=False)
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
    question_author_data = pd.merge(question_author_data, clean_author_data.dropna(subset=dynamic_author_vars, how='all').loc[:, ['date_day', 'author', ] + dynamic_author_vars],
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
    post_data = pd.read_csv('../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz', sep='\t', compression='gzip', index_col=False, usecols=['id', 'selftext', 'title'])
    post_data.rename(columns={'id': 'parent_id', 'selftext': 'post', 'title': 'post_title'}, inplace=True)
    ## sample data
    if(sample_type=='paired'):
        # sample N pair of questions per reader group per post
        author_vars = ['expert_pct_bin', 'relative_time_bin', 'location_region']
        flat_question_data = pd.melt(question_author_data, id_vars=['author', 'parent_id', 'question_id', 'question', 'created_utc', 'subreddit'],
                                     value_vars=author_vars, var_name='group_category',
                                     value_name='author_group')
        flat_question_data.dropna(subset=['author_group'], inplace=True)
        flat_question_data = flat_question_data[flat_question_data.loc[:, 'author_group'] != 'UNK']
        ## get paired data
        paired_group_question_data = []
        num_groups_per_category = 2
        for category_i, data_i in flat_question_data.groupby('group_category'):
            author_groups_i = data_i.loc[:, 'author_group'].unique()
            for id_j, data_j in tqdm(data_i.groupby('parent_id')):
                np.random.shuffle(data_j.values)
                ## get max(group_count) questions for each group, and oversample
                if (data_j.loc[:, 'author_group'].nunique() == num_groups_per_category):
                    data_j_1 = data_j[data_j.loc[:, 'author_group'] == author_groups_i[0]]
                    data_j_2 = data_j[data_j.loc[:, 'author_group'] == author_groups_i[1]]
                    max_group_count_j = data_j.loc[:, 'author_group'].value_counts().max()
                    data_j_1 = data_j_1.loc[np.random.choice(data_j_1.index, max_group_count_j, replace=(data_j_1.shape[0] < max_group_count_j))]
                    data_j_2 = data_j_2.loc[np.random.choice(data_j_2.index, max_group_count_j, replace=(data_j_2.shape[0] < max_group_count_j))]
                    paired_group_question_data.extend([data_j_1, data_j_2])
                    # min-sample => too few data!!
                    # min_group_count_j = data_j.loc[:, 'author_group'].value_counts().min()
                    # paired_group_question_data.append(data_j_1.iloc[:min_group_count_j, :])
                    # paired_group_question_data.append(data_j_2.iloc[:min_group_count_j, :])
        question_author_data = pd.concat(paired_group_question_data,axis=0)
        print(f'after paired sampling: question data has label distribution = {question_author_data.loc[:, "author_group"].value_counts()}')
        print(f'after paired sampling: question data has subreddit distribution = {question_author_data.loc[:, "subreddit"].value_counts()}')
        # print(f'question author data sample = {question_author_data.head()}')
    elif(sample_type=='all'):
        group_vars = ['location_region', 'expert_pct_bin', 'relative_time_bin']
        sample_question_data = []
        for group_var in group_vars:
            # drop UNK vals FML
            if (group_var == 'location_region'):
                data_to_sample = question_author_data[question_author_data.loc[:, group_var] != 'UNK']
            else:
                data_to_sample = question_author_data.copy()
            sample_question_data_i = sample_by_subreddit_author_group(data_to_sample, group_var, sample_size=sample_size)
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
        question_author_data = pd.concat(sample_question_data, axis=0)
    # sample by subreddit, article ID
    # remove this because it downsamples rare author groups
    # subreddit_sample_count = 10000
    # subreddit_sample_data = []
    # for subreddit_i, data_i in question_author_data.groupby('subreddit'):
    #     sample_data_i = []
    #     sample_data_ctr_i = 0
    #     np.random.shuffle(data_i.values)
    #     # article_ids_i = np.random.choice(data_i.loc[:, 'article_id'].values, subreddit_sample_count, replace=(data_i.shape[0] < subreddit_sample_count))
    #     # keep collecting data from each post/question group until we hit sample size
    #     for id_j in cycle(data_i.loc[:, 'parent_id'].unique()):
    #         data_j = data_i[data_i.loc[:, 'parent_id']==id_j]
    #         sample_data_i.append(data_j)
    #         sample_data_ctr_i += data_j.shape[0]
    #         if(sample_data_ctr_i >= subreddit_sample_count):
    #             break
    #     subreddit_sample_data += sample_data_i
    # question_author_data = pd.concat(subreddit_sample_data, axis=0)

    # tmp debugging
    print(f'question data has subreddit distribution = {question_author_data.loc[:, "subreddit"].value_counts()}')
    post_question_data = pd.merge(post_data, question_author_data,
                                  on='parent_id', how='right')
    print(f'post/question data has label distribution = {post_question_data.loc[:, "author_group"].value_counts()}')
    return post_question_data

## tokenize data
def combine_post_question_text(data, tokenizer, max_length=1024):
    post_tokens = tokenizer.tokenize(data.loc['post'])
    question_tokens = tokenizer.tokenize(data.loc['question'])
    combined_tokens = post_tokens + ['<QUESTION>'] + question_tokens
    if(len(combined_tokens) > max_length):
        combined_tokens = combined_tokens[-max_length:]
    combined_text = tokenizer.convert_tokens_to_string(combined_tokens)
    return combined_text

from transformers import TrainingArguments, Trainer, \
    BartForSequenceClassification, AdamW, get_scheduler, BartTokenizer, \
    get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

def compute_metrics(pred, predictions=None, labels=None):
    if(pred is not None):
        labels = pred.label_ids
        predictions = pred.predictions[0]
        if(type(labels) is torch.Tensor):
            labels = labels.numpy()
            if(labels.dim() == 2):
                labels = labels.squeeze(1)
    # print(f'final preds = {[x.shape for x in pred.predictions]}')
    # tmp debugging
    # print(f'prediction sample = {predictions}')
    preds = np.argmax(predictions, axis=-1)
    # tmp debugging
    # print(f'preds have shape {preds.shape}')
    # print(f'labels have shape {labels.shape}')
    # preds = preds.squeeze(0)
    # print(f'preds have shape {preds.shape}')
    acc = float((preds == labels).sum()) / labels.shape[0]
    # get F1 for 1 class and 0 class
    pred_f1_score_1 = f1_score(labels, preds)
    pred_f1_score_0 = f1_score((1-labels), (1-preds))
    # TP = ((labels==1) & (preds==1)).sum()
    # FP = ((labels==1) & (preds==0)).sum()
    # FN = ((labels==0) & (preds==1)).sum()
    # pred_precision = TP / (FP + TP)
    # pred_recall = TP / (FN + TP)
    #metrics = {'F1': pred_f1_score, 'precision' : float(pred_precision), 'recall' : float(pred_recall)}
    metrics = {'acc' : acc, 'F1_class_1' : pred_f1_score_1, 'F1_class_0' : pred_f1_score_0}
    return metrics

def train_transformer_model(train_dataset, test_dataset, tokenizer, out_dir,
                            n_gpu=1, num_labels=2, model_weight_file=None):
    # tmp debugging
    # print(f'train data has labels = ')
    # get model
    # num_labels = data.loc[:, pred_var].nunique()
    model_name = 'facebook/bart-base'
    model = BartForSequenceClassification.from_pretrained(model_name,
                                                          cache_dir='../../data/model_cache/',
                                                          num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))
    if(model_weight_file is not None):
        model_weights = torch.load(model_weight_file)
        model.load_state_dict(model_weights)
    # set up training regime
    training_args = load_training_args(out_dir)
    # training_args.n_gpu = n_gpu
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def load_training_args(out_dir):
    return TrainingArguments(
        output_dir=out_dir,
        # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=3,  # batch size per device during training
        per_device_eval_batch_size=3,  # batch size for evaluation
        warmup_steps=1000,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        load_best_model_at_end=True,
        # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=1000,  # log & save weights each logging_steps
        # evaluation_strategy="epoch",  # evaluate each `epoch`
        # tmp debugging: shorter evals => figure out why prediction is broken
        evaluation_strategy="steps",  # evaluate each `eval_steps`
        eval_steps=1000,  # evaluate each `eval_steps`
        save_total_limit=1,
        eval_accumulation_steps=100,
        ## TODO: multiple gpus
        # local_rank=(-1 if n_gpu > 1 else 0)
    )

PARALLEL_MODEL_ARGS= {
    'per_device_train_batch_size' : 2,
    'weight_decay' : 0.01,
    'learning_rate' : 5e-5,
    'gradient_accumulation_steps' : 1,
    'num_train_epochs' : 5,
    'lr_scheduler_type' : 'linear',
    'num_warmup_steps' : 1000,
    'max_train_steps' : 100000,
    'max_target_length' : 64,
}
def train_model_parallel(train_dataset, test_dataset, tokenizer, out_dir, num_labels=2,
                        args=PARALLEL_MODEL_ARGS):
    model_name = 'facebook/bart-base'
    model = BartForSequenceClassification.from_pretrained(model_name,
                                                          cache_dir='../../data/model_cache/',
                                                          num_labels=num_labels)
    from transformers.data.data_collator import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)
    # training_args = load_training_args(out_dir)
    train_data_loader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args['per_device_train_batch_size'])
    test_data_loader = DataLoader(test_dataset, shuffle=True, collate_fn=data_collator, batch_size=args['per_device_train_batch_size'])
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay)],
            "weight_decay": args['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'])
    accelerator = Accelerator()
    model, optimizer, train_data_loader, eval_data_loader = accelerator.prepare(
        model, optimizer, train_data_loader, test_data_loader
    )
    num_update_steps_per_epoch = ceil(
        len(train_data_loader) / args['gradient_accumulation_steps'])
    if(args.get('max_train_steps') is None):
        args['max_train_steps'] = args['num_train_epochs'] * num_update_steps_per_epoch
    else:
        args['num_train_epochs'] = ceil(
            args['max_train_steps'] / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=args['num_warmup_steps'],
        num_training_steps=args['max_train_steps'],
    )
    metric = load_metric('accuracy')
    ## train lol
    total_batch_size = args['per_device_train_batch_size'] * accelerator.num_processes * args['gradient_accumulation_steps']

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args['num_train_epochs']}")
    print(
        f"  Instantaneous batch size per device = {args['per_device_train_batch_size']}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(
        f"  Gradient Accumulation steps = {args['gradient_accumulation_steps']}")
    print(f"  Total optimization steps = {args['max_train_steps']}")
    progress_bar = tqdm(range(args['max_train_steps']), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args['num_train_epochs']):
        print(f'epoch={epoch}')
        model.train()
        for step, batch in enumerate(train_data_loader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args['gradient_accumulation_steps']
            accelerator.backward(loss)
            if (step % args['gradient_accumulation_steps'] == 0 or step == len(train_data_loader) - 1):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if(completed_steps >= args['max_train_steps:']):
                break

        model.eval()
        if(args['val_max_target_length'] is None):
            args['val_max_target_length'] = args['max_target_length']

        for step, batch in enumerate(test_data_loader):
            with torch.no_grad():
                model_output = model(**batch)
                labels = accelerator.gather(batch['labels']).cpu()
                pred = np.argmax(accelerator.gather(model_output.logits.cpu()), axis=-1)
                metric.add_batch(pred, labels)

        result = metric.compute()

        print(f'test result: {result}')

    if(out_dir is not None):
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(out_dir,
                                        save_function=accelerator.save)

def split_data(data, max_length, pred_var, text_var, tokenizer, train_pct):
    # tmp debugging
    print(f'data label dist = {data.loc[:, pred_var].value_counts()}')
    # get train/test data
    np.random.shuffle(data.values)
    N = data.shape[0]
    N_train = int(train_pct * N)
    train_data = data.iloc[:N_train, :]
    test_data = data.iloc[N_train:, :]
    # get train/test encoding
    train_encodings = tokenizer(train_data.loc[:, text_var].values.tolist(),
                                truncation=True, padding=True,
                                max_length=max_length)
    test_encodings = tokenizer(test_data.loc[:, text_var].values.tolist(),
                               truncation=True, padding=True,
                               max_length=max_length)
    # get train, test datasets
    train_dataset = BasicDataset(train_encodings,
                                 train_data.loc[:, pred_var].values.tolist())
    test_dataset = BasicDataset(test_encodings,
                                test_data.loc[:, pred_var].values.tolist())
    return test_dataset, train_dataset

def load_model_tokenizer(model_name, model_dir, num_labels=2, model_weight_file=None, tokenizer=None):
    if(tokenizer is None):
        tokenizer = torch.load(os.path.join(model_dir,'BART_tokenizer.pt'))
        # add special token for combining post + question
        tokenizer.add_special_tokens({'cls_token': '<QUESTION>'})
    model = BartForSequenceClassification.from_pretrained(model_name,
                                                          cache_dir='../../data/model_cache/',
                                                          num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))
    if(model_weight_file is not None):
        model_weights = torch.load(model_weight_file)
        model.load_state_dict(model_weights)
        # tmp debugging
        print(f'loaded model weights')
    # move to GPU
    model = model.to('cuda:0')
    return model, tokenizer

def test_transformer_model(test_dataset, out_dir, model_weight_file, tokenizer, num_labels, pred_var):
    model_name = 'facebook/bart-base'
    model_dir = '../../data/model_cache'
    model, _ = load_model_tokenizer(model_name, model_dir, num_labels=num_labels,
                                            model_weight_file=model_weight_file,
                                            tokenizer=tokenizer)
    # predict
    model.eval()
    pred_labels = []
    pred_probs = []
    # tmp debugging
    print(f'label distribution = {pd.Series([int(x["labels"][0]) for x in test_dataset]).value_counts()}')
    with torch.no_grad():
        for data_i in tqdm(test_dataset):
            input_ids = data_i['input_ids'].unsqueeze(0).to(model.device)
            attention_mask = data_i['attention_mask'].unsqueeze(0).to(model.device)
            labels = data_i['labels'].unsqueeze(0).to(model.device)
            test_data_clean = {'input_ids' : input_ids, 'attention_mask' : attention_mask, 'labels' : labels}
            model_pred = model(**test_data_clean)
            ##print(f'model pred has vals {dir(model_pred)}')
            pred_labels.append(labels.to('cpu'))
            pred_probs.append(model_pred.logits.to('cpu'))
    pred_labels = torch.cat(pred_labels)
    pred_probs = torch.cat(pred_probs)
    # tmp debugging
    #print(f'pred labels = {pred_labels.shape}')
    #print(f'pred probs = {pred_probs.shape}')
    # evaluate
    test_output = compute_metrics(pred=None, labels=pred_labels, predictions=pred_probs)
    test_output = pd.Series(test_output)
    ## save to file!!
    test_output_file = os.path.join(out_dir, f'{pred_var}_prediction_results.csv')
    test_output.to_csv(test_output_file)

def train_test_model_with_encoding(data, group_var, out_dir, text_var='question_encoded', post_var='post_encoded', n_folds=10):
    if(post_var is not None):
        X = np.hstack([np.vstack(data.loc[:, text_var].values), np.vstack(data.loc[:, post_var].values)])
    else:
        X = np.vstack(data.loc[:, text_var].values)
    layer_size = X.shape[1]
    Y = data.loc[:, group_var].values
    group_category = data.loc[:, 'group_category'].unique()[0]
    Y_vals = list(set(Y))
    # convert to binary
    class_1 = Y_vals[0]
    Y = (Y==class_1).astype(int)
    # tmp debugging
    # print(f'X = {X}')
    # print(f'Y = {Y}')
    # fit models across all folds
    model_scores = []
    k_folds = StratifiedKFold(n_splits=n_folds, random_state=123, shuffle=True)
    # score_vars = ['model_acc', f'F1_{Y_vals[0]}', f'F1_{Y_vals[1]}']
    subreddits = data.loc[:, 'subreddit']
    for i, (train_idx, test_idx) in tqdm(enumerate(k_folds.split(X, Y))):
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        # fit model
        model = MLPClassifier(hidden_layer_sizes=[layer_size, ], activation='relu', max_iter=1000, random_state=123)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
#             model_score = f1_score(Y_pred, Y_test)
        model_acc = (Y_pred==Y_test).sum() / len(Y_test)
        # get F1 for both classes...there must be a better way to do this
        model_f1_class_1 = f1_score(Y_pred, Y_test)
        model_f1_class_0 = f1_score((1-Y_pred), (1-Y_test))
        model_scores_i = {'model_acc' : model_acc, f'F1_{Y_vals[0]}' : model_f1_class_1, f'F1_{Y_vals[1]}' : model_f1_class_0, 'fold' : i}
        ## get scores per subreddit!!
        # test_idx_lookup = {idx_i : i for idx_i in enumerate(test_idx)}
        for subreddit_j in subreddits:
            idx_j = list(set(np.where(data.loc[:, 'subreddit']==subreddit_j)[0]) & set(test_idx))
            if(len(idx_j) > 0):
                # test_idx_j = list(map(lambda x: test_idx_lookup[x], idx_j))
                Y_pred_j = model.predict(X[idx_j, :])
                model_acc_j = (Y[idx_j]==Y_pred_j).sum() / len(Y_pred_j)
                model_scores_i[f'model_acc_{subreddit_j}'] = model_acc_j
        model_scores.append(model_scores_i)
    # model_scores = pd.DataFrame(model_scores, columns=score_vars+['fold'])
    model_scores = pd.DataFrame(model_scores)
    # compute mean, sd for all scores
    model_agg_scores = {}
    score_vars = ['model_acc', f'F1_{Y_vals[0]}', f'F1_{Y_vals[1]}'] + [f'model_acc_{x}' for x in subreddits]
    for score_var_i in score_vars:
        scores_i = model_scores.loc[:, score_var_i].dropna()
        model_agg_scores[f'{score_var_i}_mean'] = scores_i.mean()
        model_agg_scores[f'{score_var_i}_SD'] = scores_i.std()
    model_agg_scores = pd.Series(model_agg_scores)
    model_out_file = os.path.join(out_dir, f'MLP_prediction_group={group_category}_class1={class_1}.pkl')
    if (not os.path.exists(model_out_file)):
        # fit model on full data for later use
        full_model = MLPClassifier(hidden_layer_sizes=[layer_size, ], activation='relu', max_iter=1000, random_state=123)
        full_model.fit(X, Y)
        # save model
        with open(model_out_file, 'wb') as model_output:
            pickle.dump(full_model, model_output)
    else:
        full_model = pickle.load(open(model_out_file, 'rb'))
    return full_model, class_1, model_agg_scores

def train_test_basic_classifier(group_categories, sample_size, out_dir, sample_type='sample'):
    ## get sentence encodings for all posts/questions
    post_question_data_file = os.path.join(out_dir, f'sample_type={sample_type}_post_question_data.gz')
    embed_vars = ['question', 'post']
    if(not os.path.exists(post_question_data_file)):
        post_question_data = load_sample_data(sample_type=sample_type, sample_size=sample_size)
        sentence_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        # tmp debugging
        # post_question_data = post_question_data.iloc[:1000, :]
        # question_encoding = sentence_model.encode(post_question_data.loc[:, 'question'], batch_size=8, device=torch.cuda.current_device(), show_progress_bar=True)
        # post_question_data = post_question_data.assign(**{
        #     'question_encoded': [question_encoding[i, :] for i in range(question_encoding.shape[0])],
        #     # 'question_encoded': post_question_data.loc[:, 'question'].progress_apply(lambda x: sentence_model.encode(x))
        # })
        for embed_var_i in embed_vars:
            encode_var_i = f'{embed_var_i}_encoded'
            encoding_i = sentence_model.encode(post_question_data.loc[:, embed_var_i], batch_size=16, device=torch.cuda.current_device(), show_progress_bar=True)
            post_question_data = post_question_data.assign(**{
                encode_var_i : [encoding_i[i, :] for i in range(encoding_i.shape[0])],
            })
        ## compress via PCA => prevent overfitting
        embed_dim = 100
        for embed_var_i in embed_vars:
            encode_var_i = f'{embed_var_i}_encoded'
            mat_i = np.vstack(post_question_data.loc[:, encode_var_i])
            pca_model_i = PCA(n_components=embed_dim, random_state=123)
            reduce_mat_i = pca_model_i.fit_transform(mat_i)
            post_question_data = post_question_data.assign(**{
                f'PCA_{encode_var_i}': [reduce_mat_i[i, :] for i in range(reduce_mat_i.shape[0])]
            })
            # save PCA model file for later data transformation FML
            pca_model_file_i = os.path.join(out_dir, f'PCA_model_embed={encode_var_i}.pkl')
            pickle.dump(pca_model_i, open(pca_model_file_i, 'wb'))
        post_question_data.to_csv(post_question_data_file, sep='\t', compression='gzip', index=False)
    else:
        arr_matcher = re.compile('[\[\]\n]')
        embed_var_converters = {
            f'PCA_{embed_var}_encoded': lambda x: np.fromstring(arr_matcher.sub('', x).strip(), sep=' ', dtype=float)
            for embed_var in embed_vars
        }
        post_question_data = pd.read_csv(post_question_data_file, sep='\t', compression='gzip', index_col=False,
                                         converters=embed_var_converters)
        # tmp debugging
        # print(f'post ')

    # author_group_scores = []
    text_var = 'PCA_question_encoded'
    post_var = 'PCA_post_encoded'
    # for group_var_i, data_i in post_question_data.groupby('group_category'):
    question_post_out_dir = os.path.join(out_dir, 'question_post_data')
    question_out_dir = os.path.join(out_dir, 'question_data')
    out_dirs = [question_out_dir, question_post_out_dir]
    post_vars = [None, post_var]
    for group_var_i in group_categories:
        data_i = post_question_data[post_question_data.loc[:, 'group_category'] == group_var_i]
        print(f'testing var = {group_var_i}')
        ## model with question only
        for post_var_j, out_dir_j in zip(post_vars, out_dirs):
            if(not os.path.exists(out_dir_j)):
                os.mkdir(out_dir_j)
            full_model_i, class_var_1_i, model_scores = train_test_model_with_encoding(data_i, 'author_group', out_dir_j, text_var=text_var, post_var=post_var_j)
            model_scores.loc['author_group'] = group_var_i
            # write scores
            author_group_score_out_file = os.path.join(out_dir_j, f'MLP_prediction_group={group_var_i}_class1={class_var_1_i}_scores.tsv')
            model_scores.to_csv(author_group_score_out_file, sep='\t', index=True)

def train_test_transformer_classification(group_categories, group_var,
                                          max_length, n_gpu, num_labels,
                                          out_dir, post_question_data,
                                          sample_size, text_var, tokenizer,
                                          train_pct, retrain=False):
    group_label_lookup = {
        'expert_pct_bin=0.0': 0,
        'expert_pct_bin=1.0': 1,
        'relative_time_bin=0.0': 0,
        'relative_time_bin=1.0': 1,
        'location_region=NONUS': 0,
        'location_region=US': 1,
    }
    ## split data
    train_data_file_i = os.path.join(out_dir, 'train_data.pt')
    test_data_file_i = os.path.join(out_dir, 'test_data.pt')
    if (not os.path.exists(train_data_file_i)):
        if (post_question_data is None):
            post_question_data = load_sample_data(sample_size=sample_size)
            post_question_data = post_question_data.assign(**{
                'post_question': post_question_data.apply(
                    lambda x: combine_post_question_text(x, tokenizer,
                                                         max_length=max_length),
                    axis=1)
            })
        data_i = post_question_data[
            post_question_data.loc[:, 'group_category'].isin(group_categories)]
        data_i = data_i.assign(**{
            group_var: (data_i.loc[:, 'author_group'].apply(
                lambda x: group_label_lookup[x])).astype(int)
        })
        test_dataset, train_dataset = split_data(data_i, max_length,
                                                 group_var, text_var,
                                                 tokenizer, train_pct)
        torch.save(test_dataset, test_data_file_i)
        torch.save(train_dataset, train_data_file_i)
    ## train
    model_checkpoint_dirs_i = list(
        filter(lambda x: x.startswith('checkpoint'), os.listdir(out_dir)))
    model_checkpoint_dirs_i = list(
        map(lambda x: os.path.join(out_dir, x), model_checkpoint_dirs_i))
    print(f'model checkpoints = {model_checkpoint_dirs_i}')
    if (len(model_checkpoint_dirs_i) == 0 or retrain):
        # load data
        train_dataset = torch.load(train_data_file_i)
        test_dataset = torch.load(test_data_file_i)
        ## down-sample test data because model can't handle it all boo hoo
        test_sample_size = 5000
        test_idx = np.random.choice(list(range(len(test_dataset.labels))),
                                    test_sample_size, replace=False)
        test_dataset = select_from_dataset(test_dataset, test_idx)
        # optional: load model
        model_weight_file_i = None
        if (retrain):
            most_recent_checkpoint_dir_i = max(model_checkpoint_dirs_i,
                                               key=lambda x: int(
                                                   x.split('-')[1]))
            model_weight_file_i = os.path.join(most_recent_checkpoint_dir_i,
                                               'pytorch_model.bin')
        train_transformer_model(train_dataset, test_dataset, tokenizer,
                                out_dir, n_gpu=n_gpu, num_labels=num_labels,
                                model_weight_file=model_weight_file_i)
        # parallel training
        # TODO: debug device sharing
        # train_model_parallel(train_dataset, test_dataset, tokenizer, out_dir_i, num_labels=num_labels)
    ## test
    test_output_file_i = os.path.join(out_dir,
                                      f'{group_var}_prediction_results.csv')
    if (not os.path.exists(test_output_file_i)):
        print(f'testing var = {group_var}')
        test_dataset = torch.load(test_data_file_i)
        # tmp debugging
        # test_dataset = test_dataset.select(list(range(1000)), keep_in_memory=True, load_from_cache_file=False)
        # group_vals_i = data_i.loc[:, 'author_group'].unique()
        # print(f'var has dist = {data_i.loc[:, group_var_i].value_counts()}')
        # load weights from most recent model
        model_checkpoint_dirs_i = list(
            filter(lambda x: x.startswith('checkpoint'), os.listdir(out_dir)))
        model_checkpoint_dirs_i = list(
            map(lambda x: os.path.join(out_dir, x), model_checkpoint_dirs_i))
        most_recent_checkpoint_dir_i = max(model_checkpoint_dirs_i,
                                           key=lambda x: int(x.split('-')[1]))
        model_weight_file_i = os.path.join(most_recent_checkpoint_dir_i,
                                           'pytorch_model.bin')
        # test model
        test_transformer_model(test_dataset, out_dir, model_weight_file_i,
                               tokenizer,
                               num_labels, group_var)

QUESTION_TOKEN='[QUESTION]'
EXTRA_PADDING_TOKENS=3
def combine_post_question(data, tokenizer, max_length=1024):
    post_txt = data.loc["post"]
    question_txt = data.loc["question"]
    post_question_txt = ' '.join([post_txt, QUESTION_TOKEN, question_txt])
    post_question_txt_tokens = tokenizer.tokenize(post_question_txt)
    if(len(post_question_txt_tokens) > max_length):
        # shorten post to fit question
        post_tokens = tokenizer.tokenize(post_txt)
        question_tokens = tokenizer.tokenize(question_txt)
        post_tokens = post_tokens[:(max_length - len(question_tokens) - EXTRA_PADDING_TOKENS)] # subtract extra tokens for CLS, QUESTION, EOS
        post_txt = tokenizer.convert_tokens_to_string(post_tokens)
        post_question_txt = ' '.join([post_txt, QUESTION_TOKEN, question_txt])
    return post_question_txt


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(timedelta(seconds=elapsed_rounded))

def train_test_full_transformer(group_categories, sample_size, sample_type, out_dir):
    """
    Train/test model to predict group from question+post text, with full transformer BART model. FUN

    :param group_categories:
    :param out_dir:
    """
    ## set random seeds!!
    seed_val = 123
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    ## organize data
    post_question_data_file = os.path.join(out_dir, f'sample_type={sample_type}_post_question_data.gz')
    if (not os.path.exists(post_question_data_file)):
        post_question_data = load_sample_data(sample_type=sample_type, sample_size=sample_size)
        post_question_data.to_csv(post_question_data_file, sep='\t', compression='gzip', index=False)
    else:
        post_question_data = pd.read_csv(post_question_data_file, sep='\t', compression='gzip', index_col=False)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[QUESTION]']})
    post_question_data = post_question_data.assign(**{
        'post_question': post_question_data.apply(lambda x: combine_post_question(x, tokenizer), axis=1)
    })
    default_group_values = {
        'expert_pct_bin' : 1.0,
        'relative_time_bin' : 1.0,
        'location_region' : 'US',
    }
    for group_category_i in group_categories:
        print(f'about to process data for category={group_category_i}')
        post_question_data_i = post_question_data[post_question_data.loc[:, 'group_category']==group_category_i]
        default_group_val_i = default_group_values[group_category_i]
        labels_i = (post_question_data_i.loc[:, 'author_group']==default_group_val_i).astype(int).values
        # Tokenize all of the sentences and map the tokens to word IDs.
        # max_length = 1024 # TOO LONG!! NO CAPES
        max_length = 512
        input_data_i = list(map(lambda x: tokenizer.encode_plus(x, add_special_tokens=True, return_attention_mask=True,
                                                              return_tensors='pt', max_length=max_length,
                                                              padding='max_length', truncation=True),
                                tqdm(post_question_data_i.loc[:, 'post_question'])))
        input_ids_i = list(map(lambda x: x['input_ids'], input_data_i))
        attention_masks_i = list(map(lambda x: x['attention_mask'], input_data_i))
        # Convert the lists into tensors.
        input_ids_i = torch.cat(input_ids_i, dim=0)
        attention_masks_i = torch.cat(attention_masks_i, dim=0)
        labels_i = torch.LongTensor(labels_i)
        dataset = TensorDataset(input_ids_i, attention_masks_i, labels_i)
        # Create a 90-10 train-validation split.
        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        print('{:d} training samples'.format(train_size))
        print('{:d} validation samples'.format(val_size))
        model_out_dir_i = os.path.join(out_dir, f'question_post_group={group_category_i}_transformer_model')
        if (not os.path.exists(model_out_dir_i)):
            os.mkdir(model_out_dir_i)
        model_out_file_i = os.path.join(model_out_dir_i, 'pytorch_model.bin')
        if(not os.path.exists(model_out_file_i)):
            batch_size = 4
            train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler=RandomSampler(train_dataset),  # Select batches randomly
                batch_size=batch_size  # Trains with this batch size.
            )
            validation_dataloader = DataLoader(
                val_dataset,  # The validation samples.
                sampler=SequentialSampler(val_dataset),
                batch_size=batch_size  # Evaluate with this batch size.
            )
            ## load model
            model = BartForSequenceClassification.from_pretrained(
                "facebook/bart-base",
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False,
            )
            model.resize_token_embeddings(len(tokenizer))
            model = model.cuda()
            optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
            # epochs = 4 # NOTE: this leads to ~55% accuracy on validation w/ N=10000 data, which was increasing before it ended
            epochs = 8
            total_steps = len(train_dataloader) * epochs
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=0,
                                                        num_training_steps=total_steps)
            ## train!! and validate
            training_stats = []
            # Measure the total training time for the whole run.
            total_t0 = time.time()
            device = torch.cuda.current_device()
            # For each epoch...
            for epoch_i in range(0, epochs):
                # ========================================
                #               Training
                # ========================================
                print("")
                print(
                    '======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
                print('Training...')
                t0 = time.time()
                total_train_loss = 0

                model.train()
                for step, batch in enumerate(train_dataloader):
                    if step % 40 == 0 and not step == 0:
                        elapsed = format_time(time.time() - t0)
                        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                            step, len(train_dataloader), elapsed))
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)
                    model.zero_grad()
                    result = model(input_ids=b_input_ids,
                                   attention_mask=b_input_mask,
                                   labels=b_labels,
                                   return_dict=True)
                    b_input_ids = b_input_ids.to('cpu')
                    b_input_mask = b_input_mask.to('cpu')
                    b_labels = b_labels.to('cpu')
                    loss = result.loss
                    logits = result.logits
                    total_train_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                # Calculate the average loss over all of the batches.
                avg_train_loss = total_train_loss / len(train_dataloader)
                # Measure how long this epoch took.
                training_time = format_time(time.time() - t0)
                print("")
                print("  Average training loss: {0:.2f}".format(avg_train_loss))
                print("  Training epoch took: {:}".format(training_time))
                # ========================================
                #               Validation
                # ========================================
                # After the completion of each training epoch, measure our performance on
                # our validation set.
                print("")
                print("Running Validation...")
                t0 = time.time()
                # Put the model in evaluation mode--the dropout layers behave differently
                # during evaluation.
                model.eval()
                # Tracking variables
                total_eval_accuracy = 0
                total_eval_loss = 0
                nb_eval_steps = 0
                # Evaluate data for one epoch
                for batch in validation_dataloader:
                    # Unpack this training batch from our dataloader.
                    #
                    # As we unpack the batch, we'll also copy each tensor to the GPU using
                    # the `to` method.
                    #
                    # `batch` contains three pytorch tensors:
                    #   [0]: input ids
                    #   [1]: attention masks
                    #   [2]: labels
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)
                    # Tell pytorch not to bother with constructing the compute graph during
                    # the forward pass, since this is only needed for backprop (training).
                    with torch.no_grad():
                        # Forward pass, calculate logit predictions.
                        # token_type_ids is the same as the "segment ids", which
                        # differentiates sentence 1 and 2 in 2-sentence tasks.
                        result = model(b_input_ids,
                                       attention_mask=b_input_mask,
                                       labels=b_labels,
                                       return_dict=True)
                        # fix device => GPU memory errors
                        b_input_ids = b_input_ids.to('cpu')
                        b_input_mask = b_input_mask.to('cpu')
                        b_labels = b_labels.to('cpu')
                    # Get the loss and "logits" output by the model. The "logits" are the
                    # output values prior to applying an activation function like the
                    # softmax.
                    loss = result.loss
                    logits = result.logits
                    # Accumulate the validation loss.
                    total_eval_loss += loss.item()
                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    # Calculate the accuracy for this batch of test sentences, and
                    # accumulate it over all batches.
                    total_eval_accuracy += flat_accuracy(logits, label_ids)
                # Report the final accuracy for this validation run.
                avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
                print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
                # Calculate the average loss over all of the batches.
                avg_val_loss = total_eval_loss / len(validation_dataloader)
                # Measure how long the validation run took.
                validation_time = format_time(time.time() - t0)
                print("  Validation Loss: {0:.2f}".format(avg_val_loss))
                print("  Validation took: {:}".format(validation_time))
                # Record all statistics from this epoch.
                training_stats.append(
                    {
                        'epoch': epoch_i + 1,
                        'Training Loss': avg_train_loss,
                        'Valid. Loss': avg_val_loss,
                        'Valid. Accur.': avg_val_accuracy,
                        'Training Time': training_time,
                        'Validation Time': validation_time
                    }
                )
            print("")
            print("Training complete!")
            print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
            ## save trained model
            model.save_model(model_out_dir_i)
            ## save training stats
            training_stats_data = pd.DataFrame(training_stats)
            training_stats_data_file = os.path.join(model_out_dir_i, f'training_stats.tsv')
            training_stats_data.to_csv(training_stats_data_file, sep='\t', index=False)
        ## compute accuracy on validation data
        model = BartForSequenceClassification.from_pretrained(
            "facebook/bart-base",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )
        model_weights = torch.load(model_out_file_i)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(model_weights)
        # get validation accuracy
        model.eval()
        total_eval_accuracy = 0
        val_preds = []
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
                result = model(b_input_ids,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)
                b_input_ids = b_input_ids.to('cpu')
                b_input_mask = b_input_mask.to('cpu')
                b_labels = b_labels.to('cpu')
                label_ids = b_labels.numpy()
            logits = result.logits
            logits = logits.detach().cpu().numpy()
            preds = list(logits.argmax(axis=1))
            val_preds.extend(preds)
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        # compute mean, std F1
        val_labels = list(map(lambda x: x[2], val_dataset))
        val_f1 = f1_score(val_labels, val_preds)
        val_accuracy = np.sum(val_preds == val_labels) / len(val_labels)
        val_scores = pd.Series([val_f1, val_accuracy], index=['F1', 'acc'])
        # save scores
        score_out_file_i = os.path.join(model_out_dir_i, 'acc_scores.tsv')
        val_scores.to_csv(score_out_file_i, sep='\t', index=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('--group_categories', nargs='+', default=['location_region', 'expert_pct_bin', 'relative_time_bin'])
    parser.add_argument('--retrain', dest='feature', action='store_true', default=False)
    parser.add_argument('--out_dir', default='../../data/reddit_data/group_classification_model/')
    args = vars(parser.parse_args())
    #sample_size = 0 # no-replacement sampling
    sample_size = 10000 # sampling with replacement
    # n_gpu = 1
    # tmp debugging
    ## set up model etc.
    #tokenizer = BartTokenizer.from_pretrained(model_name,
    #                                          cache_dir='../../data/model_cache/')
    #tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir='../../data/model_cache/')
    post_question_data = None # only need data if we need to split train/test
    # text_var = 'post_question'
    # tmp debugging: only use question
    # text_var = 'question'
    # train_pct = 0.8
    # num_labels = 2
    group_categories = args['group_categories']
    # retrain = 'retrain' in args and args['retrain']
    #group_categories = ['location_region', 'expert_pct_bin', 'relative_time_bin']
    # group_categories = ['location_region']
    # post_question_data = post_question_data[post_question_data.loc[:, 'group_category'].isin(group_categories)]
    ## simple neural network approach
    out_dir = args['out_dir']
    sample_type = 'paired'
    # sample_type = 'sample'
    # train_test_basic_classifier(group_categories, sample_size, out_dir, sample_type=sample_type)
    train_test_full_transformer(group_categories, sample_size, sample_type, out_dir)

    ## transformer code: doesn't learn anything? same P(Y|X) regardless of input
    # for group_var_i in group_categories:
    #     print(f'processing group var {group_var_i}')
    #     out_dir_i = f'../../data/reddit_data/group_classification_model/group={group_var_i}/'
    #     if(not os.path.exists(out_dir_i)):
    #         os.mkdir(out_dir_i)
    #     tokenizer = torch.load('../../data/model_cache/BART_tokenizer.pt')
    #     # add special token for combining post + question
    #     tokenizer.add_special_tokens({'cls_token': '<QUESTION>'})
    #     max_length = 1024
    #     train_test_transformer_classification(group_categories,
    #                                           group_var_i, max_length, n_gpu,
    #                                           num_labels, out_dir_i,
    #                                           post_question_data, sample_size,
    #                                           text_var, tokenizer, train_pct, retrain=retrain)

if __name__ == '__main__':
    main()
