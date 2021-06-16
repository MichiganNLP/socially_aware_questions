"""
Test how easily we can predict whether a question
was written by a member of group 1 or 2.
"""
import os
from math import ceil

import pandas as pd
import numpy as np
from accelerate import Accelerator 
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_helpers import BasicDataset
from datasets import load_metric

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

def load_sample_data(sample_size=0):
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
            data_to_sample, group_var, sample_size=sample_size)
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
    BartForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import f1_score

def compute_metrics(pred, predictions=None, labels=None):
    if(pred is not None):
        labels = pred.label_ids
        predictions = pred.predictions[0]
    # print(f'final preds = {[x.shape for x in pred.predictions]}')
    # tmp debugging
    print(f'prediction sample = {predictions}')
    preds = np.argmax(predictions, axis=-1)
    labels = labels.squeeze(1)
    preds = preds.squeeze(0)
    print(f'preds have shape {preds.shape}')
    print(f'labels have shape {labels.shape}')
    acc = float((preds == labels).sum()) / labels.shape[0]
    # get F1 for 1 class and 0 class
    pred_f1_score_1 = f1_score(labels, preds)
    pred_f1_score_0 = f1_score((1-labels), (1-preds))
    TP = ((labels==1) & (preds==1)).sum()
    FP = ((labels==1) & (preds==0)).sum()
    FN = ((labels==0) & (preds==1)).sum()
    pred_precision = TP / (FP + TP)
    pred_recall = TP / (FN + TP)
    #metrics = {'F1': pred_f1_score, 'precision' : float(pred_precision), 'recall' : float(pred_recall)}
    metrics = {'acc' : acc, 'F1_class_1' : pred_f1_score_1, 'F1_class_0' : pred_f1_score_0}
    return metrics

def train_transformer_model(train_dataset, test_dataset, tokenizer, out_dir,
                            n_gpu=1, num_labels=2):
    # tmp debugging
    # print(f'train data has labels = ')
    # get model
    # num_labels = data.loc[:, pred_var].nunique()
    model_name = 'facebook/bart-base'
    model = BartForSequenceClassification.from_pretrained(model_name,
                                                          cache_dir='../../data/model_cache/',
                                                          num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))
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
        num_train_epochs=5,  # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,  # batch size for evaluation
        warmup_steps=1000,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        load_best_model_at_end=True,
        # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=1000,  # log & save weights each logging_steps
        evaluation_strategy="epoch",  # evaluate each `epoch`
        save_total_limit=1,
        eval_accumulation_steps=100,
        ## TODO: multiple gpus
        # local_rank=(-1 if n_gpu > 1 else 0)
    )


def train_model_parallel(train_dataset, test_dataset, tokenizer, out_dir, num_labels=2):
    model_name = 'facebook/bart-base'
    model = BartForSequenceClassification.from_pretrained(model_name,
                                                          cache_dir='../../data/model_cache/',
                                                          num_labels=num_labels)
    from transformers.data.data_collator import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)
    training_args = load_training_args(out_dir)
    train_data_loader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size)
    test_data_loader = DataLoader(test_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    accelerator = Accelerator()
    model, optimizer, train_data_loader, eval_data_loader = accelerator.prepare(
        model, optimizer, train_data_loader, test_data_loader
    )
    num_update_steps_per_epoch = ceil(
        len(train_data_loader) / training_args.gradient_accumulation_steps)
    if training_args.max_train_steps is None:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    else:
        training_args.num_train_epochs = ceil(
            training_args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=training_args.max_train_steps,
    )
    metric = load_metric('accuracy')
    ## train lol
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {training_args.num_train_epochs}")
    print(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(
        f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {training_args.max_train_steps}")
    progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(training_args.num_train_epochs):
        print(f'epoch={epoch}')
        model.train()
        for step, batch in enumerate(train_data_loader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % training_args.gradient_accumulation_steps == 0 or step == len(
                    train_data_loader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= training_args.max_train_steps:
                break

        model.eval()
        if training_args.val_max_target_length is None:
            training_args.val_max_target_length = training_args.max_target_length

        for step, batch in enumerate(test_data_loader):
            with torch.no_grad():
                model_output = model(**batch)
                labels = accelerator.gather(batch['labels']).cpu()
                pred = np.argmax(accelerator.gather(model_output.logits.cpu()), axis=-1)
                metric.add_batch(pred, labels)

        result = metric.compute()

        print(f'test result: {result}')

    if training_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(training_args.output_dir,
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
    test_output_file = os.path.join(out_dir,
                                    f'{pred_var}_prediction_results.csv')
    test_output.to_csv(test_output_file)

def main():
    #sample_size = 0 # no-replacement sampling
    sample_size = 20000 # sampling with replacement
    n_gpu = 1
    # tmp debugging
    ## set up model etc.
    #tokenizer = BartTokenizer.from_pretrained(model_name,
    #                                          cache_dir='../../data/model_cache/')
    #tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir='../../data/model_cache/')
    tokenizer = torch.load('../../data/model_cache/BART_tokenizer.pt')
    # add special token for combining post + question
    tokenizer.add_special_tokens({'cls_token': '<QUESTION>'})
    max_length = 1024
    post_question_data = None # only need data if we need to split train/test
    text_var = 'post_question'
    group_label_lookup = {
        'expert_pct_bin=0.0': 0,
        'expert_pct_bin=1.0' : 1,
        'relative_time_bin=0.0' : 0,
        'relative_time_bin=1.0' : 1,
        'location_region=NONUS' : 0,
        'location_region=US' : 1,
    }
    train_pct = 0.8
    num_labels = 2
    #group_categories = ['location_region', 'expert_pct_bin', 'relative_time_bin']
    group_categories = ['location_region']
    # post_question_data = post_question_data[post_question_data.loc[:, 'group_category'].isin(group_categories)]
    for group_var_i in group_categories:
        print(f'processing group var {group_var_i}')
        out_dir_i = f'../../data/reddit_data/group_classification_model/group={group_var_i}/'
        if(not os.path.exists(out_dir_i)):
            os.mkdir(out_dir_i)
        ## split data
        train_data_file_i = os.path.join(out_dir_i, 'train_data.pt')
        test_data_file_i = os.path.join(out_dir_i, 'test_data.pt')
        if (not os.path.exists(train_data_file_i)):
            if(post_question_data is None):
                post_question_data = load_sample_data(sample_size=sample_size)
                post_question_data = post_question_data.assign(**{
                    'post_question': post_question_data.apply(
                        lambda x: combine_post_question_text(x, tokenizer,
                                                             max_length=max_length),
                        axis=1)
                })
            data_i = post_question_data[post_question_data.loc[:, 'group_category'].isin(group_categories)]
            data_i = data_i.assign(**{
                group_var_i: (data_i.loc[:, 'author_group'].apply(
                    lambda x: group_label_lookup[x])).astype(int)
            })
            test_dataset, train_dataset = split_data(data_i, max_length,
                                                     group_var_i, text_var,
                                                     tokenizer, train_pct)
            torch.save(test_dataset, test_data_file_i)
            torch.save(train_dataset, train_data_file_i)
        ## train
        model_checkpoint_dirs_i = list(filter(lambda x: x.startswith('checkpoint'), os.listdir(out_dir_i)))
        model_checkpoint_dirs_i = list(map(lambda x: os.path.join(out_dir_i, x), model_checkpoint_dirs_i))
        print(f'model checkpoints = {model_checkpoint_dirs_i}')
        # train data if we don't already have model
        if (len(model_checkpoint_dirs_i) == 0):
            # load data
            train_dataset = torch.load(train_data_file_i)
            test_dataset = torch.load(test_data_file_i)
            #train_transformer_model(train_dataset, test_dataset, tokenizer,
            #                        out_dir_i, n_gpu=n_gpu, num_labels=num_labels)
            # parallel training
            train_model_parallel(train_dataset, test_dataset, tokenizer, out_dir_i, num_labels=num_labels)
        ## test
        test_output_file_i = os.path.join(out_dir_i, f'{group_var_i}_prediction_results.csv')
        if(not os.path.exists(test_output_file_i)):
            print(f'testing var = {group_var_i}')
            test_dataset = torch.load(test_data_file_i)
            # tmp debugging
            #test_dataset = test_dataset.select(list(range(1000)), keep_in_memory=True, load_from_cache_file=False)
            # group_vals_i = data_i.loc[:, 'author_group'].unique()
            # print(f'var has dist = {data_i.loc[:, group_var_i].value_counts()}')
            # load weights from most recent model
            most_recent_checkpoint_dir_i = max(model_checkpoint_dirs_i, key=lambda x: int(x.split('-')[1]))
            model_weight_file_i = os.path.join(most_recent_checkpoint_dir_i, 'pytorch_model.bin')
            # test model
            test_transformer_model(test_dataset, out_dir_i, model_weight_file_i, tokenizer,
                                   num_labels, group_var_i)

if __name__ == '__main__':
    main()
