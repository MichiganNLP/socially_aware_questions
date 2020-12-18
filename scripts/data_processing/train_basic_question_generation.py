"""
Train basic question generation on top of
pre-trained language models (e.g. BART).
"""
import numpy as np
import pandas as pd
import nlp
import torch
from transformers import BartTokenizer
import sys
if ('question_generation' not in sys.path):
    sys.path.append('question_generation')
from data_collator import T2TDataCollator
from transformers import AutoModelForSeq2SeqLM
import os
from trainer import Trainer
from data_helpers import DataProcessor, DataArguments, round_date_to_day
from datetime import datetime
np.random.seed(123)

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
    ## TODO: increase max input length!!
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
        for author_token in author_tokens:
            tokenizer.add_special_tokens({'cls_token': author_token})
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
        # add author variable value to each source text
        for data_idx, data_i in clean_data.iterrows():
            # tokenize, fit to max length
            source_text_i = data_i.loc[source_text_var]
            source_text_tokens_i = tokenizer.tokenize(source_text_i)
            source_text_tokens_i = source_text_tokens_i[:(max_source_length-1)]
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
                source_text_j = tokenizer.convert_tokens_to_string(source_text_tokens_j)
                data_j.loc[source_text_var] = source_text_j
                author_txt_data.append(data_j)
        clean_data = pd.concat(author_txt_data, axis=1).transpose()
    # split train/val
    N = clean_data.shape[0]
    N_train = int(N * train_pct)
    np.random.shuffle(clean_data.values)
    clean_data_train = clean_data.iloc[:N_train, :]
    clean_data_val = clean_data.iloc[N_train:, :]
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
    train_data.set_format(type='torch', columns=columns)
    val_data.set_format(type='torch', columns=columns)
    #     print(f'train data {train_data}')
    train_data_out_file = os.path.join(out_dir, f'{data_name}_train_data.pt')
    val_data_out_file = os.path.join(out_dir, f'{data_name}_val_data.pt')
    torch.save(train_data, train_data_out_file)
    torch.save(val_data, val_data_out_file)
    # save tokenizer?? sure
    tokenizer_out_file = os.path.join(out_dir, 'BART_tokenizer.pt')
    torch.save(tokenizer, tokenizer_out_file)

def load_training_args(model_out_dir, train_data_file, val_data_file, out_dir, max_source_len, max_target_len):
    training_args = DataArguments(model_out_dir)
    training_args.train_file_path = train_data_file
    training_args.valid_file_path = val_data_file
    training_args.data_dir = out_dir
    training_args.task = 'qg'
    training_args.max_source_length = max_source_len
    training_args.max_target_length = max_target_len
    training_args.n_gpu = 1
    # training_args.device = 'cuda:1'
    training_args.seed = 123
    training_args.disable_tqdm = False
    training_args.local_rank = -1
    training_args.output_dir = model_out_dir
    training_args.num_train_epochs = 20
    # training_args.max_steps = 1
    training_args.fp16 = False
    training_args.label_names = None
    training_args.per_device_train_batch_size = 2
    training_args.per_device_eval_batch_size = 2
    # training_args.train_batch_size = 32
    # training_args.eval_batch_size = 32
    training_args.gradient_accumulation_steps = 4
    training_args.learning_rate = 1e-4
    training_args.dataloader_drop_last = False
    training_args.dataloader_num_workers = 8
    # training_args.evaluate_during_training = True
    training_args.do_eval = True
    training_args.evaluation_strategy = 'epoch'
    training_args.eval_steps = 500
    # default values from here lol https://github.com/huggingface/transformers/blob/49759c0cda29ab614b81e0869972c99f2edba7aa/src/transformers/training_args.py
    training_args.weight_decay = 0.01
    training_args.adam_beta1 = 0.9
    training_args.adam_beta2 = 0.999
    training_args.adam_epsilon = 1e-8
    training_args.warmup_steps = 0
    # limits number of checkpoints => 1 GB per optimizer file ;_;
    training_args.save_total_limit = 2
    return training_args

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('train_data') # ../../data/CNN_articles/cnn/article_question_data.tsv
    parser.add_argument('out_dir') # ../../data/CNN_articles/cnn/
    parser.add_argument('--device', default='cpu') # cuda:0 => GPU #0
    parser.add_argument('--model_type', default='bart')
    parser.add_argument('--author_data', default=None) # ../../data/nyt_comments/author_comment_social_data.tsv
    parser.add_argument('--sample_pct', type=float, default=1.0)
    args = vars(parser.parse_args())
    raw_train_data_file = args['train_data']
    out_dir = args['out_dir']
    device_name = args['device']
    model_type = args['model_type']
    author_data = args['author_data']
    sample_pct = args['sample_pct']
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    ## prepare data
    article_question_data = pd.read_csv(raw_train_data_file, sep='\t', index_col=False)
    if(sample_pct < 1.0):
        N_sample = int(article_question_data.shape[0] * sample_pct)
        article_question_data_idx = np.random.choice(article_question_data.index, N_sample, replace=False)
        article_question_data = article_question_data.loc[article_question_data_idx, :]
    train_pct = 0.8
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # TODO: change vocab to include named entities?
    # tmp debugging: small data
    # article_question_data = article_question_data.copy().head(2000)
    data_name = os.path.basename(raw_train_data_file).replace('.tsv', '')
    if(author_data is not None):
        data_name = f'author_type_{data_name}'
        author_data = pd.read_csv(author_data, sep='\t', index_col=False)
        # fix date
        date_day_fmt = '%Y-%m-%d'
        author_data = author_data.assign(**{
            'date_day' : author_data.loc[:, 'date_day'].apply(lambda x: datetime.strptime(x, date_day_fmt))
        })
    # tmp debugging: small data
    # data_name = f'mini_{data_name}'
    train_data_file = os.path.join(out_dir, f'{data_name}_train_data.pt')
    val_data_file = os.path.join(out_dir, f'{data_name}_val_data.pt')
    if(not os.path.exists(train_data_file)):
        prepare_question_data(article_question_data, out_dir, data_name,
                              tokenizer=tokenizer, train_pct=train_pct,
                              author_data=author_data)

    ## train model
    cache_dir = os.path.join(out_dir, 'model_cache/')
    ## TODO: why doesn't the earlier torch import work??
    import torch
    # tokenizer = torch.load('../../data/CNN_articles/cnn/BART_tokenizer.pt')
    model_type_path_lookup = {
        'bart' : 'facebook/bart-base'
    }
    model_path = model_type_path_lookup[model_type]
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device(device_name)
    model.to(device)

    ## load data
    train_dataset = torch.load(train_data_file)
    val_dataset = torch.load(val_data_file)
    train_dataset = train_dataset['train']
    val_dataset = val_dataset['train']

    # get max source/target len
    max_source_len = len(train_dataset['source_ids'][0])
    max_target_len = len(train_dataset['target_ids'][0])
    # data collator
    data_collator = T2TDataCollator(
        tokenizer=tokenizer,
        model_type=model_type,
        mode="training",
        using_tpu=False
    )
    # tmp debugging
    # model_out_dir = os.path.join(out_dir, 'mini_question_generation_model/')
    model_out_dir = os.path.join(out_dir, 'question_generation_model/')
    if (not os.path.exists(model_out_dir)):
        os.mkdir(model_out_dir)

    training_args = load_training_args(model_out_dir, train_data_file, model_out_dir, val_data_file, max_source_len, max_target_len)
    model_args = {
        'label_smoothing': 0,
    }
    ## TODO: prevent model from saving optimizer during every 500 training steps!!
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        #     prediction_loss_only=True,
        label_smoothing=model_args['label_smoothing'],
    )

    ## tmp debugging
    # print(f'evaluation strategy = {trainer.args.eval_steps}')

    ## train
    torch.cuda.empty_cache()
    trainer.train(
        model_path=model_out_dir,
    )
    trainer.save_model()

if __name__ == '__main__':
    main()