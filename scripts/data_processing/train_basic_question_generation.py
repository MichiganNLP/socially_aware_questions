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
from data_helpers import DataProcessor, DataArguments

def prepare_question_data(data, out_dir, data_name, tokenizer, train_pct=0.8):
    # change to clean source/target format
    clean_data = data.loc[:, ['article_text', 'question', 'article_id']].rename(
        columns={'article_text': 'source_text', 'question': 'target_text'})
    print(clean_data.head())
    # split train/val
    np.random.seed(123)
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
    # tmp debugging: shorten source/target
    ## TODO: increase max input length!!
    max_source_length = 1024
    max_target_length = 64
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
    args = vars(parser.parse_args())
    raw_train_data_file = args['train_data']
    out_dir = args['out_dir']
    device_name = args['device']
    model_type = args['model_type']

    ## prepare data
    article_question_data = pd.read_csv(raw_train_data_file, sep='\t', index_col=False)
    train_pct = 0.8
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # TODO: increase vocab size to include named entities?
    # tmp debugging: small data
    # article_question_data = article_question_data.copy().head(2000)
    data_name = os.path.basename(raw_train_data_file).replace('.tsv', '')
    # tmp debugging: small data
    # data_name = f'mini_{data_name}'
    train_data_file = os.path.join(out_dir, f'{data_name}_train_data.pt')
    val_data_file = os.path.join(out_dir, f'{data_name}_val_data.pt')
    if(not os.path.exists(train_data_file)):
        prepare_question_data(article_question_data, out_dir, data_name,
                              tokenizer=tokenizer, train_pct=train_pct)

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