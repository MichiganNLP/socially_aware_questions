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
from dataclasses import field
from typing import Optional
from transformers.training_args import TrainingArguments
from transformers import AutoModelForSeq2SeqLM
import os
from trainer import Trainer
from data_helpers import DataProcessor

def prepare_question_data(data, out_dir, tokenizer, train_pct=0.8):
    # change to clean source/target format
    clean_data = data.loc[:, ['article_text', 'question']].rename(
        columns={'article_text': 'source_text', 'question': 'target_text'})
    # split train/val
    np.random.seed(123)
    N = clean_data.shape[0]
    N_train = int(N * train_pct)
    np.random.shuffle(clean_data.values)
    clean_data_train = clean_data.iloc[:N_train, :]
    clean_data_val = clean_data.iloc[N_train:, :]
    clean_data_train_out_file = os.path.join(out_dir, 'article_question_generation_train_data.csv')
    clean_data_val_out_file = os.path.join(out_dir, 'article_question_generation_val_data.csv')
    clean_data_train.to_csv(clean_data_train_out_file, sep=',', index=False)
    clean_data_val.to_csv(clean_data_val_out_file, sep=',', index=False)
    # reload data into correct format lol
    train_data_set = nlp.load_dataset('csv', data_files=clean_data_train_out_file)
    val_data_set = nlp.load_dataset('csv', data_files=clean_data_val_out_file)
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # get max lengths
    source_text_tokens = clean_data.loc[:, 'source_text'].apply(lambda x: tokenizer.tokenize(x))
    target_text_tokens = clean_data.loc[:, 'target_text'].apply(lambda x: tokenizer.tokenize(x))
    #     max_source_length = max(source_text_tokens.apply(lambda x: len(x)))
    #     max_target_length = max(target_text_tokens.apply(lambda x: len(x)))
    # tmp debugging: shorten source/target
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
    train_data_out_file = os.path.join(out_dir, 'article_question_generation_train_data.pt')
    val_data_out_file = os.path.join(out_dir, 'article_question_generation_val_data.pt')
    torch.save(train_data, train_data_out_file)
    torch.save(val_data, val_data_out_file)
    # save tokenizer?? sure
    tokenizer_out_file = os.path.join(out_dir, 'BART_tokenizer.pt')
    torch.save(tokenizer, tokenizer_out_file)

def main():
    ## prepare data
    cnn_article_question_data = pd.read_csv('../../data/CNN_articles/cnn/article_question_data.tsv', sep='\t',
                                            index_col=False)

    out_dir = '../../data/CNN_articles/cnn/'
    train_pct = 0.8
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # TODO: increase vocab size to include named entities?
    # TODO: shrink data to debug training
    # tmp debugging
    cnn_article_question_data = cnn_article_question_data.copy().head(2000)
    train_data_out_file = os.path.join(out_dir, 'article_question_generation_train_data.pt')
    if(not os.path.exists(train_data_out_file)):
        prepare_question_data(cnn_article_question_data, out_dir, tokenizer=tokenizer,
                              train_pct=train_pct)


    ## train model
    cache_dir = '../../data/CNN_articles/cnn/model_cache/'
    ## TODO: why doesn't the earlier torch import work??
    import torch
    tokenizer = torch.load('../../data/CNN_articles/cnn/BART_tokenizer.pt')
    # print(len(tokenizer))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        'facebook/bart-base',
        cache_dir=cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda:0')
    model.to(device)

    ## load data
    data_dir = '../../data/CNN_articles/cnn/'
    train_file = os.path.join(data_dir, 'article_question_generation_train_data.pt')
    val_file = os.path.join(data_dir, 'article_question_generation_val_data.pt')
    train_dataset = torch.load(train_file)
    val_dataset = torch.load(val_file)
    train_dataset = train_dataset['train']
    val_dataset = val_dataset['train']

    # get max source/target len
    max_source_len = len(train_dataset['source_ids'][0])
    max_target_len = len(train_dataset['target_ids'][0])

    # initialize data collator
    model_type = 'bart'
    data_collator = T2TDataCollator(
        tokenizer=tokenizer,
        model_type=model_type,
        mode="training",
        using_tpu=False
    )
    # #  Initialize Trainer
    # need data argument class ;_;
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

    # training_arg_dict = {
    #     'train_file_path' : 'article_question_generation_train_data.pt',
    #     'valid_file_path' : 'article_question_generation_val_data.pt',
    #     'data_dir' : data_dir,
    #     'task' : 'qg',
    #     'max_source_length' : max_source_len,
    #     'max_target_length' : max_target_len,
    #     'n_gpu' : 1,
    #     'seed' : 123,
    # }
    data_dir = '../../data/CNN_articles/cnn/'
    out_dir = '../../data/CNN_articles/cnn/question_generation_model/'
    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    training_args = DataArguments(out_dir)
    training_args.train_file_path = 'article_question_generation_train_data.pt',
    training_args.valid_file_path = 'article_question_generation_val_data.pt'
    training_args.data_dir = data_dir
    training_args.task = 'qg'
    training_args.max_source_length = max_source_len
    training_args.max_target_length = max_target_len
    training_args.n_gpu = 1
    # training_args.device = 'cuda:1'
    training_args.seed = 123
    training_args.disable_tqdm = False
    training_args.local_rank = -1
    training_args.output_dir = out_dir
    training_args.num_train_epochs = 20
    # training_args.max_steps = 1
    training_args.fp16 = False
    training_args.label_names = None
    training_args.per_device_train_batch_size = 4
    training_args.per_device_eval_batch_size = 4
    # training_args.train_batch_size = 32
    # training_args.eval_batch_size = 32
    training_args.gradient_accumulation_steps = 4
    training_args.learning_rate = 1e-4
    training_args.dataloader_drop_last = False
    training_args.dataloader_num_workers = 8
    # default values from here lol https://github.com/huggingface/transformers/blob/49759c0cda29ab614b81e0869972c99f2edba7aa/src/transformers/training_args.py
    training_args.weight_decay = 0.01
    training_args.adam_beta1 = 0.9
    training_args.adam_beta2 = 0.999
    training_args.adam_epsilon = 1e-8
    training_args.warmup_steps = 0
    # limits number of checkpoints => 1 GB per optimizer file ;_;
    training_args.save_total_limit = 2
    model_args = {
        'label_smoothing': 0,
    }
    ## TODO: prevent model from saving optimizer during every 500 training steps!!
    ## TODO: log train/val loss in same way
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        #     prediction_loss_only=True,
        label_smoothing=model_args['label_smoothing'],
    )

    ## train
    import torch
    torch.cuda.empty_cache()
    model_dir = '../../data/CNN_articles/cnn/question_generation_model/'
    trainer.train(
        model_path=model_dir,
    )
    trainer.save_model()

if __name__ == '__main__':
    main()