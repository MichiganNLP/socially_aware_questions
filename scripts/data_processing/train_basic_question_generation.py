"""
Train basic question generation on top of
pre-trained language models (e.g. BART).
"""
import numpy as np
import pandas as pd
import torch
import sys
if ('question_generation' not in sys.path):
    sys.path.append('question_generation')
from data_collator import T2TDataCollator
from transformers import AutoModelForSeq2SeqLM
import os
# tmp debugging
from trainer import Trainer
from data_helpers import DataArguments
from argparse import ArgumentParser
np.random.seed(123)
torch.manual_seed(123)

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
    training_args.local_rank = -1 # something with parallelization
    training_args.output_dir = model_out_dir
    training_args.num_train_epochs = 5 # 5 = longformer, 20 = BART
    # training_args.max_steps = 1
    training_args.fp16 = False
    training_args.label_names = None
    ## TODO: bigger batches with LongFormer!! training takes too long
    training_args.per_device_train_batch_size = 1
    training_args.per_device_eval_batch_size = 2
    # training_args.train_batch_size = 32
    # training_args.eval_batch_size = 32
    training_args.gradient_accumulation_steps = 4
    ## TODO: increase training size
    training_args.learning_rate = 1e-4
    training_args.dataloader_drop_last = False
    training_args.dataloader_num_workers = 8
    training_args.evaluate_during_training = True
    training_args.do_eval = True
    training_args.evaluation_strategy = 'steps'
    training_args.eval_steps = 100
    # default values from here lol https://github.com/huggingface/transformers/blob/49759c0cda29ab614b81e0869972c99f2edba7aa/src/transformers/training_args.py
    training_args.weight_decay = 0.01
    training_args.adam_beta1 = 0.9
    training_args.adam_beta2 = 0.999
    training_args.adam_epsilon = 1e-8
    training_args.warmup_steps = 500
    # limits number of checkpoints => 1 GB per optimizer file ;_;
    training_args.save_total_limit = 2
    return training_args

def main():
    parser = ArgumentParser()
    parser.add_argument('train_data')
    parser.add_argument('val_data')
    parser.add_argument('out_dir') # ../../data/CNN_articles/cnn/
    parser.add_argument('--model_type', default='bart')
    parser.add_argument('--model_cache_dir', default=None)
    # parser.add_argument('--author_data', default=None) # ../../data/nyt_comments/author_comment_social_data.tsv
    parser.add_argument('--sample_pct', type=float, default=1.0)
    parser.add_argument('--pretrained_model', default=None)
    args = vars(parser.parse_args())
    train_data_file = args['train_data']
    val_data_file = args['val_data']
    out_dir = args['out_dir']
    model_type = args['model_type']
    # author_data = args['author_data']
    # sample_pct = args['sample_pct']
    model_cache_dir = args['model_cache_dir']
    pretrained_model = args['pretrained_model']
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    ## prepare data
    # NOTE: data preprocessing moved to clean_data_for_generation.py because this is dumb
    # article_question_data = pd.read_csv(raw_train_data_file, sep='\t', index_col=False)
    # if(sample_pct < 1.0):
    #     N_sample = int(article_question_data.shape[0] * sample_pct)
    #     article_question_data_idx = np.random.choice(article_question_data.index, N_sample, replace=False)
    #     article_question_data = article_question_data.loc[article_question_data_idx, :]
    # train_pct = 0.8
    # data_name = os.path.basename(raw_train_data_file).replace('.tsv', '')
    # if(author_data is not None):
    #     data_name = f'author_type_{data_name}'
    #     author_data = pd.read_csv(author_data, sep='\t', index_col=False)
    #     # fix date
    #     date_day_fmt = '%Y-%m-%d'
    #     author_data = author_data.assign(**{
    #         'date_day' : author_data.loc[:, 'date_day'].apply(lambda x: datetime.strptime(x, date_day_fmt))
    #     })
    # tmp debugging: small data
    # data_name = f'mini_{data_name}'
    # train_data_file = os.path.join(out_dir, f'{data_name}_train_data.pt')
    # val_data_file = os.path.join(out_dir, f'{data_name}_val_data.pt')
    # print(f'train data = {train_data_file}')
    # if(not os.path.exists(train_data_file)):
    #     tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    #     prepare_question_data(article_question_data, out_dir, data_name,
    #                           tokenizer=tokenizer, train_pct=train_pct,
    #                           author_data=author_data)
    # tmp debugging
    # import sys
    # sys.exit()
    # reload tokenizer with all processed tokens
    model_type_tokenizer_lookup = {
        'bart' : 'BART',
        'longformer': 'LongFormer',
        'bart_copy' : 'BART',
    }
    data_dir = os.path.dirname(train_data_file)
    tokenizer_name = model_type_tokenizer_lookup[model_type]
    tokenizer_file = os.path.join(data_dir, f'{tokenizer_name}_tokenizer.pt')
    tokenizer = torch.load(tokenizer_file)

    ## train model
    if(model_cache_dir is None):
        model_cache_dir = os.path.join(out_dir, 'model_cache/')
    # tokenizer = torch.load('../../data/CNN_articles/cnn/BART_tokenizer.pt')
    model_type_path_lookup = {
        'bart' : 'facebook/bart-base',
        'longformer' : 'allenai/led-base-16384',
        'bart_copy' : 'facebook/bart-base',
    }
    if (model_type == 'bart_copy'):
        ## custom loading
        pass
    else:
        model_path = model_type_path_lookup[model_type]
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            cache_dir=model_cache_dir,
        )
    if(pretrained_model is not None):
        pretrained_model_weights = torch.load(pretrained_model)
        model.load_state_dict(pretrained_model_weights)
    model.resize_token_embeddings(len(tokenizer))
    # TODO: save model to cache again to update embedding size
    # device = torch.device(device_name)
    # model.to(device)

    ## load data
    # train_data_file = os.path.join(out_dir, f'{data_name}_train_data.pt')
    # val_data_file = os.path.join(out_dir, f'{data_name}_val_data.pt')
    train_dataset = torch.load(train_data_file)
    val_dataset = torch.load(val_data_file)
    train_dataset = train_dataset['train']
    val_dataset = val_dataset['train']
    # get max source/target len
    max_source_len = len(train_dataset['source_ids'][0])
    max_target_len = len(train_dataset['target_ids'][0])
    tokenizer.model_max_length = max_source_len
    # data collator
    data_collator = T2TDataCollator(
        tokenizer=tokenizer,
        model_type=model_type,
        mode="training",
        using_tpu=False
    )
    model_out_dir = os.path.join(out_dir, 'question_generation_model/')
    if (not os.path.exists(model_out_dir)):
        os.mkdir(model_out_dir)

    training_args = load_training_args(model_out_dir, train_data_file, model_out_dir, val_data_file, max_source_len, max_target_len)
    model_args = {
        'label_smoothing': 0,
    }
    # tmp debugging
    # print(f'training device {training_args.device}')
    ## TODO: prevent model from saving optimizer after every 500 training steps!!
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        #     prediction_loss_only=True,
        label_smoothing=model_args['label_smoothing'],
        # optimizer=(),
    )

    ## tmp debugging
    print(f'evaluation strategy = {trainer.args.evaluation_strategy}')

    ## train
    torch.cuda.empty_cache()
    trainer.train(
        model_path=model_out_dir,
    )
    trainer.save_model()

if __name__ == '__main__':
    main()