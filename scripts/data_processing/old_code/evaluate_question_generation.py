"""
Evaluate question generation based on accuracy in matching test data and inherent quality.

(1) question overlap: BLEU-N scores, ROUGE (?)
(2) repetition: uniqueness of generated questions
(3) copying: exact overlap with train data
"""
from argparse import ArgumentParser
import torch
from data_helpers import compute_text_bleu, generate_predictions, convert_ids_to_clean_str # compute_max_sent_score
from rouge import Rouge
from tqdm import tqdm
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import numpy as np

def compute_all_match_scores(pred_text, target_text, tokenizer, rouge_evaluator):
    # target_text = tokenizer.convert_ids_to_tokens(target_ids,
    #                                               skip_special_tokens=True)
    pred_tokens = tokenizer.tokenize(pred_text)
    bleu_1_score = compute_text_bleu(pred_tokens, target_text,
                                     weights=[1.0, 0., 0., 0., ])
    bleu_2_score = compute_text_bleu(pred_tokens, target_text,
                                     weights=[0.0, 1.0, 0., 0., ])
    rouge_score = rouge_evaluator.get_scores([pred_text], [target_text])
    rouge_l_score = rouge_score['rouge-l']['f']
    return bleu_1_score, bleu_2_score, rouge_l_score

def count_exact_matches(pred_text, compare_text):
    match_counts = []
    for text_i in tqdm(pred_text):
        matches_i = list(filter(lambda x: x==text_i, compare_text))
        match_counts.append(len(matches_i))
    return match_counts

def evaluate_model(model, train_data, test_data, tokenizer, out_dir, data_name=None):
    generation_method = 'beam_search'  # TODO: experiment with sample
    num_beams = 8
    # temperature = 1.0
    pred_text = generate_predictions(model, test_data, tokenizer,
                                     generation_method=generation_method,
                                     num_beams=num_beams)
    ### matching test data: BLEU and ROUGE
    # use ROUGE for longest common subsequence
    rouge_evaluator = Rouge(metrics=['rouge-l'], max_n=4,
                            alpha=0.5,  # default F1 score
                            stemming=True)
    combined_overlap_scores = []
    for pred_text_i, target_text_i in zip(pred_text, test_data['target_text']):
        scores = compute_all_match_scores(pred_text_i, target_text_i, tokenizer,
                                          rouge_evaluator)
        combined_overlap_scores.append(scores)
    combined_overlap_scores = pd.DataFrame(combined_overlap_scores, columns=['bleu-1', 'bleu-2', 'rouge-l'])
    mean_overlap_scores = combined_overlap_scores.mean(axis=0)
    ### repetition
    pred_text_clean = list(map(lambda x: x.lower().strip(), pred_text))
    unique_question_pct = len(set(pred_text_clean)) / len(pred_text_clean)
    ### copying
    target_text_clean = list(
        map(lambda x: convert_ids_to_clean_str(x, tokenizer).lower(),
            train_data['target_ids']))
    print(f'computing match counts:\nexample pred text {pred_text_clean[:10]}\nexample target text {target_text_clean}')
    exact_match_counts = count_exact_matches(pred_text_clean, target_text_clean)
    exact_match_pct = sum(
        list(filter(lambda x: x > 0, exact_match_counts))) / len(train_data)
    ### combine all scores
    eval_scores = mean_overlap_scores.append(
        pd.Series([unique_question_pct, exact_match_pct],
                  index=['pct_unique', 'pct_match_train_data']))
    ### write to file
    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    out_file_name = 'question_scores.tsv'
    if(data_name is not None):
        out_file_name = f'{data_name}_question_scores.tsv'
    out_file = os.path.join(out_dir, out_file_name)
    eval_scores.to_csv(out_file, sep='\t', index=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('out_dir')
    parser.add_argument('train_data')
    parser.add_argument('test_data')
    parser.add_argument('--model_type', default='bart')
    parser.add_argument('--model_cache_dir', default=None)
    parser.add_argument('--post_metadata', default=None)
    parser.add_argument('--data_name', default=None)
    args = vars(parser.parse_args())
    model_file = args['model_file']
    out_dir = args['out_dir']
    train_data_file = args['train_data']
    test_data_file = args['test_data']
    model_type = args['model_type']
    model_cache_dir = args['model_cache_dir']
    data_name = args.get('data_name')

    ## load data
    # model
    model_type_path_lookup = {
        'bart': 'facebook/bart-base',
        'longformer' : 'allenai/led-base-16384'
    }
    model_path = model_type_path_lookup[model_type]
    model_file_dir = os.path.dirname(os.path.dirname(model_file))
    if(model_cache_dir is None):
        model_cache_dir = os.path.join(model_file_dir, 'model_cache')
    tokenizer_name_lookup = {
        'bart': 'BART',
        'longformer' : 'LongFormer'
    }
    data_dir = os.path.dirname(train_data_file)
    tokenizer_path = os.path.join(data_dir, f'{tokenizer_name_lookup[model_type]}_tokenizer.pt')
    # tokenizer_dir = os.path.abspath(os.path.join(model_file, '../../../'))
    # tokenizer_path = os.path.join(tokenizer_dir, f'{tokenizer_name}_tokenizer.pt')
    # print(tokenizer_path)
    if(os.path.exists(tokenizer_path)):
        tokenizer = torch.load(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        cache_dir=model_cache_dir,
    )
    pretrained_model_weights = torch.load(model_file)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(pretrained_model_weights)
    # train/test data
    train_data = torch.load(train_data_file)['train']
    test_data = torch.load(test_data_file)['train']
    ## set device
    # device = torch.device(device_name)
    # model.to(device)

    ## evaluation
    # generate predictions
    out_file_name = 'question_scores.tsv'
    if (data_name is not None):
        out_file_name = f'{data_name}_question_scores.tsv'
    if(not os.path.exists(out_file_name)):
        evaluate_model(model, train_data, test_data, tokenizer, out_dir, data_name=data_name)

    ## optional: same thing but for different subsets of post data
    if(args.get('post_metadata') is not None):
        post_metadata = pd.read_csv(args['post_metadata'], sep='\t', compression='gzip', index_col=False, usecols=['id', 'subreddit'])
        community_var = 'subreddit'
        post_article_ids = np.array(post_metadata['article_id'])
        for community_i, metadata_i in post_metadata.groupby(community_var):
            article_ids_i = set(metadata_i.loc[:, 'article_id'].unique())
            idx_i = np.where(np.apply_along_axis(lambda x: x in article_ids_i, 0, post_article_ids.reshape(1, -1)))
            # idx_i = np.where(post_metadata['article_id'].isin(metadata_i.loc[:, 'id'].unique()))
            test_data_i = test_data.select(idx_i)
            data_name_i = f'{community_i}_{data_name}'
            evaluate_model(model, train_data, test_data_i, tokenizer, out_dir, data_name=data_name_i)

if __name__ == '__main__':
    main()