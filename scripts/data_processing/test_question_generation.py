"""
Test question generation with trained model.
Generate raw output for inspection AND compute
aggregate scores (BLEU, ROUGE).
"""
import gzip
import os
from argparse import ArgumentParser
from rouge_score.rouge_scorer import RougeScorer
from data_helpers import generate_predictions, compute_text_bleu
import torch
from transformers import AutoModelForSeq2SeqLM, BartTokenizer
import pandas as pd
import numpy as np

def main():
    parser = ArgumentParser()
    parser.add_argument('test_data')
    parser.add_argument('--model_file', default=None)
    parser.add_argument('--model_cache_dir', default='../../data/model_cache/')
    parser.add_argument('--model_type', default='bart')
    parser.add_argument('--out_dir', default='../../data/model_cache/')
    parser.add_argument('--post_metadata', default=None)
    args = vars(parser.parse_args())
    model_file = args['model_file']
    model_cache_dir = args['model_cache_dir']
    model_type = args['model_type']
    test_data = args['test_data']
    out_dir = args['out_dir']
    post_metadata = args.get('post_metadata')
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    ## load model, data
    model_name_lookup = {
        'bart' : 'facebook/bart-base'
    }
    model_full_name_lookup = {
        'bart' : 'BART',
    }
    full_model_name = model_name_lookup[model_type]
    data_dir = os.path.dirname(test_data)
    tokenizer_file = os.path.join(data_dir, f'{model_full_name_lookup[model_type]}_tokenizer.pt')
    if(os.path.exists(tokenizer_file)):
        model_tokenizer = torch.load(tokenizer_file)
    else:
        model_tokenizer = BartTokenizer.from_pretrained(full_model_name, cache_dir=model_cache_dir)
    generation_model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name,
                                                             cache_dir=model_cache_dir)
    generation_model.resize_token_embeddings(len(model_tokenizer))
    if (model_file is not None):
        model_weights = torch.load(model_file)
        generation_model.load_state_dict(model_weights)
    test_data = torch.load(test_data)['train']

    ## generate lol
    generated_text_out_file = os.path.join(out_dir, 'test_data_output_text.gz')
    if(not os.path.exists(generated_text_out_file)):
        generation_method = 'beam_search'
        num_beams = 8
        pred_data = generate_predictions(generation_model, test_data, model_tokenizer,
                                             generation_method=generation_method,
                                             num_beams=num_beams,)
        with gzip.open(generated_text_out_file, 'wt') as generated_text_out:
            generated_text_out.write('\n'.join(pred_data))
    else:
        pred_data = np.array(list(map(lambda x: x.strip(), gzip.open(generated_text_out_file, 'rt'))))

    ## get aggregate scores
    generated_text_score_out_file = os.path.join(out_dir,
                                                 'test_data_output_scores.tsv')
    if(not os.path.exists(generated_text_score_out_file)):
        generation_score_data = get_generation_scores(pred_data, test_data)
        ## write things to file
        generation_score_data.to_csv(generated_text_score_out_file, sep='\t', index=False)
    ## optional: same thing but for different subsets of post data
    if(post_metadata is not None):
        post_metadata = pd.read_csv(post_metadata, sep='\t',
                                    compression='gzip', index_col=False,
                                    usecols=['id', 'subreddit'])
        community_var = 'subreddit'
        # post_article_ids = np.array(test_data['article_id'])
        post_article_ids = test_data['article_id']
        community_scores = []
        for community_i, metadata_i in post_metadata.groupby(community_var):
            article_ids_i = set(metadata_i.loc[:, 'id'].unique())
            idx_i = [i for i,x in enumerate(post_article_ids) if x in article_ids_i]
            # this gives hash errors for some reason
            # idx_i = np.where(
            #     np.apply_along_axis(lambda x: x in article_ids_i, 0,
            #                         post_article_ids.reshape(1, -1)))[0]
            test_data_i = test_data.select(idx_i, keep_in_memory=True, load_from_cache_file=False)
            pred_data_i = pred_data[idx_i]
            generation_score_data_i = get_generation_scores(pred_data_i, test_data_i)
            generation_score_data_i = generation_score_data_i.assign(**{'community' : community_i})
            community_scores.append(generation_score_data_i)
        community_scores = pd.concat(community_scores, axis=0)
        community_score_out_file = os.path.join(out_dir, f'test_data_output_scores_communities.tsv')
        community_scores.to_csv(community_score_out_file, sep='\t', index=False)


def get_generation_scores(pred_data, test_data):
    generation_score_data = test_question_overlap(pred_data, test_data)
    # compute mean/sd
    generation_score_means = generation_score_data.mean(axis=0)
    generation_score_sd = generation_score_data.std(axis=0)
    generation_score_data = pd.concat([
        generation_score_means,
        generation_score_sd,
    ], axis=1).transpose()
    generation_score_data.index = ['mean', 'sd']
    generation_score_data = generation_score_data.reset_index().rename(
        columns={'index': 'stat'})
    return generation_score_data


def test_question_overlap(pred_data, test_data):
    ## TODO: need semantic overlap score! for input and for target text
    generation_scores = []
    bleu_weights = [1.0, 0., 0., 0.]  # 100% 1-grams, 0% 2-grams, etc.
    rouge_scorer = RougeScorer(['rougeL'], use_stemmer=True)
    for test_data_i, pred_data_i in zip(test_data['target_text'], pred_data):
        bleu_score_i = compute_text_bleu(test_data_i, pred_data_i,
                                         weights=bleu_weights)
        rouge_score_data_i = rouge_scorer.score(test_data_i, pred_data_i)
        rouge_score_i = rouge_score_data_i['rougeL'].fmeasure
        generation_scores.append([bleu_score_i, rouge_score_i])
    generation_score_data = pd.DataFrame(generation_scores,
                                         columns=['BLEU-1', 'ROUGE-L'])
    return generation_score_data


if __name__ == '__main__':
    main()