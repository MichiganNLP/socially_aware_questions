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

def main():
    parser = ArgumentParser()
    parser.add_argument('test_data')
    parser.add_argument('--model_file', default=None)
    parser.add_argument('--model_cache_dir', default='../../data/model_cache/')
    parser.add_argument('--model_type', default='bart')
    parser.add_argument('--out_dir', default='../../data/model_cache/')
    args = vars(parser.parse_args())
    model_file = args['model_file']
    model_cache_dir = args['model_cache_dir']
    model_type = args['model_type']
    test_data = args['test_data']
    out_dir = args['out_dir']

    ## load model, data
    model_name_lookup = {
        'bart' : 'facebook/bart-base'
    }
    full_model_name = model_name_lookup[model_type]
    generation_model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name, cache_dir=model_cache_dir)
    if(model_file is not None):
        model_weights = torch.load(model_file)
        generation_model.load_state_dict(model_weights)
    model_tokenizer = BartTokenizer.from_pretrained(full_model_name, cache_dir=model_cache_dir)
    test_data = torch.load(test_data)['train']

    ## generate lol
    generation_method = 'beam_search'
    num_beams = 8
    pred_data = generate_predictions(generation_model, test_data, model_tokenizer,
                                         generation_method=generation_method,
                                         num_beams=num_beams,
                                     )

    ## get aggregate scores
    ## TODO: need semantic overlap score! for input and for target text
    generation_scores = []
    bleu_weights = [1.0, 0., 0., 0.] # 100% 1-grams, 0% 2-grams, etc.
    rouge_scorer = RougeScorer(['rougeL'], use_stemmer=True)
    for test_data_i, pred_data_i in zip(test_data['target_text'], pred_data):
        bleu_score_i = compute_text_bleu(test_data_i, pred_data_i, weights=bleu_weights)
        rouge_score_data_i = rouge_scorer.score(test_data_i, pred_data_i)
        rouge_score_i = rouge_score_data_i['rougeL'].fmeasure
        generation_scores.append([bleu_score_i, rouge_score_i])
    generation_score_data = pd.DataFrame(generation_scores, columns=['BLEU-1', 'ROUGE-L'])
    # compute mean/sd
    generation_score_means = generation_score_data.mean(axis=0)
    generation_score_sd = generation_score_data.std(axis=0)
    generation_score_data = pd.concat([
        generation_score_means,
        generation_score_sd,
    ], axis=1).transpose()
    generation_score_data.index = ['mean', 'sd']
    generation_score_data = generation_score_data.reset_index().rename(columns={'index' : 'stat'})

    ## write things to file
    generated_text_out_file = os.path.join(out_dir, 'test_data_output_text.gz')
    generated_text_score_out_file = os.path.join(os.path.dirname(model_file), 'test_data_output_scores.tsv')
    with gzip.open(generated_text_out_file, 'wt') as generated_text_out:
        generated_text_out.write('\n'.join(pred_data))
    generation_score_data.to_csv(generated_text_score_out_file, sep='\t', index=False)

if __name__ == '__main__':
    main()