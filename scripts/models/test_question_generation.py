"""
Test question generation with trained model.
Generate raw output for inspection AND compute
aggregate scores (BLEU, ROUGE).
"""
import gzip
import os
import sys
from argparse import ArgumentParser
from rouge_score.rouge_scorer import RougeScorer
from sklearn.metrics.pairwise import cosine_distances
from stop_words import get_stop_words
from torch import Tensor
from tqdm import tqdm

from model_helpers import generate_predictions, compute_text_bleu, load_vectors
import torch
import sys
from author_group_attention_model import AuthorGroupAttentionModel, AuthorGroupAttentionModelConditionalGeneration

CPU_COUNT=10
torch.set_num_threads(CPU_COUNT)
from transformers import AutoModelForSeq2SeqLM, BartTokenizer, BartConfig
import pandas as pd
import numpy as np
from author_aware_model import AuthorTextGenerationModel
from nltk.tokenize import WordPunctTokenizer

STOP_WORDS = set(get_stop_words('en'))
# remove question words from stops
question_words = {'where', 'when', 'what', 'how', 'why', 'which', 'did', 'could', 'can', 'would', 'should'}
referent_words = {'he', 'she', 'they', 'his', 'her', 'their'}
STOP_WORDS = STOP_WORDS - question_words
STOP_WORDS = STOP_WORDS - referent_words
def get_generation_scores(pred_data, test_data, model, model_type='bart', word_embed_file=None, sample_size=1000, train_data=None):
    """
    Get generation scores for all predicted data, compute
    mean and SDS

    :param pred_data:
    :param test_data:
    :return:
    """
    generation_score_data = test_question_overlap(pred_data, test_data, word_embed_file=word_embed_file, stop_words=STOP_WORDS)
    # compute mean/sd
    generation_score_means = generation_score_data.mean(axis=0)
    generation_score_sd = generation_score_data.std(axis=0)
    generation_score_data = pd.concat([
        generation_score_means,
        generation_score_sd,
    ], axis=1).transpose()
    generation_score_data.index = ['mean', 'sd']
    # compute diversity = % unique data
    diversity_score = len(set(pred_data)) / len(pred_data)
    diversity_score = pd.DataFrame([diversity_score, 0.], index=['mean', 'sd'], columns=['diversity'])
    generation_score_data = pd.concat([generation_score_data, diversity_score], axis=1)
    # compute redundancy = copying from train data
    if(train_data is not None):
        train_data_text = set(train_data['target_text'])
        pred_data_overlap = list(filter(lambda x: x in train_data_text, pred_data))
        redundancy_score = len(pred_data_overlap) / len(pred_data)
        redundancy_score = pd.DataFrame([redundancy_score, 0.], index=['mean', 'sd'], columns=['redundancy'])
        generation_score_data = pd.concat([generation_score_data, redundancy_score], axis=1)
    # compute perplexity!
    log_likelihoods = []
    model_tensor_data_cols = ['source_ids', 'attention_mask', 'target_ids']
    model_float_data_cols = []
    model_extra_data_cols = []
    model_data_col_lookup = {'source_ids' : 'input_ids', 'target_ids' : 'labels'}
    if(model_type == 'bart_author_attention'):
        model_extra_data_cols.append('reader_token')
    elif(model_type == 'bart_author_embed'):
        model_float_data_cols.append('author_embed')
    # sample data to save time on perplexity
    sample_size = min(sample_size, len(test_data))
    sample_test_data = test_data.select(np.random.choice(list(range(len(test_data))), sample_size, replace=False))
    device = torch.cuda.current_device()
    for data_i in tqdm(sample_test_data):
        # remove padding tokens from output => don't care about PPL for pad tokens
        # print(f'data before filtering target IDs ({non_pad_target_ids_i})')
        # print(f'pad ID = {model.config.pad_token_id}')
        data_i['target_ids'] = list(filter(lambda x: x!=model.config.pad_token_id, data_i['target_ids']))

        # tmp debugging
        # print(f'data after filtering target IDs: ({data_i["target_ids"]})')
        # reshape tensors for model
        data_dict_i = {data_col : torch.LongTensor(data_i.get(data_col)).unsqueeze(0).to(device) for data_col in model_tensor_data_cols}
        for data_col in model_float_data_cols:
            data_dict_i[data_col] = torch.Tensor(data_i.get(data_col).unsqueeze(0).to(device))
        for data_col in model_extra_data_cols:
            data_dict_i[data_col] = [data_i.get(data_col)] 
        # data_dict_i = {data_col: torch.LongTensor(data_i.get(data_col)).unsqueeze(0).cpu() for data_col in model_data_cols}
        # rename column to match model input FML
        for k,v in model_data_col_lookup.items():
            data_dict_i[v] = data_dict_i[k]
            data_dict_i.pop(k)
        # tmp debugging
        # print(f'data dict {data_dict_i}')
        # output_i = model(**data_dict_i)
        with torch.no_grad():
            model.eval()
            data_dict_i.update({k: v.to(device) for k, v in data_dict_i.items() if type(v) is Tensor})
            # tmp debugging
            print(f'data dict before passing to model =\n{data_dict_i}')
            # output_i = model(input_ids=data_dict_i['input_ids'], attention_mask=data_dict_i['attention_mask'], labels=data_dict_i['labels'])
            output_i = model(**data_dict_i)
            data_dict_i.update({k: v.to('cpu') for k, v in data_dict_i.items() if type(v) is Tensor})
            ll = output_i[0].cpu()
            # print(f'log likelihood = {ll}')
            log_likelihoods.append(ll)
            # clear cache??
            del(output_i)
            # torch.cuda.empty_cache()
    log_likelihoods = torch.stack(log_likelihoods)
    perplexity = torch.exp(log_likelihoods).mean()
    perplexity_std = torch.exp(log_likelihoods).std()
    perplexity_data = pd.DataFrame([perplexity, perplexity_std], columns=['PPL'], index=['mean', 'sd'])
    generation_score_data = pd.concat([generation_score_data, perplexity_data], axis=1)
    # fix score format
    generation_score_data = generation_score_data.reset_index().rename(columns={'index': 'stat'})
    return generation_score_data

def test_question_overlap(pred_data, test_data, word_embed_file=None, stop_words=[]):
    text_overlap_scores = []
    bleu_weights = [1.0, 0., 0., 0.]  # 100% 1-grams, 0% 2-grams, etc.
    rouge_scorer = RougeScorer(['rougeL'], use_stemmer=True)
    score_cols = ['BLEU-1', 'ROUGE-L']
    # load embeddings for word mover distance
    if(word_embed_file is not None):
        word_embeds = load_vectors(word_embed_file)
        tokenizer = WordPunctTokenizer()
        score_cols.append('WMD')
        word_embed_vocab = set(word_embeds.index) - stop_words
    for test_data_i, pred_data_i in zip(test_data['target_text'], pred_data):
        bleu_score_i = compute_text_bleu(test_data_i, pred_data_i,
                                         weights=bleu_weights)
        # bleu_score_i = 0.
        rouge_score_data_i = rouge_scorer.score(test_data_i, pred_data_i)
        rouge_score_i = rouge_score_data_i['rougeL'].fmeasure
        generation_scores_i = [bleu_score_i, rouge_score_i]
        if(word_embed_file is not None):
            # tokenize/normalize data
            pred_tokens_i = list(map(lambda x: x.lower(), tokenizer.tokenize(pred_data_i)))
            test_tokens_i = list(map(lambda x: x.lower(), tokenizer.tokenize(test_data_i)))
            clean_pred_tokens_i = list(filter(lambda x: x in word_embed_vocab, pred_tokens_i))
            clean_test_tokens_i = list(filter(lambda x: x in word_embed_vocab, test_tokens_i))
            word_mover_dist_i = 1.
            if(len(clean_pred_tokens_i) > 0 and len(clean_test_tokens_i) > 0):
                word_mover_dist_i = compute_word_mover_dist(clean_pred_tokens_i, clean_test_tokens_i, word_embeds)
            # tmp debugging: which tokens do we drop from word embeddings?
            else:
                print(f'missing tokens: pred tokens={pred_tokens_i}; test tokens={test_tokens_i}')
            generation_scores_i.append(word_mover_dist_i)
        text_overlap_scores.append(generation_scores_i)
    # tmp debugging
    # print(f'generation score sample {generation_scores[:10]}')
    generation_score_data = pd.DataFrame(text_overlap_scores, columns=score_cols)
    return generation_score_data

def compute_word_mover_dist(tokens_1, tokens_2, word_embeds):
    embed_1 = word_embeds.loc[tokens_1, :]
    embed_2 = word_embeds.loc[tokens_2, :]
    # remove nan values
    # embed_1.dropna(axis=0, how='any', inplace=True)
    # embed_2.dropna(axis=0, how='any', inplace=True)
    mean_embed_1 = embed_1.mean(axis=0).values.reshape(1, -1)
    mean_embed_2 = embed_2.mean(axis=0).values.reshape(1, -1)
    word_mover_dist_i = 1.
    try:
        word_mover_dist_i = cosine_distances(mean_embed_1, mean_embed_2)[0][0]
    except Exception as e:
        print(f'WMD exception {e}')
        # print(f'bad tokens:\n1={tokens_1};\n2={tokens_2}')
    return word_mover_dist_i

def load_model(model_cache_dir, model_file, model_type, data_dir):
    if (model_type.startswith('bart_')):
        base_model_type = 'bart'
    else:
        base_model_type = model_type
    model_name_lookup = {
        'bart': 'facebook/bart-base',
    }
    model_full_name_lookup = {
        'bart': 'BART',
    }
    full_model_name = model_name_lookup[base_model_type]
    tokenizer_file = os.path.join(data_dir, f'{model_full_name_lookup[base_model_type]}_tokenizer.pt')
    if (os.path.exists(tokenizer_file)):
        model_tokenizer = torch.load(tokenizer_file)
    else:
        model_tokenizer = BartTokenizer.from_pretrained(full_model_name, cache_dir=model_cache_dir)
    # get config file from same directory as model
    config_file = os.path.join(os.path.dirname(model_file), 'config.json')
    config = BartConfig.from_json_file(config_file)
    if (model_type == 'bart_author_embeds'):
        ## custom loading
        # config_file = os.path.join(model_cache_dir, 'BART_author_model_config.json')
        # config = BartConfig.from_json_file(config_file)
        # config.author_embeds = 100
        generation_model = AuthorTextGenerationModel(config)
    elif(model_type == 'bart_author_attention'):
        ## tmp debugging
        # config_file = os.path.join(model_cache_dir, 'BART_author_model_config.json')
        # config = BartConfig.from_json_file(config_file)
        reader_group_types = config.__dict__['reader_group_types']
        generation_model = AuthorGroupAttentionModelConditionalGeneration(config, reader_group_types=reader_group_types)
    else:
        generation_model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name,
                                                                 cache_dir=model_cache_dir)
    generation_model.resize_token_embeddings(len(model_tokenizer))
    if (model_file is not None):
        if(torch.cuda.is_available()):
            model_weights = torch.load(model_file)
        else:
            model_weights = torch.load(model_file, map_location=torch.device('cpu'))
        # optional: reset vocab size
        if(model_weights['lm_head.weight'].shape[0] != generation_model.config.vocab_size):
            generation_model.resize_token_embeddings(model_weights['lm_head.weight'].shape[0])
        generation_model.load_state_dict(model_weights)
    return generation_model, model_tokenizer

def main():
    parser = ArgumentParser()
    parser.add_argument('test_data')
    parser.add_argument('--train_data', default=None)
    parser.add_argument('--model_file', default=None)
    parser.add_argument('--model_cache_dir', default='../../data/model_cache/')
    parser.add_argument('--model_type', default='bart')
    parser.add_argument('--out_dir', default='../../data/model_cache/')
    parser.add_argument('--post_metadata', default=None)
    parser.add_argument('--word_embed_file', default='../../data/embeddings/wiki-news-300d-1M.vec.gz')
    args = vars(parser.parse_args())
    model_file = args['model_file']
    model_cache_dir = args['model_cache_dir']
    model_type = args['model_type']
    test_data = args['test_data']
    train_data = args.get('train_data')
    out_dir = args['out_dir']
    post_metadata = args.get('post_metadata')
    word_embed_file = args.get('word_embed_file')
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    ## load model, data
    data_dir = os.path.dirname(test_data)
    generation_model, model_tokenizer = load_model(model_cache_dir, model_file, model_type, data_dir)
    # tmp debugging
    generation_model.to(torch.cuda.current_device())
    test_data = torch.load(test_data)#['train']
    if('train' in test_data):
        test_data = test_data['train']
    # fix source IDs for author-token model
    if(model_type == 'bart_author_token'):
        test_data.remove_column_('source_ids')
        test_data.rename_column_('source_ids_reader_token', 'source_ids')
    data_cols = ['source_ids', 'target_ids', 'attention_mask']
    if(model_type == 'bart_author_embeds'):
        # choose appropriate column
        test_data.rename_column_(generation_model.config.__dict__['author_embed_type'], 'author_embeds')
        data_cols.append('author_embeds')
    test_data.set_format('torch', columns=data_cols, output_all_columns=True)
    if(train_data is not None):
        train_data = torch.load(train_data)
    ## get extra args
    model_kwargs = []
    if(model_type == 'bart_author_embeds'):
        model_kwargs.append('author_embeds')
        # tmp debugging
        # print(f'data has cols {test_data.column_names}')
    elif(model_type == 'bart_author_attention'):
        test_data.remove_column_('reader_token')
        test_data.rename_column_('reader_token_str', 'reader_token')
        model_kwargs.append('reader_token')
        # tmp debugging
        # for data_i in test_data:
        #     if(data_i['reader_token'] is None):
        #         print(f'bad test data with no reader token {data_i}')
        #         sys.exit(0)
    
    ## generate lol
    generated_text_out_file = os.path.join(out_dir, 'test_data_output_text.gz')
    if(not os.path.exists(generated_text_out_file)):
        generation_method = 'beam_search'
        num_beams = 8
        pred_data = generate_predictions(generation_model, test_data, model_tokenizer,
                                         generation_method=generation_method,
                                         num_beams=num_beams, model_kwargs=model_kwargs)
        pred_data = np.array(pred_data)
        with gzip.open(generated_text_out_file, 'wt') as generated_text_out:
            generated_text_out.write('\n'.join(pred_data))
    else:
        pred_data = np.array(list(map(lambda x: x.strip(), gzip.open(generated_text_out_file, 'rt'))))

    ## get aggregate scores
    generated_text_score_out_file = os.path.join(out_dir,
                                                 'test_data_output_scores.tsv')
    # print(f'generated score file {generated_text_score_out_file}')
    if(not os.path.exists(generated_text_score_out_file)):
        # tmp debugging
        # test_data = test_data.select(list(range(100)))
        # pred_data = pred_data[:100]
        generation_score_data = get_generation_scores(pred_data, test_data, generation_model, model_type=model_type, word_embed_file=word_embed_file, train_data=train_data)
        ## write things to file
        generation_score_data.to_csv(generated_text_score_out_file, sep='\t', index=False)
    ## optional: same thing but for different subsets of post data
    ## reader groups
    reader_group_score_out_file = os.path.join(out_dir, 'test_data_scores_reader_groups.tsv')
    if(not os.path.exists(reader_group_score_out_file)):
        if(model_type == 'bart_author_attention'):
            reader_groups = list(set(test_data['reader_token']))
        else:
            reader_groups = list(set(test_data['reader_token_str']))
        reader_group_scores = []
        for reader_group_i in reader_groups:
            idx_i = np.where(np.array(test_data['reader_token_str'])==reader_group_i)[0]
            # tmp debugging
            # print(f'reader group {reader_group_i} has idx={idx_i}')
            test_data_i = test_data.select(idx_i, keep_in_memory=True, load_from_cache_file=False)
            pred_data_i = pred_data[idx_i]
            generation_score_data_i = get_generation_scores(pred_data_i, test_data_i, generation_model, model_type=model_type, word_embed_file=word_embed_file, train_data=train_data)
            generation_score_data_i.assign(**{'reader_group' : reader_group_i})
            reader_group_scores.append(generation_score_data_i)
        reader_group_scores = pd.concat(reader_group_scores, axis=0)
        reader_group_scores.to_csv(reader_group_score_out_file, sep='\t', index=False)
    ## per-community
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
            generation_score_data_i = get_generation_scores(pred_data_i, test_data_i, generation_model, word_embed_file=word_embed_file, train_data=train_data)
            generation_score_data_i = generation_score_data_i.assign(**{'community' : community_i})
            community_scores.append(generation_score_data_i)
        community_scores = pd.concat(community_scores, axis=0)
        community_score_out_file = os.path.join(out_dir, f'test_data_output_scores_communities.tsv')
        community_scores.to_csv(community_score_out_file, sep='\t', index=False)


if __name__ == '__main__':
    main()
