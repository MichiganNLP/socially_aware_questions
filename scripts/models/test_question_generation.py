"""
Test question generation with trained model.
Generate raw output for inspection AND compute
aggregate scores (BLEU, ROUGE).
"""
import gzip
import json
import os
import pickle
from argparse import ArgumentParser
from itertools import product

from stop_words import get_stop_words
from torch import Tensor
from tqdm import tqdm

from model_helpers import generate_predictions, load_model, compute_word_mover_dist, test_question_overlap
import sys
if('answerability_metric' not in sys.path):
    sys.path.append('answerability_metric')
from answerability_metric.answerability_score import get_answerability_scores
import torch
from sentence_transformers import SentenceTransformer
from nlp import Dataset

CPU_COUNT=10
torch.set_num_threads(CPU_COUNT)
import pandas as pd
import numpy as np
np.random.seed(123)
torch.manual_seed(123)
## suppress BLEU warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

STOP_WORDS = set(get_stop_words('en'))
# remove question words from stops
question_words = {'where', 'when', 'what', 'how', 'why', 'which', 'did', 'could', 'can', 'would', 'should'}
referent_words = {'he', 'she', 'they', 'his', 'her', 'their'}
STOP_WORDS = STOP_WORDS - question_words
STOP_WORDS = STOP_WORDS - referent_words
def get_generation_scores(pred_data, test_data, model, model_type='bart', word_embed_file=None, sample_size=5000, train_data=None):
    """
    Get generation scores for all predicted data, compute
    mean and SD

    :param pred_data:
    :param test_data:
    :return:
    """
    full_generation_score_data = test_question_overlap(pred_data, test_data, word_embed_file=word_embed_file, stop_words=STOP_WORDS)
    # compute mean/sd
    generation_score_means = full_generation_score_data.mean(axis=0)
    generation_score_sd = full_generation_score_data.std(axis=0)
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
    perplexity_data = compute_perplexity(model, model_type, sample_size, test_data)
    generation_score_data = pd.concat([generation_score_data, perplexity_data], axis=1)
    # fix score format
    generation_score_data = generation_score_data.reset_index().rename(columns={'index': 'stat'})
    return full_generation_score_data, generation_score_data

def get_subreddit_group_generation_scores(pred_data, test_data,
                                          model, model_type='bart',
                                          word_embed_file=None, sample_size=5000,
                                          train_data=None):
    subreddit_group_combos = list(product(test_data.loc[:, 'subreddit'].unique(), test_data.loc[:, 'group_category'].unique()))
    per_subreddit_group_scores = []
    for (subreddit_i, group_i) in subreddit_group_combos:
        idx_i = np.where((test_data.loc[:, 'subreddit']==subreddit_i) & (test_data.loc[:, 'group_category']==group_i))[0]
        pred_data_i = pred_data[idx_i]
        test_data_i = test_data.select(idx_i, keep_in_memory=True, load_from_cache_file=False)
        _, score_data_i = get_generation_scores(pred_data_i, test_data_i, model,
                                                model_type=model_type, word_embed_file=word_embed_file,
                                                sample_size=sample_size, train_data=train_data)
        score_data_i = score_data_i.assign(**{
            'subreddit' : subreddit_i,
            'group_category' : group_i,
        })
        per_subreddit_group_scores.append(score_data_i)
    per_subreddit_group_scores = pd.concat(per_subreddit_group_scores, axis=0)
    return per_subreddit_group_scores

def compute_perplexity(model, model_type, sample_size, test_data, return_log_likelihoods=False):
    log_likelihoods = []
    model_tensor_data_cols = ['source_ids', 'attention_mask', 'target_ids']
    model_float_data_cols = []
    model_extra_data_cols = []
    model_data_col_lookup = {'source_ids': 'input_ids', 'target_ids': 'labels'}
    if (model_type == 'bart_author_attention'):
        model_extra_data_cols.append('reader_token')
    elif (model_type == 'bart_author_embed'):
        model_float_data_cols.append('author_embed')
    # tmp debugging
    # print(f'model type = {model_type}')
    # print(f'model extra data cols {model_extra_data_cols}')
    # optional: restrict to valid data
    # if(model_type == 'bart_author_attention'):
    #     test_data = test_data.filter(lambda x: 'reader_token' in x.keys())
    # sample data to save time on perplexity
    # sample_size = min(sample_size, len(test_data))
    if(sample_size < len(test_data)):
        sample_test_data = test_data.select(
            np.random.choice(list(range(len(test_data))), sample_size,
                             replace=False))
    else:
        sample_test_data = test_data
    device = torch.cuda.current_device()
    for data_i in tqdm(sample_test_data):
        data_dict_i = prepare_data_for_model_forward_pass(data_i, device, model.config,
                                                          model_data_col_lookup,
                                                          model_extra_data_cols,
                                                          model_float_data_cols,
                                                          model_tensor_data_cols)
        # tmp debugging
        # print(f'data dict {data_dict_i}')
        # output_i = model(**data_dict_i)
        # try:
        with torch.no_grad():
            model.eval()
            data_dict_i.update(
                {k: v.to(device) for k, v in data_dict_i.items() if
                 type(v) is Tensor})
            # tmp debugging
            # print(f'data dict before passing to model =\n{data_dict_i}')
            # output_i = model(input_ids=data_dict_i['input_ids'], attention_mask=data_dict_i['attention_mask'], labels=data_dict_i['labels'])
            output_i = model(**data_dict_i)
            data_dict_i.update(
                {k: v.to('cpu') for k, v in data_dict_i.items() if
                 type(v) is Tensor})
            ll = output_i[0].cpu()
            # print(f'log likelihood = {ll}')
            log_likelihoods.append(ll)
            # clear cache??
            del (output_i)
            # torch.cuda.empty_cache()
        # except Exception as e:
        #     print(f'could not process batch {data_dict_i} because of error {e}')
    log_likelihoods = torch.stack(log_likelihoods)
    perplexity = torch.exp(log_likelihoods).mean()
    perplexity_std = torch.exp(log_likelihoods).std()
    perplexity_data = pd.DataFrame([perplexity, perplexity_std],
                                   columns=['PPL'], index=['mean', 'sd'])
    if(return_log_likelihoods):
        return log_likelihoods, perplexity_data
    else:
        return perplexity_data


def prepare_data_for_model_forward_pass(data, device, model_config, model_data_col_lookup,
                                        model_extra_data_cols,
                                        model_float_data_cols,
                                        model_tensor_data_cols):
    # remove padding tokens from output => don't care about PPL for pad tokens
    # print(f'data before filtering target IDs ({non_pad_target_ids_i})')
    # print(f'pad ID = {model.config.pad_token_id}')
    data['target_ids'] = list(
        filter(lambda x: x != model_config.pad_token_id,
               data['target_ids']))
    # tmp debugging
    # print(f'data after filtering target IDs: ({data_i["target_ids"]})')
    # tmp debugging
    # print(f'raw data before converting to dict {data_i}')
    # reshape tensors for model
    data_dict = {
        data_col: torch.LongTensor(data.get(data_col)).unsqueeze(0).to(
            device) for data_col in model_tensor_data_cols}
    for data_col in model_float_data_cols:
        data_dict[data_col] = torch.Tensor(
            data.get(data_col).unsqueeze(0).to(device))
    for data_col in model_extra_data_cols:
        data_dict[data_col] = [data.get(data_col)]
        # data_dict_i = {data_col: torch.LongTensor(data_i.get(data_col)).unsqueeze(0).cpu() for data_col in model_data_cols}
    # rename column to match model input FML
    for k, v in model_data_col_lookup.items():
        data_dict[v] = data_dict[k]
        data_dict.pop(k)
    return data_dict

def prepare_test_data_for_generation(model_config, model_type, test_data):
    ## fix metadata for reader-aware models
    # fix reader token for all models! because we need to
    # compare performance between reader groups
    if('reader_token' in test_data.column_names):
        test_data.remove_column_('reader_token')
    test_data.rename_column_('reader_token_str', 'reader_token')
    if (model_type == 'bart_author_token'):
        test_data.remove_column_('source_ids')
        test_data.rename_column_('source_ids_reader_token', 'source_ids')
        # test_data.remove_column_('reader_token')
    # if(model_type in {'bart_author_token', 'bart_author_attention'}):
    #     if('reader_token' in test_data.column_names):
    #         test_data.remove_column_('reader_token')
    #     # fix reader token data
    #     test_data.rename_column_('reader_token_str', 'reader_token')
    data_cols = ['source_ids', 'target_ids', 'attention_mask']
    data_cols = list(filter(lambda x: x in test_data.column_names, data_cols))
    if (model_type == 'bart_author_embeds'):
        # choose appropriate column for embeds
        test_data.rename_column_(
            model_config.__dict__['author_embed_type'],
            'author_embeds')
        data_cols.append('author_embeds')
    ## get extra args
    model_kwargs = []
    if (model_type == 'bart_author_embeds'):
        model_kwargs.append('author_embeds')
        # tmp debugging
        # print(f'data has cols {test_data.column_names}')
    elif (model_type == 'bart_author_attention'):
        model_kwargs.append('reader_token')
    test_data.set_format('torch', columns=data_cols, output_all_columns=True)
    return model_kwargs

def main():
    parser = ArgumentParser()
    parser.add_argument('test_data')
    parser.add_argument('--train_data', default=None)
    parser.add_argument('--model_file', default=None)
    parser.add_argument('--model_cache_dir', default='../../data/model_cache/')
    parser.add_argument('--model_type', default='bart')
    parser.add_argument('--out_dir', default='../../data/reddit_data/')
    parser.add_argument('--post_metadata', default=None)
    parser.add_argument('--word_embed_file', default='../../data/embeddings/wiki-news-300d-1M.vec.gz')
    parser.add_argument('--generation_params', default='../../data/model_cache/beam_search_generation_params.json')
    #parser.add_argument('--generate_classify', dest='generate_classify', action='store_true')
    parser.add_argument('--post_subgroup_file', default=None)
    args = vars(parser.parse_args())
    model_file = args['model_file']
    model_cache_dir = args['model_cache_dir']
    model_type = args['model_type']
    test_data = args['test_data']
    train_data = args.get('train_data')
    out_dir = args['out_dir']
    post_metadata = args.get('post_metadata')
    word_embed_file = args.get('word_embed_file')
    generation_param_file = args.get('generation_params')
    #generate_classify = args.get('generate_classify')
    post_subgroup_file = args.get('post_subgroup_file')
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    ## load model, data
    data_dir = os.path.dirname(train_data)
    generation_model, model_tokenizer = load_model(model_cache_dir, model_file, model_type, data_dir)
    generation_model.to(torch.cuda.current_device())
    test_data = torch.load(test_data)
    if('train' in test_data):
        test_data = test_data['train']
    model_kwargs = prepare_test_data_for_generation(generation_model.config, model_type, test_data)
    if(train_data is not None):
        train_data = torch.load(train_data)
    ## add group category to test data
    test_data = add_reader_group_category(test_data)
    # tmp debugging: shuffle test data
    # test_data = test_data.shuffle(seed=123, keep_in_memory=True, cache_file_name=None)
    # tmp debugging: less test data
    # test_data = test_data.select(list(range(500)), keep_in_memory=True, cache_file_name=None)
    # get data name: based on model generation parameters
    generation_params = json.load(open(generation_param_file))
    # generation_method = generation_params['generation_method']
    generate_classify = generation_params['generate_classify']
    # note: get rid of generation string on output name because it makes file names annoying
    # generation_str = f'{generation_method}_{"_".join(k+"="+str(v) for k,v in generation_params.items() if k!= "generation_method")}'
    # output_name = f'test_data_{generation_str}_output'
    output_name = 'test_data_output'
    generated_text_out_file = os.path.join(out_dir, f'{output_name}_text.gz')
    # get classifiers etc. for generate + classify
    if(generate_classify and not os.path.exists(generated_text_out_file)):
        model_classifier_dir = '../../data/reddit_data/group_classification_model/'
        reader_groups = ['location_region', 'expert_pct_bin', 'relative_time_bin']
        reader_group_class_defaults = {
            'location_region' : 'NONUS', 'expert_pct_bin' : 0.0, 'relative_time_bin' : 0.0,
        }
        model_classifiers = {
            #reader_group : pickle.load(open(os.path.join(model_classifier_dir, f'question_post_data/MLP_prediction_group=author_group_class1={reader_group}={reader_group_class_defaults[reader_group]}.pkl'), 'rb'))
            reader_group: pickle.load(open(os.path.join(model_classifier_dir, f'question_post_data/MLP_prediction_group={reader_group}_class1={reader_group_class_defaults[reader_group]}.pkl'), 'rb'))
            for reader_group in reader_groups
        }
        sentence_encoder = SentenceTransformer('paraphrase-distilroberta-base-v1')
        pca_question_model = pickle.load(open(os.path.join(model_classifier_dir, 'PCA_model_embed=question_encoded.pkl'), 'rb'))
        pca_post_model = pickle.load(open(os.path.join(model_classifier_dir, 'PCA_model_embed=post_encoded.pkl'), 'rb'))
        generate_classify_tools = (model_classifiers, sentence_encoder, pca_question_model, pca_post_model)
    else:
        generate_classify_tools = None

    ## generate lol
    # tmp debugging
    #if(model_type == 'bart_author_embeds'):
    #    print(f'model has input for embeds = {generation_model.model.author_embed_module}')
    if(not os.path.exists(generated_text_out_file)):
        pred_data = generate_predictions(generation_model, test_data, model_tokenizer,
                                         generation_params=generation_params,
                                         model_kwargs=model_kwargs,
                                         generate_classify_tools=generate_classify_tools)
        pred_data = np.array(pred_data)
        with gzip.open(generated_text_out_file, 'wt') as generated_text_out:
            generated_text_out.write('\n'.join(pred_data))
    else:
        pred_data = np.array(list(map(lambda x: x.strip(), gzip.open(generated_text_out_file, 'rt'))))
    # tmp debugging: align test/pred data
    if(len(test_data) < len(pred_data)):
        pred_data = pred_data[:len(test_data)]

    ## get aggregate scores
    generated_text_score_out_file = os.path.join(out_dir, f'{output_name}_scores.tsv')
    full_generated_text_score_out_file = os.path.join(out_dir, f'{output_name}_scores_full.gz')
    # print(f'generated score file {generated_text_score_out_file}')
    if(not os.path.exists(generated_text_score_out_file)):
        # tmp debugging
        # test_data = test_data.select(list(range(100)))
        # pred_data = pred_data[:100]
        full_generation_score_data, generation_score_data = get_generation_scores(pred_data, test_data, generation_model, model_type=model_type, word_embed_file=word_embed_file, train_data=train_data)
        ## write things to file
        full_generation_score_data.to_csv(full_generated_text_score_out_file, sep='\t', index=False, compression='gzip')
        generation_score_data.to_csv(generated_text_score_out_file, sep='\t', index=False)
    ## optional: same thing but for different subsets of post data
    ## reader groups
    reader_group_score_out_file = os.path.join(out_dir, f'{output_name}_scores_reader_groups.tsv')
    # full_generated_text_score_out_file = os.path.join(out_dir, f'{output_name}_scores_reader_groups_full.tsv')
    # print(f'reader group file = {reader_group_score_out_file}')
    if(not os.path.exists(reader_group_score_out_file)):
        # if(model_type == 'bart_author_attention'):
        #     reader_groups = list(set(test_data['reader_token']))
        # else:
        # tmp debugging
        reader_groups = list(set(test_data['reader_token']))
        reader_group_scores = []
        for reader_group_i in reader_groups:
            if(reader_group_i == 'UNK'):
                idx_i = np.where(np.array(test_data['reader_token'])!=reader_group_i)[0]
                reader_group_i = 'non_UNK'
            else:
                idx_i = np.where(np.array(test_data['reader_token'])==reader_group_i)[0]
            test_data_i = test_data.select(idx_i, keep_in_memory=True, load_from_cache_file=False)
            pred_data_i = pred_data[idx_i]
            full_generation_score_data_i, generation_score_data_i = get_generation_scores(pred_data_i, test_data_i, generation_model, model_type=model_type, word_embed_file=word_embed_file, train_data=train_data)
            generation_score_data_i = generation_score_data_i.assign(**{'reader_group' : reader_group_i})
            reader_group_scores.append(generation_score_data_i)
        # also: get scores for readers with embeddings
        embed_groups = ['author_has_subreddit_embed', 'author_has_text_embed']
        for embed_group_i in embed_groups:
            idx_i = np.where(np.array(test_data[embed_group_i]))[0]
            test_data_i = test_data.select(idx_i, keep_in_memory=True, load_from_cache_file=False)
            pred_data_i = pred_data[idx_i]
            full_generation_score_data_i, generation_score_data_i = get_generation_scores(pred_data_i, test_data_i, generation_model, model_type=model_type, word_embed_file=word_embed_file, train_data=train_data)
            generation_score_data_i = generation_score_data_i.assign(**{'reader_group': embed_group_i})
            reader_group_scores.append(generation_score_data_i)
        reader_group_scores = pd.concat(reader_group_scores, axis=0)
        reader_group_scores.to_csv(reader_group_score_out_file, sep='\t', index=False)
    ## per-community
    community_score_out_file = os.path.join(out_dir, f'{output_name}_scores_communities.tsv')
    if(post_metadata is not None and not os.path.exists(community_score_out_file)):
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
            full_generation_score_data_i, generation_score_data_i = get_generation_scores(pred_data_i, test_data_i, generation_model, model_type=model_type, word_embed_file=word_embed_file, train_data=train_data)
            generation_score_data_i = generation_score_data_i.assign(**{'community' : community_i})
            community_scores.append(generation_score_data_i)
        community_scores = pd.concat(community_scores, axis=0)
        community_scores.to_csv(community_score_out_file, sep='\t', index=False)
    ## post sub-group
    post_subgroup_name = os.path.basename(post_subgroup_file).replace('_data.gz', '')
    post_subgroup_score_out_file = os.path.join(out_dir, f'{output_name}_scores_subgroup={post_subgroup_name}.tsv')
    full_post_subgroup_score_out_file = os.path.join(out_dir, f'{output_name}_scores_subgroup={post_subgroup_name}_full.gz')
    if(post_subgroup_file is not None and not os.path.exists(post_subgroup_score_out_file)):
        #subgroup_test_data_out_file = post_subgroup_file.replace('.gz', '_test.pt')
        #if(not os.path.exists(subgroup_test_data_out_file)):
        post_subgroup_data = pd.read_csv(post_subgroup_file, sep='\t', index_col=False, compression='gzip')
        post_subgroup_data.rename(columns={'parent_id' : 'article_id', 'author_id' : 'author'}, inplace=True)
        # print(f'post subgroup data and test data have shared columns {set(post_subgroup_data.columns) & set(test_data.column_names)}')
        ## merge data
        test_data_df = test_data.data.to_pandas()
        # tmp debugging
        # pred_data = pred_data[:len(test_data)]
        test_data_df = test_data_df.assign(**{'pred_data': pred_data})
        subgroup_test_data_df = pd.merge(test_data_df, post_subgroup_data, on=['article_id', 'id', 'author', 'question_id'], how='inner')
        subgroup_pred_data = subgroup_test_data_df.loc[:, 'pred_data'].values
        subgroup_test_data_df.drop('pred_data', axis=1, inplace=True)
        subgroup_test_data = Dataset.from_pandas(subgroup_test_data_df)
        #    # save for posterity
        #    torch.save(subgroup_test_data, subgroup_test_data_out_file)
        #else:
        #    subgroup_test_data = torch.load(subgroup_test_data_out_file)
        # tmp debugging
        # print(f'subgroup test data = {len(subgroup_test_data)}')
        full_subgroup_generation_score_data, subgroup_generation_score_data = get_generation_scores(subgroup_pred_data, subgroup_test_data, generation_model, model_type=model_type, word_embed_file=word_embed_file, train_data=train_data)
        full_subgroup_generation_score_data.to_csv(full_post_subgroup_score_out_file, sep='\t', index=False, compression='gzip')
        subgroup_generation_score_data.to_csv(post_subgroup_score_out_file, sep='\t', index=False)
        # same thing but with data subgroups
        subgroup_per_subreddit_group_scores = get_subreddit_group_generation_scores(post_subgroup_data, subgroup_test_data,
                                                                                    generation_model, model_type=model_type,
                                                                                    word_embed_file=None, sample_size=5000,
                                                                                    train_data=None)
        subgroup_per_subreddit_group_score_file = os.path.join(out_dir, f'{output_name}_scores_subgroup={post_subgroup_name}_subreddit_readergroup.tsv')
        subgroup_per_subreddit_group_scores.to_csv(subgroup_per_subreddit_group_score_file, sep='\t', index=False)


def add_reader_group_category(test_data):
    reader_group_lookup = {
        'expert': ['<EXPERT_PCT_0_AUTHOR>', '<EXPERT_PCT_1_AUTHOR>'],
        'time': ['<RESPONSE_TIME_0_AUTHOR>', '<RESPONSE_TIME_1_AUTHOR>'],
        'location': ['<US_AUTHOR>', '<NONUS_AUTHOR>'],
        'UNK': ['UNK'],
    }
    reader_group_lookup = {
        v1: k for k, v in reader_group_lookup.items() for v1 in v
    }
    test_data = test_data.data.to_pandas()
    test_data = test_data.assign(**{
        'group_category': test_data.loc[:, 'reader_token'].apply(
            lambda x: reader_group_lookup[x])
    })
    test_data = Dataset.from_pandas(test_data)
    return test_data


if __name__ == '__main__':
    main()
