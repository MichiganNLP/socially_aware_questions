"""
Extract questions that are specific to reader groups,
using the classifiers trained on predicting group category.
"""
import os
import pickle
import re
from argparse import ArgumentParser
from ast import literal_eval
from itertools import product
from pathlib import Path
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import torch
import  sys
if('..' not in sys.path):
    sys.path.append('..')
    sys.path.append('../models/')
from model_helpers import load_sentence_embed_model

def get_group_classification_models(group_vars, model_dir, subreddits):
    """
    Load group classification models from file.

    :param group_vars:
    :param model_dir:
    :param subreddits:
    :return:
    """
    subreddit_group_model_lookup = {}
    default_class_matcher = re.compile('(?<=class1\=).*(?=\.pkl)')
    for subreddit_i in subreddits:
        model_dir_i = os.path.join(model_dir, subreddit_i)
        # load model
        for group_var_j in group_vars:
            model_file_matcher_i = re.compile(f'.*group={group_var_j}.*\.pkl')
            model_file_i = list(filter(lambda x: model_file_matcher_i.match(x) is not None, os.listdir(model_dir_i)))
            if (len(model_file_i) > 0):
                model_file_i = model_file_i[0]
                model_file_i = os.path.join(model_dir_i, model_file_i)
                # get default class
                default_class_i = default_class_matcher.search(model_file_i).group(0)
                if (default_class_i.isdigit()):
                    default_class_i = literal_eval(default_class_i)
                model_i = pickle.load(open(model_file_i, 'rb'))
                subreddit_group_model_lookup[f'{subreddit_i};{group_var_j}'] = (model_i, default_class_i)
    return subreddit_group_model_lookup

def get_model_class_prob(data, model_lookup, reader_group_other_class_lookup, pred_var='PCA_question_post_encoded'):
    """
    Compute model classification prob for group.

    :param data:
    :param model_lookup:
    :param reader_group_other_class_lookup:
    :param pred_var:
    :return:
    """
    subreddit = data.loc["subreddit"]
    group_category = data.loc["group_category"]
    model, default_class = model_lookup[f'{subreddit};{group_category}']
    model_prob = model.predict_proba(data.loc[pred_var].reshape(1,-1))[0, :]
    max_class_idx = model_prob.argmax()
    max_class_prob = model_prob.max()
    if(max_class_idx == 1):
        max_class = default_class
    else:
        max_class = reader_group_other_class_lookup.loc[group_category][default_class]
    return max_class, max_class_prob

def main():
    parser = ArgumentParser()
    parser.add_argument('test_data_file')
    parser.add_argument('out_dir')
    parser.add_argument('--post_data_file', default='../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz')
    parser.add_argument('--model_dir', default='../../data/reddit_data/group_classification_model/question_post_data/')
    parser.add_argument('--prob_cutoff_pct', type=int, default=90)
    args = vars(parser.parse_args())
    out_dir = args['out_dir']
    test_data_file = args['test_data_file']
    post_data_file = args['post_data_file']
    model_dir = args['model_dir']
    prob_cutoff_pct = args['prob_cutoff_pct']

    ## generate reader group prob scores
    reader_group_prob_score_file = os.path.join(out_dir, 'reader_group_prob_scores.gz')
    if(not os.path.exists(reader_group_prob_score_file)):
        test_data = torch.load(test_data_file).data.to_pandas().drop(['source_ids', 'target_ids', 'attention_mask', 'subreddit_embed', 'text_embed', 'source_ids_reader_token', 'source_text_reader_token'],axis=1)
        post_data = pd.read_csv(post_data_file, sep='\t', compression='gzip', usecols=['id', 'subreddit']).rename(
            columns={'id': 'article_id'})
        test_data = pd.merge(test_data, post_data, on='article_id')
        ## add reader group categories
        reader_group_category_lookup = {
            'expert_pct_bin': ['<EXPERT_PCT_0_AUTHOR>', '<EXPERT_PCT_1_AUTHOR>'],
            'relative_time_bin': ['<RESPONSE_TIME_0_AUTHOR>', '<RESPONSE_TIME_1_AUTHOR>'],
            'location_region': ['<NONUS_AUTHOR>', '<US_AUTHOR>'],
            'UNK': ['UNK']
        }
        reader_group_category_lookup = {
            v1: k for k, v in reader_group_category_lookup.items() for v1 in v
        }
        # fix column names
        test_data.rename(columns={'reader_token_str': 'reader_group', 'source_text': 'post', 'target_text': 'question'}, inplace=True)
        test_data = test_data.assign(**{
            'group_category': test_data.loc[:, 'reader_group'].apply(reader_group_category_lookup.get)
        })
        ## generate text representations
        sentence_embed_model = load_sentence_embed_model()
        # encode data
        embed_vars = ['question', 'post']
        for embed_var_i in embed_vars:
            encode_var_i = f'{embed_var_i}_encoded'
            encoding_i = sentence_embed_model.encode(test_data.loc[:, embed_var_i].values, batch_size=16, device=torch.cuda.current_device(), show_progress_bar=True)
            test_data = test_data.assign(**{
                encode_var_i: [encoding_i[i, :] for i in range(encoding_i.shape[0])],
            })
        ## use PCA to compress data
        model_home_dir = Path(model_dir).parent.absolute()
        PCA_question_embed_model = pickle.load(open(os.path.join(model_home_dir, 'PCA_model_embed=question_encoded.pkl'), 'rb'))
        PCA_post_embed_model = pickle.load(open(os.path.join(model_home_dir, 'PCA_model_embed=post_encoded.pkl'), 'rb'))
        PCA_models = [PCA_question_embed_model, PCA_post_embed_model]
        for embed_var_i, PCA_model_i in zip(embed_vars, PCA_models):
            encode_var_i = f'{embed_var_i}_encoded'
            mat_i = np.vstack(test_data.loc[:, encode_var_i].values)
            reduce_mat_i = PCA_model_i.transform(mat_i)
            test_data = test_data.assign(**{
                f'PCA_{encode_var_i}': [reduce_mat_i[i, :] for i in range(reduce_mat_i.shape[0])]
            })
        # combine representations
        test_data = test_data.assign(**{
            'PCA_question_post_encoded': test_data.apply(lambda x: np.hstack([x.loc['PCA_question_encoded'], x.loc['PCA_post_encoded']]), axis=1)
        })
        # fix class name for classification UGH
        reader_group_category_class_lookup = {
            'expert_pct_bin': {'<EXPERT_PCT_0_AUTHOR>': 0, '<EXPERT_PCT_1_AUTHOR>': 1},
            'relative_time_bin': {'<RESPONSE_TIME_0_AUTHOR>': 0, '<RESPONSE_TIME_1_AUTHOR>': 1},
            'location_region': {'<NONUS_AUTHOR>': 'NONUS', '<US_AUTHOR>': 'US'},
            'UNK': {'UNK': 'UNK'}
        }
        test_data = test_data.assign(**{
            'reader_group_class': test_data.apply(lambda x: reader_group_category_class_lookup[x.loc['group_category']][x.loc['reader_group']], axis=1)
        })
        print(f'group category counts\n{test_data.loc[:, ["group_category", "reader_group_class"]].value_counts().sort_index()}')
        ## load all classification models
        subreddits = test_data.loc[:, 'subreddit'].unique()
        group_vars = list(set(test_data.loc[:, 'group_category'].unique()) - {'UNK'})
        subreddit_group_model_lookup = get_group_classification_models(group_vars, model_dir, subreddits)
        ## compute P(group)
        reader_group_other_class_lookup = test_data.groupby('group_category').apply(lambda x: dict([(z[0], z[1]) for z in product(x.loc[:, 'reader_group_class'].unique(), x.loc[:, 'reader_group_class'].unique()) if z[0] != z[1]]))
        pred_var = 'PCA_question_post_encoded'
        reader_group_test_data = test_data[test_data.loc[:, 'group_category'] != 'UNK']
        pred_class_prob_data = reader_group_test_data.progress_apply(lambda x: get_model_class_prob(x, subreddit_group_model_lookup, reader_group_other_class_lookup, pred_var=pred_var), axis=1)
        # fix variables etc.
        reader_group_test_data = reader_group_test_data.assign(**{
            'pred_class': list(map(lambda x: x[0], pred_class_prob_data)),
            'pred_class_prob': list(map(lambda x: x[1], pred_class_prob_data)),
        })
        # restrict to useful columns
        reader_group_test_data_cols = ['post', 'question', 'article_id', 'id', 'author', 'question_id',
                                       'subreddit', 'group_category',
                                       'reader_group_class', 'pred_class', 'pred_class_prob']
        reader_group_test_data.loc[:, reader_group_test_data_cols].to_csv(reader_group_prob_score_file, sep='\t', compression='gzip', index=False)
    else:
        reader_group_test_data = pd.read_csv(reader_group_prob_score_file, sep='\t', compression='gzip')

    ## cut off at specified probability, save to file
    per_group_prob_cutoff_data = []
    for (group_i, subreddit_i), data_i in reader_group_test_data.groupby(['group_category', 'subreddit']):
        # tmp debugging
        # print(f'data class probs = {data_i.loc[:, "pred_class_prob"].dtype}')
        prob_cutoff_val_i = np.percentile(data_i.loc[:, 'pred_class_prob'], prob_cutoff_pct)
        # get true positives
        per_group_prob_cutoff_data_i = data_i[(data_i.loc[:, 'pred_class_prob'] >= prob_cutoff_val_i) &
                                              (data_i.loc[:, 'reader_group_class'] == data_i.loc[:, 'pred_class'])]
        per_group_prob_cutoff_data.append(per_group_prob_cutoff_data_i)
    per_group_prob_cutoff_data = pd.concat(per_group_prob_cutoff_data, axis=0)
    prob_cutoff_data_file = os.path.join(out_dir, f'reader_group_cutoff_pct={prob_cutoff_pct}.gz')
    per_group_prob_cutoff_data.to_csv(prob_cutoff_data_file, sep='\t', compression='gzip', index=False)

if __name__ == '__main__':
    main()