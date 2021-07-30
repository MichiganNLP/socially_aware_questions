"""
Identify author questions from different
reader groups for the same post that are
highly different (based on semantic representations).
"""
import os
from argparse import ArgumentParser
from data_helpers import load_sample_data
import pandas as pd
import numpy as np
np.random.seed(123)
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    parser = ArgumentParser()
    parser.add_argument('out_dir')
    args = vars(parser.parse_args())
    out_dir = args['out_dir']

    ## load all data
    sample_type = None
    post_question_data = load_sample_data(sample_type=sample_type)
    author_vars = ['expert_pct_bin', 'relative_time_bin', 'location_region']
    flat_question_data = pd.melt(post_question_data,
                                 id_vars=['author', 'parent_id', 'id', 'question_id', 'question', 'created_utc', 'subreddit'],
                                 value_vars=author_vars, var_name='group_category', value_name='author_group')
    flat_question_data.dropna(subset=['author_group'], inplace=True)
    flat_question_data = flat_question_data[flat_question_data.loc[:, 'author_group'] != 'UNK']
    ## get paired data
    paired_group_question_data = []
    num_groups_per_category = 2
    for category_i, data_i in flat_question_data.groupby('group_category'):
        author_groups_i = data_i.loc[:, 'author_group'].unique()
        for id_j, data_j in tqdm(data_i.groupby('parent_id')):
            np.random.shuffle(data_j.values)
            ## get max(group_count) questions for each group, and oversample
            if (data_j.loc[:, 'author_group'].nunique() == num_groups_per_category):
                data_j_1 = data_j[data_j.loc[:, 'author_group'] == author_groups_i[0]]
                data_j_2 = data_j[data_j.loc[:, 'author_group'] == author_groups_i[1]]
                max_group_count_j = data_j.loc[:, 'author_group'].value_counts().max()
                data_j_1 = data_j_1.loc[np.random.choice(data_j_1.index, max_group_count_j, replace=(data_j_1.shape[0] < max_group_count_j))]
                data_j_2 = data_j_2.loc[np.random.choice(data_j_2.index, max_group_count_j, replace=(data_j_2.shape[0] < max_group_count_j))]
                paired_group_question_data.extend([data_j_1, data_j_2])
    paired_group_question_data = pd.concat(paired_group_question_data, axis=0)
    ## reorganize data
    paired_sample_size = 500
    paired_sample_data = []
    pair_data_cols = ['question', 'author_group', 'id', 'question_id', 'author']
    for (subreddit_i, group_i), data_i in paired_group_question_data.groupby(['subreddit', 'group_category']):
        sample_ids_i = np.random.choice(data_i.loc[:, 'parent_id'].unique(), paired_sample_size, replace=(data_i.loc[:, 'parent_id'].nunique() < paired_sample_size))
        group_vals = data_i.loc[:, 'author_group'].unique()
        for id_j in sample_ids_i:
            data_j = data_i[data_i.loc[:, 'parent_id'] == id_j]
            data_j_1 = data_j[data_j.loc[:, 'author_group'] == group_vals[0]].iloc[0, :]
            data_j_2 = data_j[data_j.loc[:, 'author_group'] == group_vals[1]].iloc[0, :]
            # fix col names
            data_j_1 = data_j_1.loc[pair_data_cols]
            data_j_2 = data_j_2.loc[pair_data_cols]
            data_j_1.rename({x: f'{x}_1' for x in pair_data_cols}, inplace=True)
            data_j_2.rename({x: f'{x}_2' for x in pair_data_cols}, inplace=True)
            pair_data_j = pd.concat([data_j_1, data_j_2], axis=0)
            pair_data_j = pair_data_j.append(pd.Series({'parent_id': id_j, 'subreddit': subreddit_i, 'group_category': group_i}))
            paired_sample_data.append(pair_data_j)
    paired_sample_data = pd.concat(paired_sample_data, axis=1).transpose()
    # add post text
    paired_sample_data = pd.merge(paired_sample_data, post_question_data.loc[:, ['parent_id', 'post']], on='parent_id', how='left')
    paired_sample_data.drop_duplicates(['parent_id', 'group_category'], inplace=True)

    ## compute sentence representations
    sentence_encoder = SentenceTransformer('paraphrase-distilroberta-base-v1')
    question_vars = ['question_1', 'question_2']
    for question_var in question_vars:
        question_embed = sentence_encoder.encode(paired_sample_data.loc[:, question_var].values.tolist())
        paired_sample_data = paired_sample_data.assign(**{
            f'{question_var}_embed': [question_embed[i, :] for i in range(len(question_embed))]
        })
    ## compute similarity
    paired_sample_data = paired_sample_data.assign(**{
        'question_sim': paired_sample_data.apply(lambda x: cosine_similarity(
            x.loc['question_1_embed'].reshape(1, -1),
            x.loc['question_2_embed'].reshape(1, -1))[0][0], axis=1)
    })
    ## compute similarity cutoff (per subreddit)
    max_sim_pct = 10
    similarity_cutoff_sample_data = []
    for subreddit_i, data_i in paired_sample_data.groupby('subreddit'):
        similarity_cutoff_i = np.percentile(data_i.loc[:, 'question_sim'], max_sim_pct)
        valid_data_i = data_i[data_i.loc[:, 'question_sim'] <= similarity_cutoff_i]
        similarity_cutoff_sample_data.append(valid_data_i)
    similarity_cutoff_sample_data = pd.concat(similarity_cutoff_sample_data, axis=0)
    # flatten ID data
    flat_sample_data = pd.melt(similarity_cutoff_sample_data, id_vars=['parent_id', 'question_id_1', 'question_id_2', 'author_1', 'author_2'], value_vars=['id_1', 'id_2'], var_name='id_type', value_name='id')
    flat_sample_data = pd.melt(flat_sample_data, id_vars=['parent_id', 'author_1', 'author_2', 'id'], value_vars=['question_id_1', 'question_id_2'], var_name='question_id_type', value_name='question_id')
    flat_sample_data = pd.melt(flat_sample_data, id_vars=['parent_id', 'id', 'question_id'], value_vars=['author_1', 'author_2'], var_name='author_id_type', value_name='author_id')
    flat_sample_data.drop('author_id_type', axis=1, inplace=True)
    ## save to file
    out_file = os.path.join(out_dir, 'paired_question_low_sim_data.gz')
    flat_sample_data.to_csv(out_file, sep='\t', index=False, compression='gzip')

if __name__=='__main__':
    main()