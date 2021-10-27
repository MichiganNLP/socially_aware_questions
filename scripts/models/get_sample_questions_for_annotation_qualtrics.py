"""
Get sample questions for annotation on Qualtrics.

Format: subreddit=SUBREDDIT_readergroup=GROUP.txt
"""
import gzip
import json
import os
import re
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from nltk import WordPunctTokenizer
from nlp.arrow_dataset import Dataset

import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_helpers import load_sample_data
import random
random.seed(123)
from test_question_generation import prepare_test_data_for_generation
from model_helpers import load_model, generate_predictions

DEFAULT_GROUP_VAL_LOOKUP = {
    'location': 'US',
    'expert': 1,
    'time': 1,
}
DEFAULT_GROUP_VAL_NAME_LOOKUP = {
    'location': 'US',
    'expert': 'expert',
    'time': 'slow-response',
}
GROUP_VAL_ALL_NAME_LOOKUP = {
    'location': ['NONUS', 'US'],
    'expert': ['novice', 'expert'],
    'time': ['fast-response', 'slow-response']

}
GROUP_EXPLANATION_LOOKUP = {
    'location': 'A NONUS reader is someone who does not live in the United States, and a US reader is someone who currently lives in the US.',
    'expert': 'A novice reader is someone who does not spend very much time discussing topics like this, and a expert reader is someone who spends a lot of time discussing topics like this.',
    'time': 'A fast-response reader is someone who responds to posts quickly, and a slow-response reader is someone who typically responds to posts after a long time.',

}

def convert_question_data_to_txt(data, question_vals=['Q1.1', 'Q1.2', 'Q1.3'], question_id_vars=[]):
    # header text
    question_num = data.loc['question_num']
    subreddit = f"r/{data.loc['subreddit']}"
    text = [f"""
    [[Question:DB]]
    Subreddit: <b>{subreddit}</b> </br>
    Please read the following post.</br></br>

    Post:\n\n{data.loc['post_text']}</br></br>
    """]
    # all_question_vals = ['text_model_output'] + [f'{q}_{question_num}' for q in question_vals]
    # combined_question_id = ','.join(data.loc[['question_id_1', 'question_id_2']].values)
    reader_group_category = data.loc['group_category']
    # question_id_base = f'post={data.loc["post_id"]}_question={combined_question_id}_group={reader_group_category}_'
    # question quality
    question_quality_txt = """
    First, <b>rate the following questions</b> according to the following factors: (1) if the question is <b>relevant</b> to the post, (2) if the question is <b>understandable</b> (if it makes sense to you), and (3) if the question is <b>answerable</b> (if the post author could answer the question).
    """
    text.append(question_quality_txt)
    q_ctr = 1
    # [[ID:{question_id_base + 'question=' + question_val_i + '_quality_' + str(i + 1)}]]
    if(len(question_id_vars) > 0):
        question_id_base_str = '_'.join(list(map(lambda x: f'{x}={data.loc[x]}', question_id_vars)))
    else:
        question_id_base_str = ''
    for i, question_val_i in enumerate(question_vals):
        question_id_str = '_'.join([f'{question_num}.{q_ctr}', question_id_base_str])
        question_txt_i = f"""
        [[Question:Matrix]]
        [[ID:{question_id_str}]]
        {q_ctr}. {data.loc[question_val_i]}

        [[Choices]]
        Relevant
        Understandable
        Answerable

        [[Answers]]
        Very
        Somewhat
        Neutral
        Not very
        Not at all
        """
        text.append(question_txt_i)
        q_ctr += 1
    # reader groups
    num_reader_groups = 2
    # default_group_val = DEFAULT_GROUP_VAL_LOOKUP[reader_group_category]
    default_group_val_name = DEFAULT_GROUP_VAL_NAME_LOOKUP[reader_group_category]
    group_val_names = GROUP_VAL_ALL_NAME_LOOKUP[reader_group_category]
    group_explanation = GROUP_EXPLANATION_LOOKUP[reader_group_category]
    # [[ID:{question_id_base + 'group'}]]
    reader_group_question_txt = f"""
    [[Question:MC]]
    [[ID:{question_num}.{q_ctr}]]

    Next, please read the following two questions that were asked about the post.</br>
    One of the following questions was written by a <b>{group_val_names[0]}</b> reader and the other question was written by a <b>{group_val_names[1]}</b> reader.
    {group_explanation}</br></br>
    Which question do you think was written by a <b>{default_group_val_name}</b>?

    [[Choices]]
    """
    for i in range(1, num_reader_groups + 1):
        q_i = f'Q2.{i}'
        reader_group_question_txt += f"""
        Q{i}: {data.loc[q_i]}"""
    text.append(reader_group_question_txt)
    text = ''.join(text)
    text = re.sub('(?<=\n)( ){3,}', '', text)
    return text

def group_question_txt_by_subgroup(data, subgroup_vars=['subreddit']):
    subgroup_txt = []
    for subgroup_vals_i, data_i in data.groupby(subgroup_vars):
        subgroup_var_str_i = list(map(lambda x: f'{x[0]}={x[1]}', zip(subgroup_vars, subgroup_vals_i)))
        ## add question index
        data_i = data_i.assign(**{'question_num' : list(range(data_i.shape[0]))})
        subgroup_question_data_txt_i = convert_question_data_to_txt_frame(data_i, subgroup_var_str_i)
        subgroup_txt.append([subgroup_vals_i, subgroup_question_data_txt_i])
    subgroups, subgroup_question_data_txt = zip(*subgroup_txt)
    return subgroups, subgroup_question_data_txt

def convert_question_data_to_txt_frame(data, block_name=None):
    question_id_vars = ['subreddit', 'group_category']
    question_data = data.apply(lambda x: convert_question_data_to_txt(x, question_id_vars=question_id_vars), axis=1).values
    question_data_txt = '[[AdvancedFormat]]'
    question_data_txt = '\n'.join([question_data_txt, '\n\n[[PageBreak]]\n\n'.join(question_data)])
    if(block_name is not None):
        block_txt = f'[[Block:{block_name}]]'
    else:
        block_txt = '[[Block]]'
    question_data_txt = '\n'.join([block_txt, question_data_txt])
    return question_data_txt

def generate_opposite_group_questions(data_dir, model_cache_dir, model_type,
                                      paired_group_data, reader_model_file):
    """
    Generate questions from the opposite reader group: e.g.
    for a question written by "US" reader, generate question by "NONUS" reader.

    :param data_dir:
    :param model_cache_dir:
    :param model_type:
    :param paired_group_data:
    :param reader_model_file:
    :return:
    """
    if('reader_group_2' not in paired_group_data.columns):
        reader_group_category_lookup = {
            'expert': ['<EXPERT_PCT_0_AUTHOR>', '<EXPERT_PCT_1_AUTHOR>'],
            'time': ['<RESPONSE_TIME_0_AUTHOR>', '<RESPONSE_TIME_1_AUTHOR>'],
            'location': ['<US_AUTHOR>', '<NONUS_AUTHOR>'],
        }
        reader_group_pair_lookup = {x: y for vs in reader_group_category_lookup.values()
                                    for (x, y) in zip(vs, list(reversed(vs)))}
        paired_group_data = paired_group_data.assign(**{
            'reader_group_2': paired_group_data.loc[:, 'reader_group_1'].apply(reader_group_pair_lookup.get)
        })
    paired_group_test_data = paired_group_data.loc[:, ['reader_group_2', 'source_ids', 'attention_mask']]
    inv_test_data = Dataset.from_pandas(paired_group_test_data)
    inv_test_data.rename_column_('reader_group_2', 'reader_token_str')
    model, model_tokenizer = load_model(model_cache_dir, reader_model_file,
                                        model_type, data_dir)
    model.to(torch.cuda.current_device())
    model_kwargs = prepare_test_data_for_generation(model.config, model_type, inv_test_data)
    generation_param_file = os.path.join(model_cache_dir, 'sample_generation_params.json')
    generation_params = json.load(open(generation_param_file))
    inv_test_data_pred = generate_predictions(model, inv_test_data,
                                              model_tokenizer,
                                              generation_params=generation_params,
                                              model_kwargs=model_kwargs)
    # paired_group_data = paired_group_data.assign(**{
    #     'reader_model_group_2': inv_test_data_pred,
    # })
    return inv_test_data_pred

def main():
    parser = ArgumentParser()
    parser.add_argument('test_data_file')
    parser.add_argument('text_model_data_file')
    parser.add_argument('reader_model_data_file')
    parser.add_argument('reader_model_file')
    parser.add_argument('out_dir')
    args = vars(parser.parse_args())
    test_data_file = args['test_data_file']
    text_model_data_file = args['text_model_data_file']
    reader_model_data_file = args['reader_model_data_file']
    reader_model_file = args['reader_model_file']
    out_dir = args['out_dir']

    ## load data
    sample_question_data = load_sample_data(sample_type='all')
    # load generated data
    test_data = torch.load(test_data_file)
    test_data_df = test_data.data.to_pandas()
    test_data_df = test_data_df.loc[:, ['article_id', 'question_id', 'author', 'id', 'reader_token_str', 'reader_token', 'source_ids', 'source_text', 'attention_mask', 'target_text']]
    text_only_model_data = list(map(lambda x: x.strip(), gzip.open(text_model_data_file, 'rt')))
    reader_model_data = list(map(lambda x: x.strip(), gzip.open(reader_model_data_file,'rt')))
    test_data_df = test_data_df.assign(**{
        'text_model': text_only_model_data,
        'reader_model': reader_model_data,
    })
    test_data_df.rename(columns={'article_id': 'parent_id', 'source_text' : 'post_text'}, inplace=True)
    # copy reader token to separate column for later
    test_data_df = test_data_df.assign(**{'reader_group' : test_data_df.loc[:, 'reader_token_str']})
    ## get N questions per reader group, generate questions for other reader group from reader-aware model
    N_questions_per_group = 20 # need > 5 because we might generate the same text
    reader_group_category_lookup  = {
        'expert' : ['<EXPERT_PCT_0_AUTHOR>', '<EXPERT_PCT_1_AUTHOR>'],
        'time' : ['<RESPONSE_TIME_0_AUTHOR>', '<RESPONSE_TIME_1_AUTHOR>'],
        'location': ['<US_AUTHOR>', '<NONUS_AUTHOR>'],
    }
    reader_group_pair_lookup = {x: y for vs in reader_group_category_lookup.values() for (x,y) in zip(vs, list(reversed(vs)))}
    reader_group_category_lookup = {v : k for k, vs in reader_group_category_lookup.items() for v in vs}
    test_data_df = test_data_df[test_data_df.loc[:, 'reader_group'] != 'UNK']
    test_data_df = test_data_df.assign(**{
        'group_category' : test_data_df.loc[:, 'reader_group'].apply(reader_group_category_lookup.get)
    })
    # add subreddit data
    post_data = pd.read_csv('../../data/reddit_data/subreddit_submissions_2018-01_2019-12.gz', sep='\t', compression='gzip', usecols=['id', 'subreddit'])
    post_data.rename(columns={'id' : 'parent_id'}, inplace=True)
    # print(f'test data columns = {list(sorted(test_data_df.columns))}')
    test_data_df = pd.merge(post_data, test_data_df, on='parent_id', how='right')
    sample_test_data = []
    for (group_category_i, subreddit_i), data_i in test_data_df.groupby(['group_category', 'subreddit']):
        for group_j, data_i in data_i.groupby('reader_group'):
            sample_data_i = data_i.loc[np.random.choice(data_i.index, N_questions_per_group, replace=False), :]
            sample_test_data.append(sample_data_i)
    paired_group_data = pd.concat(sample_test_data, axis=0)
    paired_group_data.rename(columns={'reader_group':'reader_group_1'}, inplace=True)
    paired_group_data = paired_group_data.assign(**{
        'reader_model_group_1' : paired_group_data.loc[:, 'reader_model'].values
    })
    paired_group_data = paired_group_data.assign(**{
        'reader_group_2' : paired_group_data.loc[:, 'reader_group_1'].apply(reader_group_pair_lookup.get)
    })
    ## generate text for the other reader group
    # get inverted text data: e.g. "US" => "NONUS"
    model_cache_dir = '../../data/model_cache'
    model_type = 'bart_author_attention'
    data_dir = '../../data/reddit_data/'
    paired_group_data = generate_opposite_group_questions(data_dir, model_cache_dir,
                                                          model_type, paired_group_data,
                                                          reader_model_file)
    paired_group_data = paired_group_data[paired_group_data.loc[:, 'reader_model_group_1']!=paired_group_data.loc[:, 'reader_model_group_2']]
    # filter long posts
    tokenizer = WordPunctTokenizer()
    max_post_word_count = 500
    paired_group_data = paired_group_data.assign(**{
        'post_len': paired_group_data.loc[:, 'post_text'].apply(lambda x: len(tokenizer.tokenize(x)))
    })
    paired_group_data = paired_group_data[paired_group_data.loc[:, 'post_len']<=max_post_word_count]
    ## convert to standard format
    ## one file per reader group per subreddit
    Q1_vars = ['target_text', 'text_model', 'reader_model']
    Q2_group_idx = [1, 2]
    annotation_data_cols = ['parent_id', 'post_text', 'subreddit', 'group_category', 'reader_group'] + ['Q1.1', 'Q1.2', 'Q1.3'] + ['Q1.1.type', 'Q1.2.type', 'Q1.3.type'] + ['Q2.1', 'Q2.2'] + ['Q2.1.type', 'Q2.2.type']
    subgroup_vars = ['group_category', 'subreddit']
    annotation_sample_size = 10
    for (group_category_i, subreddit_i), data_i in paired_group_data.groupby(subgroup_vars):
        ## organize
        # parent_id | subreddit | reader_group_category | Q1.1 | Q1.2 | Q1.3 | type.1 | type.2 | type.3 | Q1.reader_group | Q2.1 | Q2.2 | Q2.1.reader_group | Q2.2.reader_group
        annotation_data_i = []
        data_i = data_i.loc[np.random.choice(data_i.index, annotation_sample_size, replace=False)]
        for idx_j, data_j in data_i.iterrows():
            random.shuffle(Q1_vars)
            random.shuffle(Q2_group_idx)
            flat_data_j = data_j.loc[['parent_id', 'post_text', 'subreddit', 'group_category', 'reader_group_1']].tolist()
            flat_data_j.extend(data_j.loc[Q1_vars].tolist())
            flat_data_j.extend(Q1_vars)
            flat_data_j.extend(data_j.loc[[f'reader_model_group_{x}' for x in Q2_group_idx]].tolist())
            flat_data_j.extend(data_j.loc[[f'reader_group_{x}' for x in Q2_group_idx]].tolist())
            annotation_data_i.append(flat_data_j)
        annotation_data_i = pd.DataFrame(annotation_data_i, columns=annotation_data_cols)
        # add question index
        annotation_data_i = annotation_data_i.assign(**{
            'question_num' : list(range(annotation_data_i.shape[0]))
        })
        # print(f'annotation data has shape {annotation_data_i.shape}')
        # save original data => separate file per subreddit/group category
        annotation_data_file = os.path.join(out_dir, f'subreddit={subreddit_i}_group={group_category_i}_annotation_data.tsv')
        annotation_data_i.to_csv(annotation_data_file, sep='\t', index=False)
        ## convert to Qualtrics text etc.
        # question_txt_output_i = group_question_txt_by_subgroup(annotation_data_i, subgroup_vars=subgroup_vars)
        # question_txt_i = question_txt_output_i[0]
        block_name = f'subreddit={subreddit_i}_group={group_category_i}'
        question_txt_i = convert_question_data_to_txt_frame(annotation_data_i, block_name=block_name)
        question_txt_file_i = os.path.join(out_dir, f'subreddit={subreddit_i}_group={group_category_i}_annotation_data.txt')
        with open(question_txt_file_i, 'w') as question_txt_out:
            question_txt_out.write(question_txt_i)

    # combine with sample data
    # tmp debugging
    # print(f'test data cols = {list(sorted(test_data_df.columns))}')
    # print(f'question data cols = {list(sorted(sample_question_data.columns))}')
    # sample_data = pd.merge(sample_question_data, test_data_df, on=['question_id', 'parent_id', 'author'], how='inner')
    # author_group_category_vals = {
    #     'relative_time_bin': [0, 1],
    #     'expert_pct_bin': [0, 1],
    #     'location_region': ['US', 'NONUS'],
    # }
    # num_group_vals = 2
    # # tmp debugging
    # # print(f'sample data cols={sample_data.columns}')
    # # print(f'sample data author groups = {sample_data.loc[:, "author_group"].unique()}')
    # paired_group_data = []
    # # print(f'sample data cols = {list(sorted(sample_data.columns))}')
    # for (parent_id_i, group_category_i), data_i in sample_data.groupby(['parent_id', 'group_category']):
    #     if (len(set(data_i.loc[:, 'author_group'].unique()) & set(author_group_category_vals[group_category_i])) == num_group_vals):
    #         group_vals_i = author_group_category_vals[group_category_i]
    #         random.shuffle(group_vals_i)
    #         # randomly swap values for annotation!!
    #         post_i = data_i.loc[:, 'post'].iloc[0]
    #         text_output_i = data_i.loc[:, 'text_model'].iloc[0]
    #         subreddit_i = data_i.loc[:, 'subreddit'].iloc[0]
    #         paired_group_data_i = []
    #         for j, group_val_j in enumerate(group_vals_i):
    #             data_j = data_i[data_i.loc[:, 'author_group'] == group_val_j].iloc[0, :]
    #             # get real text, reader-aware text
    #             paired_group_data_i.append(pd.Series([group_val_j, data_j.loc['reader_model'], data_j.loc['question'], data_j.loc['id']],
    #                                                  index=[f'reader_group_{j + 1}', f'reader_model_output_group_{j + 1}', f'question_group_{j + 1}', f'question_id_{j + 1}']))
    #         paired_group_data_i = pd.concat(paired_group_data_i, axis=0)
    #         paired_group_data_i = paired_group_data_i.append(pd.Series([parent_id_i, post_i, subreddit_i, text_output_i, group_category_i],
    #                                                                    index=['post_id', 'post_text', 'subreddit', 'text_model_output', 'reader_group_category']))
    #         #         paired_group_data_i = paired_group_data_i.append(pd.Series(group_vals_i, index=[f'group_{x+1}' for x in range(len(group_vals_i))]))
    #         paired_group_data.append(paired_group_data_i)
    # paired_group_data = pd.concat(paired_group_data, axis=1).transpose()
    # # remove duplicate questions
    # paired_group_data = paired_group_data[paired_group_data.loc[:, 'question_group_1'] != paired_group_data.loc[:, 'question_group_2']]
    # paired_group_data = paired_group_data[paired_group_data.loc[:,'reader_model_output_group_1'] != paired_group_data.loc[:,'reader_model_output_group_2']]
    # add per-pair ID
    # paired_group_data = paired_group_data.assign(**{
    #     'pair_id': paired_group_data.apply(lambda x: hash(x.loc['question_group_1'] + x.loc['question_group_2']), axis=1)
    # })
    # # filter long posts
    # tokenizer = WordPunctTokenizer()
    # max_post_word_count = 300
    # paired_group_data = paired_group_data.assign(**{
    #     'post_len': paired_group_data.loc[:, 'post_text'].apply(lambda x: len(tokenizer.tokenize(x)))
    # })
    # paired_group_data = paired_group_data[paired_group_data.loc[:, 'post_len'] <= max_post_word_count]
    # print(paired_group_data.loc[:, 'subreddit'].value_counts())


if __name__ == '__main__':
    main()