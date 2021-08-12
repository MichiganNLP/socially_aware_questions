"""
Get sample questions for annotation on Qualtrics.

Format: subreddit=SUBREDDIT_readergroup=GROUP.txt
"""
import gzip
import re
from argparse import ArgumentParser

import pandas as pd
import torch
from nltk import WordPunctTokenizer

from data_helpers import load_sample_data
import random
random.seed(123)

DEFAULT_GROUP_VAL_LOOKUP = {
    'location_region': 'US',
    'expert_pct_bin': 1,
    'relative_time_bin': 1,
}
DEFAULT_GROUP_VAL_NAME_LOOKUP = {
    'location_region': 'US',
    'expert_pct_bin': 'expert',
    'relative_time_bin': 'slow-response',
}
GROUP_VAL_ALL_NAME_LOOKUP = {
    'location_region': ['NONUS', 'US'],
    'expert_pct_bin': ['novice', 'expert'],
    'relative_time_bin': ['fast-response', 'slow-response']

}
GROUP_EXPLANATION_LOOKUP = {
    'location_region': 'A NONUS reader is someone who does not live in the United States, and a US reader is someone who currently lives in the US.',
    'expert_pct_bin': 'A novice reader is someone who does not spend very much time discussing topics like this, and a expert reader is someone who spends a lot of time discussing topics like this.',
    'relative_time_bin': 'A fast-response reader is someone who responds to posts quickly, and a slow-response reader is someone who typically responds to posts after a long time.',

}


def convert_question_data_to_txt(data, question_vals=['question_group', 'reader_model_output_group'], question_num=1):
    # header text
    subreddit = f"r/{data.loc['subreddit']}"
    text = [f"""
    [[Question:DB]]
    Subreddit: <b>{subreddit}</b> </br>
    Please read the following post.</br></br>

    Post:\n\n{data.loc['post_text']}</br></br>
    """]
    all_question_vals = ['text_model_output'] + [f'{q}_{question_num}' for q in
                                                 question_vals]
    combined_question_id = ','.join(
        data.loc[['question_id_1', 'question_id_2']].values)
    reader_group_category = data.loc['reader_group_category']
    # question_id_base = f'post={data.loc["post_id"]}_question={combined_question_id}_group={reader_group_category}_'
    # question quality
    question_quality_txt = """
    First, <b>rate the following questions</b> according to the following factors: (1) if the question is <b>relevant</b> to the post, (2) if the question is <b>understandable</b> (if it makes sense to you), and (3) if the question is <b>answerable</b> (if the post author could answer the question).
    """
    text.append(question_quality_txt)
    q_ctr = 1
    # [[ID:{question_id_base + 'question=' + question_val_i + '_quality_' + str(i + 1)}]]
    for i, question_val_i in enumerate(all_question_vals):
        question_txt_i = f"""
        [[Question:Matrix]]
        [[ID:{question_num}.{q_ctr}]]
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
    default_group_val = DEFAULT_GROUP_VAL_LOOKUP[reader_group_category]
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
        reader_group_question_txt += f"""
        Q{i}: {data.loc['reader_model_output_group_' + str(i)]}"""
    text.append(reader_group_question_txt)
    text = ''.join(text)
    text = re.sub('(?<=\n)( ){3,}', '', text)
    return text

def group_question_txt_by_subreddit(data):
    subreddit_txt = []
    for subreddit_i, data_i in data.groupby('subreddit'):
        subreddit_question_data_i = data_i.apply(lambda x: convert_question_data_to_txt(x), axis=1).values
        subreddit_question_data_txt_i = '[[AdvancedFormat]]'
        subreddit_question_data_txt_i = '\n'.join([subreddit_question_data_txt_i, '\n\n[[PageBreak]]\n\n'.join(subreddit_question_data_i)])
        subreddit_question_data_txt_i = '\n'.join([f'[[Block:subreddit={subreddit_i}]]', subreddit_question_data_txt_i])
        subreddit_txt.append([subreddit_i, subreddit_question_data_txt_i])
    subreddits, subreddit_question_data_txt = zip(*subreddit_txt)
    return subreddits, subreddit_question_data_txt

def main():
    parser = ArgumentParser()
    parser.add_argument('test_data_file')
    parser.add_argument('text_model_data_file')
    parser.add_argument('reader_model_data_file')
    parser.add_argument('out_dir')
    args = vars(parser.parse_args())
    test_data_file = args['test_data_file']
    text_model_data_file = args['text_model_data_file']
    reader_model_data_file = args['reader_model_data_file']
    out_dir = args['out_dir']

    ## load data
    sample_question_data = load_sample_data(sample_type='all')
    # load generated data
    test_data = torch.load(test_data_file)
    test_data_df = test_data.data.to_pandas()
    test_data_df = test_data_df.loc[:, ['article_id', 'question_id', 'author', 'id']]
    text_only_model_data = list(map(lambda x: x.strip(), gzip.open(text_model_data_file, 'rt')))
    reader_model_data = list(map(lambda x: x.strip(), gzip.open(reader_model_data_file,'rt')))
    test_data_df = test_data_df.assign(**{
        'text_model': text_only_model_data,
        'reader_model': reader_model_data,
    })
    test_data_df.rename(columns={'article_id': 'parent_id'}, inplace=True)
    # combine with sample data
    # tmp debugging
    # print(f'test data cols = {list(sorted(test_data_df.columns))}')
    # print(f'question data cols = {list(sorted(sample_question_data.columns))}')
    sample_data = pd.merge(sample_question_data, test_data_df, on=['question_id', 'parent_id', 'author'], how='inner')
    paired_group_data = []
    author_group_category_vals = {
        'relative_time_bin': [0, 1],
        'expert_pct_bin': [0, 1],
        'location_region': ['US', 'NONUS'],
    }
    num_group_vals = 2
    for (parent_id_i, group_category_i), data_i in sample_data.groupby(['parent_id', 'group_category']):
        if (len(set(data_i.loc[:, 'author_group'].unique()) & set(author_group_category_vals[group_category_i])) == num_group_vals):
            group_vals_i = author_group_category_vals[group_category_i]
            random.shuffle(group_vals_i)
            # randomly swap values for annotation!!
            post_i = data_i.loc[:, 'post'].iloc[0]
            text_output_i = data_i.loc[:, 'text_model'].iloc[0]
            subreddit_i = data_i.loc[:, 'subreddit'].iloc[0]
            paired_group_data_i = []
            for j, group_val_j in enumerate(group_vals_i):
                data_j = data_i[data_i.loc[:, 'author_group'] == group_val_j].iloc[0, :]
                # get real text, reader-aware text
                paired_group_data_i.append(pd.Series([group_val_j, data_j.loc['reader_model'], data_j.loc['question'], data_j.loc['id']],
                                                     index=[f'reader_group_{j + 1}', f'reader_model_output_group_{j + 1}', f'question_group_{j + 1}', f'question_id_{j + 1}']))
            paired_group_data_i = pd.concat(paired_group_data_i, axis=0)
            paired_group_data_i = paired_group_data_i.append(pd.Series([parent_id_i, post_i, subreddit_i, text_output_i, group_category_i],
                                                                       index=['post_id', 'post_text', 'subreddit', 'text_model_output', 'reader_group_category']))
            #         paired_group_data_i = paired_group_data_i.append(pd.Series(group_vals_i, index=[f'group_{x+1}' for x in range(len(group_vals_i))]))
            paired_group_data.append(paired_group_data_i)
    paired_group_data = pd.concat(paired_group_data, axis=1).transpose()
    # remove duplicate questions
    paired_group_data = paired_group_data[paired_group_data.loc[:, 'question_group_1'] != paired_group_data.loc[:, 'question_group_2']]
    paired_group_data = paired_group_data[paired_group_data.loc[:,'reader_model_output_group_1'] != paired_group_data.loc[:,'reader_model_output_group_2']]
    # add per-pair ID
    paired_group_data = paired_group_data.assign(**{
        'pair_id': paired_group_data.apply(lambda x: hash(x.loc['question_group_1'] + x.loc['question_group_2']), axis=1)
    })
    # filter long posts
    tokenizer = WordPunctTokenizer()
    max_post_word_count = 300
    paired_group_data = paired_group_data.assign(**{
        'post_len': paired_group_data.loc[:, 'post_text'].apply(lambda x: len(tokenizer.tokenize(x)))
    })
    paired_group_data = paired_group_data[paired_group_data.loc[:, 'post_len'] <= max_post_word_count]
    print(paired_group_data.loc[:, 'subreddit'].value_counts())

if __name__ == '__main__':
    main()