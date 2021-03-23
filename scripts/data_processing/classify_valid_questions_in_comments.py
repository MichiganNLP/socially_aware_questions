"""
Classify valid questions in comments
using text of question.
E.g. "why" questions are more often classified
as "not valid clarification" as compared to
"what" or "where" questions.
More info in look_for_valid_questions_in_comments.ipynb.
"""
from argparse import ArgumentParser
import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('valid_question_data_files', nargs='+') #
    valid_question_data = list(map(lambda x: pd.read_csv(x, sep='\t', index_col=False), args['valid_question_data_files']))
    valid_question_data = pd.concat(valid_question_data, axis=0)

    ## load label data
    label_cols = ['question_is_relevant', 'question_is_clarification',
                  'submission_contains_answer']
    valid_question_data.fillna({x: 0. for x in label_cols},
                                        inplace=True)
    text_cols = ['selftext', 'parent_text', 'question']
    valid_question_data.fillna({x: '' for x in text_cols},
                                        inplace=True)
    # fix inconsistent columns
    valid_question_data = valid_question_data.assign(**{
        'submission_contains_answer': valid_question_data.loc[:,
                                      'submission_contains_answer'] + valid_question_data.loc[
                                                                      :,
                                                                      'post_contains_answer']
    })
    valid_question_data = valid_question_data.assign(**{
        'selftext': valid_question_data.loc[:, 'selftext'].astype(
            str) + valid_question_data.loc[:, 'parent_text'].astype(
            str)
    })
    valid_question_data = valid_question_data.assign(**{
        'question': valid_question_data.apply(
            lambda x: x.loc['question'] if x.loc['question'] != '' else x.loc[
                'body'], axis=1)
    })
    ## remove bot authors
    bot_authors = ['LocationBot']
    valid_question_data = pd.merge(valid_question_data,
                                            question_comment_data.loc[:,
                                            ['id', 'author']], on='id')
    valid_question_data = valid_question_data[
        ~valid_question_data.loc[:, 'author'].isin(bot_authors)]
    for label_col in label_cols:
        print(valid_question_data.loc[:, label_col].value_counts())
    ## compute validity
    valid_question_data = valid_question_data.assign(**{
        'question_is_valid': ((valid_question_data.loc[:,
                               'question_is_relevant'] == 1) &
                              (valid_question_data.loc[:,
                               'question_is_clarification'] == 1) &
                              (valid_question_data.loc[:,
                               'submission_contains_answer'] == 0)).astype(int)
    })

if __name__ == '__main__':
    main()