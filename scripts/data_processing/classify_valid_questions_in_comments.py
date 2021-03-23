"""
Train classifier on valid questions in comments
using text of question.
E.g. "why" questions are more often considered
"not valid clarification" as compared to
"what" or "where" questions (which look
for concrete details to clarify).
More info in look_for_valid_questions_in_comments.ipynb.
"""
import os
import pickle
import re
from argparse import ArgumentParser
from collections import defaultdict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from data_helpers import remove_edit_data
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def main():
    parser = ArgumentParser()
    parser.add_argument('valid_question_data_files', nargs='+')
    parser.add_argument('--out_dir', default='../../data/reddit_data/valid_question_model/')
    args = vars(parser.parse_args())
    valid_question_data = list(map(lambda x: pd.read_csv(x, sep='\t', index_col=False), args['valid_question_data_files']))
    valid_question_data = pd.concat(valid_question_data, axis=0)

    ## load label data
    label_cols = ['question_is_relevant', 'question_is_clarification',
                  'submission_contains_answer']
    valid_question_data.fillna({x: 0. for x in label_cols}, inplace=True)
    text_cols = ['selftext', 'parent_text', 'question']
    valid_question_data.fillna({x: '' for x in text_cols}, inplace=True)
    # fix inconsistent columns
    valid_question_data = valid_question_data.assign(**{
        'submission_contains_answer': valid_question_data.loc[:, 'submission_contains_answer'] + valid_question_data.loc[:, 'post_contains_answer']
    })
    valid_question_data = valid_question_data.assign(**{
        'selftext': valid_question_data.loc[:, 'selftext'].astype(str) + valid_question_data.loc[:, 'parent_text'].astype(str)
    })
    valid_question_data = valid_question_data.assign(**{
        'question': valid_question_data.apply(
            lambda x: x.loc['question'] if x.loc['question'] != '' else x.loc['body'], axis=1)
    })
    ## remove bot authors
    # bot_authors = ['LocationBot']
    # valid_question_data = pd.merge(valid_question_data,
    #                                         question_comment_data.loc[:,
    #                                         ['id', 'author']], on='id')
    # valid_question_data = valid_question_data[~valid_question_data.loc[:, 'author'].isin(bot_authors)]
    # for label_col in label_cols:
    #     print(valid_question_data.loc[:, label_col].value_counts())
    # ## remove edits from posts
    # valid_question_data = valid_question_data.assign(**{
    #     'clean_parent_text': valid_question_data.loc[:, 'parent_text'].apply(
    #         lambda x: remove_edit_data(x))
    # })
    ## remove quotations from all questions
    quote_matcher = re.compile('&gt;[^\n]+$')
    test_question = '&gt;legally, what can I do to attempt to get the best sentence possible?'
    # print(quote_matcher.sub('', test_question))
    valid_question_data = valid_question_data.assign(**{
        'question': valid_question_data.loc[:, 'question'].apply(lambda x: quote_matcher.sub('', x))
    })
    # drop null questions
    valid_question_data = valid_question_data[valid_question_data.loc[:, 'question'] != '']

    ## extract text
    ## convert data to useful format
    word_tokenizer = WordPunctTokenizer()
    question_vocab = {'does', 'did', 'what', 'how', 'were', 'are', 'was', 'is',
                      'where', 'when', 'why', 'could', 'can', 'who', 'would'}
    cv = CountVectorizer(vocabulary=question_vocab, min_df=0.,
                         tokenizer=word_tokenizer.tokenize)
    question_dtm = cv.fit_transform(valid_question_data.loc[:, 'question'].values)
    question_labels = np.array(valid_question_data.loc[:, 'question_is_clarification'].values)

    ## fit model
    model = LogisticRegression(max_iter=1000)
    model.fit(question_dtm, question_labels)
    ## compute accuracy
    n_folds = 10
    k_fold = StratifiedKFold(n_folds, shuffle=True)
    total_acc_scores = []
    subreddit_acc_scores = defaultdict(list)
    subreddits = valid_question_data.loc[:, 'subreddit'].unique()
    for train_idx, test_idx in k_fold.split(question_dtm, question_labels):
        X_train, X_test = question_dtm[train_idx, :], question_dtm[test_idx, :]
        Y_train, Y_test = question_labels[train_idx], question_labels[test_idx]
        model.fit(X_train, Y_train)
        # total accuracy
        Y_pred = model.predict(X_test)
        acc = f1_score(Y_test, Y_pred)
        total_acc_scores.append(acc)
        # per-subreddit accuracy
        for subreddit_i in subreddits:
            subreddit_idx_i = np.where(valid_question_data.loc[:, 'subreddit'] == subreddit_i)[0]
            subreddit_idx_i = list(set(subreddit_idx_i) & set(test_idx))
            if (len(subreddit_idx_i) > 0):
                X_test_i = question_dtm[subreddit_idx_i, :]
                Y_test_i = question_labels[subreddit_idx_i]
                Y_pred_i = model.predict(X_test_i)
                acc = f1_score(Y_test_i, Y_pred_i)
                subreddit_acc_scores[subreddit_i].append(acc)
    # compute mean accuracy, combine
    mean_acc = np.mean(total_acc_scores)
    mean_subreddit_acc = list(map(lambda x: (x[0], np.mean(x[1])), subreddit_acc_scores.items()))
    acc_scores = [('total', mean_acc)] + mean_subreddit_acc
    acc_scores = pd.DataFrame(acc_scores, columns=['score_type', 'score'])

    ## save
    out_dir = args['out_dir']
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    model_out_file = os.path.join(out_dir, 'valid_question_detection_model.pkl')
    pickle.dump(model, open(model_out_file, 'wb'))
    # save accuracy
    acc_out_file = os.path.join(out_dir, 'valid_question_detection_scores.csv')
    acc_scores.to_csv(acc_out_file, sep=',', index=False)

if __name__ == '__main__':
    main()