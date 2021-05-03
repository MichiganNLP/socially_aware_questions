"""
Process annotated data to use for filtering the collected
data based on (1) relevance; (2) clarification question.
"""
import logging
import os
import pickle
import re
from argparse import ArgumentParser
from functools import reduce
from math import ceil

import numpy as np
import pandas as pd
np.random.seed(123)
from data_helpers import compute_sent_word_overlap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def assign_next_annotation(annotation_list):
    if (len(annotation_list) > 1):
        return annotation_list[1]
    else:
        return np.nan

from statsmodels.stats.inter_rater import cohens_kappa, to_table
def compute_pairwise_annotator_agreement(label_data):
    label_table, label_bins = to_table(label_data)
    agreement_results = cohens_kappa(label_table)
    return agreement_results['kappa']

def resample_data(data, label_var):
    label_counts = data.loc[:, label_var].value_counts()
    minority_label = label_counts.idxmin()
    N_majority = label_counts.max()
    N_minority = label_counts.min()
    N_resample = N_majority - N_minority
    resample_multiplier = int(ceil(N_resample / N_minority))
    minority_data = data[data.loc[:, label_var]==minority_label]
    resample_data_candidate_idx = list(reduce(lambda x,y: x+y, [minority_data.index.tolist(),]*resample_multiplier))
    resample_data_idx = np.random.choice(resample_data_candidate_idx, N_resample, replace=False)
    new_data = pd.concat([data, data.loc[resample_data_idx, :]], axis=0)
    return new_data

def train_test_model_multi_folds(X, Y, model, N_folds=5):
    k_fold = StratifiedKFold(n_splits=N_folds, shuffle=True, random_state=123)
    prediction_scores = []
    for train_idx, test_idx in k_fold.split(X, Y):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        f1 = f1_score(Y_test, Y_pred)
        TP = ((Y_test==1) & (Y_pred==1)).sum()
        FP = ((Y_test==0) & (Y_pred==1)).sum()
        FN = ((Y_test==1) & (Y_pred==0)).sum()
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        prediction_scores.append([prec, rec, f1])
    prediction_scores = pd.DataFrame(prediction_scores, columns=['prec', 'rec', 'f1'])
    # train model on full data
    full_data_model = model.fit(X, Y)
    return full_data_model, prediction_scores

def test_question_relevance(annotation_data, annotation_dir):
    more_info_question_annotation_data = annotation_data[annotation_data.loc[:, 'question_asks_for_more_info_agree']]
    more_info_label_counts = more_info_question_annotation_data.loc[:, 'question_asks_for_more_info_maj_label'].value_counts()
    logging.info(f'more_info label counts =\n{more_info_label_counts}')
    # resample data
    label_var = 'question_asks_for_more_info_maj_label'
    resample_more_info_data = resample_data(more_info_question_annotation_data, label_var)
    # extract text features
    from sklearn.feature_extraction.text import CountVectorizer
    max_features = 50
    max_df = 0.25
    cv = CountVectorizer(max_df=max_df, max_features=max_features)
    dtm = cv.fit_transform(resample_more_info_data.loc[:, 'question'].values)
    # top features
    cv_vocab = list(sorted(cv.vocabulary_, key=cv.vocabulary_.get))
    dtm_vocab_counts = pd.Series(np.array(dtm.sum(axis=0))[0], index=cv_vocab).sort_values(ascending=False)
    logging.info(f'top vocab in questions = {dtm_vocab_counts.iloc[:10]}')
    label_var = 'question_asks_for_more_info_maj_label'
    models = [
        SVC(C=1., kernel='rbf', max_iter=1000),
        LogisticRegression(penalty='l2', C=1., max_iter=1000),
        RandomForestClassifier(n_estimators=25)
    ]
    info_labels = resample_more_info_data.loc[:, label_var].values
    more_info_classification_scores = []
    for model in models:
        full_model, prediction_scores = train_test_model_multi_folds(dtm, info_labels, model, N_folds=10)
        mean_prediction_scores = prediction_scores.mean(axis=0)
        model_type = type(model).__name__
        mean_prediction_scores.loc['model_type'] = model_type
        more_info_classification_scores.append(mean_prediction_scores)
        # save model
        model_file = os.path.join(annotation_dir, f'{model_type}.pkl')
        with open(model_file, 'wb') as model_out:
            pickle.dump(full_model, model_out)
    mean_prediction_scores = pd.concat(more_info_classification_scores, axis=1).transpose()
    mean_prediction_score_file = os.path.join(annotation_dir, 'more_info_classification_scores.tsv')
    mean_prediction_scores.to_csv(mean_prediction_score_file, sep='\t', index=False)
    # save vocab from CV for model features
    vocab_out_file = os.path.join(annotation_dir, 'model_vocab.txt')
    with open(vocab_out_file, 'w') as vocab_out:
        vocab_out.write('\n'.join(cv_vocab))

def test_post_question_overlap(annotation_data, annotation_dir):
    question_relevant_annotation_data = annotation_data[annotation_data.loc[:, 'question_is_relevant_agree']]
    question_relevant_annotation_data = question_relevant_annotation_data.assign(**{
        'post_question_overlap': question_relevant_annotation_data.apply(lambda x: compute_sent_word_overlap(x.loc['post_sent_tokens'], [x.loc['question_tokens']])[0], axis=1)
    })
    overlap_bound_combos = [
        [0.05, 0.9],  # wide
        [0.1, 0.5],  # medium
        [0.125, 0.3],  # narrow
    ]
    overlap_scores = []
    for overlap_bounds_i in overlap_bound_combos:
        question_relevant_annotation_data = question_relevant_annotation_data.assign(**{
            'question_post_overlap': (question_relevant_annotation_data.loc[:, 'post_question_overlap'] >= overlap_bounds_i[0]) & (question_relevant_annotation_data.loc[:, 'post_question_overlap'] <= overlap_bounds_i[1])
        })
        TP = ((question_relevant_annotation_data.loc[:, 'question_is_relevant_maj_label'] == 1) & (question_relevant_annotation_data.loc[:, 'question_post_overlap'] == 1)).sum()
        FP = ((question_relevant_annotation_data.loc[:, 'question_is_relevant_maj_label'] == 0) & (question_relevant_annotation_data.loc[:, 'question_post_overlap'] == 1)).sum()
        FN = ((question_relevant_annotation_data.loc[:, 'question_is_relevant_maj_label'] == 1) & (question_relevant_annotation_data.loc[:, 'question_post_overlap'] == 0)).sum()
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        overlap_scores.append([overlap_bounds_i, prec, rec])
        logging.info(f'overlap bounds {"{:.3f}".format(overlap_bounds_i[0])}-{"{:.3f}".format(overlap_bounds_i[1])}: prec={"{:.3f}".format(prec)}; rec={"{:.3f}".format(rec)}')
    # save to file
    overlap_scores = pd.DataFrame(overlap_scores, columns=['overlap_bounds', 'prec', 'rec'])
    overlap_score_file = os.path.join(annotation_dir, 'relevant_overlap_scores.tsv')
    overlap_scores.to_csv(overlap_score_file, sep='\t', index=False)


def assign_majority_label(annotation_data):
    label_cols = ['question_is_relevant', 'question_asks_for_more_info']
    num_annotators_per_question = 2
    annotation_cols = [f'{label_col}_num_{i}' for label_col in label_cols for i in range(num_annotators_per_question)]
    annotation_data = annotation_data.dropna(subset=annotation_cols, axis=0, how='any')
    ## restrict to questions with perfect agreement
    from scipy.stats import mode
    for label_col_i in label_cols:
        annotation_cols_i = [f'{label_col_i}_num_{i}' for i in range(num_annotators_per_question)]
        agree_col_i = f'{label_col_i}_agree'
        annotation_data = annotation_data.assign(**{
            agree_col_i: annotation_data.loc[:, annotation_cols_i].apply(lambda x: x.min() == x.max() and x.min() != -1, axis=1)
        })
        logging.info(f'label={label_col_i}; {annotation_data.loc[:, agree_col_i].sum()}/{annotation_data.shape[0]} data with agreement')
        ## assign majority label
        annotation_data = annotation_data.assign(**{
            f'{label_col_i}_maj_label': annotation_data.loc[:, annotation_cols_i].apply(lambda x: mode(x.values).mode[0], axis=1)
        })
    return annotation_data

from nltk.tokenize import PunktSentenceTokenizer, WordPunctTokenizer
def preprocess_text(annotation_data):
    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = WordPunctTokenizer()
    annotation_data = annotation_data.assign(**{
        'post_sent_tokens': annotation_data.loc[:, 'post_text'].apply(lambda x: list(map(lambda y: word_tokenizer.tokenize(y), sent_tokenizer.tokenize(x)))),
        'question_tokens': annotation_data.loc[:, 'question'].apply(lambda x: word_tokenizer.tokenize(x)),
    })
    return annotation_data


def organize_data(annotation_dir, no_label_data):
    annotation_data = pd.read_csv(no_label_data, sep='\t', index_col=False)
    label_cols = ['question_is_relevant', 'question_asks_for_more_info']
    annotation_data.drop(label_cols, axis=1, inplace=True)
    annotator_file_matcher = re.compile('(?<=_sample_)(\d+)(?=\.tsv)')
    annotator_files = list(filter(lambda x: annotator_file_matcher.search(x) is not None, os.listdir(annotation_dir)))
    annotator_files = list(map(lambda x: os.path.join(annotation_dir, x), annotator_files))
    num_annotators = len(annotator_files)
    for annotator_file_i in annotator_files:
        annotator_data_i = pd.read_csv(annotator_file_i, sep='\t', index_col=0)
        label_i = annotator_file_matcher.search(annotator_file_i).group(0)
        ## join data
        annotation_data = pd.merge(annotation_data, annotator_data_i.loc[:, ['parent_id'] + label_cols], on=['parent_id'], how='left')
        annotation_data.rename(columns={
            label_col: f'{label_col}_{label_i}'
            for label_col in label_cols
        }, inplace=True)
    # debugging: which questions have 0 annotators??
    # for label_col in label_cols:
    #     all_null_data = annotation_data[annotation_data.loc[:, [f'{label_col}_{i}' for i in range(num_annotators)]].isna().apply(lambda x: all(x), axis=1)]
    #     display(all_null_data.loc[:, 'question'].values)
    # move labels to A, B columns
    ## combine per-question annotations (e.g. 2 annotation per question => num_0 and num_1)
    annotator_nums = list(range(num_annotators))
    for label_col_i in label_cols:
        annotator_label_cols_i = list(map(lambda x: f'{label_col_i}_{x}', annotator_nums))
        annotation_data = annotation_data.assign(**{
            label_col_i: annotation_data.loc[:, annotator_label_cols_i].apply(lambda x: list(filter(lambda y: not np.isnan(y), x.tolist())), axis=1)
        })
        # which questions have 0 annotators?
        logging.info(f'label col = {label_col_i} which has 0 annotators: {annotation_data[annotation_data.loc[:, label_col_i].apply(lambda x: len(x) == 0)]}')
        annotation_data = annotation_data.assign(**{
            f'{label_col_i}_num_0': annotation_data.loc[:, label_col_i].apply(lambda x: x[0]),
            f'{label_col_i}_num_1': annotation_data.loc[:, label_col_i].apply(lambda x: assign_next_annotation(x)),
        })
    logging.info(f'reorganized annotations:\n{annotation_data.head()}')
    # in case some annotators didn't finish all questions, remove unfinished rows
    num_annotators_per_question = 2
    annotation_cols = [f'{label_col}_num_{i}' for label_col in label_cols for i in range(num_annotators_per_question)]
    cutoff_annotation_data = annotation_data.dropna(subset=annotation_cols, axis=0, how='any')
    logging.info(f'{cutoff_annotation_data.shape[0]}/{annotation_data.shape[0]} after removing invalid annotations')
    if (cutoff_annotation_data.shape[0] < annotation_data.shape[0]):
        annotation_data = cutoff_annotation_data.copy()
    return annotation_data, annotator_files, label_cols


def compute_agreement(annotation_data, annotation_dir, annotator_files, label_cols):
    from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
    import krippendorff
    for label_col_i in label_cols:
        label_col_matcher = re.compile(f'{label_col_i}_num_\d')
        annotator_cols_i = list(filter(lambda x: label_col_matcher.match(x) is not None, annotation_data.columns))
        label_data_i = annotation_data.loc[:, annotator_cols_i].values
        category_table_i, n_cat_i = aggregate_raters(label_data_i)
        kappa_i = fleiss_kappa(category_table_i)
        alpha_i = krippendorff.alpha(label_data_i.transpose())
        logging.info(f'label {label_col_i} has kappa={"{:.3f}".format(kappa_i)}; alpha={"{:.3f}".format(alpha_i)}')
    # without unsure answers
    agreement_scores = []
    for label_col_i in label_cols:
        label_col_matcher = re.compile(f'{label_col_i}_num_\d')
        annotator_cols_i = list(filter(lambda x: label_col_matcher.match(x) is not None, annotation_data.columns))
        label_data_i = annotation_data.loc[:, annotator_cols_i]
        clean_label_data_i = label_data_i[label_data_i.loc[:, annotator_cols_i].min(axis=1) != -1]
        logging.info(f'label {label_col_i} has {clean_label_data_i.shape[0]}/{label_data_i.shape[0]} clean values')
        category_table_i, n_cat_i = aggregate_raters(clean_label_data_i.values)
        kappa_i = fleiss_kappa(category_table_i)
        alpha_i = krippendorff.alpha(clean_label_data_i.transpose())
        # raw agreement
        total_agreement_i = clean_label_data_i.loc[:, annotator_cols_i].apply(lambda x: x.min() == x.max(), axis=1).sum() / clean_label_data_i.shape[0]
        logging.info(f'without unsure labels: label {label_col_i} has kappa={"{:.3f}".format(kappa_i)}; alpha={"{:.3f}".format(alpha_i)}')
        agreement_scores.append([label_col_i, kappa_i, alpha_i, total_agreement_i])
    agreement_scores = pd.DataFrame(agreement_scores, columns=['label', 'kappa', 'alpha', 'raw_agreement'])
    # write to file
    agreement_score_file = os.path.join(annotation_dir, 'agreement_scores.tsv')
    agreement_scores.to_csv(agreement_score_file, sep='\t', index=False)
    ## compute pairwise agreement
    num_annotators = len(annotator_files)
    pairwise_annotator_scores = []
    for label_col_i in label_cols:
        logging.info(f'**** pairwise agreement label={label_col_i} ****')
        for j in range(num_annotators):
            label_col_j = f'{label_col_i}_{j}'
            for k in range(j + 1, num_annotators):
                label_col_k = f'{label_col_i}_{k}'
                # remove nan labels
                clean_label_data_j_k = annotation_data.dropna(subset=[label_col_j, label_col_k], axis=0, how='any')
                agreement_j_k = compute_pairwise_annotator_agreement(clean_label_data_j_k.loc[:, [label_col_j, label_col_k]])
                raw_agreement_j_k = (annotation_data.loc[:, label_col_j] == annotation_data.loc[:, label_col_k]).sum() / annotation_data.shape[0]
                logging.info(f'annotator {j} vs. annotator {k}: agreement={"{:.3f}".format(raw_agreement_j_k)}; kappa={"{:.3f}".format(agreement_j_k)}')

def main():
    parser = ArgumentParser()
    parser.add_argument('annotation_dir') # '../../data/reddit_data/annotation_data/round_2_annotation_sample/test_data/'
    parser.add_argument('no_label_data') # advice_subreddit_no_filter_comment_question_data_annotation_sample.tsv
    args = vars(parser.parse_args())
    annotation_dir = args['annotation_dir']
    no_label_data = args['no_label_data']
    logging.basicConfig(filename=f'../../logs/process_annotated_data.txt',
                        filemode='w', format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.INFO)

    ## load data
    annotation_data, annotator_files, label_cols = organize_data(annotation_dir, no_label_data)
    ## compute agreement
    compute_agreement(annotation_data, annotation_dir, annotator_files, label_cols)
    ## preprocess text
    annotation_data = preprocess_text(annotation_data)
    ## add majority label
    annotation_data = assign_majority_label(annotation_data)
    ## test overlap
    test_post_question_overlap(annotation_data, annotation_dir)
    ## test relevance
    test_question_relevance(annotation_data, annotation_dir)

if __name__ == '__main__':
    main()