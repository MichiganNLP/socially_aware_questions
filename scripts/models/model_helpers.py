import os
import pickle
import re
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding, BartTokenizer, BartConfig, \
    AutoModelForSeq2SeqLM
from transformers.training_args import TrainingArguments
from rouge_score.rouge_scorer import RougeScorer
from nltk.tokenize import WordPunctTokenizer
from dataclasses import field
from typing import Optional
from sklearn.metrics.pairwise import cosine_distances

from author_aware_model import AuthorTextGenerationModel
from author_group_attention_model import AuthorGroupAttentionModelConditionalGeneration
from scripts.data_processing.data_helpers import load_vectors


class DataArguments(TrainingArguments):
    train_file_path: str = field(
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: str = field(
        metadata={"help": "Path for cached valid dataset"},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path for data files"},
    )
    task: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which task 'qa', 'qg', 'e2e_qg', 'ans_ext', 'multi'. 'multi' means 'qa', 'qg', 'ans_ext' tasks"},
    )
    qg_format: Optional[str] = field(
        default='prepend_qg_format',
        metadata={"help": "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )
    n_gpu: Optional[int] = field(
        default=1,
    )

def generate_predictions(model, data, tokenizer,
                         generation_params=[],
                         model_kwargs=[],
                         generate_classify_tools=None):
    """
    Generate predicted text from transformer model.

    :param model:
    :param data:
    :return:
    """
    # generation_method = 'beam_search', num_beams = 4,
    # temperature = 1.0, top_p = 1.0,
    max_decoding_length = 64
    length_penalty = 1
    # device = torch.device(device_name)
    device = torch.cuda.current_device()
    model.to(device)
    pred_text = []
    generation_method = generation_params['generation_method']
    if(generate_classify_tools is not None):
        (model_classifiers, sentence_encoder, pca_question_model, pca_post_model) = generate_classify_tools
        reader_group_class_lookup = {
            '<EXPERT_PCT_0_AUTHOR>' : 0,
            '<EXPERT_PCT_1_AUTHOR>': 1,
            '<RESPONSE_TIME_0_AUTHOR>' : 0,
            '<RESPONSE_TIME_1_AUTHOR>': 1,
            '<US_AUTHOR>': 1,
            '<NONUS_AUTHOR>': 0,
        }
        reader_group_category_lookup = {
            'location_region' : ['<US_AUTHOR>', '<NONUS_AUTHOR>'],
            'expert_pct_bin' : ['<EXPERT_PCT_0_AUTHOR>', '<EXPERT_PCT_1_AUTHOR>'],
            'relative_time_bin' : ['<RESPONSE_TIME_0_AUTHOR>', '<RESPONSE_TIME_1_AUTHOR>'],
        }
        reader_group_category_lookup = {v : k for k,vs in reader_group_category_lookup.items() for v in vs}
    # for decoder-modified model, need to add extra attention mask
    # rename_kwargs = {}
    # if(type(model) is AuthorGroupAttentionModelConditionalGeneration and model.config.__dict__['reader_group_attention_location']=='decoder'):
    #     rename_kwargs['attention_mask'] = 'decoder_attention_mask'
    for batch_i in tqdm(data):
        source_i = batch_i['source_ids']
        attention_i = batch_i['attention_mask']
        # fix type in case of difference
        if (type(source_i) is list):
            source_i = torch.LongTensor(source_i)
        if (type(attention_i) is list):
            attention_i = torch.Tensor(attention_i)
        source_i = source_i.unsqueeze(0).to(device)
        attention_i = attention_i.unsqueeze(0).to(device)
        # handle model kwargs: reader tokens, embeddings, etc.
        model_kwargs_i = prepare_model_kwargs_for_generation(batch_i, model_kwargs)
        # tmp debugging
        #print(f'model kwargs = {model_kwargs_i}')
        #if ('author_embeds' in model_kwargs_i):
            # model_kwargs_i['author_embeds'] =model_kwargs_i['author_embeds'].unsqueeze(0)
            # source_i = source_i.unsqueeze(0)
            # attention_i = attention_i.unsqueeze(0)
        #    print(f'author embed data shape={model_kwargs_i["author_embeds"].shape}')
        #    print(f'input ids shape={source_i.shape}')
        #    print(f'input ids  {source_i.cpu().numpy()}')
        # print(f'model kwargs after type fix has type: {model_kwargs_i["author_embeds"].dtype}')
        if(generation_method == 'beam_search'):
                output_i = model.generate(
                input_ids=source_i,
                attention_mask=attention_i,
                num_beams=generation_params['num_beams'],
                max_length=max_decoding_length,
                length_penalty=length_penalty,
                num_return_sequences=1,
                output_attentions=True,
                **model_kwargs_i
            )
        elif(generation_method == 'sample'):
            # tmp debugging
            # if('author_embeds' in model_kwargs_i):
            #     print(f'author embed data = {model_kwargs_i["author_embeds"].shape}')
            #     print(f'input ids = {source_i.shape}')
            #     print(f'input ids  {source_i}')
            num_return_sequences = 1
            num_beams = None
            if(generate_classify_tools is not None and batch_i['reader_token'] != 'UNK'):
                num_return_sequences = 10
                num_beams = 1
            # tmp debugging
            # tmp debugging: do regular pass
            #print(f'about to do regular forward pass as test; source shape = {source_i.shape}')
            #test_output_i = model(input_ids=source_i, attention_mask=attention_i, **model_kwargs_i)
            #print('about to generate')
            #print(f'pre-generate attention mask={attention_i}; source shape = {source_i.shape}')
            output_i = model.generate(
                input_ids=source_i,
                attention_mask=attention_i,
                temperature=generation_params['temperature'],
                top_p=generation_params['top_p'],
                max_length=max_decoding_length,
                length_penalty=length_penalty,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                do_sample=True,
                output_attentions=True,
                output_hidden_states=True,
                **model_kwargs_i
            )
            ## for generate-classify model, rerank generated text based on P(class | text)
            # print(f'reader token = {batch_i["reader_token_str"]}')
            if(generate_classify_tools is not None and batch_i['reader_token'] != 'UNK'):
                # print(f'output before sorting = {output_i}')
                # print(f'classify generation results')
                # encode post, question
                source_txt_i = batch_i['source_text']
                output_txt_i = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(x.squeeze(0), skip_special_tokens=True)) for x in output_i]
                # output_txt_i = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(output_i, skip_special_tokens=True))
                output_txt_i = pd.DataFrame([output_txt_i, output_i], index=['output_txt', 'output_ids']).transpose()
                output_txt_i.drop_duplicates('output_txt', inplace=True)
                source_txt_embed_i = sentence_encoder.encode(source_txt_i, device=torch.cuda.current_device())
                output_txt_embed_i = sentence_encoder.encode(output_txt_i.loc[:, 'output_txt'].values.tolist(), device=torch.cuda.current_device())
                # print(f'source text embed has shape {source_txt_embed_i.shape[0]}')
                source_txt_embed_i = np.repeat(source_txt_embed_i.reshape(1,-1), output_txt_i.shape[0], axis=0)
                # print(f'source text embed repeat has shape {source_txt_embed_i.shape[0]}')
                source_txt_embed_i = pca_post_model.transform(source_txt_embed_i)
                output_txt_embed_i = pca_question_model.transform(output_txt_embed_i)
                txt_embed_i = np.hstack([source_txt_embed_i, output_txt_embed_i])
                # classify according to the reader group
                reader_group_i = batch_i['reader_token']
                reader_group_category_i = reader_group_category_lookup[reader_group_i]
                reader_group_class_i = reader_group_class_lookup[reader_group_i]
                model_classifier_i = model_classifiers[reader_group_category_i]
                group_probs_i = model_classifier_i.predict_proba(txt_embed_i)
                group_probs_i = pd.DataFrame(group_probs_i, columns=[0,1]).assign(**{'output_ids' : output_txt_i.loc[:, 'output_ids'].values})
                # get output IDs with highest score for reader group
                group_probs_i.sort_values(reader_group_class_i, ascending=False, inplace=True)
                # print(f'group probs = {group_probs_i.head()}')
                output_i = [group_probs_i.iloc[0, :].loc['output_ids']]
                # print(f'output after sorting = {output_i}')
        prediction = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_i]
        pred_text.extend(prediction)
    return pred_text

def prepare_model_kwargs_for_generation(data, model_kwargs):
    model_kwargs = {
        model_kwarg: data[model_kwarg]
        for model_kwarg in model_kwargs
    }
    # optional: rename kwargs (e.g. "attention_mask" => "decoder_attention_mask")
    # model_kwargs.update({v : data[k] for k,v in rename_kwargs})
    # fix type, shape of model kwargs
    # tmp debugging
    # print(f'model kwargs before type fix {model_kwargs_i}')
    # e.g. int => Tensor(int)
    model_kwargs.update({
        kwarg: [kwarg_val]
        for kwarg, kwarg_val in model_kwargs.items()
        if (type(kwarg_val) is not list and type(kwarg_val) is not torch.Tensor)
    })
    # fix lists
    model_kwargs.update({
        int_kwarg: torch.LongTensor([kwarg_val]).unsqueeze(0).unsqueeze(0)
        for int_kwarg, kwarg_val in list(filter(lambda x: type(x[1]) is list and type(x[1][0]) is int, model_kwargs.items()))
    })
    model_kwargs.update({
        float_kwarg: torch.Tensor(kwarg_val).unsqueeze(0).unsqueeze(0)
        for float_kwarg, kwarg_val in list(filter(lambda x: type(x[1]) is list and type(x[1][0]) is float, model_kwargs.items()))
    })
    # fix tensors
    model_kwargs.update({
        kwarg : kwarg_val.unsqueeze(0).to(torch.cuda.current_device())
        for kwarg, kwarg_val in model_kwargs.items()
        if type(kwarg_val) is torch.Tensor and kwarg_val.dim()==1
    })
    # fix data type (convert double to float to match model weights)
    # tmp debugging
    # print(f'data type before {model_kwargs["author_embeds"].dtype}')
    model_kwargs.update({
        kwarg: kwarg_val.float()
        for kwarg, kwarg_val in model_kwargs.items()
        if type(kwarg_val) is torch.Tensor and kwarg_val.dtype is torch.float64
    })
    # print(f'data type after {model_kwargs["author_embeds"].dtype}')
    return model_kwargs


def compute_text_bleu(txt_1, txt_2, weights):
    score = sentence_bleu([txt_1], txt_2, weights=weights)
    return score

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

def test_question_overlap(pred_data, test_data, word_embed_file=None, stop_words=[], tokenizer=None):
    text_overlap_scores = []
    bleu_weights = [1.0, 0., 0., 0.]  # 100% 1-grams, 0% 2-grams, etc.
    rouge_scorer = RougeScorer(['rougeL'], use_stemmer=True)
    sentence_embed_model = load_sentence_embed_model()
    score_cols = ['BLEU-1', 'ROUGE-L', 'sentence_dist']
    # answerability coefficients
    # ner_weight, qt_weight, re_weight, delta, ngram_metric = (0.6, 0.2, 0.1, 0.7, 'Bleu_3') # default coefficients
    ner_weight, qt_weight, re_weight, delta, ngram_metric = (0.41, 0.20, 0.36, 0.66, 'Bleu_1') # optimal coefficients for SQUAD
    if(tokenizer is None):
        tokenizer = WordPunctTokenizer()
    # load embeddings for word mover distance
    if(word_embed_file is not None):
        word_embeds = load_vectors(word_embed_file)
        score_cols.append('WMD')
        word_embed_vocab = set(word_embeds.index) - stop_words
        # tmp debugging
    # print(f'mid pred data sample = ({pred_data[:10]}):({pred_data[-10:]})')
    for i, x in enumerate(pred_data):
        try:
            tokenizer.tokenize(x)
        except Exception as e:
            print(f'bad pred data (i={i}) = {x}')
            break
    # print(f'late pred data sample = ({pred_data[:10]}):({pred_data[-10:]})')
    for test_data_i, pred_data_i in tqdm(zip(test_data['target_text'], pred_data)):
        # get tokens first
        pred_tokens_i = list(map(lambda x: x.lower(), tokenizer.tokenize(pred_data_i)))
        test_tokens_i = list(map(lambda x: x.lower(), tokenizer.tokenize(test_data_i)))
        bleu_score_i = compute_text_bleu(test_tokens_i, pred_tokens_i, weights=bleu_weights)
        rouge_score_data_i = rouge_scorer.score(test_data_i, pred_data_i)
        rouge_score_i = rouge_score_data_i['rougeL'].fmeasure
        sentence_embeds_i = sentence_embed_model.encode([test_data_i, pred_data_i])
        sentence_embed_dist_i = cosine_distances(sentence_embeds_i)[1][0]
        generation_scores_i = [bleu_score_i, rouge_score_i, sentence_embed_dist_i]
        if(word_embed_file is not None):
            # tokenize/normalize data
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
    ## also add answerability; we do it separately because it's bad to do in serial
    # target_text = [x['target_text'] for x in test_data]
    # tmp debugging
    # print(f'target text N={len(target_text)}')
    # print(f'pred data N={len(pred_data)}')
    # question answerability scores
    # answerability_scores, fluent_scores = get_answerability_scores(pred_data, ner_weight, qt_weight, re_weight, target_text, ngram_metric=ngram_metric, delta=delta, return_all_scores=False)
    # output_dir = 'tmp/'
    # answerability_scores, fluent_scores = get_answerability_scores(pred_data, ner_weight, qt_weight, re_weight, target_text, ngram_metric=ngram_metric, delta=delta, return_all_scores=False, output_dir=output_dir)
    # tmp debugging
    # print(f'answerability scores = {answerability_scores}')
    ## TODO: how to get answerability scores for each Q separately??
    # generation_score_data = generation_score_data.assign(**{
    #     'answer_score' : answerability_scores,
    # })
    return generation_score_data

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        bs = pad_mask.long().sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        bs = lprobs.shape[0]

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss / bs, nll_loss / bs


class BasicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # item = {k: torch.Tensor(v[idx]) for k, v in self.encodings.items()}
        item = {
            'input_ids' : torch.LongTensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.Tensor(self.encodings['attention_mask'][idx]),
        }
        item["labels"] = torch.LongTensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

def select_from_dataset(dataset, idx):
    dataset.encodings = BatchEncoding({
        'input_ids': [dataset.encodings['input_ids'][x] for x in idx],
        'attention_mask' : [dataset.encodings['attention_mask'][x] for x in idx],
        })
    dataset.labels = [dataset.labels[x] for x in idx]
    return dataset


def load_model(model_cache_dir, model_file, model_type, data_dir):
    """
    Load model and tokenizer from file; need this for generation!

    :param model_cache_dir:
    :param model_file:
    :param model_type:
    :param data_dir:
    :return:
    """
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
    # add extra token for author embeds
    if(model_type == 'bart_author_embeds'):
        # add extra token to tokenizer
        model_tokenizer.add_tokens({'<AUTHOR_EMBED>': len(model_tokenizer)}, special_tokens=True)
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
    # fix model device
    device = torch.cuda.current_device()
    generation_model.to(device)
    return generation_model, model_tokenizer

def load_sentence_embed_model():
    sentence_embed_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    return sentence_embed_model


def subsample_data_by_class(class_count, class_var, data):
    data_class_counts = data.loc[:, class_var].value_counts()
    if (class_count == 'min'):
        data_class_count_base = data_class_counts.min()
        data_class = data_class_counts.sort_values(ascending=True).index[0]
    elif (class_count == 'max'):
        data_class_count_base = data_class_counts.max()
        data_class = data_class_counts.sort_values(ascending=False).index[0]
    # tmp debugging
    # print(f'data class counts = {data_class_counts}')
    data = pd.concat(
        [
            data[data.loc[:, class_var] == data_class],
            data[data.loc[:, class_var] != data_class].sample(data_class_count_base, replace=(class_count == 'max'), random_state=123)],
        axis=0)
    return data

def resample_by_class(data, class_var='author_group', class_count='min', subgroup_var=None):
    """
    Resample data by min/max of variable distribution.

    :param data:
    :param class_var:
    :param class_count:
    :return:
    """
    if(subgroup_var is not None):
        subgroup_data = []
        for subgroup_i, data_i in data.groupby(subgroup_var):
            data_i = subsample_data_by_class(class_count, class_var, data_i)
            subgroup_data.append(data_i)
        data = pd.concat(subgroup_data, axis=0)
    else:
        data = subsample_data_by_class(class_count, class_var, data)
    return data


def train_test_reader_group_classification(data, class_var='reader_group_class',
                                           text_var='PCA_question_encoded', post_var='PCA_post_encoded',
                                           subgroup_var='subreddit'):
    """
    Train/test reader group classification based on encoding
    of question and post data.

    :param data:
    :param text_var:
    :param post_var:
    :return:
    """
    # non_default_reader_group_class = list(set(Y) - {default_reader_group_class})[0]
    # combine text and post var
    if(post_var is not None):
        X = np.hstack([np.vstack(data.loc[:, text_var].values), np.vstack(data.loc[:, post_var].values)])
    else:
        X = np.vstack(data.loc[:, text_var].values)
    #     X = np.vstack(data.loc[:, text_var].values)
    layer_size = X.shape[1]
    # assume reader group var is already binarized etc.
    Y = data.loc[:, class_var].values
    # fit models across all folds
    model_scores = []
    if(subgroup_var is not None):
        subgroups = data.loc[:, subgroup_var].unique()
    data = data.assign(**{'idx': list(range(data.shape[0]))})
    parent_id_i = data.loc[:, 'parent_id'].unique()
    train_pct = 0.8
    train_N_i = int(len(parent_id_i) * train_pct)
    n_folds = 10
    max_train_iter = 1000
    for j in tqdm(range(n_folds)):
        # split by parent ID
        train_id_j = set(np.random.choice(parent_id_i, train_N_i, replace=False))
        test_id_j = list(set(parent_id_i) - train_id_j)
        train_idx = np.where(data.loc[:, 'parent_id'].isin(train_id_j))[0]
        test_idx = np.where(data.loc[:, 'parent_id'].isin(test_id_j))[0]
        # resample data to avoid class distribution imbalance
        train_data = data.iloc[train_idx, :]
        train_data = resample_by_class(train_data, class_var=class_var, class_count='max', subgroup_var=subgroup_var)
        train_idx = train_data.loc[:, 'idx'].values
        # tmp debug
        # print(f'train data has class distribution {train_data.loc[:, class_var].value_counts()}')
        test_data = data.iloc[test_idx, :]
        test_data = resample_by_class(test_data, class_var=class_var, class_count='max', subgroup_var=subgroup_var)
        test_idx = test_data.loc[:, 'idx'].values
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        # tmp debug
        # print(f'Y train has class distribution {pd.Series(Y_train).value_counts()}')
        # print(f'Y test has class distribution {pd.Series(Y_test).value_counts()}')
        # fit model
        model = MLPClassifier(hidden_layer_sizes=[layer_size, ],
                              activation='relu', max_iter=max_train_iter,
                              random_state=123)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        Y_prob = model.predict_proba(X_test)
        model_acc = (Y_pred == Y_test).sum() / len(Y_test)
        # get F1 for both classes...there must be a better way to do this
        model_f1_class_1 = f1_score(Y_pred, Y_test)
        model_f1_class_0 = f1_score((1 - Y_pred), (1 - Y_test))
        model_f1_macro = f1_score(Y_pred, Y_test, average='macro')
        model_auc = roc_auc_score(Y_test, Y_prob[:, 1])
        model_scores_j = {
            'model_acc': model_acc,
            f'F1_class=1': model_f1_class_1,
            f'F1_class=0': model_f1_class_0,
            'F1_macro': model_f1_macro,
            'AUC': model_auc,
            'fold': j}
        ## get scores per subreddit!!
        if(subgroup_var is not None and len(subgroups) > 1):
            for subgroup_k in subgroups:
                idx_k = list(set(np.where(data.loc[:, 'subreddit'] == subgroup_k)[0]) & set(test_idx))
                if (len(idx_k) > 0):
                    Y_pred_k = model.predict(X[idx_k, :])
                    # tmp debug
                    print(f'subreddit {subgroup_k} has class distribution {pd.Series(Y_pred_k).value_counts()}')
                    model_acc_k = (Y[idx_k] == Y_pred_k).sum() / len(idx_k)
                    model_scores_j[f'model_acc_{subgroup_k}'] = model_acc_k
            model_scores_j['model_acc_subreddit_mean'] = np.mean([model_scores_j[f'model_acc_{subreddit_k}'] for subreddit_k in subreddits])
        #         print(f'model scores = {model_scores_j}')
        model_scores.append(model_scores_j)
    model_scores = pd.DataFrame(model_scores)
    return model_scores

def load_reader_group_classifiers(model_dir, reader_groups, subreddits):
    subreddit_group_model_lookup = {}
    default_class_matcher = re.compile('(?<=class1\=).*(?=\.pkl)')
    for subreddit_i in subreddits:
        model_dir_i = os.path.join(model_dir, subreddit_i)
        # load model
        for group_var_j in reader_groups:
            model_file_matcher_i = re.compile(f'.*group={group_var_j}.*\.pkl')
            model_file_i = list(filter(lambda x: model_file_matcher_i.match(x) is not None, os.listdir(model_dir_i)))
            if (len(model_file_i) > 0):
                model_file_i = model_file_i[0]
                model_file_i = os.path.join(model_dir_i, model_file_i)
                # get default class
                default_class_i = default_class_matcher.search(
                    model_file_i).group(0)
                if (default_class_i.isdigit()):
                    default_class_i = literal_eval(default_class_i)
                model_i = pickle.load(open(model_file_i, 'rb'))
                subreddit_group_model_lookup[f'{subreddit_i};{group_var_j}'] = (
                model_i, default_class_i)
    return subreddit_group_model_lookup

def get_model_class_prob(data, models,
                         reader_group_other_class_lookup,
                         pred_var='PCA_question_post_encoded'):
    subreddit = data.loc["subreddit"]
    group_category = data.loc["group_category"]
    model, default_class = models[f'{subreddit};{group_category}']
    model_prob = model.predict_proba(data.loc[pred_var].reshape(1,-1))[0, :]
    max_class_idx = model_prob.argmax()
    max_class_prob = model_prob.max()
    if(max_class_idx == 1):
        max_class = default_class
    else:
        max_class = reader_group_other_class_lookup.loc[group_category][default_class]
    return max_class, max_class_prob