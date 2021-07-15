import gzip

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding
from transformers.training_args import TrainingArguments
from dataclasses import field
from typing import Optional

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

def load_vectors(embed_file):
    """
    Load word embedding vectors from file

    :param fname:
    :return:
    """
    data = {}
    for i, line in enumerate(gzip.open(embed_file, 'rt')):
        # skip first line
        if(i > 0):
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
    # convert to dataframe
    data = pd.DataFrame(data).transpose()
    return data

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
        model_classifiers, sentence_encoder, pca_question_model, pca_post_model = generate_classify_tools
        reader_group_class_lookup = {
            '<EXPERT_PCT_0_BIN>' : 0,
            '<EXPERT_PCT_1_BIN>': 1,
            '<RELATIVE_TIME_0_BIN>' : 1,
            '<RELATIVE_TIME_1_BIN>': 0,
            '<US_AUTHOR>': 0,
            '<NONUS_AUTHOR>': 1,
        }
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
                **model_kwargs_i
            )            
        elif(generation_method == 'sample'):
            # tmp debugging
            if('author_embeds' in model_kwargs_i):
                print(f'author embed data = {model_kwargs_i["author_embeds"].shape}')
                print(f'input ids = {source_i.shape}')
                print(f'input ids  {source_i}')
            num_return_sequences = 1
            num_beams = None
            if(generate_classify_tools is not None and batch_i['reader_token_str'] is not 'UNK'):
                num_return_sequences = 10
                num_beams = 1
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
                **model_kwargs_i
            )
            ## for generate-classify model, rerank generated text based on P(class | text)
            if(generate_classify_tools is not None and batch_i['reader_token'] is not 'UNK'):
                # encode post, question
                source_txt_i = batch_i['source_text']
                output_txt_i = tokenizer.convert_tokens_to_str(tokenizer.convert_ids_to_tokens(output_i, skip_special_tokens=True))
                output_txt_i = pd.DataFrame([output_txt_i, output_i], index=['output_txt', 'output_ids']).transpose()
                output_txt_i.drop_duplicates('output_txt', inplace=True)
                source_txt_embed_i = sentence_encoder.encode(source_txt_i, device=torch.cuda.current_device())
                output_txt_embed_i = sentence_encoder.encode(output_txt_i.loc[:, 'output_txt'], device=torch.cuda.current_device())
                source_txt_embed_i = np.repeat(source_txt_embed_i, len(output_txt_i), axis=0)
                source_txt_embed_i = pca_post_model.transform(source_txt_embed_i)
                output_txt_embed_i = pca_question_model.transform(output_txt_embed_i)
                txt_embed_i = np.hstack([source_txt_embed_i, output_txt_embed_i])
                # classify according to the reader group
                reader_group_i = batch_i['reader_token']
                reader_group_class_i = reader_group_class_lookup[reader_group_i]
                model_classifier_i = model_classifiers[reader_group_i]
                group_probs_i = model_classifier_i.predict_proba(txt_embed_i)
                group_probs_i = pd.DataFrame(group_probs_i, columns=[0,1]).assign(**{'output_ids' : output_txt_i.loc[:, 'output_ids']})
                # get output IDs with highest score for reader group
                group_probs_i.sort_values(reader_group_class_i, ascending=False, inplace=True)
                output_i = [group_probs_i.iloc[0, :].loc['output_ids']]
>>>>>>> 9bfaffad6109ecb12be11b8f9295d23718e840a5
        prediction = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_i]
        pred_text.extend(prediction)
    return pred_text

def prepare_model_kwargs_for_generation(data, model_kwargs):
    model_kwargs = {
        model_kwarg: data[model_kwarg]
        for model_kwarg in model_kwargs
    }
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
