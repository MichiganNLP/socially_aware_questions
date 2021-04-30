import gzip
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
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

def generate_predictions(model, data, tokenizer, device_name='cuda:0',
                         generation_method='beam_search', num_beams=4,
                         temperature=1.0, top_p=1.0, model_kwargs=[]):
    """
    Generate predicted text from transformer model.

    :param model:
    :param data:
    :param device_name:
    :return:
    """
    max_decoding_length = 64
    length_penalty = 1
    device = torch.device(device_name)
    model.to(device)
    pred_text = []
    for batch_i in tqdm(data):
        source_i = batch_i['source_ids']
        attention_i = batch_i['attention_mask']
        # handle model kwargs: reader tokens, embeddings, etc.
        model_kwargs_i = prepare_model_kwargs_for_generation(batch_i, model_kwargs)
        # print(f'model kwargs after type fix {model_kwargs_i}')
        # fix type in case of difference
        if(type(source_i) is list):
            source_i = torch.LongTensor(source_i)
        if(type(attention_i) is list):
            attention_i = torch.Tensor(attention_i)
        if(generation_method == 'beam_search'):
            output_i = model.generate(
                input_ids=source_i.to(device).reshape(1,-1),
                attention_mask=attention_i.to(device).reshape(1,-1),
                num_beams=num_beams,
                temperature=temperature,
                max_length=max_decoding_length,
                length_penalty=length_penalty,
                num_return_sequences=1,
                **model_kwargs_i
            )
        elif(generation_method == 'sample'):
            output_i = model.generate(
                input_ids=source_i.to(device).reshape(1, -1),
                attention_mask=attention_i.to(device).reshape(1, -1),
                temperature=temperature,
                top_p=top_p,
                max_length=max_decoding_length,
                length_penalty=length_penalty,
                num_return_sequences=1,
                do_sample=True,
            )
        prediction = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_i]
        pred_text.extend(prediction)
    return pred_text

def prepare_model_kwargs_for_generation(data, model_kwargs):
    model_kwargs_i = {
        model_kwarg: data[model_kwarg]
        for model_kwarg in model_kwargs
    }
    # fix type, shape of model kwargs
    # tmp debugging
    # print(f'model kwargs before type fix {model_kwargs_i}')
    # e.g. int => Tensor(int)
    model_kwargs_i.update({
        kwarg: [kwarg_val]
        for kwarg, kwarg_val in model_kwargs_i.items()
        if (type(kwarg_val) is not list and type(kwarg_val) is not torch.Tensor)
    })
    model_kwargs_i.update({
        int_kwarg: torch.LongTensor([kwarg_val]).reshape(1, -1)
        for int_kwarg, kwarg_val in list(filter(lambda x: type(x[1]) is list and type(x[1][0]) is int, model_kwargs_i.items()))
    })
    model_kwargs_i.update({
        float_kwarg: torch.Tensor(kwarg_val).reshape(1, -1)
        for float_kwarg, kwarg_val in list(filter(lambda x: type(x[1]) is list and type(x[1][0]) is float, model_kwargs_i.items()))
    })
    return model_kwargs_i


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
