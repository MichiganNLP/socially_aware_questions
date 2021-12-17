from nltk.tokenize.casual import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
tqdm.pandas()
# transformers overhead
from datasets import Dataset
from transformers import Trainer
from torch.utils.data import DataLoader
from typing import Optional, List
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize, EvalPrediction
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify, IterableDataset, IterableDatasetShard, nested_truncate
from transformers.deepspeed import deepspeed_init
import collections
from transformers.utils import logging
logger = logging.get_logger(__name__)
if is_torch_tpu_available():
    import torch_xla.distributed.parallel_loader as pl
import torch
import numpy as np
import pandas as pd

def keyword_score(pred, labels):
    pred = set(pred)
    labels = set(labels)
    TP = pred & labels
    FP = pred - labels
    FN = labels - pred
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    F1 = (prec + rec) / 2.
    return prec, rec, F1

def update_duplicate_labels(pred_scores, labels, inputs, ignore_val=-100):
    # remove scores for duplicate input
    # for 1-label only keep max score; for 0-label, only keep min score
    norm_pred_scores = torch.softmax(pred_scores, axis=1)
    print(f'norm scores = {norm_pred_scores}')
    drop_idx = []
    for input_i in list(set(inputs)):
        idx_i = np.where(inputs==input_i)[0]
        if(len(idx_i) > 1):
            labels_i = labels[idx_i]
            # get relevant idx in order of pred score
            high_score_idx_i = [x for x in norm_pred_scores[:, 1].argsort(descending=True).tolist() if x in idx_i]
            # 1-label: remove scores below max-score
            if(any(labels_i==1)):
                drop_idx.extend(high_score_idx_i[1:])
            # 0-label: remove scores above min-score
            else:
                drop_idx.extend(high_score_idx_i[:-1])
#     print(f'drop idx = {drop_idx}')
    fix_labels = np.array(labels)
    fix_labels[drop_idx] = ignore_val
#     print(f'valid idx = {valid_idx}')
    ## TODO: if this throws error in loss because of reindex, we will re-assign scores
    return fix_labels
class KeywordLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
    # override https://github.com/huggingface/transformers/blob/bef1e3e4a00bd0863f804ba0a4e05dc77676a341/src/transformers/trainer.py#compute_loss
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # tmp debugging
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            input_ids = inputs['input_ids']
            labels = inputs['labels']
            # update labels
            clean_labels = []
            for idx_j in range(len(input_ids)):
                outputs_j = outputs[idx_j]
                pred_scores_j = outputs.logits[idx_j, :, :]
                input_j = input_ids[idx_j].tolist()
                labels_j = labels[idx_j].tolist()
                # assign -100 to duplicate labels
                labels_j = update_duplicate_labels(pred_scores_j, labels_j, input_j, ignore_val=-100)
                clean_labels.append(labels_j)
            labels = torch.cat(clean_labels)
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # get the ids for words that model output
        if(return_outputs):
            return loss, outputs
        else:
            return loss
    # custom eval function
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.
        # need all inputs for evaluation
        all_inputs = []

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None
            all_inputs.append(inputs)
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), all_inputs)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    def compute_metrics(p):
        predictions, labels, inputs = p
        predictions = np.argmax(predictions, axis=2)
        pred = [i for i,o in zip(inputs, predictions) if o==1]
        prec, rec, F1 = keyword_score(pred, labels)
        return {
            'precision' : prec,
            'rec' : rec,
            'F1' : F1
        }

# tokenize, etc.
def convert_keyword_to_output_label(input_tokens, keywords):
    output_labels = pd.Series(input_tokens).isin(keywords).astype(int).values
    return output_labels

def build_dataset(tokenizer, inputs, outputs, max_length=100):
    input_token_data = tokenizer(inputs, padding=True, max_length=max_length)
    input_tokens = [tokenizer.convert_ids_to_tokens(x) for x in input_token_data['input_ids']]
    output_ids = list(map(lambda x: convert_keyword_to_output_label(x[0], x[1]), zip(input_tokens, outputs)))
    data_dict = {
    'input_ids' : input_token_data['input_ids'],
    'attention_mask' : input_token_data['attention_mask'],
    'labels' : output_ids,
    }
    dataset = Dataset.from_dict(data_dict)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

def extract_keywords_from_data(data, tokenizer=None):
    if(tokenizer is None):
        tokenizer = TweetTokenizer()
    # lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    # get stops and stop lemmas!
    punct = list(',!/&|-:\'"()[]\\/’?.') + ['...']
    stops = stopwords.words('english')
    stops += punct
    stops += list(set(map(stemmer.stem, stops)))
    stops = set(stops)
    data = data.assign(**{
        'title_tokens': data.loc[:, 'title'].progress_apply(lambda x: tokenizer.tokenize(x)),
        'question_tokens': data.loc[:, 'reply_question'].progress_apply(
            lambda x: list(map(lambda y: stemmer.stem(y), tokenizer.tokenize(x)))),
    })
    # stem words
    data = data.assign(**{
        'title_tokens_clean': data.loc[:, 'title_tokens'].progress_apply(lambda x: list(map(lambda y: stemmer.stem(y), x))),
        'question_tokens_clean': data.loc[:, 'question_tokens'].progress_apply(
            lambda x: list(map(lambda y: stemmer.stem(y), x))),
    })
    ## get overlap
    data = data.assign(**{
        'title_question_overlap': data.progress_apply(
            lambda x: (set(x.loc['title_tokens_clean']) & set(x.loc['question_tokens_clean'])) - stops, axis=1)
    })
    ## match overlap tokens with original post tokens
    data = data.assign(**{
        'title_question_overlap_clean': data.progress_apply(
            lambda x: [t1 for t1, t2 in zip(x.loc['title_tokens'], x.loc['title_tokens_clean']) if t2 in x.loc['title_question_overlap']], axis=1)
    })
    return data