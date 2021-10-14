import os
import re
from typing import Any, Dict, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from torch import nn
from torch.cuda.amp import autocast

from transformers import Trainer as HFTrainer
from transformers.file_utils import is_apex_available, is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

if is_apex_available():
    from apex import amp

from model_helpers import label_smoothed_nll_loss

class Trainer(HFTrainer):
    def __init__(self, label_smoothing: float = 0, accelerator = None, **kwargs):
        super().__init__(**kwargs)
        ## TODO: figure out amp to enable fp16 data encoding
        #if(is_apex_available()):
        #    self.model = amp.initialize(self.model, self.optimizer)
        self.label_smoothing = label_smoothing
        self.accelerator = accelerator
    
    # override to support label smoothing
    def _training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)


        # Our model outputs do not work with DataParallel, so forcing return tuple.
        if isinstance(model, nn.DataParallel):
            inputs["return_tuple"] = True

        if self.label_smoothing == 0:
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        else:
            labels = inputs.pop("labels")
            labels[labels == -100] = model.config.pad_token_id
            outputs = model(**inputs)
            lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.label_smoothing, ignore_index=model.config.pad_token_id
            )

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # parallel loss
        if(self.accelerator is not None):
            self.accelerator.backward(loss)
        elif self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def plot_grad_flow(self, named_parameters, path, step):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class / vanilla training loop after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters(), output_path, step)" to visualize the gradient flow
        and save a plot in the output path as a series of .png images.
        Adapted from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
        '''
        plt_file = os.path.join(path, f"{step:04d}.png")
        # debug: only display reader-aware attention and generic attention
        reader_generic_attn_matcher = re.compile('self_attn_per_group|self_attn_general')
        if(not os.path.exists(plt_file)):
            ave_grads = []
            max_grads = []
            layers = []
            for n, p in named_parameters:
                if ((p.requires_grad) and ("bias" not in n) and (p.grad is not None) and (reader_generic_attn_matcher.search(n) is not None)):
                    layers.append(n)
                    ave_grads.append(p.grad.abs().mean())
                    max_grads.append(p.grad.abs().max())
            plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1,
                    color="c")
            plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=1,
                    color="b")
            plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
            plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
            plt.xlim(left=0, right=len(ave_grads))
            plt.ylim(bottom=1e-8, top=20)  # zoom in on the lower gradient regions
            plt.yscale("log")
            plt.xlabel("Layers")
            plt.ylabel("average gradient")
            plt.title(f"Gradient flow; step={step:04d}")
            plt.grid(True)
            plt.legend([Line2D([0], [0], color="c", lw=4),
                        Line2D([0], [0], color="b", lw=4),
                        Line2D([0], [0], color="k", lw=4)],
                       ['max-gradient', 'mean-gradient', 'zero-gradient'])
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            plt.savefig(plt_file, bbox_inches='tight', dpi=200)
            fig.clear()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        UPDATE: plots gradient after back-prop

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.use_amp else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        ## visualize gradient
        if(self.args.plot_gradient):
            step = self.state.global_step ## TODO: which step will increase every time we call training step??
            output_path = os.path.join(self.args.output_dir, 'gradient_plots/')
            if (not os.path.exists(output_path)):
                os.mkdir(output_path)
            if(step % 100 == 0):
                self.plot_grad_flow(self.model.named_parameters(), output_path, step)

        return loss.detach()