"""
Author-group attention model,
i.e. different attention module
for different reader groups.
"""
## reader-specific encoder
import math
import warnings
from math import sqrt
import random
from typing import Optional, Tuple, Union, Callable, List, Iterable, Dict, Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BartConfig, BeamSearchScorer
from transformers.file_utils import ModelOutput
from transformers.generation_utils import GreedySearchOutput, SampleOutput, \
    BeamSearchOutput, BeamSampleOutput
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, \
    Seq2SeqModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.modeling_bart import BartAttention, ACT2FN, \
    BartLearnedPositionalEmbedding, BartEncoderLayer, _expand_mask, \
    BartPretrainedModel, BartEncoder, BartModel, BartDecoder, \
    BartForConditionalGeneration, \
    shift_tokens_right, BartDecoderLayer
import torch.nn.functional as F
from transformers.utils import logging
logger = logging.get_logger(__name__)

class AuthorGroupAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        reader_group_types : list = [],
        reader_group_weight: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        ## reader-specific attention types
        self.reader_group_weight = reader_group_weight
        self.reader_k_proj = nn.ModuleDict({
            reader_group_type : nn.Linear(embed_dim, embed_dim, bias=bias)
            for reader_group_type in reader_group_types
        })
        self.reader_q_proj = nn.ModuleDict({
            reader_group_type: nn.Linear(embed_dim, embed_dim, bias=bias)
            for reader_group_type in reader_group_types
        })
        self.reader_v_proj = nn.ModuleDict({
            reader_group_type: nn.Linear(embed_dim, embed_dim, bias=bias)
            for reader_group_type in reader_group_types
        })

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        reader_token: list = [],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get reader-specific probs
        reader_attn_probs = []
        # tmp debugging
        # print(f'attention mask has shape = {attention_mask.shape}')
        # print(f'hidden states has shape = {hidden_states.shape}')
        # if(key_value_states is not None):
        #     print(f'key value states has shape = {key_value_states.shape}')
        # if(past_key_value is not None):
        #     print(f'past key value has shape = {past_key_value.shape}')
        reader_bsz = 1
        # TODO edge case: reader tokens < hidden states
        # if(len(reader_token) < hidden_states.shape[0]):
        #     reader_bsz = bsz
        reader_past_key_value = past_key_value
        # tmp debugging
        print(f'hidden states shape = {hidden_states.shape}')
        print(f'past key value = {past_key_value}')
        print(f'layer head mask = {layer_head_mask}')
        print(f'reader tokens = {reader_token}')
        for i, reader_token_i in enumerate(reader_token):
            if(attention_mask is not None):
                attention_mask_i = attention_mask[[i], :, :, :]
            else:
                attention_mask_i = None
            if(hidden_states is not None):
                hidden_states_i = hidden_states[[i], :, :]
            else:
                hidden_states_i = None
            if(reader_past_key_value is not None):
                past_key_value_i = reader_past_key_value[[i], :, :]
            else:
                past_key_value_i = None
            attn_probs_i, attn_weights_reshaped_i, past_key_value_i, value_states_i = self.get_attn_probs(
                attention_mask_i,
                reader_bsz,
                hidden_states_i,
                is_cross_attention,
                key_value_states, layer_head_mask, output_attentions,
                past_key_value_i, tgt_len, reader_token=reader_token_i)
            print(f'reader attn probs shape = {attn_probs_i.shape}')
            reader_attn_probs.append(attn_probs_i)
        reader_attn_probs = torch.cat(reader_attn_probs, axis=0)
        print(f'final reader attn probs shape = {reader_attn_probs.shape}')
        # generic probs
        generic_attn_probs, generic_attn_weights_reshaped, generic_past_key_value, generic_value_states = self.get_attn_probs(
                attention_mask, bsz, hidden_states, is_cross_attention,
                key_value_states, layer_head_mask, output_attentions,
                past_key_value, tgt_len, reader_token=None)
        print(f'generic attn probs shape = {generic_attn_probs.shape}')
        ## attn_probs = reader_attn_probs + general_attn_probs
        attn_probs = (self.reader_group_weight * reader_attn_probs + (1-self.reader_group_weight) * generic_attn_probs) / 2.
        attn_output = torch.bmm(attn_probs, generic_value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        # tmp debugging
        # print(f'generic past key value = {generic_past_key_value}')

        return attn_output, generic_attn_weights_reshaped, generic_past_key_value

    def get_attn_probs(self, attention_mask, bsz, hidden_states,
                       is_cross_attention, key_value_states, layer_head_mask,
                       output_attentions, past_key_value, tgt_len, reader_token=None):
        if(reader_token is None):
            q_proj, k_proj, v_proj = self.q_proj, self.k_proj, self.v_proj
        else:
            q_proj, k_proj, v_proj = self.reader_q_proj[reader_token], self.reader_k_proj[reader_token], self.reader_v_proj[reader_token]
        query_states = q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(k_proj(key_value_states), -1, bsz)
            value_states = self._shape(v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(k_proj(hidden_states), -1, bsz)
            value_states = self._shape(v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(k_proj(hidden_states), -1, bsz)
            value_states = self._shape(v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = F.dropout(attn_weights, p=self.dropout,
                               training=self.training)
        return attn_probs, attn_weights_reshaped, past_key_value, value_states


class AuthorGroupAttentionEncoderLayer(BartEncoderLayer):
    def __init__(self, config: BartConfig, reader_group_types):
        super().__init__(config)
        self.embed_dim = config.d_model
        self.reader_attn_weight = config.__dict__['reader_attn_weight']
        self.reader_attn_config = config.__dict__['reader_attn_config']
        # NOTE we need to include "OTHER" in reader_groups
        # to provide catch-all category
        if(self.reader_attn_config == 'attn_prob'):
            self.self_attn = AuthorGroupAttention(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                reader_group_types=reader_group_types,
                reader_group_weight=self.reader_attn_weight,
                is_decoder=False,
            )
        elif (self.reader_attn_config.startswith('attn_full')):
            self.self_attn_per_group = nn.ModuleDict({
                reader_group : BartAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout)#.to(torch.cuda.current_device())
                for reader_group in reader_group_types
            })
            # tmp debugging
            # self.self_attn_per_group_2 = nn.ModuleList([BartAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout).to(torch.cuda.current_device()),]*len(reader_group_types))
            self.self_attn_general = BartAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout)  # .to(torch.cuda.current_device())
            if(self.reader_attn_config == 'attn_full_concat'):
                self.self_attn_combiner = nn.Linear(2 * config.encoder_attention_heads, config.encoder_attention_heads)
                self.self_attn_combiner_norm = nn.LayerNorm(config.encoder_attention_heads)
                self.hidden_state_combiner = nn.Linear(2 * self.embed_dim, self.embed_dim)
                self.hidden_state_combiner_norm = nn.LayerNorm(self.embed_dim)
        # self.self_attn = BartAttention(
        #     embed_dim=self.embed_dim,
        #     num_heads=config.encoder_attention_heads,
        #     dropout=config.attention_dropout,
        # )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            reader_token : torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            reader_groups : reader group labels for specific attention layer (e.g. "US_AUTHOR")
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        # tmp debugging
        # print(f'encoder: reader tokens {reader_token}')
        # tmp debugging
        # print(f'attention per group = {self.self_attn_per_group}')
        if(self.reader_attn_config == 'attn_prob'):
            hidden_states, attn_weights, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                reader_token=reader_token,
            )
        elif(self.reader_attn_config.startswith('attn_full')):
            # run the hidden states, attention masks through each
            # reader-group specific module separately, then
            # recombine after
            combined_hidden_states = []
            combined_attn_weights = []
            for i, reader_group_i in enumerate(reader_token):
                # print(f'reader group = {reader_group_i}')
                # reader_group_attn_i = self.self_attn_per_group[int(reader_group_i)]
                reader_group_attn_i = self.self_attn_per_group[reader_group_i]
                # tmp debug
                # print(f'encoder layer: hidden states {hidden_states[[i], :, :].shape}')
                # print(f'encoder layer: attention {attention_mask[[i], :, :, :].shape}')
                # print(f'attention module {reader_group_attn_i}')
                # print(f'attention forward = {help(reader_group_attn_i.forward)}')
                hidden_states_i, attn_weights_i, _ = reader_group_attn_i(
                    hidden_states=hidden_states[[i], :, :],
                    attention_mask=(attention_mask[[i], :, :, :] if attention_mask is not None else None),
                    # key_value_states=None,
                    # past_key_value=None,
                    layer_head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )

                # print(f'output = attention weights {attn_weights_i}')
                combined_hidden_states.append(hidden_states_i)
                combined_attn_weights.append(attn_weights_i)
            reader_hidden_states = torch.cat(combined_hidden_states, axis=0)
            if(not any(list(map(lambda x: x is None, combined_attn_weights)))):
                reader_attn_weights = torch.cat(combined_attn_weights, axis=0)
            else:
                reader_attn_weights = None
            ## combine reader states with "regular" attention (non-weighted? sure)
            general_hidden_states, general_attn_weights, _ = self.self_attn_general(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            ## attn combinination = addition
            if(self.reader_attn_config == 'attn_full_mean'):
                # tmp debugging
                # print(f'computing mean reader and general attention states')
                hidden_states = (self.reader_attn_weight * reader_hidden_states + (1 - self.reader_attn_weight) * general_hidden_states) / 2.
                attn_weights = (self.reader_attn_weight * reader_attn_weights + (1 - self.reader_attn_weight) * general_attn_weights) / 2.
            elif(self.reader_attn_config == 'attn_full_concat'):
                ## attn combination = concat + normalize
                combined_hidden_states = torch.cat([reader_hidden_states, general_hidden_states], axis=2)
                combined_hidden_states = self.hidden_state_combiner(combined_hidden_states)
                hidden_states = self.hidden_state_combiner_norm(combined_hidden_states)
                # tmp debugging
                # print(f'encoder: after combining, hidden states have shape = {hidden_states.shape}')
                combined_attn_weights = torch.cat([reader_attn_weights, general_attn_weights], axis=1)
                # print(f'reader attn weights has shape = {reader_attn_weights.shape}; general attn weights has shape = {general_attn_weights.shape}')
                # print(f'combined attn weights has shape = {combined_attn_weights.shape}')
                # print(f'self attn combiner weights has shape {self.self_attn_combiner.weight.shape}')
                combined_attn_weights = self.self_attn_combiner(combined_attn_weights.transpose(1,3))
                attn_weights = self.self_attn_combiner_norm(combined_attn_weights).transpose(1,3)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class AuthorGroupAttentionEncoder(BartEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None, reader_group_types = [], reader_attn_position=0):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        encoder_list = []
        self.reader_attn_position = reader_attn_position
        for i in range(config.encoder_layers):
            if(i==self.reader_attn_position):
                encoder_list.append(AuthorGroupAttentionEncoderLayer(config, reader_group_types=reader_group_types))
            else:
                encoder_list.append(BartEncoderLayer(config))
        # encoder_list = [AuthorGroupAttentionEncoderLayer(config, reader_group_types=reader_group_types)]
        # # tmp debug
        # # encoder_list = [BartEncoderLayer(config)]
        # for i in range(config.encoder_layers-1):
        #     encoder_list.append(BartEncoderLayer(config))
        self.layers = nn.ModuleList(encoder_list)
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        reader_token=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        # tmp debugging
        # print(f'model forward pass: reader token = {reader_token}')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    # use reader token on specific layer to get
                    # reader-specific attention
                    if(idx == self.reader_attn_position):
                        layer_outputs = encoder_layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            reader_token=reader_token,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                        )
                    else:
                        layer_outputs = encoder_layer(
                            hidden_states,
                            attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                        )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class AuthorGroupAttentionDecoderLayer(BartDecoderLayer):
    def __init__(self, config: BartConfig, reader_group_types = []):
        super().__init__(config)
        self.embed_dim = config.d_model

        # self.self_attn = BartAttention(
        #     embed_dim=self.embed_dim,
        #     num_heads=config.decoder_attention_heads,
        #     dropout=config.attention_dropout,
        #     is_decoder=True,
        # )
        self.reader_attn_config = config.__dict__['reader_attn_config']
        self.reader_attn_weight = config.__dict__['reader_attn_weight']
        # NOTE we need to include "OTHER" in reader_groups
        # to provide catch-all category
        if (self.reader_attn_config == 'attn_prob'):
            self.self_attn = AuthorGroupAttention(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                reader_group_types=reader_group_types,
                reader_group_weight=self.reader_attn_weight,
                is_decoder=True
            )
        elif(self.reader_attn_config.startswith('attn_full')):
            self.self_attn_per_group = nn.ModuleDict({
                reader_group: BartAttention(
                    embed_dim=self.embed_dim,
                    num_heads=config.encoder_attention_heads,
                    dropout=config.attention_dropout,
                    is_decoder=True
                )
                for reader_group in reader_group_types
            })
            self.self_attn_general = BartAttention(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True
            )
            # tmp debugging
            # self.self_attn_per_group_2 = nn.ModuleList([BartAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout).to(torch.cuda.current_device()),]*len(reader_group_types))
            # self.self_attn_general = BartAttention(embed_dim=self.embed_dim,
            #                                        num_heads=config.encoder_attention_heads,
            #                                        dropout=config.attention_dropout)  # .to(torch.cuda.current_device())
            # extra weights for combining reader + general attn
            if(self.reader_attn_config == 'attn_full_concat'):
                self.self_attn_combiner = nn.Linear(2 * config.encoder_attention_heads, config.encoder_attention_heads)
                self.self_attn_combiner_norm = nn.LayerNorm(config.encoder_attention_heads)
                self.hidden_state_combiner = nn.Linear(2 * self.embed_dim, self.embed_dim)
                self.hidden_state_combiner_norm = nn.LayerNorm(self.embed_dim)

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        reader_token = [],
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (:obj:`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple

        if (self.reader_attn_config == 'attn_prob'):
            # tmp debugging
            print(f'in decoder: before attention, hidden state has shape = {hidden_states.shape}')
            hidden_states, attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                reader_token=reader_token,
            )
        elif (self.reader_attn_config.startswith('attn_full')):
            combined_hidden_states = []
            combined_attn_weights = []
            # tmp debugging
            # if(self_attn_past_key_value is not None):
            #     print(f'self-attn past key value shapes = {list(map(lambda x: x.shape, self_attn_past_key_value))}')
            for i, reader_group_i in enumerate(reader_token):
                reader_group_attn_i = self.self_attn_per_group[reader_group_i]
                # fix past key values
                if(self_attn_past_key_value is not None):
                    self_attn_past_key_value_i = [v[[i], :, :, :] for v in self_attn_past_key_value]
                    #print(f'reader token idx={i}; self-attention past key = {self_attn_past_key_value_i}')
                else:
                    self_attn_past_key_value_i = None
                hidden_states_i, attn_weights_i, _ = reader_group_attn_i(
                    hidden_states=hidden_states[[i], :, :],
                    attention_mask=(attention_mask[[i], :, :, :] if attention_mask is not None else None),
                    past_key_value=self_attn_past_key_value_i,
                    layer_head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )
                combined_hidden_states.append(hidden_states_i)
                combined_attn_weights.append(attn_weights_i)
            reader_hidden_states = torch.cat(combined_hidden_states, axis=0)
            if(not any(list(map(lambda x: x is None, combined_attn_weights)))):
                reader_attn_weights = torch.cat(combined_attn_weights, axis=0)
            else:
                reader_attn_weights = None
            ## combine reader states with "regular" attention
            # tmp debugging
            # print(f'before generic attention: hidden state shape = {hidden_states.shape}')
            general_hidden_states, general_attn_weights, present_key_value = self.self_attn_general(
                hidden_states=hidden_states,
                past_key_value=self_attn_past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            attn_weights = general_attn_weights
            if(reader_attn_weights is not None):
                # TODO: make mean-attention an option
                if(self.reader_attn_config == 'attn_full_mean'):
                    hidden_states = (self.reader_attn_weight * reader_hidden_states + (1 - self.reader_attn_weight) * general_hidden_states) / 2.
                    attn_weights = (self.reader_attn_weight * reader_attn_weights + (1 - self.reader_attn_weight) * general_attn_weights) / 2.
                # concat hidden states, attentions; normalize
                elif(self.reader_attn_config == 'attn_full_concat'):
                    # print(f'reader hidden state shape = {reader_hidden_states.shape}')
                    # print(f'generic hidden state shape = {general_hidden_states.shape}')
                    # print(f'reader attn shape = {reader_attn_weights.shape}')
                    # print(f'generic attn shape = {general_attn_weights.shape}')
                    # fix misalignment in reader/general during decoding
                    # copy reader states to match general states
                    if(reader_hidden_states.shape[0] != general_hidden_states.shape[0]):
                        reader_hidden_states = reader_hidden_states.repeat(general_hidden_states.shape[0], 1, 1)
                        reader_attn_weights = reader_attn_weights.repeat(general_attn_weights.shape[0], 1, 1, 1)
                    combined_hidden_states = torch.cat([reader_hidden_states, general_hidden_states], axis=2)
                    combined_hidden_states = self.hidden_state_combiner(combined_hidden_states)
                    hidden_states = self.hidden_state_combiner_norm(combined_hidden_states)
                    combined_attn_weights = torch.cat([reader_attn_weights, general_attn_weights], axis=1)
                    # print(f'reader attn weights has shape = {reader_attn_weights.shape}; general attn weights has shape = {general_attn_weights.shape}')
                    # print(f'combined attn weights has shape = {combined_attn_weights.shape}')
                    # print(f'self attn combiner weights has shape {self.self_attn_combiner.weight.shape}')
                    combined_attn_weights = self.self_attn_combiner(combined_attn_weights.transpose(1, 3))
                    attn_weights = self.self_attn_combiner_norm(combined_attn_weights).transpose(1, 3)
            else:
                hidden_states = general_hidden_states

        # hidden_states, self_attn_weights, present_key_value = self.self_attn(
        #     hidden_states=hidden_states,
        #     past_key_value=self_attn_past_key_value,
        #     attention_mask=attention_mask,
        #     layer_head_mask=layer_head_mask,
        #     output_attentions=output_attentions,
        # )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            # tmp debugging
            # print(f'present key value = {present_key_value}')
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class AuthorGroupAttentionDecoder(BartDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None,
                 reader_group_types=[], reader_attn_position=0):
        ## TODO: port from encoder, proceed as expected
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        decoder_list = []
        self.reader_attn_position = reader_attn_position
        for i in range(config.decoder_layers):
            if (i == self.reader_attn_position):
                decoder_list.append(AuthorGroupAttentionDecoderLayer(config, reader_group_types=reader_group_types))
            else:
                decoder_list.append(BartDecoderLayer(config))
        # self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layers = nn.ModuleList(decoder_list)
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        reader_token=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # tmp debugging
        #print(f'decoder attention mask before preparing = {attention_mask}; input shape = {input_shape}')
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        #print(f'decoder attention mask after preparing = {attention_mask}')
        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # tmp debugging
        # print(f'before decoder: hidden states = {hidden_states.shape}')
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning("`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting `use_cache=False`...")
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                # tmp debugging
                #print(f'decoder layer = {idx}; attention mask = {attention_mask}')
                if (idx == self.reader_attn_position):
                    # tmp debugging
                    # print(f'input shape = {input_shape}')
                    # print(f'reader token = {reader_token}')
                    layer_outputs = decoder_layer(
                        hidden_states,
                        reader_token=reader_token,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                # layer_outputs = decoder_layer(
                #     hidden_states,
                #     attention_mask=attention_mask,
                #     encoder_hidden_states=encoder_hidden_states,
                #     encoder_attention_mask=encoder_attention_mask,
                #     layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                #     cross_attn_layer_head_mask=(
                #         cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                #     ),
                #     past_key_value=past_key_value,
                #     output_attentions=output_attentions,
                #     use_cache=use_cache,
                # )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class AuthorGroupAttentionModel(BartModel):
    def __init__(self, config: BartConfig, reader_group_types=[]):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.reader_group_attention_location = config.__dict__['reader_group_attention_location']
        if(self.reader_group_attention_location == 'encoder'):
            self.encoder = AuthorGroupAttentionEncoder(config, self.shared, reader_group_types=reader_group_types, reader_attn_position=config.__dict__['reader_attn_position'])
            self.decoder = BartDecoder(config, self.shared)
        elif(self.reader_group_attention_location == 'decoder'):
            self.encoder = BartEncoder(config, self.shared)
            self.decoder = AuthorGroupAttentionDecoder(config, self.shared, reader_group_types=reader_group_types, reader_attn_position=config.__dict__['reader_attn_position'])

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        reader_token=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # tmp debugging
        # print(f'mid model forward pass: reader token = {reader_token}')

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            if(self.reader_group_attention_location == 'encoder'):
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    reader_token=reader_token,
                )
            else:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        if(self.reader_group_attention_location == 'decoder'):
            # tmp debugging
            #print(f'decoder attention mask = {decoder_attention_mask}')
            #print(f'attention mask = {attention_mask}')
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                # head_mask=decoder_head_mask,
                # encoder_head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                reader_token=reader_token,
            )
        else:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                # head_mask=decoder_head_mask,
                # encoder_head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class AuthorGroupAttentionModelConditionalGeneration(BartForConditionalGeneration):

    def __init__(self, config: BartConfig, reader_group_types=[]):
        super().__init__(config)
        self.model = AuthorGroupAttentionModel(config, reader_group_types=reader_group_types)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        ## reader group-specific LM for final decoding
        ## more parameters but more straightforward to learn
        if(self.model.config.__dict__['reader_group_attention_location']=='lm_head'):
            self.reader_lm_heads = nn.ModuleDict({
                reader_group_type : nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
                for reader_group_type in reader_group_types
            })
            self.reader_attn_weight = self.model.config.__dict__['reader_attn_weight']

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        reader_token=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # tmp debugging
        # print(f'top model forward pass: reader token = {reader_token}')

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            reader_token=reader_token,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        if (self.model.config.__dict__['reader_group_attention_location'] == 'lm_head'):
            reader_group_logits = []
            print(f'output weights have shape = {outputs[0].shape}')
            for i, reader_token_i in enumerate(reader_token):
                output_i = outputs[0][[i], :, :]
                reader_group_logit_i = self.reader_lm_heads[reader_token_i](output_i) + self.final_logits_bias
                reader_group_logits.append(reader_group_logit_i)
            reader_group_logits = torch.cat(reader_group_logits, axis=0)
            # tmp debugging
            print(f'reader group LM logits have shape = {reader_group_logits.shape}')
            print(f'generic LM logits have shape = {lm_logits.shape}')
            lm_logits = (lm_logits * (1-self.reader_attn_weight) + reader_group_logits * self.reader_attn_weight) / 2.
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            'reader_token' : kwargs['reader_token'],
            #'decoder_attention_mask' : (attention_mask if self.config.__dict__['reader_group_attention_location']=='decoder' else None), # for decoder model: add decoder attention mask
        }

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor, model_kwargs) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
            }
            # for decoder model: remove reader_token from encoder args
            if(self.config.__dict__['reader_group_attention_location'] == 'decoder'):
                del(encoder_kwargs['reader_token'])
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs

    # @torch.no_grad()
    # def generate(
    #         self,
    #         input_ids: Optional[torch.LongTensor] = None,
    #         max_length: Optional[int] = None,
    #         min_length: Optional[int] = None,
    #         do_sample: Optional[bool] = None,
    #         early_stopping: Optional[bool] = None,
    #         num_beams: Optional[int] = None,
    #         temperature: Optional[float] = None,
    #         top_k: Optional[int] = None,
    #         top_p: Optional[float] = None,
    #         repetition_penalty: Optional[float] = None,
    #         bad_words_ids: Optional[Iterable[int]] = None,
    #         bos_token_id: Optional[int] = None,
    #         pad_token_id: Optional[int] = None,
    #         eos_token_id: Optional[int] = None,
    #         length_penalty: Optional[float] = None,
    #         no_repeat_ngram_size: Optional[int] = None,
    #         encoder_no_repeat_ngram_size: Optional[int] = None,
    #         num_return_sequences: Optional[int] = None,
    #         max_time: Optional[float] = None,
    #         max_new_tokens: Optional[int] = None,
    #         decoder_start_token_id: Optional[int] = None,
    #         use_cache: Optional[bool] = None,
    #         num_beam_groups: Optional[int] = None,
    #         diversity_penalty: Optional[float] = None,
    #         prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    #         output_attentions: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         output_scores: Optional[bool] = None,
    #         return_dict_in_generate: Optional[bool] = None,
    #         forced_bos_token_id: Optional[int] = None,
    #         forced_eos_token_id: Optional[int] = None,
    #         remove_invalid_values: Optional[bool] = None,
    #         synced_gpus: Optional[bool] = None,
    #         **model_kwargs,
    # ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
    #
    #     # set init values
    #     if max_length is None and max_new_tokens is None:
    #         # Both are None, default
    #         max_length = self.config.max_length
    #     elif max_length is not None and max_new_tokens is not None:
    #         # Both are set, this is odd, raise a warning
    #         warnings.warn("Both `max_length` and `max_new_tokens` have been set but they serve the same purpose.", UserWarning)
    #
    #     max_length = max_length if max_length is not None else self.config.max_length
    #     num_beams = num_beams if num_beams is not None else self.config.num_beams
    #     num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
    #     do_sample = do_sample if do_sample is not None else self.config.do_sample
    #     num_return_sequences = (
    #         num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    #     )
    #
    #     pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    #     bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    #     eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    #
    #     output_scores = output_scores if output_scores is not None else self.config.output_scores
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict_in_generate = (
    #         return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    #     )
    #
    #     model_kwargs["output_attentions"] = output_attentions
    #     model_kwargs["output_hidden_states"] = output_hidden_states
    #
    #     if input_ids is None and "inputs_embeds" not in model_kwargs:
    #         # init `input_ids` with bos_token_id
    #         input_ids = self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))
    #
    #     if model_kwargs.get("attention_mask", None) is None:
    #         # init `attention_mask` depending on `pad_token_id`
    #         model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id)
    #
    #     # special case if pad_token_id is not defined
    #     if pad_token_id is None and eos_token_id is not None:
    #         logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
    #         pad_token_id = eos_token_id
    #
    #     # Storing encoder_input_ids for logits_processor that could use them
    #     encoder_input_ids = input_ids if self.config.is_encoder_decoder else None
    #
    #     if self.config.is_encoder_decoder:
    #         # add encoder_outputs to model_kwargs
    #         ## NOTE: avoid issue with decoder if possible
    #         if(self.config.__dict__['reader_group_attention_location']=='decoder'):
    #             model_kwargs['decoder_reader_token'] = model_kwargs['reader_token']
    #             del(model_kwargs['reader_token'])
    #         model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)
    #         if (self.config.__dict__['reader_group_attention_location'] == 'decoder'):
    #             model_kwargs['reader_token'] = model_kwargs['decoder_reader_token']
    #             del(model_kwargs['decoder_reader_token'])
    #         # set input_ids as decoder_input_ids
    #         if "decoder_input_ids" in model_kwargs:
    #             input_ids = model_kwargs.pop("decoder_input_ids")
    #         else:
    #             input_ids = self._prepare_decoder_input_ids_for_generation(
    #                 input_ids, decoder_start_token_id=decoder_start_token_id,
    #                 bos_token_id=bos_token_id
    #             )
    #
    #         if "encoder_outputs" not in model_kwargs or not isinstance(
    #                 model_kwargs["encoder_outputs"], ModelOutput):
    #             raise ValueError(
    #                 "Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")
    #
    #     if input_ids.shape[-1] >= max_length:
    #         input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    #         logger.warning(
    #             f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
    #             "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
    #         )
    #
    #     # determine generation mode
    #     is_greedy_gen_mode = (num_beams == 1) and (
    #                 num_beam_groups == 1) and do_sample is False
    #     is_sample_gen_mode = (num_beams == 1) and (
    #                 num_beam_groups == 1) and do_sample is True
    #     is_beam_gen_mode = (num_beams > 1) and (
    #                 num_beam_groups == 1) and do_sample is False
    #     is_beam_sample_gen_mode = (num_beams > 1) and (
    #                 num_beam_groups == 1) and do_sample is True
    #     is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
    #     if num_beam_groups > num_beams:
    #         raise ValueError(
    #             "`num_beam_groups` has to be smaller or equal to `num_beams`")
    #     if is_group_beam_gen_mode and do_sample is True:
    #         raise ValueError(
    #             "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
    #         )
    #
    #     # set model_kwargs
    #     model_kwargs["use_cache"] = use_cache
    #
    #     # get distribution pre_processing samplers
    #     logits_processor = self._get_logits_processor(
    #         repetition_penalty=repetition_penalty,
    #         no_repeat_ngram_size=no_repeat_ngram_size,
    #         encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
    #         encoder_input_ids=encoder_input_ids,
    #         bad_words_ids=bad_words_ids,
    #         min_length=min_length,
    #         max_length=max_length,
    #         eos_token_id=eos_token_id,
    #         forced_bos_token_id=forced_bos_token_id,
    #         forced_eos_token_id=forced_eos_token_id,
    #         prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    #         num_beams=num_beams,
    #         num_beam_groups=num_beam_groups,
    #         diversity_penalty=diversity_penalty,
    #         remove_invalid_values=remove_invalid_values,
    #     )
    #
    #     cur_len = input_ids.shape[-1]
    #     stopping_criteria = self._get_stopping_criteria(
    #         max_length=max_length, max_time=max_time,
    #         max_new_tokens=max_new_tokens, start_length=cur_len
    #     )
    #
    #     if is_greedy_gen_mode:
    #         if num_return_sequences > 1:
    #             raise ValueError(f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search.")
    #
    #         # greedy search
    #         return self.greedy_search(
    #             input_ids,
    #             logits_processor=logits_processor,
    #             stopping_criteria=stopping_criteria,
    #             pad_token_id=pad_token_id,
    #             eos_token_id=eos_token_id,
    #             output_scores=output_scores,
    #             return_dict_in_generate=return_dict_in_generate,
    #             synced_gpus=synced_gpus,
    #             **model_kwargs,
    #         )
    #
    #     elif is_sample_gen_mode:
    #         # get probability distribution warper
    #         logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, temperature=temperature,num_beams=num_beams)
    #
    #         # expand input_ids with `num_return_sequences` additional sequences per batch
    #         input_ids, model_kwargs = self._expand_inputs_for_generation(
    #             input_ids, expand_size=num_return_sequences,
    #             is_encoder_decoder=self.config.is_encoder_decoder,
    #             **model_kwargs,
    #         )
    #
    #         # sample
    #         return self.sample(
    #             input_ids,
    #             logits_processor=logits_processor,
    #             logits_warper=logits_warper,
    #             stopping_criteria=stopping_criteria,
    #             pad_token_id=pad_token_id,
    #             eos_token_id=eos_token_id,
    #             output_scores=output_scores,
    #             return_dict_in_generate=return_dict_in_generate,
    #             synced_gpus=synced_gpus,
    #             **model_kwargs,
    #         )
    #
    #     elif is_beam_gen_mode:
    #         batch_size = input_ids.shape[0]
    #
    #         length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
    #         early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
    #
    #         if num_return_sequences > num_beams:
    #             raise ValueError(
    #                 "`num_return_sequences` has to be smaller or equal to `num_beams`.")
    #
    #         if stopping_criteria.max_length is None:
    #             raise ValueError(
    #                 "`max_length` needs to be a stopping_criteria for now.")
    #
    #         beam_scorer = BeamSearchScorer(
    #             batch_size=batch_size,
    #             num_beams=num_beams,
    #             device=self.device,
    #             length_penalty=length_penalty,
    #             do_early_stopping=early_stopping,
    #             num_beam_hyps_to_keep=num_return_sequences,
    #         )
    #         # interleave with `num_beams`
    #         input_ids, model_kwargs = self._expand_inputs_for_generation(
    #             input_ids, expand_size=num_beams,
    #             is_encoder_decoder=self.config.is_encoder_decoder,
    #             **model_kwargs
    #         )
    #         return self.beam_search(
    #             input_ids,
    #             beam_scorer,
    #             logits_processor=logits_processor,
    #             stopping_criteria=stopping_criteria,
    #             pad_token_id=pad_token_id,
    #             eos_token_id=eos_token_id,
    #             output_scores=output_scores,
    #             return_dict_in_generate=return_dict_in_generate,
    #             synced_gpus=synced_gpus,
    #             **model_kwargs,
    #         )
    #
    #     elif is_beam_sample_gen_mode:
    #         logits_warper = self._get_logits_warper(
    #             top_k=top_k, top_p=top_p, temperature=temperature,
    #             num_beams=num_beams
    #         )
    #
    #         batch_size = input_ids.shape[0] * num_return_sequences
    #
    #         length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
    #         if stopping_criteria.max_length is None:
    #             raise ValueError(
    #                 "`max_length` needs to be a stopping_criteria for now.")
    #         beam_scorer = BeamSearchScorer(
    #             batch_size=batch_size,
    #             num_beams=num_beams,
    #             device=self.device,
    #             length_penalty=length_penalty,
    #             do_early_stopping=early_stopping,
    #         )
    #
    #         # interleave with `num_beams * num_return_sequences`
    #         input_ids, model_kwargs = self._expand_inputs_for_generation(
    #             input_ids,
    #             expand_size=num_beams * num_return_sequences,
    #             is_encoder_decoder=self.config.is_encoder_decoder,
    #             **model_kwargs,
    #         )
    #
    #         return self.beam_sample(
    #             input_ids,
    #             beam_scorer,
    #             logits_processor=logits_processor,
    #             logits_warper=logits_warper,
    #             stopping_criteria=stopping_criteria,
    #             pad_token_id=pad_token_id,
    #             eos_token_id=eos_token_id,
    #             output_scores=output_scores,
    #             return_dict_in_generate=return_dict_in_generate,
    #             synced_gpus=synced_gpus,
    #             **model_kwargs,
    #         )
    #
    #     elif is_group_beam_gen_mode:
    #         batch_size = input_ids.shape[0]
    #
    #         length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
    #         early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
    #
    #         if num_return_sequences > num_beams:
    #             raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
    #
    #         if num_beams % num_beam_groups != 0:
    #             raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")
    #
    #         if stopping_criteria.max_length is None:
    #             raise ValueError("`max_length` needs to be a stopping_criteria for now.")
    #
    #         diverse_beam_scorer = BeamSearchScorer(
    #             batch_size=batch_size,
    #             num_beams=num_beams,
    #             max_length=stopping_criteria.max_length,
    #             device=self.device,
    #             length_penalty=length_penalty,
    #             do_early_stopping=early_stopping,
    #             num_beam_hyps_to_keep=num_return_sequences,
    #             num_beam_groups=num_beam_groups,
    #         )
    #         # interleave with `num_beams`
    #         input_ids, model_kwargs = self._expand_inputs_for_generation(
    #             input_ids, expand_size=num_beams,
    #             is_encoder_decoder=self.config.is_encoder_decoder,
    #             **model_kwargs
    #         )
    #         return self.group_beam_search(
    #             input_ids,
    #             diverse_beam_scorer,
    #             logits_processor=logits_processor,
    #             stopping_criteria=stopping_criteria,
    #             pad_token_id=pad_token_id,
    #             eos_token_id=eos_token_id,
    #             output_scores=output_scores,
    #             return_dict_in_generate=return_dict_in_generate,
    #             synced_gpus=synced_gpus,
    #             **model_kwargs,
    #         )

