"""
Author-aware model: uses latent author representation
to generate text.
"""
## transformer boilerplate
from math import sqrt
from typing import Optional, Union, Callable, List, Iterable, Dict, Any

import numpy as np
import torch
from torch import nn
import random
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BartPretrainedModel, BartConfig
from transformers.file_utils import ModelOutput
from transformers.generation_utils import GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.modeling_bart import BartDecoder, \
    shift_tokens_right, _expand_mask, BartEncoderLayer, \
    BartLearnedPositionalEmbedding, BartForConditionalGeneration, BartDecoderLayer, BartEncoder, BartModel, _make_causal_mask


class AuthorTextEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
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
            #self.padding_idx,
       	)
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.author_embed_network = nn.Linear(self.config.author_embeds, embed_dim)
        self.author_embed_layernorm = nn.LayerNorm(embed_dim)
        self.author_embed_token_id = self.config.author_embed_token_id
        # self.author_text_combine_network = nn.Linear(self.config.max_position_embeddings+1, self.config.max_position_embeddings)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        author_embeds=None,
        attention_mask=None,
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

        ## add special embed token to input IDs
        if(author_embeds is not None):
            # clip final input tokens to make room
            clean_input_ids = []
            # print(f'pre-clean input IDs has shape {input_ids.shape}')
            for input_id_row_i in input_ids:
                # tmp debugging
                # print(f'input ID row = {input_id_row_i}')
                padding_idx_i = np.where(input_id_row_i.cpu() == self.padding_idx)[0]
                # print(f'padding idx = {padding_idx_i}')
                # handling input without padding
                if(len(padding_idx_i) < 2):
                    # clip final tokens on max-length input
                    if (len(input_id_row_i) > (self.config.max_position_embeddings - 3)):
                        clip_len_i = len(input_id_row_i) - self.config.max_position_embeddings + 3
                        input_id_row_i = input_id_row_i[:-clip_len_i]
                        end_of_input_idx_i = len(input_id_row_i)
                    else:
                        end_of_input_idx_i = padding_idx_i[-1] if len(padding_idx_i) > 0 else len(input_id_row_i)-1
                    # append EOS + padding + padding to end of non-padded input
                    # print(f'end of input idx = {end_of_input_idx_i}')
                    input_id_row_i = torch.cat([input_id_row_i[:end_of_input_idx_i],
                                                torch.LongTensor([self.config.eos_token_id, self.padding_idx, self.padding_idx]).to(input_id_row_i.device)])
                # add embed token at end of sequence
                padding_idx_i = np.where(input_id_row_i.cpu() == self.padding_idx)[0]
                if (len(padding_idx_i) > 0):
                    input_id_row_i[padding_idx_i[0]] = self.author_embed_token_id
                    # print(f'input with embed token {input_id_row_i}')
                clean_input_ids.append(input_id_row_i)
            # add extra padding to all short input ID rows
            # print(f'clean input IDs have len {[len(x) for x in clean_input_ids]}')
            max_input_len = max([len(x) for x in clean_input_ids])
            clean_input_ids = list(map(lambda x: torch.cat([x, torch.LongTensor([self.padding_idx,]*(max_input_len-len(x))).to(x.device)]), clean_input_ids))
            # re-combine input IDs
            clean_input_ids = list(map(lambda x: x.unsqueeze(0), clean_input_ids))
            input_ids = torch.cat(clean_input_ids, axis=0)
            # update shape
            input_shape = input_ids.size()
            # print(f'clean input ids has shape {input_ids.shape}')

            # for input_id_row_i in input_ids:
            #     padding_idx_i = np.where(input_id_row_i.cpu()==self.padding_idx)[0]
            #     if(len(padding_idx_i) > 0):
            #         input_id_row_i[padding_idx_i[0]] = self.author_embed_token_id
            #         print(f'input with embed token {input_id_row_i}')

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        ## add author embeddings
        if(author_embeds is not None):
            author_embeds_hidden = self.author_embed_network(author_embeds.unsqueeze(1))
            author_embeds_hidden = self.author_embed_layernorm(author_embeds_hidden)
            # tmp debugging
            # print(f'input embeds has shape {inputs_embeds.shape}')
            # print(f'author embed hidden has shape {author_embeds_hidden.shape}')
            clean_input_embeds = []
            # print(f'text-only input embeds has shape {inputs_embeds.shape}')
            for input_embeds_row_i, input_ids_row_i, author_embed_row_i, attention_mask_row_i in zip(inputs_embeds, input_ids, author_embeds_hidden, attention_mask):
                # replace first padding index with author embeds
                # print(f'input embed row has shape {input_embeds_row_i.shape}')
                # print(f'author embed row has shape {author_embed_row_i.shape}')
                # print(f'input embeds without author embed has shape {input_embeds_row_i.shape}')
                padding_idx_i = np.where(input_ids_row_i.cpu()==self.padding_idx)[0]
                # print(f'padding idx = {padding_idx_i}')
                # insert author embedding
                # inputs_embeds_row_i[padding_idx_i[0], :] = author_embed_row_i[0, :].to(inputs_embeds_row_i.device)
                input_embeds_row_i = torch.cat([input_embeds_row_i[:padding_idx_i[0], :], author_embed_row_i, input_embeds_row_i[(padding_idx_i[0]+1):, :]])
                clean_input_embeds.append(input_embeds_row_i)
                # tmp debugging
                # print(f'input embeds with author embed {input_embeds_row_i}')
                # print(f'input embeds has shape {input_embeds_row_i.shape}')
            inputs_embeds = torch.cat([x.unsqueeze(0) for x in clean_input_embeds], axis=0)
            # print(f'text+author input embeds has shape {inputs_embeds.shape}')
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        ## add author embeddings directly to hidden state
        # if(author_embeds is not None):
        # ## add author embeddings
        #     # remove final token from input, replace with author embedding
        #     # hidden_states = hidden_states[:, :-1, :]
        #     # author_embeds = author_embeds.reshape(author_embeds.shape[0], 1, author_embeds.shape[1])#.double()
        #     # # tmp debugging
        #     # # print(f'author embeds {author_embeds}')
        #     # author_embeds_hidden = self.author_embed_network(author_embeds)
        #     # author_embeds_hidden = self.author_embed_layernorm(author_embeds_hidden)
        #     # combine author embeds with hidden states
        #     # pass through ANOTHER network to combine
        #     author_embeds_hidden = self.author_embed_network(author_embeds.unsqueeze(1))
        #     author_embeds_hidden = self.author_embed_layernorm(author_embeds_hidden)
        #     if(hidden_states.shape[1] == self.config.max_position_embeddings):
        #         hidden_states = hidden_states[:, :-1, :]
        #     # tmp debugging
        #     # print(f'hidden states have dimensions={hidden_states.shape}')
        #     # print(f'author embeds have dimensions={author_embeds_hidden.shape}')
        #     hidden_states = torch.cat([hidden_states, author_embeds_hidden], axis=1)
        #     # tmp debugging
        #     # print(f'hidden states have dimensions={hidden_states.shape}')
        #     # print(f'author embeds have dimensions={author_embeds_hidden.shape}')
        #     # text_author_combined = torch.cat([hidden_states, author_embeds_hidden], dim=1).transpose(1,2)
        #     # print(f'combined embeds have dimensions={text_author_combined.shape}')
        #     # hidden_states = self.author_text_combine_network(text_author_combined).transpose(1,2)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
            # tmp debugging
            # print(f'expanded attention mask dimensions={attention_mask.shape}')

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
                    # tmp debugging
                    # print(f'encoder layer {idx}: {encoder_layer}')
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None), # only in new version??
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

## same thing but with decoder
class AuthorTextDecoder(BartDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            #self.padding_idx,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        ## autho-specific networks
        embed_dim = config.d_model
        self.author_embed_network = nn.Linear(self.config.author_embeds, embed_dim)
        self.author_embed_layernorm = nn.LayerNorm(embed_dim)
        self.author_attention_combine_network = nn.Linear(embed_dim*2, embed_dim)
        self.author_attention_combine_layernorm = nn.LayerNorm(embed_dim)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        author_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
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
            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
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

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        # print(f'before modification: attention mask has shape {attention_mask.shape}')

        if attention_mask is not None and combined_attention_mask is not None:
            if (author_embeds is not None):
                combined_attention_mask[:, -1] = 1
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if (author_embeds is not None):
                # encoder_attention_mask[:, -1] = 1
                # print(f'encoder attention before expanding has shape={encoder_attention_mask.shape}')
                ## less terrible strategy: modify each dimension of encoder attention
                ## according to author embedding
                author_embeds = self.author_embed_network(author_embeds)
                author_embeds = self.author_embed_layernorm(author_embeds)
                # if(len(encoder_hidden_states.shape)==3):
                #     author_embed_repeat = author_embeds.unsqueeze(1).repeat(1, encoder_hidden_states.shape[1], 1)
                #     if (author_embeds.shape[0] != encoder_hidden_states[0]):
                #         author_embed_repeat = author_embed_repeat.repeat(encoder_hidden_states.shape[0], 1, 1)
                # tmp debugging
                #print(f'encoder hidden state has shape {encoder_hidden_states.shape}')
                #print(f'author embeds has shape {author_embeds.shape}')
                #if(len(author_embeds.shape)==2):
                #    author_embeds = author_embeds.unsqueeze(0)
                if(len(encoder_hidden_states.shape)==3):
                    author_embed_repeat = author_embeds.unsqueeze(1).repeat(1, encoder_hidden_states.shape[1], 1)
                    if (author_embeds.shape[0] != encoder_hidden_states.shape[0]):
                        author_embed_repeat = author_embed_repeat.repeat(encoder_hidden_states.shape[0], 1, 1)
                    # tmp debugging
                    #print(f'author embed repeat has shape {author_embed_repeat.shape}')
                    author_embed_hidden_combine = torch.cat([author_embed_repeat, encoder_hidden_states], dim=2)
                    author_embed_attention_combine = self.author_attention_combine_network(author_embed_hidden_combine)
                    encoder_hidden_states = self.author_attention_combine_layernorm(author_embed_attention_combine)
                ## old dumb strategy: concatenate author embedding as extra word
                # get final index of input ids in each row; set idx+1 to 1 (for author embed)
                ## easy case: 1 input means set last valid idx to 1; add extra row for embed
                # if(len(encoder_attention_mask.shape) == 2):
                #     expanded_input_size =  input_ids.shape[0]+1
                #     author_attention_row = torch.LongTensor([1,]*(expanded_input_size) + [0,]*(encoder_attention_mask.shape[1]-(expanded_input_size))).unsqueeze(0)
                #     author_attention_row = author_attention_row.to(encoder_attention_mask.device)
                #     encoder_attention_mask = torch.cat([encoder_attention_mask, author_attention_row])
                #     encoder_attention_mask[:, input_ids.shape[0]] = 1
                # # add extra space for author embed
                # if(len(input_shape) == 2):
                #     input_shape = torch.Size((input_shape[0]+1, input_shape[1]))
                # # encoder_attention_mask = torch.cat([encoder_attention_mask, torch.])
                # print(f'before expanding: encoder hidden state = {encoder_hidden_states.shape}')
                # author_hidden_state_row = torch.ones(1, encoder_hidden_states.shape[1], encoder_hidden_states.shape[2])
                # author_hidden_state_row = author_hidden_state_row.to(encoder_hidden_states.device)
                # encoder_hidden_states = torch.cat([encoder_hidden_states, author_hidden_state_row])
                # print(f'after expanding: encoder hidden state = {encoder_hidden_states.shape}')
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # print(f'input shape = {input_shape}')
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            ## also expand encoder hidden state lol

        # print(f'encoder attention after expanding = {encoder_attention_mask.shape}')

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        ## add author info to input data
        ## TODO: how to get these hidden states to match with encoder_hidden_states (src_len * max_len * dim)
        # if (author_embeds is not None):
        #     ## add author embeddings
        #     # remove final token from input, replace with author embedding
        #     # TODO: add author embedding before padding? not sure that it matters
        #     author_embeds_hidden = self.author_embed_network(author_embeds.unsqueeze(1))
        #     author_embeds_hidden = self.author_embed_layernorm(author_embeds_hidden)
        #     # drop final word if at sequence at max length
        #     if(hidden_states.shape[1] == self.config.max_position_embeddings):
        #         hidden_states = hidden_states[:, :-1, :]
        #     if (hidden_states.shape[0] == self.config.max_position_embeddings):
        #         hidden_states = hidden_states[:-1, :, :]
        #     # tmp debugging
        #     print(f'hidden states have dimensions={hidden_states.shape}')
        #     print(f'author embeds have dimensions={author_embeds_hidden.shape}')
        #     hidden_states = torch.cat([hidden_states, author_embeds_hidden], axis=0)
        #     # hidden_states = hidden_states.transpose(0,1)
        #     print(f'post-combine hidden states have dimensions={hidden_states.shape}')
        #     print(f'post-combine attention has dimensions={encoder_attention_mask.shape}')
        #     print(f'post-combine attention={encoder_attention_mask[:, :, :, :input_shape[0]]}')

            #print(f'hidden state before modification has shape={hidden_states.shape}')
            #hidden_states = hidden_states[:-1, :, :]
            #author_embeds = author_embeds.reshape(author_embeds.shape[0], 1, author_embeds.shape[1]).float()
            #author_embeds_hidden = self.author_embed_network(author_embeds)
            #author_embeds_hidden = self.author_embed_layernorm(author_embeds_hidden)
			# tmp debugging
            #print(f'author embeds has shape={author_embeds_hidden.shape}')
            #print(f'hidden state has shape={hidden_states.shape}')
            #hidden_states = torch.cat([hidden_states, author_embeds_hidden], axis=1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False):
                if use_cache:
                    raise ValueError(
                        "When using `gradient_checkpointing, make sure that `use_cache=False` and `config.use_cache=False`."
                    )

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    combined_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:
                # tmp debugging
                #print(f'before decoder: encoder hidden state shape = {encoder_hidden_states.shape}')
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    # def forward(
    #     self,
    #     input_ids=None,
    #     attention_mask=None,
    #     author_embeds=None,
    #     encoder_hidden_states=None,
    #     encoder_attention_mask=None,
    #     head_mask=None,
    #     encoder_head_mask=None,
    #     past_key_values=None,
    #     inputs_embeds=None,
    #     use_cache=None,
    #     output_attentions=None,
    #     output_hidden_states=None,
    #     return_dict=None,
    # ):
    #     r"""
    #     Args:
    #         input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
    #             Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
    #             provide it.
    #
    #             Indices can be obtained using :class:`~transformers.BartTokenizer`. See
    #             :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
    #             for details.
    #
    #             `What are input IDs? <../glossary.html#input-ids>`__
    #         attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
    #             Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
    #
    #             - 1 for tokens that are **not masked**,
    #             - 0 for tokens that are **masked**.
    #
    #             `What are attention masks? <../glossary.html#attention-mask>`__
    #         encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
    #             Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
    #             of the decoder.
    #         encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
    #             Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
    #             selected in ``[0, 1]``:
    #
    #             - 1 for tokens that are **not masked**,
    #             - 0 for tokens that are **masked**.
    #
    #             `What are attention masks? <../glossary.html#attention-mask>`__
    #         head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
    #             Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:
    #
    #             - 1 indicates the head is **not masked**,
    #             - 0 indicates the heas is **masked**.
    #
    #         encoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
    #             Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
    #             on hidden heads. Mask values selected in ``[0, 1]``:
    #
    #             - 1 indicates the head is **not masked**,
    #             - 0 indicates the heas is **masked**.
    #
    #         past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
    #             Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
    #             decoding.
    #
    #             If :obj:`past_key_values` are used, the user can optionally input only the last
    #             :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
    #             shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
    #             sequence_length)`.
    #         inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
    #             Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
    #             representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
    #             into associated vectors than the model's internal embedding lookup matrix.
    #         output_attentions (:obj:`bool`, `optional`):
    #             Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
    #             returned tensors for more detail.
    #         output_hidden_states (:obj:`bool`, `optional`):
    #             Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
    #             for more detail.
    #         return_dict (:obj:`bool`, `optional`):
    #             Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
    #     """
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #
    #     # retrieve input_ids and inputs_embeds
    #     if input_ids is not None and inputs_embeds is not None:
    #         raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    #     elif input_ids is not None:
    #         input_shape = input_ids.size()
    #         input_ids = input_ids.view(-1, input_shape[-1])
    #     elif inputs_embeds is not None:
    #         input_shape = inputs_embeds.size()[:-1]
    #     else:
    #         raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
    #
    #     # past_key_values_length
    #     past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    #
    #     if inputs_embeds is None:
    #         inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    #     # tmp debug
    #     # print(dir(super()))
    #     attention_mask = super()._prepare_decoder_attention_mask(
    #         attention_mask, input_shape, inputs_embeds, past_key_values_length
    #     )
    #
    #
    #     # expand encoder attention mask
    #     if encoder_hidden_states is not None and encoder_attention_mask is not None:
    #         if(author_embeds is not None):
    #             encoder_attention_mask[:, -1] = 1
    #         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    #         encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
    #
    #     # embed positions
    #     positions = self.embed_positions(input_shape, past_key_values_length)
    #
    #     hidden_states = inputs_embeds + positions
    #     hidden_states = self.layernorm_embedding(hidden_states)
    #     hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
    #
    #     ## add author info to input data
    #     if (author_embeds is not None):
    #         ## add author embeddings
    #         # remove final token from input, replace with author embedding
    #         # TODO: add author embedding before padding? not sure that it matters
    #         hidden_states = hidden_states[:, :-1, :]
    #         author_embeds = author_embeds.reshape(author_embeds.shape[0], 1, author_embeds.shape[1]).double()
    #         author_embeds_hidden = self.author_embed_network(author_embeds)
    #         author_embeds_hidden = self.author_embed_layernorm(author_embeds_hidden)
    #         hidden_states = torch.cat([hidden_states, author_embeds_hidden], axis=1)
    #
    #     # decoder layers
    #     all_hidden_states = () if output_hidden_states else None
    #     all_self_attns = () if output_attentions else None
    #     all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
    #     next_decoder_cache = () if use_cache else None
    #
    #     # check if head_mask has a correct number of layers specified if desired
    #     if head_mask is not None:
    #         assert head_mask.size()[0] == (
    #             len(self.layers)
    #         ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
    #     for idx, decoder_layer in enumerate(self.layers):
    #         # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
    #         if output_hidden_states:
    #             all_hidden_states += (hidden_states,)
    #         dropout_probability = random.uniform(0, 1)
    #         if self.training and (dropout_probability < self.layerdrop):
    #             continue
    #
    #         past_key_value = past_key_values[idx] if past_key_values is not None else None
    #
    #         if getattr(self.config, "gradient_checkpointing", False) and self.training:
    #
    #             if use_cache:
    #                 print(
    #                     "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
    #                     "`use_cache=False`..."
    #                 )
    #                 use_cache = False
    #
    #             def create_custom_forward(module):
    #                 def custom_forward(*inputs):
    #                     # None for past_key_value
    #                     return module(*inputs, output_attentions, use_cache)
    #
    #                 return custom_forward
    #
    #             layer_outputs = torch.utils.checkpoint.checkpoint(
    #                 create_custom_forward(decoder_layer),
    #                 hidden_states,
    #                 attention_mask,
    #                 encoder_hidden_states,
    #                 encoder_attention_mask,
    #                 head_mask[idx] if head_mask is not None else None,
    #                 encoder_head_mask[idx] if encoder_head_mask is not None else None,
    #                 None,
    #             )
    #         else:
    #
    #             layer_outputs = decoder_layer(
    #                 hidden_states,
    #                 attention_mask=attention_mask,
    #                 encoder_hidden_states=encoder_hidden_states,
    #                 encoder_attention_mask=encoder_attention_mask,
    #                 layer_head_mask=(head_mask[idx] if head_mask is not None else None),
    #                 encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
    #                 past_key_value=past_key_value,
    #                 output_attentions=output_attentions,
    #                 use_cache=use_cache,
    #             )
    #         hidden_states = layer_outputs[0]
    #
    #         if use_cache:
    #             next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
    #
    #         if output_attentions:
    #             all_self_attns += (layer_outputs[1],)
    #
    #             if encoder_hidden_states is not None:
    #                 all_cross_attentions += (layer_outputs[2],)
    #
    #     # add hidden states from the last decoder layer
    #     if output_hidden_states:
    #         all_hidden_states += (hidden_states,)
    #
    #     next_cache = next_decoder_cache if use_cache else None
    #     if not return_dict:
    #         return tuple(
    #             v
    #             for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
    #             if v is not None
    #         )
    #     return BaseModelOutputWithPastAndCrossAttentions(
    #         last_hidden_state=hidden_states,
    #         past_key_values=next_cache,
    #         hidden_states=all_hidden_states,
    #         attentions=all_self_attns,
    #         cross_attentions=all_cross_attentions,
    #     )

class BartAuthorTextModel(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.author_embed_module = config.__dict__['author_embed_module']
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        if(self.author_embed_module in {'encoder', 'encoder_decoder'}):
            self.encoder = AuthorTextEncoder(config, self.shared)
        if(self.author_embed_module in {'decoder', 'encoder_decoder'}):
            self.decoder = AuthorTextDecoder(config, self.shared)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        author_embeds=None,
        attention_mask=None,
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

        ## increase attention size (for encoder) to include author token + embed
        if (author_embeds is not None and self.author_embed_module in {'encoder', 'encoder_decoder'}):
            attention_pad_len = 2 # add 2 slots: one for special token and one for author embed
            clean_attention_mask = []
            # print(f'attention mask before cleaning has shape = {attention_mask.shape}')
            for attention_mask_row_i in attention_mask:
                # print(f'pre-clean attention mask row has shape {attention_mask_row_i.shape}')
                extra_attention_mask_i = torch.ones(attention_pad_len).to(attention_mask_row_i.device)
                padding_idx_i = np.where(attention_mask_row_i.cpu() == 0)[0]
                # print(f'attention mask padding idx = {padding_idx_i}')
                # clip max-length sequence, i.e. where there is no padding left
                if(len(attention_mask_row_i) > self.config.max_position_embeddings-attention_pad_len):
                    clip_len_i = len(attention_mask_row_i) - self.config.max_position_embeddings + attention_pad_len
                    attention_mask_row_i = attention_mask_row_i[:-clip_len_i]
                    # print(f'max-length attention mask after fix = {attention_mask_row_i.shape}')
                if(len(padding_idx_i) == 0):
                    attention_mask_row_i = torch.cat([attention_mask_row_i, extra_attention_mask_i])
                else:
                    attention_mask_row_i = torch.cat([attention_mask_row_i[:padding_idx_i[0]], extra_attention_mask_i, attention_mask_row_i[(padding_idx_i[0]):]])
                # print(f'clean attention mask row has shape {attention_mask_row_i.shape}')
                clean_attention_mask.append(attention_mask_row_i)
            attention_mask = torch.cat([x.unsqueeze(0) for x in clean_attention_mask])
            # tmp debugging
            # print(f'attention mask after cleaning has shape {attention_mask.shape}')
            # if (attention_mask.shape[1] == self.config.max_position_embeddings):
            #     attention_mask = attention_mask[:, :-1]
            # author_attn = torch.ones(attention_mask.shape[0], 1)
            # author_attn = author_attn.to(attention_mask.device)
            # # tmp debugging
            # # print(f'attention mask dimensions={attention_mask.shape}')
            # # print(f'author attention mask dimensions={author_attn.shape}')
            # attention_mask = torch.cat([attention_mask, author_attn], axis=1)
            # print(f'combined attention mask dimensions={attention_mask.shape}')

        if encoder_outputs is None:
            if(self.author_embed_module in {'encoder', 'encoder_decoder'}):
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    author_embeds=author_embeds,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
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
        if (self.author_embed_module in {'decoder', 'encoder_decoder'}):
            # tmp debugging
            # print(f'encoder hidden state has shape {encoder_outputs[0].shape}')
            # print(f'encoder attention has shape {attention_mask.shape}')
            # print(f'decoder attention = {decoder_attention_mask}')
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                author_embeds=author_embeds,
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
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     author_embeds=(author_embeds if self.author_embed_module == 'decoder' else None),
        #     attention_mask=decoder_attention_mask,
        #     encoder_hidden_states=encoder_outputs[0],
        #     encoder_attention_mask=attention_mask,
        #     # head_mask=decoder_head_mask,
        #     # encoder_head_mask=head_mask,
        #     past_key_values=past_key_values,
        #     inputs_embeds=decoder_inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

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

class AuthorTextGenerationModel(BartForConditionalGeneration):

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartAuthorTextModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()
    #
    # def get_encoder(self):
    #     return self.model.get_encoder()
    #
    # def get_decoder(self):
    #     return self.model.get_decoder()
    #
    # def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
    #     new_embeddings = super().resize_token_embeddings(new_num_tokens)
    #     self._resize_final_logits_bias(new_num_tokens)
    #     return new_embeddings
    #
    # def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
    #     old_num_tokens = self.final_logits_bias.shape[-1]
    #     if new_num_tokens <= old_num_tokens:
    #         new_bias = self.final_logits_bias[:, :new_num_tokens]
    #     else:
    #         extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
    #         new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
    #     self.register_buffer("final_logits_bias", new_bias)
    #
    # def get_output_embeddings(self):
    #     return self.lm_head
    #
    # def set_output_embeddings(self, new_embeddings):
    #     self.lm_head = new_embeddings
    #
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        author_embeds=None,
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

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            author_embeds=author_embeds,
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
            'author_embeds' : kwargs['author_embeds'],
        }

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        """
        Get encoder output for generation.
        NOTE: if model is on "decoder" mode then
        don't feed author embeds to encoder!

        :param input_ids:
        :param model_kwargs:
        :return:
        """
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or
                        argument.startswith("cross_attn") or
                        (argument=='author_embeds' and self.model.author_embed_module == 'decoder')
                        )
            }
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids,
                                                                   return_dict=True,
                                                                   **encoder_kwargs)
        return model_kwargs

    # def generate(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     max_length: Optional[int] = None,
    #     min_length: Optional[int] = None,
    #     do_sample: Optional[bool] = None,
    #     early_stopping: Optional[bool] = None,
    #     num_beams: Optional[int] = None,
    #     temperature: Optional[float] = None,
    #     top_k: Optional[int] = None,
    #     top_p: Optional[float] = None,
    #     repetition_penalty: Optional[float] = None,
    #     bad_words_ids: Optional[Iterable[int]] = None,
    #     bos_token_id: Optional[int] = None,
    #     pad_token_id: Optional[int] = None,
    #     eos_token_id: Optional[int] = None,
    #     length_penalty: Optional[float] = None,
    #     no_repeat_ngram_size: Optional[int] = None,
    #     num_return_sequences: Optional[int] = None,
    #     decoder_start_token_id: Optional[int] = None,
    #     use_cache: Optional[bool] = None,
    #     num_beam_groups: Optional[int] = None,
    #     diversity_penalty: Optional[float] = None,
    #     prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     output_scores: Optional[bool] = None,
    #     return_dict_in_generate: Optional[bool] = None,
    #     **model_kwargs,
    # ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
    #     pass
    #
    #
    # def prepare_inputs_for_generation(
    #     self,
    #     decoder_input_ids,
    #     past=None,
    #     attention_mask=None,
    #     head_mask=None,
    #     use_cache=None,
    #     encoder_outputs=None,
    #     **kwargs
    # ):
    #     # cut decoder_input_ids if past is used
    #     if past is not None:
    #         decoder_input_ids = decoder_input_ids[:, -1:]
    #
    #     return {
    #         "input_ids": None,  # encoder_outputs is defined. input_ids not needed
    #         "encoder_outputs": encoder_outputs,
    #         "past_key_values": past,
    #         "decoder_input_ids": decoder_input_ids,
    #         "attention_mask": attention_mask,
    #         "head_mask": head_mask,
    #         "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
    #     }
    #
    # def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
    #     return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
    #
    # @staticmethod
    # def _reorder_cache(past, beam_idx):
    #     reordered_past = ()
    #     for layer_past in past:
    #         # cached cross_attention states don't have to be reordered -> they are always the same
    #         reordered_past += (
    #             tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
    #         )
    #     return reordered_past
