"""
Model for question generation that includes copy-mechanism.

P(word | context) = P(word) + weight * P(copy)

Copied from tensorflow from here: https://github.com/xiongma/transformer-pointer-generator/blob/master/model.py
And from torch here: https://medium.com/@epwalsh10/incorporating-a-copy-mechanism-into-sequence-to-sequence-models-40917280b89d
And from transformers here: https://huggingface.co/transformers/_modules/transformers/models/bart/modeling_bart.html#BartForConditionalGeneration
"""
from typing import Tuple, List, Dict

import transformers
from torch.autograd.grad_mode import F
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, \
    BaseModelOutputWithPastAndCrossAttentions
import torch
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import BartDecoder, _expand_mask

# borrowing from here: https://github.com/laihuiyuan/pointer-generator/blob/6a727f4a2f314c2b47df9ce8838dca0de61bfcd4/models/layers.py
# class CopyDecoder(BartDecoder):
#
#     def __init__(self, config):
#         super().__init__(config)
#         # add copy layer
#         ## TODO: what is copy layer size??
#         # self.copy_layer = torch.nnLinear(self.config.decoder_output_dim, self.max_source_length)
#         self.copy_layer = torch.nn.Linear(self.config.hidden_dim*4 + self.config.embedding_size, 1)
#
#     def forward(self,
#                 input_ids=None,
#                 attention_mask=None,
#                 encoder_hidden_states=None,
#                 encoder_attention_mask=None,
#                 past_key_values=None,
#                 inputs_embeds=None,
#                 use_cache=None,
#                 output_attentions=None,
#                 output_hidden_states=None,
#                 return_dict=None,
#     ):
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         # retrieve input_ids and inputs_embeds
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError(
#                 "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#             input_ids = input_ids.view(-1, input_shape[-1])
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError(
#                 "You have to specify either decoder_input_ids or decoder_inputs_embeds")
#
#         # past_key_values_length
#         past_key_values_length = past_key_values[0][0].shape[
#             2] if past_key_values is not None else 0
#
#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
#
#         # create causal mask
#         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#         combined_attention_mask = None
#         if input_shape[-1] > 1:
#             combined_attention_mask = _make_causal_mask(
#                 input_shape, inputs_embeds.dtype,
#                 past_key_values_length=past_key_values_length
#             ).to(self.device)
#
#         if attention_mask is not None and combined_attention_mask is not None:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             combined_attention_mask = combined_attention_mask + _expand_mask(
#                 attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
#             )
#
#         # expand encoder attention mask
#         if encoder_hidden_states is not None and encoder_attention_mask is not None:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             encoder_attention_mask = _expand_mask(encoder_attention_mask,
#                                                   inputs_embeds.dtype,
#                                                   tgt_len=input_shape[-1])
#
#         # embed positions
#         positions = self.embed_positions(input_shape, past_key_values_length)
#
#         hidden_states = inputs_embeds + positions
#         hidden_states = self.layernorm_embedding(hidden_states)
#
#         hidden_states = F.dropout(hidden_states, p=self.dropout,
#                                   training=self.training)
#
#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         all_cross_attentions = () if output_attentions else None
#         next_decoder_cache = () if use_cache else None
#         for idx, decoder_layer in enumerate(self.layers):
#             # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)
#             dropout_probability = torch.random.uniform(0, 1)
#             if self.training and (dropout_probability < self.layerdrop):
#                 continue
#
#             past_key_value = past_key_values[
#                 idx] if past_key_values is not None else None
#
#             if getattr(self.config, "gradient_checkpointing", False):
#                 if use_cache:
#                     raise ValueError(
#                         "When using `gradient_checkpointing, make sure that `use_cache=False` and `config.use_cache=False`."
#                     )
#
#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         # None for past_key_value
#                         return module(*inputs, output_attentions, use_cache)
#
#                     return custom_forward
#
#                 layer_outputs = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(decoder_layer),
#                     hidden_states,
#                     combined_attention_mask,
#                     encoder_hidden_states,
#                     encoder_attention_mask,
#                     None,
#                 )
#             else:
#
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=combined_attention_mask,
#                     encoder_hidden_states=encoder_hidden_states,
#                     encoder_attention_mask=encoder_attention_mask,
#                     past_key_value=past_key_value,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                 )
#             hidden_states = layer_outputs[0]
#
#             if use_cache:
#                 next_decoder_cache += (
#                 layer_outputs[3 if output_attentions else 1],)
#
#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)
#                 all_cross_attentions += (layer_outputs[2],)
#
#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)
#
#         next_cache = next_decoder_cache if use_cache else None
#         if not return_dict:
#             return tuple(
#                 v
#                 for v in
#                 [hidden_states, next_cache, all_hidden_states, all_self_attns,
#                  all_cross_attentions]
#                 if v is not None
#             )
#         return BaseModelOutputWithPastAndCrossAttentions(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#             cross_attentions=all_cross_attentions,
#         )

class CopyGenerationModel(BartForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self._output_copying_layer = torch.nn.Linear(self.encoder_output_dim,
                                                     self.decoder_output_dim)

    def _get_copy_scores(self, encoder_outputs, decoder_hidden):
        trim_encoder_outputs = encoder_outputs[:, -1:1]
        copy_projection = self._output_copying_layer(trim_encoder_outputs)
        copy_projection = torch.tanh(copy_projection)
        copy_scores = copy_projection.bmm(decoder_hidden.unsqueeze(-1)).squeeze(-1)
        return copy_scores

    def _get_ll_contrib(
            self,
            generation_scores: torch.Tensor,
            generation_scores_mask: torch.BoolTensor,
            copy_scores: torch.Tensor,
            target_tokens: torch.Tensor,
            target_to_source: torch.Tensor,
            source_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the log-likelihood contribution from a single timestep.
        # Parameters
        generation_scores : `torch.Tensor`
            Shape: `(batch_size, target_vocab_size)`
        generation_scores_mask : `torch.BoolTensor`
            Shape: `(batch_size, target_vocab_size)`. This is just a tensor of 1's.
        copy_scores : `torch.Tensor`
            Shape: `(batch_size, source_sequence_length)`
        target_tokens : `torch.Tensor`
            Shape: `(batch_size,)`
        target_to_source : `torch.Tensor`
            Shape: `(batch_size, source_sequence_length)`
        source_mask : `torch.BoolTensor`
            Shape: `(batch_size, source_sequence_length)`
        # Returns
        Tuple[torch.Tensor, torch.Tensor]
            Shape: `(batch_size,), (batch_size, source_sequence_length)`
        """
        _, target_size = generation_scores.size()

        # The point of this mask is to just mask out all source token scores
        # that just represent padding. We apply the mask to the concatenation
        # of the generation scores and the copy scores to normalize the scores
        # correctly during the softmax.
        # shape: (batch_size, target_vocab_size + source_sequence_length)
        mask = torch.cat((generation_scores_mask, source_mask), dim=-1)
        # shape: (batch_size, target_vocab_size + source_sequence_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + source_sequence_length)
        log_probs = util.masked_log_softmax(all_scores, mask)
        # Calculate the log probability (`copy_log_probs`) for each token in the source sentence
        # that matches the current target token. We use the sum of these copy probabilities
        # for matching tokens in the source sentence to get the total probability
        # for the target token. We also need to normalize the individual copy probabilities
        # to create `selective_weights`, which are used in the next timestep to create
        # a selective read state.
        # shape: (batch_size, source_sequence_length)
        copy_log_probs = (
                log_probs[:, target_size:]
                + (
                        target_to_source.to(
                            log_probs.dtype) + util.tiny_value_of_dtype(
                    log_probs.dtype)
                ).log()
        )
        # Since `log_probs[:, target_size]` gives us the raw copy log probabilities,
        # we use a non-log softmax to get the normalized non-log copy probabilities.
        selective_weights = util.masked_softmax(log_probs[:, target_size:],
                                                target_to_source)
        # This mask ensures that item in the batch has a non-zero generation probabilities
        # for this timestep only when the gold target token is not OOV or there are no
        # matching tokens in the source sentence.
        # shape: (batch_size, 1)
        gen_mask = (target_tokens != self._oov_index) | (
                    target_to_source.sum(-1) == 0)
        log_gen_mask = (gen_mask + util.tiny_value_of_dtype(
            log_probs.dtype)).log().unsqueeze(-1)
        # Now we get the generation score for the gold target token.
        # shape: (batch_size, 1)
        generation_log_probs = log_probs.gather(1, target_tokens.unsqueeze(
            1)) + log_gen_mask
        # ... and add the copy score to get the step log likelihood.
        # shape: (batch_size, 1 + source_sequence_length)
        combined_gen_and_copy = torch.cat(
            (generation_log_probs, copy_log_probs), dim=-1)
        # shape: (batch_size,)
        step_log_likelihood = torch.logsumexp(combined_gen_and_copy)

        return step_log_likelihood, selective_weights

    def _gather_final_log_probs(
            self,
            generation_log_probs: torch.Tensor,
            copy_log_probs: torch.Tensor,
            source_to_target,
            source_token_ids,
            # state: Dict[str, torch.Tensor],
            smooth_val=1e-45,
    ) -> torch.Tensor:
        """
        Combine copy probabilities with generation probabilities for matching tokens.
        # Parameters
        generation_log_probs : `torch.Tensor`
            Shape: `(group_size, target_vocab_size)`
        copy_log_probs : `torch.Tensor`
            Shape: `(group_size, source_sequence_length)`
        state : `Dict[str, torch.Tensor]`
        # Returns
        torch.Tensor
            Shape: `(group_size, target_vocab_size + source_sequence_length)`.
        """
        _, source_sequence_length = source_to_target.size()
        source_token_ids = source_token_ids

        # shape: [(batch_size, *)]
        modified_log_probs_list: List[torch.Tensor] = []
        for i in range(source_sequence_length):
            # shape: (group_size,)
            copy_log_probs_slice = copy_log_probs[:, i]
            # `source_to_target` is a matrix of shape (group_size, source_sequence_length)
            # where element (i, j) is the vocab index of the target token that matches the jth
            # source token in the ith group, if there is one, or the index of the OOV symbol otherwise.
            # We'll use this to add copy scores to corresponding generation scores.
            # shape: (group_size,)
            source_to_target_slice = source_to_target[:, i]
            # The OOV index in the source_to_target_slice indicates that the source
            # token is not in the target vocab, so we don't want to add that copy score
            # to the OOV token.
            copy_log_probs_to_add_mask = source_to_target_slice != self._oov_index
            copy_log_probs_to_add = (
                    copy_log_probs_slice
                    + (
                            copy_log_probs_to_add_mask
                            + smooth_val
                    ).log()
            )
            # shape: (batch_size, 1)
            copy_log_probs_to_add = copy_log_probs_to_add.unsqueeze(-1)
            # shape: (batch_size, 1)
            selected_generation_log_probs = generation_log_probs.gather(
                1, source_to_target_slice.unsqueeze(-1)
            )
            combined_scores = torch.logsumexp(
                torch.cat(
                    (selected_generation_log_probs, copy_log_probs_to_add),
                    dim=1)
            )
            generation_log_probs = generation_log_probs.scatter(
                -1, source_to_target_slice.unsqueeze(-1),
                combined_scores.unsqueeze(-1)
            )
            # We have to combine copy scores for duplicate source tokens so that
            # we can find the overall most likely source token. So, if this is the first
            # occurence of this particular source token, we add the log_probs from all other
            # occurences, otherwise we zero it out since it was already accounted for.
            if i < (source_sequence_length - 1):
                # Sum copy scores from future occurences of source token.
                # shape: (group_size, source_sequence_length - i)
                source_future_occurences = source_token_ids[:,
                                           (i + 1):] == source_token_ids[
                                                        :, i
                                                        ].unsqueeze(-1)
                # shape: (group_size, source_sequence_length - i)
                future_copy_log_probs = (
                        copy_log_probs[:, (i + 1):]
                        + (
                                source_future_occurences + smooth_val
                        ).log()
                )
                # shape: (group_size, 1 + source_sequence_length - i)
                combined = torch.cat(
                    (copy_log_probs_slice.unsqueeze(-1), future_copy_log_probs),
                    dim=-1
                )
                # shape: (group_size,)
                copy_log_probs_slice = torch.logsumexp(combined)
            if i > 0:
                # Remove copy log_probs that we have already accounted for.
                # shape: (group_size, i)
                source_previous_occurences = source_token_ids[:,
                                             0:i] == source_token_ids[
                                                     :, i
                                                     ].unsqueeze(-1)
                # shape: (group_size,)
                duplicate_mask = source_previous_occurences.sum(dim=-1) == 0
                copy_log_probs_slice = (
                        copy_log_probs_slice
                        + (duplicate_mask + smooth_val).log()
                )

            # Finally, we zero-out copy scores that we added to the generation scores
            # above so that we don't double-count them.
            # shape: (group_size,)
            left_over_copy_log_probs = (
                    copy_log_probs_slice
                    + (
                            ~copy_log_probs_to_add_mask
                            + smooth_val
                    ).log()
            )
            modified_log_probs_list.append(
                left_over_copy_log_probs.unsqueeze(-1))
        modified_log_probs_list.insert(0, generation_log_probs)

        # shape: (group_size, target_vocab_size + source_sequence_length)
        modified_log_probs = torch.cat(modified_log_probs_list, dim=-1)

        return modified_log_probs

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = transformers.shift_tokens_right(
                    labels, self.config.pad_token_id,
                    self.config.decoder_start_token_id
                )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        ## add copy probabilities
        copy_scores = self._get_copy_scores(encoder_outputs, decoder_hidden)
        source_token_ids = 0 # TODO: convert input IDs to [0,1,2...] format where index indicates matching
        final_log_probs = self._gather_final_log_probs(generation_log_probs, copy_log_probs, input_ids, source_token_ids)

        # generation_score_mask = outputs[1]
        # log_likelihood, selective_weights = self._get_ll_contrib(
        #     lm_logits, generation_score_mask, copy_scores,
        # )

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

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
        pass