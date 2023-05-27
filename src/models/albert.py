# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ALBERT modules that do not hog your GPU memory """

import torch
from torch import nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import AlbertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.albert.modeling_albert import AlbertForPreTrainingOutput
from transformers.utils import logging

from src.models.transformer import GradientCheckpointingMixin, LeanTransformer, LeanTransformerConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LeanAlbertConfig"
_TOKENIZER_FOR_DOC = "AlbertTokenizer"


class LeanAlbertConfig(LeanTransformerConfig):
    def __init__(
        self,
        *args,
        vocab_size: int = 30000,
        embedding_size: int = 128,
        classifier_dropout_prob: float = 0.1,
        type_vocab_size: int = 2,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs
    ):
        super().__init__(
            *args,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.classifier_dropout_prob = classifier_dropout_prob
        self.type_vocab_size = type_vocab_size


class LeanAlbertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: LeanTransformerConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)

        self.token_type_embeddings = config.get_token_type_embeddings()
        self.position_embeddings = config.get_input_position_embeddings()

        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.embedding_size != config.hidden_size:
            self.embedding_hidden_mapping = nn.Linear(config.embedding_size, config.hidden_size)

        if self.position_embeddings is not None:
            # position_ids (1, len position emb) is contiguous in memory and exported when serialized
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embeddings is not None:
            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        if hasattr(self, "embedding_hidden_mapping"):
            embeddings = self.embedding_hidden_mapping(embeddings)
        return embeddings


class LeanAlbertModel(GradientCheckpointingMixin, PreTrainedModel):
    config_class = LeanAlbertConfig
    base_model_prefix = "lean_albert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: config_class, add_pooling_layer=True):
        super().__init__(config)

        self.config = config
        self.embeddings = LeanAlbertEmbeddings(config)
        self.transformer = LeanTransformer(config)

        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _init_weights(self, module: nn.Module):
        return self.config.init_weights(module)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert head_mask is None and output_attentions is None and output_hidden_states is None, "not implemented"
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device, dtype=int)
        else:
            assert not torch.is_floating_point(attention_mask), "The model requires boolean or int mask with 0/1 entries"

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        transformer_outputs = self.transformer(embedding_output, extended_attention_mask)

        sequence_output = transformer_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + transformer_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class AlbertMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layer_norm = nn.LayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = config.get_activation_callable()
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states

        return prediction_scores

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class AlbertSOPHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled_output):
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


class LeanAlbertForPreTraining(GradientCheckpointingMixin, PreTrainedModel):
    config_class = LeanAlbertConfig
    base_model_prefix = "lean_albert"

    def __init__(self, config: config_class):
        super().__init__(config)

        self.albert = LeanAlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.sop_classifier = AlbertSOPHead(config)

    def get_input_embeddings(self):
        return self.albert.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.albert.embeddings.word_embeddings = new_embeddings

    def get_output_embeddings(self):
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.predictions.decoder = new_embeddings

    def _init_weights(self, module: nn.Module):
        return self.config.init_weights(module)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        sentence_order_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]

        prediction_scores = self.predictions(sequence_output)
        sop_scores = self.sop_classifier(pooled_output)

        total_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            if sentence_order_label is not None:
                sentence_order_loss = loss_fct(sop_scores.view(-1, 2), sentence_order_label.view(-1))
                total_loss = masked_lm_loss + sentence_order_loss
            else:
                total_loss = masked_lm_loss

        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return AlbertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            sop_logits=sop_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LeanAlbertForSequenceClassification(AlbertPreTrainedModel):
    config_class = LeanAlbertConfig
    base_model_prefix = "albert"

    def __init__(self, config: LeanAlbertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.albert = LeanAlbertModel(config, add_pooling_layer=False)

        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.classifier_dropout_prob),
            nn.Linear(config.hidden_size, self.config.num_labels)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
