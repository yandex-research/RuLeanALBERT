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
from functools import lru_cache

import torch
import torch.nn as nn
from transformers.file_utils import add_start_docstrings
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.albert import AlbertConfig
from transformers.models.albert.modeling_albert import (
    ACT2FN,
    ALBERT_START_DOCSTRING,
    AlbertForPreTraining,
    AlbertLayerGroup,
    AlbertMLMHead,
    AlbertModel,
    AlbertSOPHead,
    AlbertTransformer
)
from transformers.utils import logging

from src.models.config import LeanAlbertConfig
from src.modules.attn import LeanSelfAttention, RotaryAttentionCore
from src.modules.ffn import LeanFFN
from src.modules.rotary import RotaryEmbeddings

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LeanAlbertConfig"
_TOKENIZER_FOR_DOC = "AlbertTokenizer"


def get_input_embedding(config: LeanAlbertConfig):
    if config.position_embedding_type == "absolute":
        return nn.Embedding(config.max_position_embeddings, config.embedding_size)
    elif config.position_embedding_type == "rotary":
        return None
    else:
        raise NotImplementedError(f"Unsupported embedding type: {config.position_embedding}")


@lru_cache()
def get_attention_core(config: LeanAlbertConfig):
    if config.position_embedding_type == "absolute":
        return None
    elif config.position_embedding_type == "rotary":
        rotary_emb = RotaryEmbeddings(config.hidden_size // config.num_attention_heads, config.rotary_embedding_base)
        return RotaryAttentionCore(
            config.hidden_size,
            config.num_attention_heads,
            rotary_emb,
            attention_probs_dropout=config.attention_probs_dropout_prob,
        )
    else:
        raise NotImplementedError(f"Unsupported embedding type: {config.position_embedding_type}")


class LeanAlbertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.position_embeddings = get_input_embedding(config)

        self.layernorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
                position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LeanAlbertLayer(nn.Module):
    def __init__(self, config: LeanAlbertConfig):
        super().__init__()

        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.attention = LeanSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            attention_core=get_attention_core(config),
            hidden_dropout_prob=config.hidden_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.ffn = LeanFFN(
            config.hidden_size,
            config.intermediate_size,
            activation=ACT2FN[config.hidden_act],
            gated=config.hidden_act_gated,
            layer_norm_eps=config.layer_norm_eps,
            dropout=config.hidden_dropout_prob,
        )

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attention_output, *extras = self.attention(hidden_states, attention_mask, output_attentions)
        ffn_output = self.ffn(attention_output)
        return (ffn_output, attention_output, *extras)


class LeanAlbertLayerGroup(AlbertLayerGroup):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.albert_layers = nn.ModuleList([LeanAlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False
    ):
        if any(head_mask):
            raise NotImplementedError(f"head mask was provided, but it is not supported")

        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, output_attentions)
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class LeanAlbertTransformer(AlbertTransformer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList(
            [LeanAlbertLayerGroup(config) for _ in range(config.num_hidden_groups)]
        )
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group: (group_idx + 1) * layers_per_group],
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=self.post_layer_norm(hidden_states),
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


@add_start_docstrings(
    "The bare LeanALBERT Model transformer outputting raw hidden-states without any specific head on top.",
    ALBERT_START_DOCSTRING,
)
class LeanAlbertModel(AlbertModel):
    config_class = LeanAlbertConfig

    def __init__(self, config: AlbertConfig, add_pooling_layer=True):
        PreTrainedModel.__init__(self, config)

        self.config = config
        self.embeddings = LeanAlbertEmbeddings(config)
        self.encoder = LeanAlbertTransformer(config)

        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        self.init_weights()


class LeanAlbertForPreTraining(AlbertForPreTraining, PreTrainedModel):
    config_class = LeanAlbertConfig
    base_model_prefix = "albert"

    def __init__(self, config: AlbertConfig):
        PreTrainedModel.__init__(self, config)

        self.albert = LeanAlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.sop_classifier = AlbertSOPHead(config)

        self.init_weights()
