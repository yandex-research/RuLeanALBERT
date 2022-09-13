import torch
import torch.nn as nn

from transformers import PreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super().__init__()

        layers = [nn.Dropout(dropout_rate)]
        if use_activation:
            layers.append(nn.Tanh())
        layers.append(nn.Linear(input_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SpanClassificationModel(nn.Module):
    def __init__(self, backbone: PreTrainedModel, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.config = backbone.config

        hidden_size = self.config.hidden_size

        if hasattr(self.config, "classifier_dropout"):
            dropout_rate = (
                self.config.classifier_dropout if self.config.classifier_dropout is not None
                else self.config.hidden_dropout_prob
            )
        else:
            dropout_rate = self.config.classifier_dropout_prob

        self.cls_fc_layer = FCLayer(hidden_size, hidden_size, dropout_rate)
        self.e1_fc_layer = FCLayer(hidden_size, hidden_size, dropout_rate)
        self.e2_fc_layer = FCLayer(hidden_size, hidden_size, dropout_rate)
        self.label_classifier = FCLayer(hidden_size * 3, num_labels, dropout_rate, use_activation=False)

    @staticmethod
    def _entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(2)  # [batch_size, seq_len, 1]
        length_tensor = (e_mask_unsqueeze != 0).sum(dim=1)  # [batch_size, 1]

        sum_vector = (hidden_output * e_mask_unsqueeze).sum(1)
        avg_vector = sum_vector / length_tensor
        return avg_vector

    def forward(self, input_ids, attention_mask, e1_mask, e2_mask, token_type_ids, labels=None):
        # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output, pooled_output, *outputs = self.backbone(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False,
        )

        # Average
        hidden_first = self._entity_average(sequence_output, e1_mask)
        hidden_second = self._entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)
        hidden_first = self.e1_fc_layer(hidden_first)
        hidden_second = self.e2_fc_layer(hidden_second)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, hidden_first, hidden_second], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + tuple(outputs)  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class EntityChoiceModel(nn.Module):
    def __init__(self, backbone: PreTrainedModel):
        super().__init__()
        self.backbone = backbone

        hidden_size = backbone.config.hidden_size

        if hasattr(backbone.config, "classifier_dropout"):
            dropout_rate = (
                backbone.config.classifier_dropout if backbone.config.classifier_dropout is not None
                else backbone.config.hidden_dropout_prob
            )
        else:
            dropout_rate = backbone.config.classifier_dropout_prob

        self.clf = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )

        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input_ids, attention_mask, entity_mask, token_type_ids, labels=None):
        sequence_output, pooled_output, *outputs = self.backbone(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False
        )

        entity_mask = entity_mask.unsqueeze(3)
        entity_lengths = entity_mask.sum(2)

        # [batch_size, num_entities, seq_len, hid_dim]
        embeds_for_entities = sequence_output.unsqueeze(1) * entity_mask
        aggregated_embeds = embeds_for_entities.sum(2) / (entity_lengths + 1e-8)

        # [batch_size, num_entities]
        logits = self.clf(aggregated_embeds).squeeze(2)

        # due to padding, we might have rows without entity indices at all
        # thus, we need to mask the logits/loss for them

        # [batch_size, num_entities]
        present_entities = (entity_lengths.squeeze(2) != 0).to(logits.dtype)
        logits = logits * present_entities - 10000.0 * (1 - present_entities)

        outputs = (logits,) + tuple(outputs)  # add hidden states and attention if they are here

        if labels is not None:
            # do not penalize predictions for padded entities
            label_mask = (labels != -1)

            # BCEWithLogitsLoss requires targets to be from 0 to 1
            labels[~label_mask] = 0

            loss = self.loss(logits, labels.to(logits.dtype))

            reduced_loss = ((loss * label_mask).sum(1) / (label_mask.sum(1) + 1e-8)).mean()
            outputs = (reduced_loss,) + outputs

        return outputs
