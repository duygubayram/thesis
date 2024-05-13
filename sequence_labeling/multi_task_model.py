'''
Multi task model for sequence labeling.

Adaptation of the code from:
https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240

The current version is aimed at and tested for token classification (sequence labeling) tasks only,
so support for other types of tasks (e.g. text classification, aka 'sequence classification' below)
would require some adaptation.

'''

import os
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch import nn
from transformers import AutoModel

@dataclass
class Task:
    id: int
    name: str
    type: str
    num_labels: int

class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name_or_path: str, tasks: List[Task], task_weights: Dict[int, float] = None):
        '''
        Multi task model using a transformer encoder backbone, and sequence labeling or classification heads.
        Currently the model is tested only for sequence labeling tasks.
        :param encoder_name_or_path: huggingface model name or path to model
        :param task_weights: task_id -> weight, for weighted loss
        '''
        super().__init__()
        self._task_weights = task_weights
        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        self._deberta = 'deberta' in encoder_name_or_path.lower()
        self.output_heads = nn.ModuleDict()
        for task in tasks:
            decoder = self._create_output_head(self.encoder.config.hidden_size, task)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.id)] = decoder

    @staticmethod
    def _create_output_head(encoder_hidden_size: int, task):
        if task.type == "seq_classification":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        elif task.type == "token_classification":
            return TokenClassificationHead(encoder_hidden_size, task.num_labels)
        else:
            raise NotImplementedError()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            task_ids=None,
            **kwargs,
    ):
        if not self._deberta:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
            sequence_output, pooled_output = outputs[:2]
        else: # deberta does not support head_mask
            outputs = self.encoder(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )
            sequence_output = outputs.last_hidden_state
            # this is because the pooled_output is a necessary argument, but it is used only for sequence classification
            pooled_output = sequence_output

        unique_task_ids_list = torch.unique(task_ids).tolist()

        loss_list = []
        logits = None
        for unique_task_id in unique_task_ids_list:

            task_id_filter = task_ids == unique_task_id
            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            if labels is not None:
                if self._task_weights is not None:
                    task_loss = task_loss * self._task_weights[unique_task_id]
                loss_list.append(task_loss)

        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (logits, outputs[2:])

        if loss_list:
            loss = torch.stack(loss_list)
            if self._task_weights is not None:
                outputs = (loss.sum(),) + outputs
            else:
                outputs = (loss.mean(),) + outputs

        return outputs

    def save_model(self, save_dir):
        """
        Saves the model's state and the encoder's state.

        :param save_dir: The directory where to save the model and encoder.
        """
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save model state
        model_save_path = os.path.join(save_dir, "model_state.pt")
        model_state = {
            'state_dict': self.state_dict(),
            '_deberta': self._deberta
        }
        torch.save(model_state, model_save_path)
        # Save the encoder (transformer model)
        encoder_save_path = os.path.join(save_dir, "encoder")
        self.encoder.save_pretrained(encoder_save_path)

    @classmethod
    def load_model(cls, load_dir, tasks, task_weights=None):
        """
        Loads the model's state and the encoder's state.

        :param load_dir: The directory from where to load the model and encoder.
        :param tasks: The list of tasks (required for model instantiation).
        :param task_weights: task weights (optional).

        :return: An instance of MultiTaskModel loaded from disk.
        """
        # Load the encoder (transformer model)
        encoder_load_path = os.path.join(load_dir, "encoder")
        # Create a model instance
        model = cls(encoder_load_path, tasks, task_weights)
        # Load model state
        model_load_path = os.path.join(load_dir, "model_state.pt")
        loaded_state = torch.load(model_load_path)
        model.load_state_dict(loaded_state['state_dict'])
        model._deberta = loaded_state['_deberta']
        return model


class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs
    ):
        sequence_output_dropout = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_dropout)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()

            labels = labels.long()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, 0]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.long().view(-1)
            )

        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()


