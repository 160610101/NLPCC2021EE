import torch
from torch import nn

from transformers import BertModel, BertPreTrainedModel
# from transformers import add_start_docstrings, add_start_docstrings_to_callable

from role_extraction_ner.loss import FocalLoss, DSCLoss, DiceLoss, LabelSmoothingCrossEntropy
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss


class BertForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.start_end_pos_classifier = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.start_end_pos_classifier.weight)
        self.start_end_pos_classifier.bias.data.fill_(0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_labels=None,
        end_labels=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # TODO: mask strategy, loss func, net structure
        sequence_output = outputs[0]    # (batch_size, sequence_length, hidden_size)
        sequence_output = self.dropout(sequence_output)
        start_end_pos_logits = self.start_end_pos_classifier(sequence_output)    # (batch_size, sequence_length, 2)
        start_end_pos_logits = start_end_pos_logits * attention_mask.unsqueeze(-1).expand(-1, -1, 2)
        outputs = (start_end_pos_logits,) + outputs[2:]  # add hidden states and attention if they are here

        start_positions = start_labels.unsqueeze(2)
        end_positions = end_labels.unsqueeze(2)
        start_end_pos_labels = torch.cat((start_positions, end_positions), 2)

        loss_fct = BCEWithLogitsLoss()
        total_loss = loss_fct(start_end_pos_logits, start_end_pos_labels.float())
        output = (total_loss,) + outputs
        return output       # loss, logits, _, _



        # start_logits, end_logits = logits.split(1, dim=-1)  # (batch_size, sequence_length, 1), (batch_size, sequence_length, 1)
        # start_logits = start_logits.squeeze(-1)      # (batch_size, sequence_length)
        # end_logits = end_logits.squeeze(-1)          # (batch_size, sequence_length)
        #
        # total_loss = None
        # if start_positions is not None and end_positions is not None:
        #     # If we are on multi-GPU, split add a dimension
        #     if len(start_positions.size()) > 1:
        #         start_positions = start_positions.squeeze(-1)
        #     if len(end_positions.size()) > 1:
        #         end_positions = end_positions.squeeze(-1)
        #     # sometimes the start/end positions are outside our model inputs, we ignore these terms
        #     ignored_index = start_logits.size(1)
        #     start_positions.clamp_(0, ignored_index)    # (batch_size)
        #     end_positions.clamp_(0, ignored_index)      # (batch_size)
        #
        #     loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        #     start_loss = loss_fct(start_logits, start_positions)
        #     end_loss = loss_fct(end_logits, end_positions)
        #     total_loss = (start_loss + end_loss) / 2
        #
        # output = (start_logits, end_logits) + outputs[2:]
        # return ((total_loss,) + output) if total_loss is not None else output



class BertForQuestionAnsweringMultiTask(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config)
        self.start_end_pos_classifier = nn.Linear(config.hidden_size, 2)  # TODO 双向LSTM输出，要维度乘2
        # self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True, dropout=config.hidden_dropout_prob, bidirectional=True)
        self.answerable_classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.start_end_pos_classifier.weight)
        self.start_end_pos_classifier.bias.data.fill_(0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_labels=None,
        end_labels=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # TODO: mask strategy, loss func, net structure
        sequence_output = outputs[0]  # (batch_size, sequence_length, hidden_size)
        pooled_output = outputs[1]  # (batch_size, hidden_size)

        sequence_output = self.dropout(sequence_output)
        # sequence_output, _ = self.lstm(sequence_output)
        start_end_pos_logits = self.start_end_pos_classifier(sequence_output)  # (batch_size, sequence_length, 2)
        # start_end_pos_logits = start_end_pos_logits * token_type_ids.unsqueeze(-1).expand(-1,-1,2)
        start_end_pos_logits = start_end_pos_logits * attention_mask.unsqueeze(-1).expand(-1, -1, 2)
        outputs = (start_end_pos_logits,) + outputs[2:]  # add hidden states and attention if they are here

        start_positions = start_labels.unsqueeze(2)
        end_positions = end_labels.unsqueeze(2)
        start_end_pos_labels = torch.cat((start_positions, end_positions), 2)

        loss_fct = BCEWithLogitsLoss()
        role_loss = loss_fct(start_end_pos_logits, start_end_pos_labels.float())

        output_vector = self.dropout(pooled_output)
        answerable_output = self.answerable_classifier(output_vector)  # (batch_size, 1)
        answerable_label = (torch.any(start_labels, 1) + torch.any(end_labels, 1)).float().unsqueeze(-1)
        answerable_loss = loss_fct(answerable_output, answerable_label)

        print('role_loss:', role_loss.data.item())
        print('answerable_loss:', answerable_loss.data.item())
        total_loss = (role_loss + 0.1 * answerable_loss) / 2

        output = (total_loss,) + outputs
        return output  # total_loss, start_end_pos_logits, _, _

