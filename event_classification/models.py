import torch
from torch import nn
from torch.nn import functional as F
import copy
from transformers import BertModel, BertPreTrainedModel
# from transformers import add_start_docstrings, add_start_docstrings_to_callable

# from event_classification.loss import FocalLoss, DSCLoss, DiceLoss, LabelSmoothingCrossEntropy
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss


class BertForSequenceMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, config.num_labels)
        self.max_pool = nn.MaxPool1d(512)
        self.avg_pool = nn.AvgPool1d(512)
        self.init_weights()
        for param in self.bert.parameters():
            param.requires_grad = True

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            label_mask=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        """
        pooling_layer = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]    # (batch_size, hidden_size)
        last_hidden_state = outputs[0]  # (batch_size, sequence_length, hidden_size)

        # output_vectors = [pooled_output]
        # # pooling_mean
        # input_mask_expanded = attention_mask.unsqueeze(-1)
        # input_mask_expanded = input_mask_expanded.expand(last_hidden_state.size()).float()
        # sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # sum_mask = input_mask_expanded.sum(1)
        # sum_mask = torch.clamp(sum_mask, min=1e-9)
        # output_vectors.append(sum_embeddings / sum_mask)
        # # pooling_max
        # input_mask_expanded = attention_mask.unsqueeze(-1)
        # input_mask_expanded = input_mask_expanded.expand(last_hidden_state.size()).bool()
        # input_mask_expanded = ~ input_mask_expanded
        # last_hidden_state.masked_fill(mask=input_mask_expanded, value=torch.tensor(-1e9))  # Set padding tokens to large negative value
        # max_over_time = torch.max(last_hidden_state, 1)[0]
        # output_vectors.append(max_over_time)

        output_vectors = [pooled_output]
        if pooling_layer:
            input_mask_expanded = attention_mask.unsqueeze(-1)
            input_mask_expanded = input_mask_expanded.expand(last_hidden_state.size()).float()
            masked_last_hidden_state = last_hidden_state * input_mask_expanded
            mean_pool_res = self.avg_pool(masked_last_hidden_state.transpose(1, 2)).squeeze()
            input_mask_expanded = attention_mask.unsqueeze(-1)
            input_mask_expanded = input_mask_expanded.expand(last_hidden_state.size()).bool()
            input_mask_expanded = ~ input_mask_expanded
            last_hidden_state.masked_fill(mask=input_mask_expanded, value=torch.tensor(-1e9))  # Set padding tokens to large negative value
            max_pool_res = self.max_pool(last_hidden_state.transpose(1, 2)).squeeze()
            output_vectors.append(mean_pool_res)
            output_vectors.append(max_pool_res)

        output_vector = torch.cat(output_vectors, 1)  # (batch_size, hidden_size*3)
        output_vector = self.dropout(output_vector)
        logits = self.classifier(output_vector)    # (batch_size, num_labels)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        label_count = torch.tensor([99, 63, 69, 96, 154, 299, 197, 1242, 300, 151, 109, 33, 121, 107, 61, 134, 65, 827, 79, 274, 298, 63, 99, 170, 105, 727, 82, 268, 238, 177, 128, 80, 110, 48, 75, 122, 210, 287, 462, 325, 138, 2004, 100, 145, 87, 356, 145, 82, 47, 93, 605, 197, 254, 74, 67, 63, 51, 191, 26, 64, 225, 128, 104, 83, 32], dtype=torch.float)
        label_weight = label_count.sum()/label_count
        # label_weight += min(label_weight)*3
        # label_weight /= min(label_weight).clone()
        label_weight = label_weight.cuda()

        if labels is not None:
            # loss_fct = FocalLoss()
            loss_fct = BCEWithLogitsLoss()
            # loss_fct = BCEWithLogitsLoss(weight=label_weight)
            loss = loss_fct(logits, labels.float())
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForSequenceMultiLabelClassificationWithColumn(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            label_mask=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[0]      # (batch_size, sequence_length, hidden_size)
        _batch_size, hidden_size = pooled_output.shape[0], pooled_output.shape[2]

        pooled_output = self.dropout(pooled_output)
        types_output = pooled_output.masked_select(label_mask.unsqueeze(-1).bool()).reshape(_batch_size, self.num_labels, hidden_size)  # (batch_size, num_labels, hidden_size)
        logits = self.linear(types_output).squeeze()    # (batch_size, num_labels)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        label_count = torch.tensor([99, 63, 69, 96, 154, 299, 197, 1242, 300, 151, 109, 33, 121, 107, 61, 134, 65, 827, 79, 274, 298, 63, 99, 170, 105, 727, 82, 268, 238, 177, 128, 80, 110, 48, 75, 122, 210, 287, 462, 325, 138, 2004, 100, 145, 87, 356, 145, 82, 47, 93, 605, 197, 254, 74, 67, 63, 51, 191, 26, 64, 225, 128, 104, 83, 32], dtype=torch.float)
        label_weight = label_count.sum()/label_count
        # label_weight += min(label_weight)*3
        # label_weight /= min(label_weight).clone()
        label_weight = label_weight.cuda()

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            # loss_fct = BCEWithLogitsLoss(weight=label_weight)
            loss = loss_fct(logits, labels.float())
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)