import bisect
import collections
import logging
import os
import json
import numpy as np

logger = logging.getLogger(__name__)


class SquadExample(object):
    """example的格式定义"""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 start_label_list=None,
                 end_label_list=None,
                 ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.start_label_list = start_label_list
        self.end_label_list = end_label_list

    def __str__(self):
        return self.__repr__()

    def __repr__(self): # 显示属性
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % self.question_text
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_label_list:
            s += ", start_label_list: %d" % self.start_label_list
        if self.end_label_list:
            s += ", end_label_list: %d" % self.end_label_list
        return s


class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_label_ids=None,
                 end_label_ids=None,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_label_ids = start_label_ids
        self.end_label_ids = end_label_ids


def read_squad_examples(args, mode):
    # assert mode in ['test1', 'fold_*_train', 'fold_*_dev']
    if args.withO:
        if mode.startswith('test'):
            data_dir = args.data_dir
            file_path = os.path.join(data_dir, "role_spanWithO_{}.json".format(mode))
        else:
            data_dir = args.fold_data_dir
            file_path = os.path.join(data_dir, "{}.json".format(mode))
    else:
        data_dir = args.data_dir
        file_path = os.path.join(data_dir, "role_span_{}.json".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line == '\n' or line == '':
                continue
            line_json = json.loads(line)
            query, text, start_label_list, end_label_list = line_json['query'], line_json['text'], line_json['start_label_list'], line_json['end_label_list']
            assert len(text) == len(start_label_list) == len(end_label_list)
            examples.append(SquadExample(qas_id="{}-{}".format(mode, guid_index), question_text=query, doc_tokens=text, start_label_list=start_label_list, end_label_list=end_label_list))
            guid_index += 1
    return examples


def convert_examples_to_features(
        examples,
        tokenizer,
        mode,
        label_list,
        max_seq_length,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=0,       # BCEloss没有label的padding码
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    """比赛数据集句长都小于512，所以拼接后超过max_query_length的可能性较小，一旦发生则直接截断"""
    feature = []
    input_texts = []
    label_map = {label: i for i, label in enumerate(label_list)}    # 0,1
    for (example_index, example) in enumerate(examples):
        input_ids, token_type_ids = tokenizer(example.question_text)["input_ids"], tokenizer(example.question_text)["token_type_ids"]
        start_label_ids = [label_map['0']] * len(input_ids)
        end_label_ids = [label_map['0']] * len(input_ids)
        # input_mask = [0] * len(input_ids)
        for i in range(len(example.doc_tokens)):        # 按字符编码
            if len(input_ids) == max_seq_length-1:      # 超过最大长度则截断
                break
            b_token = example.doc_tokens[i]
            input_ids.append(tokenizer.convert_tokens_to_ids(b_token.lower()))      # 英文需统一为小写
            start_label_ids.append(label_map[example.start_label_list[i]])
            end_label_ids.append(label_map[example.end_label_list[i]])
            # input_mask.append(1)
        input_ids.append(tokenizer.convert_tokens_to_ids(sep_token))
        start_label_ids.append(pad_token_label_id)
        end_label_ids.append(pad_token_label_id)
        # input_mask.append(0)

        input_mask = [1] * len(input_ids)
        token_type_ids += [1] * (len(input_ids) - len(token_type_ids))

        input_texts.append("#{}#{}#".format(example.question_text, example.doc_tokens))         # 用#代替special token，保证对应字符长度为1

        assert len(input_ids) == len(token_type_ids) == len(start_label_ids) == len(end_label_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            start_label_ids = ([pad_token_label_id] * padding_length) + start_label_ids
            end_label_ids = ([pad_token_label_id] * padding_length) + end_label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            token_type_ids += [pad_token_segment_id] * padding_length
            start_label_ids += [pad_token_label_id] * padding_length
            end_label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(start_label_ids) == max_seq_length
        assert len(end_label_ids) == max_seq_length

        if example_index < 3:
            logger.info("*** Example ***")
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("start_label_ids: %s" % " ".join([str(x) for x in start_label_ids]))
            logger.info("end_label_ids: %s" % " ".join([str(x) for x in end_label_ids]))

        feature.append(InputFeatures(input_ids=input_ids,
                                     input_mask=input_mask,
                                     segment_ids=token_type_ids,
                                     start_label_ids=start_label_ids,
                                     end_label_ids=end_label_ids,))
    return feature, input_texts


def span_metrics(out_label_ids, preds):
    start_TP, start_TP_FP, start_TP_FN= 0, 0, 0
    end_TP, end_TP_FP, end_TP_FN = 0, 0, 0

    for i in range(out_label_ids.shape[0]):
        seq_start_label = np.argwhere(out_label_ids[i][:,0]).reshape(-1)    # shape = (x,)
        seq_end_label = np.argwhere(out_label_ids[i][:,1]).reshape(-1)
        seq_start_pred = np.argwhere(preds[i][:,0]).reshape(-1)
        seq_end_pred = np.argwhere(preds[i][:,1]).reshape(-1)

        start_TP += np.intersect1d(seq_start_label, seq_start_pred).shape[0]
        end_TP += np.intersect1d(seq_end_label, seq_end_pred).shape[0]
        start_TP_FP += seq_start_pred.shape[0]
        end_TP_FP += seq_end_pred.shape[0]
        start_TP_FN += seq_start_label.shape[0]
        end_TP_FN += seq_end_label.shape[0]

    start_P = start_TP/start_TP_FP if start_TP_FP else 0
    end_P = end_TP/end_TP_FP if end_TP_FP else 0
    start_R = start_TP/start_TP_FN if start_TP_FN else 0
    end_R = end_TP/end_TP_FN if end_TP_FN else 0
    start_f1, end_f1 = 2*start_P*start_R/(start_P+start_R) if start_P+start_R else 0, 2*end_P*end_R/(end_P+end_R) if end_P+end_R else 0
    precision, recall, f1 = (start_P+end_P)/2, (start_R+end_R)/2, (start_f1+end_f1)/2

    return precision, recall, f1


def convert_span_output_to_texts(span_outputs, span_input_texts):
    """
    设定规则，把SPAN结果处理成最终的论元文本，用于评价分数和输出预测结果
    :param span_outputs: ndarray, (batch_size, sequence_length, 2), 1/0
    :param span_labels: ndarray, (batch_size, sequence_length, 2), 1/0
    :return:
    """
    span_texts = [[] for i in range(len(span_outputs))]
    for i in range(span_outputs.shape[0]):
        start_pos_list = np.argwhere(span_outputs[i][:,0]).reshape(-1)
        end_pos_list = np.argwhere(span_outputs[i][:,1]).reshape(-1)
        # if not start_pos_list or not end_pos_list:
        #     continue
        # for end_pos in end_pos_list:
        #     if end_pos in start_pos_list:
        #         span_texts[i].append(span_texts[end_pos:end_pos+1])
        #     else:
        #         search_index = bisect.bisect_left(start_pos_list, end_pos)
        #         if search_index == 0:
        #             continue
        #         start_pos = start_pos_list[search_index-1]
        #         span_texts[i].append(span_texts[start_pos:end_pos+1])
        for end_pos in end_pos_list:  # 预测的start数目往往远多于end，所以以end为基准寻找span
            start_pos = start_pos_list[start_pos_list <= end_pos]  # 寻找小于等于end_pos的第一个start_pos，前提是存在start_pos<=end_pos
            if start_pos.any():  # 如果存在
                start_pos = start_pos[-1]
                _text = span_input_texts[i][start_pos:end_pos + 1]
                span_texts[i].append(_text)
    return span_texts






