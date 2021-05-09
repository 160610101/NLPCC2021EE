import collections
import json
import string
import torch
try:
    from apex import amp
except ImportError:
    pass


class PLMConfig():
    def __init__(self):
        self.PLMpath = {
            "roberta": "G:/预训练模型/RoBERTa/RoBERTa_zh_L12_PyTorch",
            "roberta_large": "",
            "roberta_wwm": "G:/预训练模型/RoBERTa_wwm/chinese_roberta_wwm_ext",
            "roberta_wwm_large": "",
            "ernie": "G:/预训练模型/ERNIE1.0/ernie1.0",
            "macbert": "G:/预训练模型/MacBERT/MacBERT_base",
            "macbert_large": "",
        }

    def get_model_names(self):
        return self.PLMpath.keys()


def get_schema(schema_path):
    """
    读取schema文件;
    """
    all_event_types = set()
    all_classes = set()
    with open(schema_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_content = json.loads(line.strip())
            all_event_types.add(line_content["event_type"])
            all_classes.add(line_content["class"])
    print("schema定义中，共{}种事件类型，共{}种大类".format(len(all_event_types), len(all_classes)))
    all_event_types = list(all_event_types)
    all_event_types.sort()
    all_classes = list(all_classes)
    all_classes.sort()
    return all_event_types, all_classes


def token_level_metric(label_list, preds_list):
    """
    统计所有论元字符级别的PRF值。首先需要计算每个论元的PRF,而且注意label_list的每行中可能包含多个论元需要单独计算
    :param label_list: [["xxx", "xx", ...]*data_nums]，内层列表中是当前事件类型和论元角色下的所有论元字符串
    :param preds_list: [["xxx", "xx", ...]*data_nums]
    :return: token_level_precision, token_level_recall, token_level_f1
    """
    all_label_roles_num, all_pred_roles_num = 0, 0
    all_pred_role_score = 0

    for i in range(len(label_list)):
        all_label_roles_num += len(label_list[i])
    for i in range(len(preds_list)):
        all_pred_roles_num += len(preds_list[i])
    for i in range(len(label_list)):
        pred_labels = preds_list[i][:]
        for _label in label_list[i]:
            _f1 = [compute_f1(_label, _pred) for _pred in pred_labels]
            all_pred_role_score += max(_f1) if _f1 else 0

    token_level_precision = all_pred_role_score / all_pred_roles_num if all_pred_roles_num else 0
    token_level_recall = all_pred_role_score / all_label_roles_num if all_label_roles_num else 0
    token_level_f1 = 2 * token_level_precision * token_level_recall / (token_level_precision + token_level_recall) if token_level_precision + token_level_recall else 0

    return token_level_precision, token_level_recall, token_level_f1


def compute_f1(a_gold, a_pred):
    gold_toks = a_gold.lower().split()
    pred_toks = a_pred.lower().split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)  # 字典取交集
    num_same = sum(common.values()) # 相同的字的个数
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_my_span_whole_word_f1(label_list, preds_list):
    TP_FP = sum([len(_list) for _list in preds_list])
    TP_FN = sum([len(_list) for _list in label_list])
    TP = 0
    for i in range(len(label_list)):
        pred_labels = preds_list[i][:]
        for _label in label_list[i]:
            if _label in pred_labels:
                TP += 1
                pred_labels.remove(_label)
    whole_word_precision = TP / TP_FP if TP_FP else 0
    whole_word_recall = TP / TP_FN if TP_FN else 0
    whole_word_f1 = 2 * whole_word_precision * whole_word_recall / (whole_word_precision + whole_word_recall) if whole_word_precision + whole_word_recall else 0

    return whole_word_precision, whole_word_recall, whole_word_f1


def write_file(datas, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")


def remove_duplication(alist):
    res = []
    for item in alist:
        if item not in res:
            res.append(item)
    return res


def get_labels(schema_labels, task='trigger', mode="ner"):
    """

    :param path:
    :param task: trigger/role
    :param mode: classification/ner/span
    :return:
    """
    if task == 'trigger':
        labels = []
        if mode == "ner": labels.append('O')
        for event_type in schema_labels:
            if mode == "ner":
                labels.append("B-{}".format(event_type))
                labels.append("I-{}".format(event_type))
            else:   # mode == "classification"
                labels.append(event_type)
        return remove_duplication(labels)

    elif task == 'role':
        if mode == 'ner':
            return ["O", "B-ENTITY", "I-ENTITY"]
        if mode == 'span':
            return ["0", "1"]       # 是start/end， 不是start/end


def loss_backward(args, loss, optimizer):
    if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()


class FGM():
    def __init__(self, model, param_name, alpha=1.0):
        self.model = model
        self.param_name = param_name
        self.alpha = alpha

    def adversarial(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    perturbation = self.alpha * param.grad / norm
                    param.data.add_(perturbation)

    def backup_param_data(self):
        self.data = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                self.data[name] = param.data.clone()

    def restore_param_data(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                assert name in self.data
                param.data = self.data[name]
        self.data = {}

    def adversarial_training(self, args, inputs, optimizer):
        self.backup_param_data()
        self.adversarial()
        loss = self.model(**inputs)[0]
        loss_backward(args, loss, optimizer)
        self.restore_param_data()


class PGD():
    def __init__(self, model, param_name, alpha=0.3, epsilon=1.0, K=3):
        self.model = model
        self.param_name = param_name
        self.alpha = alpha
        self.epsilon = epsilon
        self.K = K

    def backup_param_data(self):
        self.data = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                self.data[name] = param.data.clone()

    def restore_param_data(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                param.data = self.data[name]

    def backup_param_grad(self):
        self.grad = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                self.grad[name] = param.grad.clone()

    def restore_param_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                param.grad = self.grad[name]


    def adversarial(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    perturbation = self.alpha * param.grad / norm
                    param.data.add_(perturbation)
                    param.data = self.project(name, param.data)

    def project(self, param_name, param_data):
        eta = param_data - self.data[param_name]
        norm = torch.norm(eta)
        if norm > self.epsilon:
            eta = self.epsilon * eta / norm
        return self.data[param_name] + eta

    def adversarial_training(self, args, inputs, optimizer):
        self.backup_param_data()
        self.backup_param_grad()
        for k in range(self.K):
            self.adversarial()
            if k != self.K - 1:
                self.model.zero_grad()
            else:
                self.restore_param_grad()
            loss = self.model(**inputs)[0]
            loss_backward(args, loss, optimizer)
        self.restore_param_data()


if __name__ == '__main__':
    label_list, pred_list =[], []
    with open('role_extraction_span/data/role_span_dev.json', 'r', encoding="utf-8") as f:
        for line in f:
            if line == '\n' or line == '':
                continue
            line_json = json.loads(line)
            line_text = line_json['text']
            start_pos, end_pos = list(map(int, line_json['start_label_list'])), list(map(int, line_json['end_label_list']))
            _label_list = []
            while sum(start_pos):
                st, en = start_pos.index(1), end_pos.index(1)
                start_pos[st], end_pos[en] = 0, 0
                _label_list.append(line_text[st:en+1])
        label_list.append(_label_list)
    with open('role_extraction_span/saved_dict_rb2/checkpoint-best/eval_predictions.json', 'r', encoding="utf-8") as f:
        for line in f:
            if line == '\n' or line == '':
                continue
            line_json = json.loads(line)
            pred_list.append(line_json[0]["pred_answers"])
    result = compute_my_span_whole_word_f1(label_list, pred_list)
    print(result)