import collections
import json
import string

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


def data_val(input_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()

    event_class_count = 0
    role_count = 0
    arg_count = 0
    arg_role_count = 0
    arg_role_one_event_count = 0
    trigger_count = 0
    argument_len_list = []

    for row in rows:
        if len(row) == 1: print(row)
        row = json.loads(row)

        arg_start_index_list = []
        arg_start_index_map = {}
        event_class_list = []
        trigger_start_index_list = []

        event_class_flag = False
        arg_start_index_flag = False
        role_flag = False
        arg_role_flag = False
        arg_role_one_event_flag = False
        trigger_flag = False

        for event in row["event_list"]:
            event_class = event["class"]
            if event_class_list == []:
                event_class_list.append(event_class)
            elif event_class not in event_class_list:
                # event_class_count += 1
                event_class_flag = True
                # print(row)

            trigger_start_index = event["trigger_start_index"]
            if trigger_start_index not in trigger_start_index_list:
                trigger_start_index_list.append(trigger_start_index)
            else:
                trigger_flag = True
                print(row)

            role_list = []
            arg_start_index_map_in_one_event = {}
            for arg in event["arguments"]:
                role = arg['role']
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                argument_len_list.append([len(argument), argument])
                if role not in role_list:
                    role_list.append(role)
                else:
                    # role_count += 1
                    arg_start_index_flag = True
                    # print(row)

                if argument_start_index not in arg_start_index_map_in_one_event:
                    arg_start_index_map_in_one_event[argument_start_index] = role
                else:
                    if role != arg_start_index_map_in_one_event[argument_start_index]:
                        arg_role_one_event_flag = True
                        # print(row)

                if argument_start_index not in arg_start_index_list:
                    arg_start_index_list.append(argument_start_index)
                    arg_start_index_map[argument_start_index] = role
                else:
                    # arg_count+= 1
                    role_flag = True
                    if role != arg_start_index_map[argument_start_index]:
                        arg_role_flag = True
                        # print(row)

        if role_flag:
            role_count += 1
            # print(row)
        if event_class_flag:
            event_class_count += 1
            # print(row)
        if arg_start_index_flag:
            arg_count += 1
            # print(row)
        if arg_role_flag:
            arg_role_count += 1
        if arg_role_one_event_flag:
            arg_role_one_event_count += 1
        if trigger_flag:
            trigger_count += 1

    print(event_class_count, role_count, arg_count, arg_role_count, arg_role_one_event_count, trigger_count)
    argument_len_list.sort(key=lambda x: x[0], reverse=True)
    print(argument_len_list[:10])


def position_val(input_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    trigger_count = 0
    arg_count = 0

    for row in rows:
        # position_flag = False

        if len(row) == 1: print(row)
        row = json.loads(row)
        text = row['text']
        for event in row["event_list"]:
            event_class = event["class"]
            trigger = event["trigger"]
            event_type = event["event_type"]
            trigger_start_index = event["trigger_start_index"]

            if text[trigger_start_index: trigger_start_index + len(trigger)] != trigger:
                print("trigger position mismatch")
                trigger_count += 1

            for arg in event["arguments"]:
                role = arg['role']
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]

                if text[argument_start_index: argument_start_index + len(argument)] != argument:
                    print("argument position mismatch")
                    arg_count += 1

    print(trigger_count, arg_count)


# 统计 event_type 分布
def data_analysis(input_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    label_list = get_labels(task='trigger', mode="classification")
    label_map = {label: i for i, label in enumerate(label_list)}
    label_count = [0 for i in range(len(label_list))]
    for row in rows:
        row = json.loads(row)
        for event in row["event_list"]:
            event_type = event["event_type"]
            label_count[label_map[event_type]] += 1
    print(label_count)


def get_num_of_arguments(input_file):
    lines = open(input_file, encoding='utf-8').read().splitlines()
    arg_count = 0
    for line in lines:
        line = json.loads(line)
        for event in line["event_list"]:
            arg_count += len(event["arguments"])
    print(arg_count)


def read_write(input_file, output_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        row = json.loads(row)
        id = row.pop('id')
        text = row.pop('text')
        # labels = row.pop('labels')
        event_list = row.pop('event_list')
        row['text'] = text
        row['id'] = id
        # row['labels'] = labels
        row['event_list'] = event_list
        results.append(row)
    write_file(results, output_file)


def schema_analysis(path="./data/event_schema/event_schema.json"):
    rows = open(path, encoding='utf-8').read().splitlines()
    argument_map = {}
    for row in rows:
        d_json = json.loads(row)
        event_type = d_json["event_type"]
        for r in d_json["role_list"]:
            role = r["role"]
            if role in argument_map:
                argument_map[role].append(event_type)
            else:
                argument_map[role] = [event_type]
    argument_unique = []
    argument_duplicate = []
    for argument, event_type_list in argument_map.items():
        if len(event_type_list) == 1:
            argument_unique.append(argument)
        else:
            argument_duplicate.append(argument)

    print(argument_unique, argument_duplicate)
    for argument in argument_duplicate:
        print(argument_map[argument])

    return argument_map


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