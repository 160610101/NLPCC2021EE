import glob
import json
from collections import defaultdict
import os, shutil
import torch
from sklearn.model_selection import KFold, StratifiedKFold


def parse_event_type(event_type, role):
    """
    eg. parse_event_type("灾害/意外-爆炸", "时间")
    returns: "爆炸的时间，年、月、日、天、周、时、分、秒等"
    """
    post_type = event_type.split('-')[1]
    post_type = "、".join(post_type.split("/"))
    desc = ''
    if role.endswith('时间') or role.endswith('时长'):
        desc = '年、月、日、天、周、时、分、秒等'
    elif role.endswith('人数') or role.endswith('人'):
        desc = '数字、人次、名'
    elif role.endswith('年龄'):
        desc = '多少岁'
    elif role.endswith('方') or role.endswith('者') or role.endswith('人'):
        desc = '人名、职业、机构'
    elif role.endswith('金额') or role.endswith('价格'):
        desc = '数字、元'
    return "{}的{}，{}".format(post_type, role, desc)


def make_data(source_data_path, target_data_path, schema_path):
    mode = source_data_path.split('_')[1].split('.')[0]
    assert mode in ['train', 'dev', 'test1', 'test2']

    schemas = []
    event_type_to_line_id = {}
    with open(schema_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            line_content = json.loads(line.strip())
            event_type_to_line_id[line_content["event_type"]] = i
            schemas.append(line_content)

    output_data = []

    with open(source_data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            # 如果有相同事件类型被分开描述，则把它们的论元合并。同时为了避免后面误用，把字典中的trigger_start_index删除
            concat_event_list = []
            concat_event_types_to_index = {}  # 用于查重
            for event in line["event_list"]:
                e_type = event["event_type"]
                del event["trigger_start_index"]
                if e_type not in concat_event_types_to_index.keys():
                    concat_event_types_to_index[e_type] = len(concat_event_list)
                    concat_event_list.append(event)
                else:
                    concat_event_list[concat_event_types_to_index[e_type]]["arguments"].extend(event["arguments"])
            text = line["text"]
            id = line["id"]
            for event in concat_event_list:
                event_type = event["event_type"]
                for schema_role in schemas[event_type_to_line_id[event_type]]["role_list"]:     # 有些句子中论元可能重复，如"结婚双方"，所以需要用标准的论元类型为基准
                    schema_role = schema_role["role"]
                    start_label_list = ['0'] * len(text)
                    end_label_list = ['0'] * len(text)
                    if not mode.startswith("test"):
                        contains_schema_role = "WithO" in target_data_path        # 句子中是否存在当前的标准论元，若不存在则不为此标准论元构造数据，即不构造全'O'的模型输入数据；若True则构造全O数据
                        # 构造start_label_list，end_label_list
                        for argument in event["arguments"]:
                            if argument["role"] == schema_role:
                                contains_schema_role = True
                                # 标签名需要与utils.get_labels()中保持一致
                                start_label_list[argument["argument_start_index"]] = "1"
                                end_label_list[argument["argument_start_index"]+len(argument["argument"])-1] = "1"
                        if contains_schema_role:
                            query = parse_event_type(event_type, schema_role)    # "爆炸的时间，年、月、日、天、周、时、分、秒等"
                            output_data.append({"query": query, "text": text, "id": id,  "event_type": event_type, "role": schema_role, "start_label_list": start_label_list,
                                            "end_label_list": end_label_list})
                    else:
                        query = parse_event_type(event_type, schema_role)  # "爆炸的时间，年、月、日、天、周、时、分、秒等"
                        output_data.append({"query": query, "text": text, "id": id,  "event_type": event_type, "role": schema_role, "start_label_list": start_label_list,
                                            "end_label_list": end_label_list})

    with open(target_data_path, 'w', encoding='utf-8') as f:
        for _data in output_data:
            f.write(json.dumps(_data, ensure_ascii=False) + '\n')


def make_test_data(event_cls_result_path, test1_path, new_test1_path):
    all_test_data = []
    with open(test1_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            all_test_data.append(line)     # [{"text":..., "id":...}, ...]

    with open(event_cls_result_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            line_event_types = json.loads(line)["labels"]
            event_list = []
            for _event_type in line_event_types:
                event_list.append({"event_type":_event_type, "trigger":"", "trigger_start_index":None, "arguments":[]})
            all_test_data[i]["event_list"] = event_list
    with open(new_test1_path, 'w', encoding='utf-8') as f:
        for _data in all_test_data:
            f.write(json.dumps(_data, ensure_ascii=False) + '\n')


def make_fused_test_data(event_cls_result_path_dir, test1_path, new_test1_path):
    """融合事件分类模型的预测结果，构造融合后对测试集统一的预测结果"""
    all_test_data = []
    with open(test1_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            all_test_data.append(line)     # [{"text":..., "id":...}, ...]

    all_test_prediction_files = glob.glob(event_cls_result_path_dir + "/**/" + '*_predictions.json', recursive=True)
    if not all_test_prediction_files:
        raise ValueError('')
    model_num = len(all_test_prediction_files)
    all_pred_results = [[] for i in range(15000)]
    for _path in all_test_prediction_files:
        with open(_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                if not line:
                    break
                all_pred_results[i].append(json.loads(line)["labels"])

    for i in range(len(all_pred_results)):
        sample_event_times = defaultdict(int)               # 设置融合规则
        for model_preds in all_pred_results[i]:
            for event_pred in model_preds:
                sample_event_times[event_pred] += 1
        event_list = []
        for _event_type in sample_event_times.keys():
            if sample_event_times[_event_type] >= model_num/2:
                event_list.append(
                        {"event_type": _event_type, "trigger": "", "trigger_start_index": None, "arguments": []})
        all_test_data[i]["event_list"] = event_list

    with open(new_test1_path, 'w', encoding='utf-8') as f:
        for _data in all_test_data:
            f.write(json.dumps(_data, ensure_ascii=False) + '\n')


def save_fold_data(train_path, dev_path, fold_data_dir, folds_num=10):
    if os.path.exists(fold_data_dir):   # 每次划分数据之前，需要删除之前划分的数据
        shutil.rmtree(fold_data_dir)
    os.makedirs(fold_data_dir)

    all_train_dev_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            all_train_dev_data.append(json.loads(line))
    with open(dev_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            all_train_dev_data.append(json.loads(line))

    gkf = KFold(n_splits=folds_num, shuffle=True, random_state=0).split(all_train_dev_data)

    for fold, (train_idx, valid_idx) in enumerate(gkf):
        train_dataset, eval_dataset = [all_train_dev_data[i] for i in train_idx], [all_train_dev_data[i] for i in valid_idx]
        fold_train_path, fold_dev_path = os.path.join(fold_data_dir, "fold_{}_train.json".format(fold)), os.path.join(fold_data_dir, "fold_{}_dev.json").format(fold)
        with open(fold_train_path, 'w', encoding='utf-8') as f:
            for obj in train_dataset:
                json.dump(obj, f, ensure_ascii=False, sort_keys=True)
                f.write("\n")
        with open(fold_dev_path, 'w', encoding='utf-8') as f:
            for obj in eval_dataset:
                json.dump(obj, f, ensure_ascii=False, sort_keys=True)
                f.write("\n")


if __name__ == '__main__':
    schema_path = '../data/duee_event_schema.json'
    train_path = '../data/duee_train.json'
    dev_path = '../data/duee_dev.json'
    test1_path = '../data/duee_test1.json'
    new_test1_path = '../data/duee_test1_with_pred_event.json'      # 需要构造成和train_path、dev_path相同的格式

    # write_span_train_path = 'data/role_span_train.json'
    # write_span_dev_path = 'data/role_span_dev.json'
    # write_span_test1_path = 'data/role_span_test1.json'

    write_spanWithO_train_path = 'data/role_spanWithO_train.json'
    write_spanWithO_dev_path = 'data/role_spanWithO_dev.json'
    write_spanWithO_test1_path = 'data/role_spanWithO_test1.json'

    # event_cls_result_path = '../event_classification/saved_dict_rb18/checkpoint-best/test1_predictions.json'
    # make_test_data(event_cls_result_path, test1_path, new_test1_path)

    event_cls_result_path_dir = '../event_classification/result_pool_fgm_256'
    make_fused_test_data(event_cls_result_path_dir, test1_path, new_test1_path)

    # make_data(train_path, write_span_train_path, schema_path)
    # make_data(dev_path, write_span_dev_path, schema_path)
    # make_data(new_test1_path, write_span_test1_path, schema_path)

    make_data(train_path, write_spanWithO_train_path, schema_path)
    make_data(dev_path, write_spanWithO_dev_path, schema_path)
    make_data(new_test1_path, write_spanWithO_test1_path, schema_path)

    fold_data_dir = '../role_extraction_span/fold_data'
    save_fold_data(write_spanWithO_train_path, write_spanWithO_dev_path, fold_data_dir, folds_num=5)