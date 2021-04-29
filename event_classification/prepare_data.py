import json
import os, shutil
import torch

from utils import get_schema
from sklearn.model_selection import KFold, StratifiedKFold

def make_data(source_data_path, target_data_path, schemas):
    mode = source_data_path.split('_')[1].split('.')[0]
    assert mode in ['train', 'dev', 'test1', 'test2']
    output_data = []
    with open(source_data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            text = line["text"]
            event_type = [0] * len(schemas)
            if not mode.startswith('test'):
                for event in line["event_list"]:
                    event_type[schemas.index(event["event_type"])] = 1
            output_data.append({"text":text, "event_type":event_type})
    with open(target_data_path, 'w', encoding='utf-8') as f:
        for _data in output_data:
            f.write(json.dumps(_data, ensure_ascii=False) + '\n')


def save_fold_data(train_path, dev_path, fold_data_dir, folds_num=10):
    if os.path.exists(fold_data_dir):   # 每次划分数据之前，需要删除之前划分的数据
        shutil.rmtree(fold_data_dir)
    os.makedirs(fold_data_dir)

    all_train_dev_data = []
    all_labels = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            all_train_dev_data.append(json.loads(line))
            all_labels.append(json.loads(line)["event_type"])
    with open(dev_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            all_train_dev_data.append(json.loads(line))
            all_labels.append(json.loads(line)["event_type"])
    all_label_ids = torch.tensor(all_labels, dtype=torch.int)

    # 样本权重
    label_count = torch.tensor([99, 63, 69, 96, 154, 299, 197, 1242, 300, 151, 109, 33, 121, 107, 61, 134, 65, 827, 79, 274, 298, 63, 99, 170, 105, 727, 82, 268, 238, 177, 128, 80, 110, 48, 75, 122, 210, 287, 462, 325, 138, 2004, 100, 145, 87, 356, 145, 82, 47, 93, 605, 197, 254, 74, 67, 63, 51, 191, 26, 64, 225, 128, 104, 83, 32], dtype=torch.float)
    label_weight = label_count.sum()/label_count

    stratified_labels = []
    for i in range(all_label_ids.shape[0]):
        label_index_list = torch.where(all_label_ids[i] == 1)[0].tolist()
        if len(label_index_list) == 1:
            stratified_labels.append(label_index_list[0])
        else:
            best_label_index = label_index_list[0]
            best_weight = 0
            for _label_index in label_index_list:
                if label_weight[_label_index] > best_weight:
                    best_weight = label_weight[_label_index]
                    best_label_index = _label_index
            stratified_labels.append(best_label_index)

    gkf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=0).split(all_train_dev_data, stratified_labels)
    # gkf = KFold(n_splits=5, shuffle=True, random_state=0).split(all_train_data)

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
    schemas, _ = get_schema(schema_path)

    write_train_path = '../event_classification/data/event_cl_train.json'
    write_dev_path = '../event_classification/data/event_cl_dev.json'
    write_test1_path = '../event_classification/data/event_cl_test1.json'

    # make_data(train_path, write_train_path, schemas)
    # make_data(dev_path, write_dev_path, schemas)
    # make_data(test1_path, write_test1_path, schemas)

    fold_data_dir = '../event_classification/fold_data'
    save_fold_data(write_train_path, write_dev_path, fold_data_dir, folds_num=10)
