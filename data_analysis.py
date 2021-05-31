import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib  # matplotlib==3.4.0
import seaborn as sns
import numpy as np
from matplotlib import font_manager as fm
from matplotlib import cm
import pandas as pd
from utils import get_schema

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

save_pic_path = 'picture/'


def count_event_type(target_data_path, event_types):
    """
    1.统计句子中事件出现的数量
    2.统计目标数据集中，各事件类型的出现数量，各大类出现的数量;
    3.统计各事件类型共现情况;
    事件类型：event_type；大类：class；触发词：trigger;
    """
    target_data_event_nums = [0]*len(event_types)     # 每种事件类型出现的数量
    target_data_sentence_nums = [0]*len(event_types)  # 每种事件类型出现的句子数量
    target_data_event_types_triggers = dict()  # 每种事件类型中含有的触发词及数量 dict(str:dict(str:int))
    event_co_occurrence_matrix = [[0]*len(event_types) for i in range(len(event_types))]    # 事件的共现矩阵
    type2id = dict()    # 事件类型to索引
    for i, _event_type in enumerate(event_types):   # 初始化一些数据结构
        target_data_event_types_triggers[_event_type] = dict()
        type2id[_event_type] = i
    sentences_num = 0   # 统计句长
    sentences_contains_events_num = [0]*20  # 统计一个句子里可以有多少事件共存（重复事件也计数），假定最多有20个事件共存

    with open(target_data_path, 'r', encoding='utf-8') as f:        # 完成对以上目标的统计
        for line in f.readlines():
            line_content = json.loads(line.strip())
            event_list = line_content["event_list"]
            sentences_contains_events_num[len(event_list)] += 1
            sentences_num += 1
            event_occurence_times = [0]*len(event_types)        # 句子中每个事件出现次数，用于计算共现矩阵
            tmp_event_set = set()
            for _event in event_list:
                _event_type = _event["event_type"]
                _class = _event["class"]
                _trigger = _event["trigger"]
                target_data_event_nums[type2id[_event_type]] += 1
                event_occurence_times[type2id[_event_type]] += 1
                tmp_event_set.add(_event_type)
                if _trigger in target_data_event_types_triggers[_event_type].keys():
                    target_data_event_types_triggers[_event_type][_trigger] += 1
                else:
                    target_data_event_types_triggers[_event_type][_trigger] = 1
            for _event_type in tmp_event_set:
                target_data_sentence_nums[type2id[_event_type]] += 1
            for i in range(len(event_types)):   # 填充共现矩阵
                for j in range(i, len(event_types)):
                    if event_occurence_times[i] and event_occurence_times[j]:
                        if i == j:
                            event_co_occurrence_matrix[i][j] = event_occurence_times[i]-1
                        else:
                            event_co_occurrence_matrix[i][j] += event_occurence_times[i]*event_occurence_times[j]
                            event_co_occurrence_matrix[j][i] += event_occurence_times[i]*event_occurence_times[j]

    data_func = target_data_path.split('_')[-1].replace(".json", "")  # train or dev
    print('当前处理数据集：', data_func, '——'*100)
    print('数据集句子总数：{}'.format(sentences_num))
    print('句子中可以出现的事件个数', sentences_contains_events_num)
    print("触发词出现次数", target_data_event_types_triggers)
    print("各事件类型出现次数", target_data_event_nums)
    print("各事件类型出现句子次数", target_data_sentence_nums)

    print("正在绘制句子中可出现的事件数的饼状图......")
    labels = ['事件数=1', '事件数=2', '事件数=3', '事件数=4', '事件数>4']
    sizes = sentences_contains_events_num[1:5] + [sum(sentences_contains_events_num[5:])]
    labels.insert(1, labels.pop())
    sizes.insert(1, sizes.pop())
    fig, axes = plt.subplots(figsize=(5, 4), ncols=2, dpi=300)
    ax1, ax2 = axes.ravel()
    patches, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%3.2f%%', startangle=170)
    proptease = fm.FontProperties()    # 重新设置字体大小
    proptease.set_size(3)
    plt.setp(autotexts, fontproperties=proptease)
    plt.setp(texts, fontproperties=proptease)
    ax1.set_title('{}数据集句子中出现的事件数'.format(data_func), loc='center')
    ax2.axis('off')    # ax2 只显示图例（legend）
    ax2.legend(patches, labels, loc='center left')
    plt.tight_layout()
    plt.savefig(save_pic_path+'{}数据集句子中出现的事件数.jpg'.format(data_func))
    plt.show()

    print("正在绘制各类事件出现次数的统计图......")
    x = np.arange(len(event_types))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
    rects1 = ax.bar(x - width/2, target_data_event_nums, width, label='事件类型出现次数')
    rects2 = ax.bar(x + width/2, target_data_sentence_nums, width, label='事件类型对应句子出现次数')
    ax.set_ylabel('出现次数')
    ax.set_title('{}数据集各事件类型出现结果'.format(data_func))
    ax.set_xticks(x)
    plt.xticks(rotation=45)
    ax.set_xticklabels(event_types, fontsize=8)
    ax.legend()
    ax.bar_label(rects1, padding=3, fontsize=8, rotation=45)
    ax.bar_label(rects2, padding=3, fontsize=8, rotation=45)
    fig.tight_layout()
    plt.savefig(save_pic_path+'{}数据集各事件类型出现结果.jpg'.format(data_func))
    plt.show()
    print('出现次数最多的事件类型为\"{}\"，共{}次；出现次数最少的事件类型为\"{}\"，共{}次'.format(event_types[target_data_event_nums.index(max(target_data_event_nums))], max(target_data_event_nums), event_types[target_data_event_nums.index(min(target_data_event_nums))], min(target_data_event_nums)))

    print("正在绘制各类事件共现矩阵......")
    plt.subplots(figsize=(30, 30), dpi=140)
    event_co_occurrence_data = pd.DataFrame(np.array(event_co_occurrence_matrix), index=event_types, columns=event_types)
    sns.heatmap(event_co_occurrence_data, annot=True, fmt="d")
    plt.xticks(fontsize=10)  # x轴刻度的字体大小（文本包含在pd_data中了）
    plt.yticks(fontsize=10)  # y轴刻度的字体大小（文本包含在pd_data中了）
    plt.title('{}数据集事件类型共现矩阵'.format(data_func), fontsize=20)  # 图片标题文本和字体大小
    cax = plt.gcf().axes[-1]       # 设置colorbar的刻度字体大小
    cax.tick_params(labelsize=20)
    plt.savefig(save_pic_path+'{}数据集事件类型共现矩阵.jpg'.format(data_func))
    plt.show()


def count_sentence_length(target_data_path):
    """
    统计目标数据集句子长度分布;
    """
    data_func = target_data_path.split('_')[-1].replace(".json", "")  # train or dev

    sentences_lengths = []
    with open(target_data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_content = json.loads(line.strip())
            text = line_content["text"]
            sentences_lengths.append(len(text))

    num_bins = 200
    fig, ax = plt.subplots()
    ax.hist(sentences_lengths, num_bins)
    ax.set_xlabel('句长')
    ax.set_ylabel('出现次数')
    ax.set_title(r'{}数据集句长分布'.format(data_func))
    fig.tight_layout()
    plt.savefig(save_pic_path+'{}数据集句长分布.jpg'.format(data_func))
    plt.show()
    print('{}数据集中，句子长度最短为{}，最长为{}，数量最多的句长为{}'.format(data_func, min(sentences_lengths), max(sentences_lengths), Counter(sentences_lengths).most_common(1)[0][0]))


if __name__ == '__main__':
    schema_path = 'data/duee_event_schema.json'
    train_path = 'data/duee_train.json'
    dev_path = 'data/duee_dev.json'
    test1_path = 'data/duee_test1.json'
    event_types, classes = get_schema(schema_path)
    print(event_types)

    # count_event_type(train_path, event_types)
    # count_event_type(dev_path, event_types)
    #
    # count_sentence_length(train_path)  # train数据集中，句子长度最短为6，最长为378，数量最多的句长为30
    # count_sentence_length(dev_path)  # dev数据集中，句子长度最短为6，最长为333，数量最多的句长为30
    # count_sentence_length(test1_path)  # test1数据集中，句子长度最短为5，最长为514，数量最多的句长为30
