import json
import numpy as np


def make_submit_data(test_input_path, test_predict_path, test_output_path):
    """
    制作用于提交的结果文件
    :param test_input_path: 包含预测前的query、text、id、event_type、role等
    :param test_predict_path: 预测结果，每行是论元列表
    :param test_output_path: 提交文件的输出路径
    :return:
    """
    submit_data = []

    pred_event_role_arguments = []
    with open(test_input_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            pred_event_role_arguments.append(line)
            del line["start_label_list"]
            del line["end_label_list"]
    with open(test_predict_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            if i == 14562:
                pass
            line = json.loads(line)['labels']
            pred_event_role_arguments[i]["arguments"] = line[0]["pred_answers"]

    for data in pred_event_role_arguments:
        id = data["id"]
        if not submit_data or id != submit_data[-1]["id"]:
            submit_data.append({"id":id, "event_list":[{"event_type":data["event_type"],
                                                        "arguments":[{"role":data["role"], "argument": _argument} for _argument in data["arguments"]]}]})
        else:
            if data["event_type"] == submit_data[-1]["event_list"][-1]["event_type"]:
                submit_data[-1]["event_list"][-1]["arguments"].extend([{"role": data["role"], "argument": _argument}
                                                                        for _argument in data["arguments"]])
            else:
                submit_data[-1]["event_list"].append({"event_type": data["event_type"],
                                                          "arguments": [{"role": data["role"], "argument": _argument}
                                                                        for _argument in data["arguments"]]})

    with open(test_output_path, 'w', encoding='utf-8') as f:
        for _data in submit_data:
            f.write(json.dumps(_data, ensure_ascii=False) + '\n')


def prediction_fusion(test_output_path, test_input_path, logits2_path, logits3_path):
    '''
    模型融合，将两个预测logits文件结果做融合
    '''
    logits2, logits3 = [], []
    with open(logits2_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            logits2.append(json.loads(line)['logits'])
    with open(logits3_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            logits3.append(json.loads(line)['logits'])
    logits2, logits3 = np.array(logits2), np.array(logits3)

    texts = []
    with open(test_input_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            texts.append([line["query"], line["text"]])

    for i, _l in enumerate(logits2):
        delete_len = len(texts[i][0])+2
        logits2[i][:delete_len][0], logits2[i][:delete_len][0] = 0, 0
        logits3[i][:delete_len][0], logits2[i][:delete_len][0] = 0, 0

    logits = (logits2+logits3)/2
    for i in range(len(texts)):
        texts[i] = "#{}#{}#".format(texts[i][0], texts[i][1])

    threshold = 0.5
    preds = logits > threshold     # (batch_size, sequence_length, 2), bool
    preds = preds + 0               # (batch_size, sequence_length, 2), 0/1

    from role_extraction_span.utils_span import convert_span_output_to_texts
    pred_results = convert_span_output_to_texts(preds, texts)

    preds_out = [[] for i in range(len(preds))]      # 写入文件的预测结果
    for i in range(len(preds)):
        preds_out[i].append({"pred_answers":pred_results[i], "pred_start_pos_list":preds[i][:, 0].tolist(), "pred_end_pos_list":preds[i][:, 1].tolist()})

    from utils import write_file
    write_file(preds_out, './out/fus_prediction.json')


def adjust_diff_test1():
    text1 = []
    with open('role_extraction_span/data/role_spanWithO_test1.json', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            text1.append(line['query'][:-1] + '#' + line['text'])
    for fold in range(5):
        new_all_result = []
        with open('role_extraction_span/result_span/fold{}_checkpoint-best/test1_predictions.json'.format(fold), 'r', encoding='utf-8') as f:
            for id, line in enumerate(f.readlines()):
                line = json.loads(line)['labels'][0]
                end_pos_list = np.argwhere(np.array(line['pred_end_pos_list'])).reshape(-1)
                start_pos_list = np.argwhere(np.array(line['pred_start_pos_list'])).reshape(-1)
                line_answer_results = []
                for end_pos in end_pos_list:  # 预测的start数目往往远多于end，所以以end为基准寻找span
                    start_pos = start_pos_list[
                        start_pos_list <= end_pos]  # 寻找小于等于end_pos的第一个start_pos，前提是存在start_pos<=end_pos
                    if start_pos.any():  # 如果存在
                        start_pos = start_pos[-1]
                        _text = text1[id][start_pos:end_pos + 1]
                        line_answer_results.append(_text)
                new_all_result.append({'labels':[{'pred_answers':line_answer_results, 'pred_end_pos_list': line['pred_end_pos_list'], 'pred_start_pos_list':line['pred_start_pos_list']}]})
        with open('role_extraction_span/result_span/fold{}_checkpoint-best/test1_predictions.json'.format(fold), 'w',
                  encoding='utf-8') as f:
            for _result in new_all_result:
                f.write(json.dumps(_result, ensure_ascii=False))
                f.write('\n')


def adjust_diff_test2():
    text1 = []
    with open('role_extraction_span/data/role_spanWithO_test2.json', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            text1.append(line['query'][:-1] + '#' + line['text'])

    new_all_result = []
    with open('role_extraction_span/result_span_test2/test2_predictions.json', 'r', encoding='utf-8') as f:
        for id, line in enumerate(f.readlines()):
            line = json.loads(line)['labels'][0]
            end_pos_list = np.argwhere(np.array(line['pred_end_pos_list'])).reshape(-1)
            start_pos_list = np.argwhere(np.array(line['pred_start_pos_list'])).reshape(-1)
            line_answer_results = []
            for end_pos in end_pos_list:  # 预测的start数目往往远多于end，所以以end为基准寻找span
                start_pos = start_pos_list[
                    start_pos_list <= end_pos]  # 寻找小于等于end_pos的第一个start_pos，前提是存在start_pos<=end_pos
                if start_pos.any():  # 如果存在
                    start_pos = start_pos[-1]
                    _text = text1[id][start_pos:end_pos + 1]
                    line_answer_results.append(_text)
            new_all_result.append({'labels':[{'pred_answers':line_answer_results, 'pred_end_pos_list': line['pred_end_pos_list'], 'pred_start_pos_list':line['pred_start_pos_list']}]})
    with open('role_extraction_span/result_span_test2/test2_predictions.json', 'w',
              encoding='utf-8') as f:
        for _result in new_all_result:
            f.write(json.dumps(_result, ensure_ascii=False))
            f.write('\n')


def span_vote_fusion():
    '''
    模型融合，对预测结果做投票融合
    '''
    fused_result = [] # 对应文件的每行，制作成make_submit_data需要的格式，列表元素是[字典]，字典只含有融合后的pred_answers
    # 投票融合规则：对于每一行的query，统计多个模型预测结果的次数，>2/5则保留
    results_combine = [[] for i in range(47983)]
    for fold in range(5):
        with open('role_extraction_span/result_span/fold{}_checkpoint-best/test1_predictions.json'.format(fold), 'r', encoding='utf-8') as f:
            for id, line in enumerate(f.readlines()):
                line = json.loads(line)['labels'][0]
                pred_ans_list = line['pred_answers']
                results_combine[id].extend(pred_ans_list)
    for _Res in results_combine:
        win_ans = set()
        for _res in _Res:
            if _Res.count(_res) > 2:
                win_ans.add(_res)
        fused_result.append([{'pred_answers':list(win_ans)}])
    with open('role_extraction_span/result_span/fused_predictions.json', 'w',
              encoding='utf-8') as f:
        for _result in fused_result:
            f.write(json.dumps(_result, ensure_ascii=False))
            f.write('\n')


def convert_span_output_to_texts(span_outputs, span_input_texts):
    """
    设定规则，把SPAN结果处理成最终的论元文本，用于评价分数和输出预测结果
    :return:
    """
    span_texts = [[] for i in range(len(span_outputs))]
    for i in range(span_outputs.shape[0]):
        start_pos_list = np.argwhere(span_outputs[i][:,0]).reshape(-1)
        end_pos_list = np.argwhere(span_outputs[i][:,1]).reshape(-1)
        for end_pos in end_pos_list:  # 预测的start数目往往远多于end，所以以end为基准寻找span
            start_pos = start_pos_list[start_pos_list <= end_pos]  # 寻找小于等于end_pos的第一个start_pos，前提是存在start_pos<=end_pos
            if start_pos.any():  # 如果存在
                start_pos = start_pos[-1]
                _text = span_input_texts[i][start_pos+2:end_pos + 3]
                span_texts[i].append(_text)
    return span_texts


if __name__ == '__main__':
    test_input_path = './role_extraction_span/data/role_spanWithO_test1.json'
    test_predict_path = './role_extraction_span/saved_dict_rb3/checkpoint-best/test1_predictions.json'
    test_output_path = './out/test1_submit.json'
    # make_submit_data(test_input_path, test_predict_path, test_output_path)

    # logits2_path = './role_extraction_span/saved_dict_rb2/checkpoint-best/test1_logits.json'
    # logits3_path = './role_extraction_span/saved_dict_rb3/checkpoint-best/test1_logits.json'
    # prediction_fusion(test_output_path, test_input_path, *(logits2_path, logits3_path))
    # make_submit_data(test_input_path, './out/fus_prediction.json', test_output_path)

    # adjust_diff_test1()       # 预测结果可能有字符偏移，需要做修正
    # adjust_diff_test2()
    # span_vote_fusion()        # 多个预测结果文件做融合
    # make_submit_data(test_input_path, 'role_extraction_span/result_span/fused_predictions.json', test_output_path)
    make_submit_data('./role_extraction_span/data/role_spanWithO_test2.json', 'role_extraction_span/result_span_test2/test2_predictions.json', './out/test2_submit.json')