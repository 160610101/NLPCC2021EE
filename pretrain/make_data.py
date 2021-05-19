import json

if __name__ == '__main__':
    input_data_paths = ['../data/duee_train.json', '../data/duee_dev.json', '../data/duee_test1.json', '../data/duee_test2.json']
    output_data_path = '../data/pretrain_data.txt'

    raw_texts = []
    for _path in input_data_paths:
        with open(_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                line_content = json.loads(line.strip())
                raw_texts.append(line_content["text"])

    with open(output_data_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(raw_texts))