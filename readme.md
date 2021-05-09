NLPCC2021信息抽取比赛，句子级事件抽取
https://aistudio.baidu.com/aistudio/competition/detail/65

当前进度：事件分类基线

环境：
Win10, CUDA11.1, pytorch1.8.1, transformers3.4.0

下载以下预训练模型，并在utils.py的PLMConfig中配置路径：
RoBERTa：https://github.com/brightmart/roberta_zh     brightmart发布的中文RoBERTa预训练模型，下载里面Pytorch版本的RoBERTa_zh_L12和RoBERTa-zh-Large
RoBERTa_wwm: https://github.com/ymcui/Chinese-BERT-wwm     科大讯飞发布的中文全词掩码RoBERTa预训练模型，下载里面Pytorch版本的RoBERTa-wwm-ext和RoBERTa-wwm-ext-large。注意下载后需要把bert_config.json文件重命名为config.json
ERNIE1.0: https://github.com/nghuyong/ERNIE-Pytorch   下载ernie-1.0(Chinese)的模型参数
MacBert: https://huggingface.co/hfl/chinese-macbert-base/tree/main
         https://huggingface.co/hfl/chinese-macbert-large/tree/main


----data/：原始数据
----event_classification/：事件多标签分类模型
    |---data/ 本模型数据
    |---fold_data/ 用于交叉验证的数据，划分好后一直保存
    |---models.py 模型
    |---prepare_data.py 由原始数据生成用于事件分类的数据（包括训练集、验证集、测试集、交叉验证数据集）
    |---run_classification.py      基线模型主程序
    |---run_classification_fold.py 使用交叉验证的主程序
    |---run_classification_fold.sh 交叉验证启动脚本
    |---...
----out/：提交文件
----role_extraction_ner/：MRC-NER论元抽取模型
----role_extraction_span/：MRC-SPAN论元抽取模型
    |---data/ 本模型数据
    |---fold_data/ 用于交叉验证的数据，划分好后一直保存
    |---models.py 模型
    |---prepare_data.py 由原始数据生成用于论元抽取的数据（包括训练集、验证集、测试集、交叉验证数据集）
    |---run_classification.py      基线模型主程序
    |---run_classification_fold.py 使用交叉验证的主程序
    |---run_classification_fold.sh 交叉验证启动脚本
    |---...
