交叉验证运行方法：

先运行prepare_data.py选择折数，会自动划分数据并保存在fold_data/目录下
然后在run_classification_fold.sh中设置参数，选择跑哪几折即可
模型参数设置：
--model_type：建议选择bert_column
--model_name_or_path：预训练模型，建议首先尝试roberta_wwm_large、ernie、macbert_large
--output_dir：每次训练时需要修改一个新的路径

折数划分，例如划分10折：

运行prepare_data.py的save_fold_data函数, 设置folds_num=10
在run_classification_fold.sh中设置参数--start_fold=0， --end_fold=9
注意：如果改变折数，务必重新运行prepare_data.py
