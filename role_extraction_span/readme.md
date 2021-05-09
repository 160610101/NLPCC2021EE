本目录：
论元抽取模型

运行```prepare_date.py```生成模型的输入数据。
首先将事件分类模型对测试集的预测结果做预处理：
```python
event_cls_result_path_dir = '事件分类模型对测试集的预测结果所在目录'
make_fused_test_data(event_cls_result_path_dir, test1_path, new_test1_path)
```
然后生成论元抽取模型的输入数据：
```python
make_data(train_path, write_spanWithO_train_path, schema_path)
make_data(dev_path, write_spanWithO_dev_path, schema_path)
make_data(new_test1_path, write_spanWithO_test1_path, schema_path)
```
若使用交叉验证，则还需要构造k折交叉验证数据，方法和事件分类模型类似
```python
fold_data_dir = '../role_extraction_span/fold_data'
save_fold_data(write_spanWithO_train_path, write_spanWithO_dev_path, fold_data_dir, folds_num=5)
```
