本目录：
论元抽取模型
* 预处理，生成模型输入数据
* 运行模型训练和预测
* 后处理，生成最终的提交文件

### 预处理
运行```prepare_date.py```生成模型的输入数据<br>
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

### 论元抽取模型 
运行脚本```run_span_fold.sh```，参数设置：

    --withO True \                                          # 必选，代表使用带O标签的数据训练
    --adv_training fgm \                                    # 启用对抗训练，可选择fgm或pgd
    --model_name_or_path roberta \                          # 配置完所有预训练模型后，可以任意选用
    --fold_data_dir ../role_extraction_span/fold_data \     # 配置交叉验证目录，下同
    --start_fold 0 \
    --end_fold 9 \
    --output_dir ../role_extraction_span/saved_dict_rb1_10fold \

由于数据量较大，建议尽量调大batch_size；或者同时调节`--gradient_accumulation_steps`和`--logging_steps`以模拟实现batch_size的扩大，需保证后者是前者的整数倍


### 后处理

待完善