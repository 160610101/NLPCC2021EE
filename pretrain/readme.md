###增量预训练
使用UER.py工具包，运行几条命令即可，模型使用RoBERTa_wwm_ext_large
步骤

0. 克隆UER仓库：https://github.com/dbiir/UER-py
1. 数据预处理

    运行```make_data.py```得到```pretrain_data.txt```，然后在UER项目中运行下面的命令得到```dataset.pt```（后续命令都是在UER中执行）
    ```
    python3 preprocess.py 
    --corpus_path
    corpora/pretrain_data.txt
    --vocab_path
    G:/预训练模型/RoBERTa_wwm/chinese_roberta_wwm_large_ext/vocab.txt
    --dataset_path
    dataset.pt
    --processes_num
    2
    --target
    mlm
    --whole_word_masking
    --span_masking
    ```
2. 把原transformers格式的pytorch模型转成UER格式
    ```
    python3 scripts/convert_bert_from_huggingface_to_uer.py 
    --input_model_path
    G:/预训练模型/RoBERTa_wwm/chinese_roberta_wwm_large_ext/pytorch_model.bin
    --output_model_path
    ../models/mymodles/chinese_roberta_wwm_large_ext/pytorch_model_uer.bin
    --layers_num
    24
    --target
    mlm
   ```
    说明：需要指定原模型路径，原模型使用RoBERTa_wwm_ext_large
3. 用语料做增量预训练
    ```
    python3 pretrain.py 
    --dataset_path
    dataset.pt
    --vocab_path
    G:/预训练模型/RoBERTa_wwm/chinese_roberta_wwm_large_ext/vocab.txt
    --pretrained_model_path
    models/mymodles/chinese_roberta_wwm_large_ext/pytorch_model_uer.bin
    --output_model_path
    models/mymodles/chinese_roberta_wwme_large_ext/pytorch_model_uer_cp.bin
    --whole_word_masking
    --span_masking
    --config_path
    G:/预训练模型/RoBERTa_wwm/chinese_roberta_wwm_large_xt/config.json
    --world_size
    1
    --gpu_ranks
    0
    --total_steps
    50000
    --save_checkpoint_steps
    10000
    --encoder
    transformer
    --target
    mlm
    --batch_size
    1
    ```
   
   说明：`dataset.pt`已经上传至本目录中，需要配置好模型路径
4. 增量预训练好的UER模型转成pytorch_transformers的格式
    ```
    python3 scripts/convert_bert_from_huggingface_to_uer.py 
    --input_model_path models/mymodles/chinese_roberta_wwm_ext/pytorch_model_uer_cp-5000.bin \
    --output_model_path models/mymodles/chinese_roberta_wwm_ext/pytorch_model_tr.bin \
    --layers_num 24
    --target mlm
    ```
                                                            
    说明：
    * `--input_model_path`参数所使用的输入模型不固定，上一步增量预训练过程会每隔`--save_checkpoint_steps`步数保留一个模型，选择其中保存下来的acc较高的模型作为这一步的输入
    * 得到`pytorch_model_tr.bin`后需重命名为`pytorch_model.bin`，然后替换掉原模型目录中的同名文件即可