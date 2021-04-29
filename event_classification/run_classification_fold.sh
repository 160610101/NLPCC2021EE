export CUDA_VISIBLE_DEVICES=0

python try_transformers.py \
--task trigger \
--model_type bert \
--model_name_or_path G:/预训练模型/RoBERTa/RoBERTa_zh_L12_PyTorch \
--do_eval \
--do_predict \
--fold_data_dir ../event_classification/fold_data \
--start_fold 3 \
--end_fold 4 \
--evaluate_during_training \
--data_dir \
../event_classification/data \
--do_lower_case \
--keep_accents \
--schema ../data/duee_event_schema.json \
--output_dir ../event_classification/saved_dict_rb1_5fold \
--overwrite_output_dir \
--max_seq_length 512 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
--save_steps 300 \
--logging_steps 300 \
--num_train_epochs 20 \
--early_stop 3 \
--learning_rate 3e-5 \
--weight_decay 0.01 \
--warmup_steps 1000 \
--seed 1