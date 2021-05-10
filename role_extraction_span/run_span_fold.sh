python run_span_fold.py \
--withO True \
--task role \
--model_type bert \
--model_name_or_path roberta \
--do_train \
--do_eval \
--do_predict \
--fold_data_dir ../role_extraction_span/fold_data \
--start_fold 0 \
--end_fold 9 \
--evaluate_during_training \
--data_dir ../role_extraction_span/data \
--adv_training fgm \
--do_lower_case \
--keep_accents \
--schema ../data/duee_event_schema.json \
--output_dir ../role_extraction_span/saved_dict_rb1_10fold \
--overwrite_output_dir \
--max_seq_length 512 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 64 \
--gradient_accumulation_steps 10 \
--save_steps 300 \
--logging_steps 300 \
--num_train_epochs 20 \
--early_stop 3 \
--learning_rate 3e-5 \
--weight_decay 0.01 \
--warmup_steps 1000 \
--seed 1