# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

import argparse
import glob
import logging
import os
import pickle
import random

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from role_extraction_span.models import BertForQuestionAnswering
from utils import get_labels, token_level_metric, compute_my_span_whole_word_f1, PLMConfig, FGM, PGD
from role_extraction_span.utils_span import convert_examples_to_features, read_squad_examples, span_metrics, \
    convert_span_output_to_texts
from utils import write_file

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
}

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]
PLM_models = PLMConfig()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train_and_eval(args, train_dataset, eval_dataset, eval_input_texts, model, tokenizer, output_dir):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.adv_training:
        if args.adv_training == 'fgm':
            adv = FGM(model, param_name='word_embeddings')
        elif args.adv_training == 'pgd':
            adv = PGD(model, param_name='word_embeddings')

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility

    best_metric = 1     # loss作为早停的评价指标时为0，f1作指标时应为1
    patience = 0
    for _ in train_iterator:
        print(_, "epoch")
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "start_labels": batch[3], "end_labels": batch[4]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            logger.info("loss: %f", loss.item())
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.adv_training:
                adv.adversarial_training(args, inputs, optimizer)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = evaluate(args, eval_dataset, eval_input_texts, model)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    current_metric = results["loss"]
                    if current_metric >= best_metric:   # f1 写的可能有问题，验证集结果一直都是0，先用loss做早停吧
                        patience += 1
                        print("=" * 80)
                        print("Best Metric", best_metric)
                        print("Current Metric", current_metric)
                        print("=" * 80)
                        if global_step > 1000 and patience > args.early_stop:
                            print("Out of patience !  Stop Training !")
                            return global_step, tr_loss / global_step
                    else:
                        patience = 0
                        print("=" * 80)
                        print("Best Metric", best_metric)
                        print("Current Metric", current_metric)
                        print("Saving Model......")
                        print("=" * 80)
                        best_metric = current_metric

                        # Save model checkpoint
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, input_texts, model, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # fix the bug when using mult-gpu during evaluating
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    input_ids = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                      "start_labels": batch[3], "end_labels": batch[4]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()   # (batch_size, sequence_length, 2)
            input_ids = inputs["input_ids"].detach().cpu().numpy()
            out_label_ids = torch.cat((inputs["start_labels"].detach().cpu().unsqueeze(2), inputs["end_labels"].detach().cpu().unsqueeze(2)), 2).numpy()     # (batch_size, sequence_length, 2)
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            input_ids = np.append(input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, torch.cat((inputs["start_labels"].detach().cpu().unsqueeze(2), inputs["end_labels"].detach().cpu().unsqueeze(2)), 2).numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    threshold = 0.5
    logits = sigmoid(preds)
    preds = logits > threshold     # (batch_size, sequence_length, 2), bool
    preds = preds + 0               # (batch_size, sequence_length, 2), 0/1

    precision, recall_score, f1_score = span_metrics(out_label_ids, preds)

    label_results = convert_span_output_to_texts(out_label_ids, input_texts)
    pred_results = convert_span_output_to_texts(preds, input_texts)

    whole_word_result = compute_my_span_whole_word_f1(label_results, pred_results)

    results = {
        "loss": eval_loss,
        "two_class_precision": precision,
        "two_class_recall": recall_score,
        "two_class_f1": f1_score,
        "token_level_precision": token_level_metric(label_results, pred_results)[0],
        "token_level_recall": token_level_metric(label_results, pred_results)[1],
        "token_level_f1": token_level_metric(label_results, pred_results)[2],  # 论元实体按字符级别匹配的f1，是比赛设定的评价标准
        "whole_word_precision": whole_word_result[0],
        "whole_word_recall": whole_word_result[1],
        "whole_word_f1": whole_word_result[2]
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    preds_out = [[] for i in range(len(preds))]      # 写入文件的预测结果
    for i in range(len(preds)):
        preds_out[i].append({"pred_answers":pred_results[i], "pred_start_pos_list":preds[i][:, 0].tolist(), "pred_end_pos_list":preds[i][:, 1].tolist()})
    return results, preds_out, logits.tolist()


def predict(args, test_dataset, test_input_texts, model, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # fix the bug when using mult-gpu during evaluating
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    preds = None
    input_ids = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                      "start_labels": batch[3], "end_labels": batch[4]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            _, logits = outputs[:2]

        if preds is None:
            preds = logits.detach().cpu().numpy()   # (batch_size, sequence_length, 2)
            input_ids = inputs["input_ids"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            input_ids = np.append(input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    threshold = 0.5
    logits = sigmoid(preds)
    preds = logits > threshold     # (batch_size, sequence_length, 2), bool
    preds = preds + 0               # (batch_size, sequence_length, 2), 0/1

    pred_results = convert_span_output_to_texts(preds, test_input_texts)

    preds_out = [[] for i in range(len(preds))]      # 写入文件的预测结果
    for i in range(len(preds)):
        preds_out[i].append({"pred_answers":pred_results[i], "pred_start_pos_list":preds[i][:, 0].tolist(), "pred_end_pos_list":preds[i][:, 1].tolist()})
    return [], preds_out, logits.tolist()


def load_and_cache_fold_examples(args, tokenizer, labels, mode):
    # mode==test1，fold_x_train，fold_x_dev

    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    if args.withO:
        cached_features_file = os.path.join(
            args.data_dir if mode.startswith('test') else args.fold_data_dir,
            "cached_{}_withO_{}_{}".format(mode, list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                     str(args.max_seq_length)
                                     ),
        )
    else:
        cached_features_file = os.path.join(
            args.data_dir if mode.startswith('test') else args.fold_data_dir,
            "cached_{}_{}_{}".format(mode, list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                     str(args.max_seq_length)
                                     ),
        )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        with open(cached_features_file+"_text.pkl", 'rb') as f:
            input_texts = pickle.load(f)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_squad_examples(args, mode)
        features, input_texts = convert_examples_to_features(
            examples,
            tokenizer,
            mode,
            labels,
            max_seq_length=args.max_seq_length,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            with open(cached_features_file + "_text.pkl", 'wb') as f:
                pickle.dump(input_texts, f, 0)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_label_ids = torch.tensor([f.start_label_ids for f in features], dtype=torch.long)
    all_end_label_ids = torch.tensor([f.end_label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_label_ids, all_end_label_ids)
    return dataset, input_texts


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        required=True,
        help="The task name.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--fold_data_dir",
        default=None,
        type=str,
        required=True,
        help="The fold data dir",
    )
    parser.add_argument(
        "--start_fold",
        default=None,
        type=int,
        required=True,
        help="Run the program on folds[start_fold:end_fold+1], eg, total 30 folds, we can set start_fold=0, end_fold=29",
    )
    parser.add_argument(
        "--end_fold",
        default=None,
        type=int,
        required=True,
        help="Run the program on folds[start_fold:end_fold+1], eg, total 30 folds, we can set start_fold=0, end_fold=29",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list" + str(list(PLM_models.get_model_names())),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--withO",
        default=False,
        type=bool,
        required=True,
        help="train the model with O label.",
    )
    # Other parameters
    parser.add_argument(
        "--schema",
        default="",
        type=str,
        help="Path to a file containing all labels.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument("--adv_training", default=None, choices=['fgm', 'pgd'], help="fgm adversarial training")
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True, help="Set this flag if model is trained with accents."
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True, help="Set this flag if model is trained without accents."
    )
    parser.add_argument("--use_fast", action="store_const", const=True, help="Set this flag to use fast tokenization.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--early_stop", default=4, type=int,
                        help="early stop when metric does not increases any more")
    parser.add_argument("--freeze",
                        action="store_true",
                        help="freeze embedding layer")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels = get_labels(args.schema, task=args.task, mode="span")  # TODO 这里直接指定mode了，后续要统一一下参数
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    model_path = PLM_models.PLMpath[args.model_name_or_path]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else model_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
    logger.info("Tokenizer arguments: %s", tokenizer_args)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else model_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        **tokenizer_args,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_test_data, all_test_input_texts = load_and_cache_fold_examples(args, tokenizer, labels, mode="test1")

    for fold in range(args.start_fold, args.end_fold+1):
        train_dataset, _ = load_and_cache_fold_examples(args, tokenizer, labels, mode="fold_{}_train".format(fold))
        eval_dataset, eval_input_texts = load_and_cache_fold_examples(args, tokenizer, labels, mode="fold_{}_dev".format(fold))

        model = model_class.from_pretrained(
            model_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        if args.freeze:
            # for param in model.bert.bert.parameters():
            #     param.requires_grad = False
            for name, param in model.named_parameters():
                if 'classifier' not in name:  # classifier layer
                    param.requires_grad = False
                else:
                    print(name)
        model.to(args.device)
        logger.info("Training/evaluation parameters %s", args)

        fold_output_dir = os.path.join(args.output_dir, "fold{}_checkpoint-best".format(fold))
        # Training
        if args.do_train:
            global_step, tr_loss = train_and_eval(args, train_dataset, eval_dataset, eval_input_texts, model, tokenizer, fold_output_dir)
            logger.info("fold %d, global_step = %s, average loss = %s", fold, global_step, tr_loss)

        # Evaluation
        if args.do_eval and args.local_rank in [-1, 0]:
            tokenizer = tokenizer_class.from_pretrained(fold_output_dir, **tokenizer_args)
            model = model_class.from_pretrained(fold_output_dir)
            model.to(args.device)
            result, predictions, logits = evaluate(args, eval_dataset, eval_input_texts, model, prefix="fold_{}_dev".format(fold))
            # Save results
            output_eval_results_file = os.path.join(fold_output_dir, "eval_results.txt")
            with open(output_eval_results_file, "w") as writer:
                for key in sorted(result.keys()):
                    writer.write("{} = {}\n".format(key, str(result[key])))
            # Save predictions
            output_eval_predictions_file = os.path.join(fold_output_dir, "eval_predictions.json")
            results = []
            for prediction in predictions:
                results.append({'labels': prediction})
            write_file(results, output_eval_predictions_file)
            # Save logits
            output_eval_logits_file = os.path.join(fold_output_dir, "eval_logits.json")
            results = []
            for logit in logits:
                results.append({'logits': logit})
            write_file(results, output_eval_logits_file)

        # predict
        if args.do_predict and args.local_rank in [-1, 0]:
            tokenizer = tokenizer_class.from_pretrained(fold_output_dir, **tokenizer_args)
            model = model_class.from_pretrained(fold_output_dir)
            model.to(args.device)
            result, predictions, logits = predict(args, all_test_data, all_test_input_texts, model, prefix="fold_{}_test".format(fold))
            # Save results
            # output_test_results_file = os.path.join(fold_output_dir, "test1_results.txt")
            # with open(output_test_results_file, "w") as writer:
            #     for key in sorted(result.keys()):
            #         writer.write("{} = {}\n".format(key, str(result[key])))
            # Save predictions
            output_test_predictions_file = os.path.join(fold_output_dir, "test1_predictions.json")
            results = []
            for prediction in predictions:
                results.append({'labels': prediction})
            write_file(results, output_test_predictions_file)
            # Save logits
            output_test_logits_file = os.path.join(fold_output_dir, "test1_logits.json")
            results = []
            for logit in logits:
                results.append({'logits': logit})
            write_file(results, output_test_logits_file)


if __name__ == "__main__":
    main()
