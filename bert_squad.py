#!/usr/bin/env python3
# Copyright 2021, Nihon Unisys, Ltd.
#
# This source code is licensed under the BSD license.

import sys
import numpy as np

import os
import argparse
import logging
import random
import pathlib
import json
import MeCab
import mojimoji
from collections import defaultdict
from scipy.special import softmax

from seqeval.metrics import precision_score, recall_score

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from transformers import BertConfig, BertTokenizer #, BertJapaneseTokenizer
from transformers import BertModel, BertPreTrainedModel

from transformers.data.processors.squad import SquadProcessor, _is_whitespace
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.file_utils import is_torch_available
from pathlib import Path
import html_util
from shinra_jp_scorer.scoring import get_annotation, get_ene, liner2dict

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default=None,
                    help='Shinra category')
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--adam_epsilon", default=1e-8, type=float)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--max_seq_length", default=384, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--model_type", default='bert', type=str)
parser.add_argument("--max_query_length", default=20, type=int)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--max_steps", default=-1, type=int)
parser.add_argument("--warmup_steps", default=0, type=int)
parser.add_argument("--logging_steps", default=50, type=int)
parser.add_argument("--save_steps", default=-1, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--n_gpu", default=1, type=int)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int)

parser.add_argument("--doc_stride", default=256, type=int,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_predict", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_formal", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument('--best_model_dir', type=str, default=None)
parser.add_argument('--data_dir', type=str, default='./data/work/bert_base_nict_bpe_htmltag')
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--not_with_negative', action='store_true',
                    help='include no answer data')
parser.add_argument('--test_case_str', type=str, default='',
                    help='test case str')
parser.add_argument('--overwrite_cache', action='store_true',
                    help='overwrite_cache')
parser.add_argument('--cache_test_case_str', type=str, default=None,
                    help='test case str')
parser.add_argument("--predict_category", default=None, type=str,
                    help="comma separated predict category")
parser.add_argument("--model_name_or_path", default='./models/NICT_BERT-base_JapaneseWikipedia_32K_BPE', type=str,
                    help="")
parser.add_argument("--base_model_name_or_path", default='./models/NICT_BERT-base_JapaneseWikipedia_32K_BPE', type=str,
                    help="")
parser.add_argument('--result_file_prefix', type=str, default='',
                    help='result file')
parser.add_argument('--group', type=str, default=None,
                    help='group name')
parser.add_argument('--categories', type=str, default=None,
                    help='categories')
parser.add_argument('--make_cache', action='store_true',
                    help='only meke cache')
parser.add_argument('--train_all_data', action='store_true',
                    help='only meke cache')
parser.add_argument('--overwrite_output_dir', action='store_true',
                    help='overwrite_output_dir')
parser.add_argument('--evaluate_during_training', action='store_true',
                    help='evaluate_during_training')
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--fp16_opt_level', type=str, default='O1')
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--tpu_ip_address', type=str, default='')
parser.add_argument('--tpu_name', type=str, default='')
parser.add_argument('--xrt_tpu_config', type=str, default='')
parser.add_argument('--do_lower_case', action='store_true')
parser.add_argument('--tokenizer_name', type=str, default='mecab_juman')
parser.add_argument('--train_file', type=str, default=None)
parser.add_argument('--predict_file', type=str, default=None)
parser.add_argument('--test_file', type=str, default=None)
parser.add_argument('--dist_file', type=str, default=None)
parser.add_argument('--html_dir', type=str, default=None)
parser.add_argument('--start_page_id', type=str, default=None)
parser.add_argument('--end_page_id', type=str, default=None)
parser.add_argument('--start_idx', type=int, default=None)
parser.add_argument('--end_idx', type=int, default=None)
# parser.add_argument('--check_mode', action='store_true',
#                     help='only meke cache')
parser.add_argument('--num_examples_split', type=int, default=10000)
parser.add_argument('--num_process', type=int, default=1000)
args = parser.parse_args()

if not args.train_file: args.train_file = 'squad_{}-train.json'.format(args.category)
if not args.predict_file: args.predict_file = 'squad_{}-dev.json'.format(args.category)
if not args.test_file: args.test_file =  'squad_{}-test.json'.format(args.category)

if args.categories and args.group:
    args.categories=args.categories.split(',')
    args.category=args.group
else:
    args.categories = [args.category]

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    logger.info("***** load dev examples *****")
    # if args.JP5:
    dev_examples = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, output_examples=True, mode='dev')
    dev_features_all = dict()
    dev_dataset_all = dict()
    for c in args.categories:

        dev_features, dev_dataset = squad_convert_examples_to_features(
            examples=dev_examples[c],
            labels=labels,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False,
            return_dataset='pt',
            pad_token_label_id=pad_token_label_id
        )
        dev_features_all[c] = dev_features
        dev_dataset_all[c] = dev_dataset

    # else:
    #     dev_examples = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, output_examples=True, mode='dev')
    #     dev_features, dev_dataset = squad_convert_examples_to_features(
    #         examples=dev_examples,
    #         labels=labels,
    #         tokenizer=tokenizer,
    #         max_seq_length=args.max_seq_length,
    #         doc_stride=args.doc_stride,
    #         max_query_length=args.max_query_length,
    #         is_training=False,
    #         return_dataset='pt',
    #         pad_token_label_id=pad_token_label_id
    #     )
    # Train!
    if args.make_cache: exit()
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    f1_scores = []
    for epoch_num in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                'input_ids':       batch[0],
                'attention_mask':  batch[1],
                'labels':  batch[3]
                # 'start_positions': batch[3],
                # 'end_positions':   batch[4]
            }

            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = None if args.model_type == 'xlm' else batch[2]

            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[6], 'p_mask': batch[7]})

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.local_rank in [-1, 0]:
            logs = {}
            if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                logger.info('epoch-{}: evaluate '.format(epoch_num)+'{}_{}_{}_{}'.format(args.output_dir, args.per_gpu_train_batch_size, args.num_train_epochs, args.learning_rate))
                for c in args.categories:
                    results, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, dev_dataset_all[c], dev_examples[c], dev_features_all[c], prefix='dev', output_shinra=False)
                    for key, value in results.items():
                        eval_key = 'eval_{}_{}'.format(c, key)
                        logs[eval_key] = value
                # else:
                #     results, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, dev_dataset, dev_examples, dev_features, prefix='dev', output_shinra=False)
                #
                #     for key, value in results.items():
                #         eval_key = 'eval_{}'.format(key)
                #         logs[eval_key] = value
                #         if key == 'f1' : f1_scores.append(value)

            output_dir = os.path.join(args.output_dir, 'epoch-{}'.format(epoch_num))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)

            for key, value in logs.items():
                tb_writer.add_scalar(key, value, global_step)
            print(json.dumps({**logs, **{'step': global_step}}))
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, f1_scores

def get_chunks(seq, begin="B", default=["O"]):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: ["O", "O", "B", "I", ...] sequence of labels

    Returns:
        list of (chunk_start, chunk_end)
    """
    # default = "O"
    chunks = []
    chunk_start = None
    for i, tok in enumerate(seq):
        if (tok in default or tok == begin) and not chunk_start is None:
            chunk = (chunk_start, i)
            chunks.append(chunk)
            chunk_start = None
        if tok == begin:
            chunk_start = i

    if chunk_start is not None:
        chunk = (chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
def simple_accuracy(preds, labels):
    return (preds == labels).mean()
def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    tp, fn, fp, tn = confusion_matrix(labels, preds).ravel()
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        # "tp, fn, fp, tn": str((tp, fn, fp, tn)),
        "acc_0": tp / (tp + fp),
        "acc_1": tn / (tn + fn)
    }

def metrics_doc(preds, labels):
    return acc_and_f1(preds, labels)

def evaluate(args, model, tokenizer, labels, pad_token_label_id, dataset, examples, features, prefix="", output_shinra=False):

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} {}*****".format(args.category, prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'labels':  batch[3]
            }

            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = None if args.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids

            # example_indices = batch[4]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[6], 'p_mask': batch[7]})

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    # preds_max = np.max(preds, axis=2)
    # preds_sum = np.sum(preds, axis=2)
    preds_argmax = np.argmax(preds, axis=2)
    # preds_max[0]
    # preds_argmax[0]
    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_score_list = [[] for _ in range(out_label_ids.shape[0])]
    # preds_max_list = [[] for _ in range(out_label_ids.shape[0])]
    # preds_sum_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds_argmax[i][j]])
                preds_score_list[i].append(preds[i][j])
                # preds_max_list[i].append(preds_max[i][j])
                # preds_sum_list[i].append(preds_sum[i][j])

    scores = {}
    # scores = {
    #     "loss": eval_loss,
    #     "precision": precision_score(out_label_list, preds_list),
    #     "recall": recall_score(out_label_list, preds_list),
    #     "f1": f1_score(out_label_list, preds_list)
    # }
    #
    # logger.info("***** Eval results %s *****", prefix)
    # for key in sorted(scores.keys()):
    #     logger.info("  %s = %s", key, str(scores[key]))

    if not output_shinra:
        return scores, preds_list

    results = []
    ex_id = -1
    word_to_char_offset = None
    answerd_entry = set()
    current_page_id = ''
    chunks_page = dict()
    chunks_gold_page = dict()
    attributes = set()
    # f_idx=0
    # f = features[f_idx]
    # f.tokens
    # len(f.tokens)
    for f_idx, f in enumerate(features):
        pred = preds_list[f_idx]
        gold = out_label_list[f_idx]
        pred_score = preds_score_list[f_idx]
        pred_score_softmax = softmax(preds_score_list[f_idx], axis=1)

        chunks = get_chunks(pred)
        # print(chunks)
        chunks_gold = get_chunks(gold)
        # print(chunks_gold)

        if not f.example_index in chunks_page: chunks_page[f.example_index] = dict()
        if not f.example_index in chunks_gold_page: chunks_gold_page[f.example_index] = dict()
        if ex_id != f.example_index:
            # added = set()
            word_to_char_offset = []
            current_w_index = -1
            for c_index, w_index in enumerate(examples[f.example_index].char_to_word_offset):
                # if w_index in added: continue
                if current_w_index == w_index or w_index == -1: continue
                word_to_char_offset.append(c_index)
                current_w_index = w_index
                # added.add(w_index)
        ex_id = f.example_index
        valid_index = np.where(np.array(f.label_ids)!=-100)[0]

        para_text_len_sum = []
        for line in examples[f.example_index].context_text.split('\n'):
            if len(para_text_len_sum) == 0:
                para_text_len_sum.append(len(line)+1)
            else:
                para_text_len_sum.append(para_text_len_sum[-1] + len(line)+1)

        qas_id_split = examples[f.example_index].qas_id.split('_')
        if len(qas_id_split) == 3:
            page_id, para_id, q_id = qas_id_split
        else:
            page_id, q_id = qas_id_split
            para_id = 0

        attr = examples[f.example_index].question_text
        attributes.add(attr)
        if not attr in chunks_page[f.example_index]: chunks_page[f.example_index][attr] = set()
        if not attr in chunks_gold_page[f.example_index]: chunks_gold_page[f.example_index][attr] = set()
        chunks_page[f.example_index][attr].update(set([(para_id, c_s, c_e) for c_s, c_e in chunks]))
        # print(chunks_page[f.example_index][attr])
        chunks_gold_page[f.example_index][attr].update(set([(para_id, c_s, c_e) for c_s, c_e in chunks_gold]))
        # print(chunks_gold_page[f.example_index][attr])

        for c_s, c_e in chunks:
            pred_score
            # score = np.mean(pred_max[c_s:c_e])
            score = np.mean(np.max(pred_score[c_s:c_e+1], axis = 1))
            score2 = np.mean(np.max(pred_score_softmax[c_s:c_e+1], axis=1))

            if f.token_to_orig_map[valid_index[c_s]] != 0 and c_s == 0: continue
            c_s = valid_index[c_s]

            if c_e == len(valid_index):
                c_e = valid_index[c_e-1]
                continue
            else:
                c_e = valid_index[c_e]

            entry = dict()
            entry['page_id'] = page_id
            if current_page_id != page_id:
                answerd_entry = set()
                current_page_id = page_id
            entry['para_id'] = para_id
            entry['q_id'] = q_id
            entry['span_start'] = f.span_start

            current_q_id = q_id
            entry['attribute'] = examples[f.example_index].question_text


            para_start_offset = word_to_char_offset[f.token_to_orig_map[c_s]]
            if f.token_to_orig_map[c_e]+1 < len(word_to_char_offset):
                para_end_offset =  word_to_char_offset[f.token_to_orig_map[c_e]]
            else:
                para_end_offset = len(examples[f.example_index].context_text)-1
            start_line = -1
            end_line = -1

            for i, line_len_sum in enumerate(para_text_len_sum):
                if start_line < 0 and para_start_offset  < line_len_sum:
                    start_line = i + f.para_start_line
                    if i > 0:
                        start_offset = para_start_offset - para_text_len_sum[i-1]
                    else:
                        start_offset = para_start_offset
                if end_line < 0 and para_end_offset  < line_len_sum:
                    end_line = i + f.para_start_line
                    if i > 0:
                        end_offset = para_end_offset - para_text_len_sum[i-1]
                    else:
                        end_offset = para_end_offset
                if start_line >= 0 and end_line >= 0:
                    break

            entry['html_offset'] = dict()
            entry['html_offset']['start'] = dict()
            entry['html_offset']['start']['line_id'] = start_line
            entry['html_offset']['start']['offset'] = start_offset

            entry['html_offset']['end'] = dict()
            entry['html_offset']['end']['line_id'] = end_line
            entry['html_offset']['end']['offset'] = end_offset

            entry['html_offset']['text'] = examples[f.example_index].context_text[para_start_offset:para_end_offset+1]
            entry['score'] = float(score * 0.1)
            entry['score2'] = float(score2)

            answer_str = '-'.join([entry['attribute'],str(start_line),str(start_offset),str(end_line),str(end_offset)])
            if not answer_str in  answerd_entry:
                results.append(entry)
                answerd_entry.add(answer_str)

    counter = defaultdict(lambda:{"TP":0, "TPFP":0, "TPFN":0})
    for ex_id,item in chunks_gold_page.items():
        for attribute in attributes:
            if chunks_page.get(ex_id) is None or chunks_page[ex_id].get(attribute) is None:
                res = []
            else:
                res = chunks_page[ex_id][attribute]
            if item.get(attribute) is None:
                ans = []
            else:
                ans = item[attribute]
            counter[attribute]["TP"] += len(set(ans) & set(res))
            counter[attribute]["TPFP"] += len(set(res))
            counter[attribute]["TPFN"] += len(set(ans))


    return scores, preds_list, results

def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, output_examples=False, mode='train'):
    if mode != 'train': evaluate = True
    else: evaluate = False
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    logger.info("Creating features from dataset file at %s", input_dir)

    processor = ShinraProcessor(tokenizer, tokenizer_name=args.tokenizer_name)
    # if not args.JP5:
    #     if mode == 'train':
    #         examples = processor.get_train_examples(args.data_dir, filename=args.train_file)
    #         if args.train_all_data:
    #             examples.extend(processor.get_train_examples(args.data_dir, filename=args.predict_file))
    #             examples.extend(processor.get_train_examples(args.data_dir, filename=args.test_file))
    #     elif mode == 'dev':
    #         examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
    #     else:
    #         examples = processor.get_dev_examples(args.data_dir, filename=args.test_file)
    # else:
    if mode == 'train':
        examples = []
    else:
        examples = dict()

    for c in args.categories:
        if mode == 'train':
            examples.extend(processor.get_train_examples(args.data_dir, filename='squad_{}-train.json'.format(c)))
            if args.train_all_data:
                examples.extend(processor.get_train_examples(args.data_dir, filename='squad_{}-dev.json'.format(c)))
                examples.extend(processor.get_train_examples(args.data_dir, filename='squad_{}-test.json'.format(c)))
        elif mode == 'dev':
            examples[c] = processor.get_dev_examples(args.data_dir, filename='squad_{}-dev.json'.format(c))
        else:
            examples[c] = processor.get_dev_examples(args.data_dir, filename='squad_{}-test.json'.format(c))

    return examples


class BertForShinraJP(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
        # a nice puppet


    """
    def __init__(self, config):
        super(BertForShinraJP, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
                # start_positions=None, end_positions=None):
        # **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
        #     Labels for computing the token classification loss.
        #     Indices should be in ``[0, ..., config.num_labels - 1]``.

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # logits = self.qa_outputs(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
        # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

def process_shinra_examples(examples, labels, tokenizer, max_seq_length,
                            doc_stride, max_query_length, is_training, pad_token_label_id=None):
    unique_id = 1000000000

    label_map = {label: i for i, label in enumerate(labels)}

    features = []

    # print(examples)
    for example_index in tqdm(reversed(range(len(examples)))):
        example = examples[example_index]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        all_doc_labels = []
        is_begin = False
        include_answer = False
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            # sub_tokens = tokenizer.subword_tokenizer.tokenize(token)
            # print('token', token)
            # print('sub_tokens', sub_tokens)
            # all_doc_labels.extend([label_map[example.answer_labels[i]]] + [pad_token_label_id] * (len(sub_tokens) - 1))
            if example.answer_labels[i] == 'B':
                is_begin = True
                include_answer = True
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
                if is_begin:
                    all_doc_labels.append(label_map[example.answer_labels[i]])
                    is_begin = False
                elif example.answer_labels[i] == 'B':
                    all_doc_labels.append(label_map['I'])
                else:
                    all_doc_labels.append(label_map[example.answer_labels[i]])

        if args.not_with_negative and not include_answer:
            continue

        spans = []

        truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
        sequence_added_tokens = tokenizer.max_len - tokenizer.max_len_single_sentence
        sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair
        # len(all_doc_tokens)
        # len(all_doc_labels)
        span_doc_tokens = all_doc_tokens
        span_doc_labels = all_doc_labels
        if span_doc_tokens == None or truncated_query == None or span_doc_tokens[0] == None or truncated_query[0] == None:
            print('span_doc_tokens', span_doc_tokens)
            print('truncated_query', truncated_query)
            print('example.doc_tokens', example.doc_tokens)
            print('example.question_text', example.question_text)
        while len(spans) * doc_stride < len(all_doc_tokens):

            encoded_dict = tokenizer.encode_plus(
                truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
                span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                pad_to_max_length=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                truncation_strategy='only_second' if tokenizer.padding_side == "right" else 'only_first'
            )

            paragraph_len = min(len(all_doc_tokens) - len(spans) * doc_stride, max_seq_length - len(truncated_query) - sequence_pair_added_tokens)

            if tokenizer.pad_token_id in encoded_dict['input_ids']:
                non_padded_ids = encoded_dict['input_ids'][:encoded_dict['input_ids'].index(tokenizer.pad_token_id)]
            else:
                non_padded_ids = encoded_dict['input_ids']
            # len(tokens)
            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
            label_ids = [pad_token_label_id] * (1 + len(truncated_query) + 1) + span_doc_labels[:paragraph_len] + [pad_token_label_id]
            if len(label_ids) < len(encoded_dict['token_type_ids']):
                label_ids = label_ids + [pad_token_label_id] * (len(encoded_dict['token_type_ids'])-len(label_ids))


            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len
            encoded_dict["label_ids"] = label_ids
            encoded_dict["para_start_line"] = example.para_start_line
            encoded_dict["para_end_line"] = example.para_end_line
            encoded_dict["is_answerable"] = 1 if label_map['B'] in label_ids else 0

            spans.append(encoded_dict)

            if "overflowing_tokens" not in encoded_dict:
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]
            span_doc_labels = span_doc_labels[doc_stride:]
            assert len(span_doc_tokens)==len(span_doc_labels)
            # assert len(encoded_dict["label_ids"]) == len(encoded_dict["input_ids"])

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["paragraph_len"]):
                is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
                index = j if tokenizer.padding_side == "left" else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context

        for span in spans:
            # Identify the position of the CLS token
            cls_index = span['input_ids'].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = np.array(span['token_type_ids'])

            p_mask = np.minimum(p_mask, 1)

            if tokenizer.padding_side == "right":
                # Limit positive values to one
                p_mask = 1 - p_mask

            p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

            # Set the CLS index to '0'
            p_mask[cls_index] = 0

            # span_is_impossible = example.is_impossible
            # start_position = 0
            # end_position = 0
            # if is_training and not span_is_impossible:
            #     # For training, if our document chunk does not contain an annotation
            #     # we throw it out, since there is nothing to predict.
            #     doc_start = span["start"]
            #     doc_end = span["start"] + span["length"] - 1
            #     out_of_span = False
            #
            #     if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
            #         out_of_span = True
            #
            #     if out_of_span:
            #         start_position = cls_index
            #         end_position = cls_index
            #         span_is_impossible = True
            #     else:
            #         if tokenizer.padding_side == "left":
            #             doc_offset = 0
            #         else:
            #             doc_offset = len(truncated_query) + sequence_added_tokens
            #
            #         start_position = tok_start_position - doc_start + doc_offset
            #         end_position = tok_end_position - doc_start + doc_offset

            yield ShinraFeatures(
                span['input_ids'],
                span['attention_mask'],
                span['token_type_ids'],
                cls_index,
                p_mask.tolist(),

                example_index=example_index,
                unique_id=unique_id,
                paragraph_len=span['paragraph_len'],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                label_ids=span['label_ids'],
                is_answerable=span['is_answerable'],
                # start_position=start_position,
                # end_position=end_position
                para_start_line=span["para_start_line"],
                para_end_line=span["para_end_line"],
                span_start=span["start"]
            )
            unique_id += 1
            # if is_training:
            #     del examples[example_index]

def squad_convert_examples_to_features(examples, labels, tokenizer, max_seq_length,
                                       doc_stride, max_query_length, is_training,
                                       return_dataset=False, pad_token_label_id=None):

    features = []
    if return_dataset == 'pt':
        if not is_torch_available():
            raise ImportError("Pytorch must be installed to return a pytorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids_l = []
        all_input_mask_l = []
        all_segment_ids_l = []
        all_label_ids_l = []
        all_is_answerable_l = []
        all_cls_index_l = []
        all_p_mask_l = []
        all_input_ids = None
        all_input_mask = None
        all_segment_ids = None
        all_label_ids = None
        all_is_answerable = None
        all_cls_index = None
        all_p_mask = None
        for f in process_shinra_examples(examples, labels, tokenizer, max_seq_length,
                                    doc_stride, max_query_length, is_training, pad_token_label_id=pad_token_label_id):
            all_input_ids_l.append(f.input_ids)
            all_input_mask_l.append(f.attention_mask)
            all_segment_ids_l.append(f.token_type_ids)
            all_label_ids_l.append(f.label_ids)
            all_is_answerable_l.append(f.is_answerable)
            all_cls_index_l.append(f.cls_index)
            all_p_mask_l.append(f.p_mask)
            if is_training:
                features.append(ShinraFeatures2(
                                example_index=f.example_index,
                                token_to_orig_map=f.token_to_orig_map,
                                label_ids=f.label_ids,
                                is_answerable=f.is_answerable,
                                para_start_line=f.para_start_line,
                                para_end_line=f.para_end_line
                            ))
            else:
                features.append(f)


            if len(all_input_ids_l) == 10000:
                if all_input_ids is None:
                    all_input_ids = torch.tensor(all_input_ids_l, dtype=torch.long)
                    all_input_mask = torch.tensor(all_input_mask_l, dtype=torch.long)
                    all_segment_ids = torch.tensor(all_segment_ids_l, dtype=torch.long)
                    all_label_ids = torch.tensor(all_label_ids_l, dtype=torch.long)
                    all_is_answerable = torch.tensor(all_is_answerable_l, dtype=torch.long)
                    all_cls_index = torch.tensor(all_cls_index_l, dtype=torch.long)
                    all_p_mask = torch.tensor(all_p_mask_l, dtype=torch.float)
                else:
                    all_input_ids = torch.cat((all_input_ids, torch.tensor(all_input_ids_l, dtype=torch.long)), dim=0)
                    all_input_mask = torch.cat((all_input_mask, torch.tensor(all_input_mask_l, dtype=torch.long)), dim=0)
                    all_segment_ids = torch.cat((all_segment_ids, torch.tensor(all_segment_ids_l, dtype=torch.long)), dim=0)
                    all_label_ids = torch.cat((all_label_ids, torch.tensor(all_label_ids_l, dtype=torch.long)), dim=0)
                    all_is_answerable = torch.cat((all_is_answerable, torch.tensor(all_is_answerable_l, dtype=torch.long)), dim=0)
                    all_cls_index = torch.cat((all_cls_index, torch.tensor(all_cls_index_l, dtype=torch.long)), dim=0)
                    all_p_mask = torch.cat((all_p_mask, torch.tensor(all_p_mask_l, dtype=torch.float)), dim=0)
                all_input_ids_l = []
                all_input_mask_l = []
                all_segment_ids_l = []
                all_label_ids_l = []
                all_is_answerable_l = []
                all_cls_index_l = []
                all_p_mask_l = []

        if len(all_input_ids_l) > 0:
            if all_input_ids is None:
                all_input_ids = torch.tensor(all_input_ids_l, dtype=torch.long)
                all_input_mask = torch.tensor(all_input_mask_l, dtype=torch.long)
                all_segment_ids = torch.tensor(all_segment_ids_l, dtype=torch.long)
                all_label_ids = torch.tensor(all_label_ids_l, dtype=torch.long)
                all_is_answerable = torch.tensor(all_is_answerable_l, dtype=torch.long)
                all_cls_index = torch.tensor(all_cls_index_l, dtype=torch.long)
                all_p_mask = torch.tensor(all_p_mask_l, dtype=torch.float)
            else:
                all_input_ids = torch.cat((all_input_ids, torch.tensor(all_input_ids_l, dtype=torch.long)), dim=0)
                all_input_mask = torch.cat((all_input_mask, torch.tensor(all_input_mask_l, dtype=torch.long)), dim=0)
                all_segment_ids = torch.cat((all_segment_ids, torch.tensor(all_segment_ids_l, dtype=torch.long)), dim=0)
                all_label_ids = torch.cat((all_label_ids, torch.tensor(all_label_ids_l, dtype=torch.long)), dim=0)
                all_is_answerable = torch.cat((all_is_answerable, torch.tensor(all_is_answerable_l, dtype=torch.long)), dim=0)
                all_cls_index = torch.cat((all_cls_index, torch.tensor(all_cls_index_l, dtype=torch.long)), dim=0)
                all_p_mask = torch.cat((all_p_mask, torch.tensor(all_p_mask_l, dtype=torch.float)), dim=0)
            del all_input_ids_l
            del all_input_mask_l
            del all_segment_ids_l
            del all_label_ids_l
            del all_is_answerable_l
            del all_cls_index_l
            del all_p_mask_l


        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_is_answerable,
                                    all_example_index, all_cls_index, all_p_mask)
        else:
            # all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            # all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_is_answerable,
                                    # all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)

        return features, dataset


    return features

# non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), ' ')
class ShinraProcessor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"
    def __init__(self, tokenizer, tokenizer_name='mecab'):
        self.tokenizer = tokenizer #MeCab.Tagger(f"-Owakati")
        self.tagger_jumandic = MeCab.Tagger(f"-Owakati -d /usr/lib/mecab/dic/jumandic")

        # self.tagger_ipadic = MeCab.Tagger(f"-Owakati")
        self.tokenizer_name = tokenizer_name

    def _create_examples(self, input_data, set_type):
        # set_type="dev"###
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            # entry=input_data[0]####
            title = entry['title']
            for paragraph in entry["paragraphs"]:
                # paragraph = entry["paragraphs"][0]###
                context_text = paragraph["context"]
                para_start_line = paragraph["start_line"]
                para_end_line = paragraph["end_line"]
                for qa in paragraph["qas"]:
                    # qa = paragraph["qas"][0]###
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    end_position_character = None
                    answer_text = None
                    answers = qa["answers"]

                    context_text = unicodedata.normalize('NFKC', context_text)
                    question_text = unicodedata.normalize('NFKC', question_text)

                    if self.tokenizer_name == 'mecab_juman':
                        question_text = mojimoji.han_to_zen(question_text).replace("\u3000", " ").rstrip("\n")
                        question_tokens = self.tagger_jumandic.parse(question_text).rstrip("\n").split()
                        context_text = mojimoji.han_to_zen(context_text).replace("\u3000", " ").rstrip("\n")
                        context_tokens = self.tagger_jumandic.parse(context_text).rstrip("\n").split()
                    else:
                        context_text = unicodedata.normalize('NFKC', context_text)
                        question_text = unicodedata.normalize('NFKC', question_text)
                        question_tokens = self.tokenizer.word_tokenizer.tokenize(question_text)
                        context_tokens = self.tokenizer.word_tokenizer.tokenize(context_text)
                    try:
                        example = ShinraExample(
                            qas_id=qas_id,
                            question_text=question_text,
                            question_tokens=question_tokens,
                            context_text=context_text,
                            context_tokens=context_tokens,
                            answers=answers,
                            para_start_line=para_start_line,
                            para_end_line=para_end_line
                        )
                        examples.append(example)
                    except:
                        print('example = ShinraExample error!')

        return examples

import unicodedata
class ShinraExample(object):
    def __init__(self,
                 qas_id,
                 question_text,
                 question_tokens,
                 context_text,
                 context_tokens,
                 answers=[],
                 para_start_line=None,
                 para_end_line=None):#,
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answers = answers
        self.para_start_line = para_start_line
        self.para_end_line = para_end_line

        doc_tokens = context_tokens
        token_i = 0
        token_i_j = 0
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        token = doc_tokens[token_i]
        # print(doc_tokens)
        # len(doc_tokens)

        for c in context_text:
            if _is_whitespace(c):
                if token_i == 0:
                    char_to_word_offset.append(-1)
                else:
                    # char_to_word_offset.append(token_i)
                    char_to_word_offset.append(-1)
            else:
                if token[token_i_j] == c:
                    char_to_word_offset.append(token_i)
                    if token_i_j+1 < len(token):
                        token_i_j += 1
                    elif token_i+1 < len(doc_tokens):
                        token_i +=1
                        token = doc_tokens[token_i]
                        token_i_j = 0
                    elif token_i+1==len(doc_tokens):
                        pass
                    else:
                        print('error1', token_i, token_i_j, token, c)
                        raise ValueError("ERROR1 ShinraExample token_i={},token_i_j={},token={},c={}".format(token_i, token_i_j, token, c))
                        # break

                else:
                    #raise ValueError("ERROR2")
                    #print('error2', token_i, token_i_j, token, c)
                    raise ValueError("ERROR2 ShinraExample token_i={},token_i_j={},token={},c={}".format(token_i, token_i_j, token, c))
                    # break

        # char_to_word_offset[-1]
        # context_text[-1]
        # len(char_to_word_offset)
        # len(context_text)
        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        # [{'answer_end': 7084, 'answer_start': 7070, 'length': 14, 'text': 'サントペコアこくさいくうこう'}]
        self.answer_labels = ['O'] * len(doc_tokens)
        # doc_tokens[char_to_word_offset[a['answer_start']]:char_to_word_offset[a['answer_end']]]
        # self.answer_labels[char_to_word_offset[a['answer_start']]:char_to_word_offset[a['answer_end']]]
        for a in answers:

            try:
                if len(char_to_word_offset) < a['answer_start'] or len(char_to_word_offset) < a['answer_end']:
                    print(self.qas_id, self.question_text, a['text'], a['answer_start'], a['answer_end'], char_to_word_offset)
                for i in range(char_to_word_offset[a['answer_start']], char_to_word_offset[a['answer_end']-1]+1):
                    if i == char_to_word_offset[a['answer_start']]:
                        self.answer_labels[i] = 'B'
                    else:
                        self.answer_labels[i] = 'I'
            except Exception as e:
                print("context_text[a['answer_start']:a['answer_end']]", context_text[a['answer_start']:a['answer_end']])
                print("error: {0}".format(e))
                print(context_text)
                print(char_to_word_offset)
                print(doc_tokens)
                print(a)
        # if start_position_character is not None and not is_impossible:
        #     self.start_position = char_to_word_offset[start_position_character]
        #     self.end_position = char_to_word_offset[end_position_character]
            # self.end_position = char_to_word_offset[start_position_character + len(answer_text) - 1]
class ShinraFeatures(object):
    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 cls_index,
                 p_mask,

                 example_index,
                 unique_id,
                 paragraph_len,
                 token_is_max_context,
                 tokens,
                 token_to_orig_map,

                 label_ids,
                 is_answerable,
                 para_start_line=None,
                 para_end_line=None,
                 span_start=None
                 # start_position,
                 # end_position
        ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.label_ids = label_ids
        self.is_answerable = is_answerable
        self.para_start_line = para_start_line
        self.para_end_line = para_end_line
        self.span_start = span_start

class ShinraFeatures2(object):
    def __init__(self,
                 example_index,
                 token_to_orig_map,
                 label_ids,
                 is_answerable,
                 para_start_line,
                 para_end_line
        ):
        self.example_index = example_index
        self.token_to_orig_map = token_to_orig_map
        self.label_ids = label_ids
        self.is_answerable = is_answerable
        self.para_start_line = para_start_line
        self.para_end_line = para_end_line

pathlib.Path(args.output_dir).mkdir(exist_ok=True)
args.output_dir = args.output_dir+'/{}_{}_train_batch{}_epoch{}_lr{}_seq{}'.format(args.category, args.test_case_str, args.per_gpu_train_batch_size, args.num_train_epochs, args.learning_rate, args.max_seq_length)

logger = logging.getLogger(__name__)
# !mkdir $args.output_dir
# !rm -r $args.output_dir
log_file = args.output_dir+'/train.log'
pathlib.Path(args.output_dir).mkdir(exist_ok=True)
fh = logging.FileHandler(log_file)
logger.addHandler(fh)


if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir))>1 and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1
args.device = device

# Setup logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

# Set seed
set_seed(args)

labels = ['B','I','O']
num_labels = len(labels)
# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
pad_token_label_id = CrossEntropyLoss().ignore_index

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

args.model_type = args.model_type.lower()
config_class, model_class = BertConfig, BertForShinraJP
config = config_class.from_pretrained(args.base_model_name_or_path,
                                      num_labels=num_labels)
if args.tokenizer_name == 'mecab_juman':
    tokenizer = BertTokenizer.from_pretrained(args.base_model_name_or_path,
                                                vocab_file=f'{args.base_model_name_or_path}/vocab.txt',
                                                do_lower_case=args.do_lower_case)
else:
    tokenizer = BertJapaneseTokenizer.from_pretrained(args.base_model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                word_tokenizer_type='mecab')


if args.do_train:
    if args.make_cache:
        model = None
    else:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config)
        model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    input_dir = args.data_dir if args.data_dir else "."
    mode = 'train'
    if args.cache_test_case_str:
        cached_features_file = os.path.join(input_dir, 'cached_{}_{}_{}_{}_{}'.format(
            args.cache_test_case_str,
            args.category,
            mode,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length))
        )
    else:
        cached_features_file = os.path.join(input_dir, 'cached_{}_{}_{}_{}_{}'.format(
            args.test_case_str,
            args.category,
            mode,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length))
        )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, train_dataset = features_and_dataset["features"], features_and_dataset["dataset"]
    else:
        examples = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, output_examples=False)
        features, train_dataset = squad_convert_examples_to_features(
            examples=examples,
            labels=labels,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            return_dataset='pt',
            pad_token_label_id=pad_token_label_id
        )
        del examples
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": train_dataset}, cached_features_file)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # train_dataset = dataset
    # global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    global_step, tr_loss, f1_scores = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    logger.info(" BEST EPOCH = {}".format(np.argmax(f1_scores)))
    sys.stdout.write(str(np.argmax(f1_scores)))
    exit()
# %% codecell
#args.do_predict=True
def split_list(l, n):
    return [l[idx:idx + n] for idx in range(0,len(l), n)]

def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)

def html_process(args, attributes, ENE, process_count=1000, page_ids=None):
    files = [f for f in iter_files(Path(args.html_dir))]
    data_size = len(files)
    squad_data = []
    start_flg = False
    end_flg = False
    if not args.start_page_id and not args.start_idx:
        start_flg = True
    for i, file in enumerate(files):
        page_id = Path(file).stem

        if not start_flg and (page_id == args.start_page_id or i == args.start_idx):
            start_flg = True
        if not start_flg: continue
        if end_flg: break
        if (args.end_page_id  and page_id == args.end_page_id) or (args.end_idx and i == args.end_idx):
            end_flg = True

        # if args.check_mode and page_ids:
        #     if str(page_id) in page_ids:  continue
        #     else:
        #         logger.info('NOT FOUND {}'.format(str(page_id)))

        with open(file) as f:
            html_content = f.read()
        try:
            content, title = html_util.replace_html_tag(html_content, html_tag=True)
        except Exception as e:
            print('ERROR! html_util', e)
            continue

        line_len = []
        for line in content:
            if len(line) == 0 or (len(line) > 0 and line[-1] != '\n'):
                line_len.append(len(line)+1)
            else:
                line_len.append(len(line))
        # line_len = [len(line) for line in content]
        flags = dict()
        paragraphs = []
        paragraph = ''
        found_answers = set()
        para_start_line_num = 0
        para_end_line_num = 0
        for line_num, line in enumerate(content):
            if not paragraph and len(line.replace(' ','').replace('\n','').strip()) == 0:
                continue
            if not paragraph:
                para_start_line_num = line_num
            paragraph += line
            if len(line) == 0 or (len(line) > 0 and line[-1] != '\n'):
                paragraph += '\n'

            if len(paragraph) > 0 and len(line) > 0 and line[-1] == '\n':
                para_end_line_num = line_num
                qas = []
                # for q, dist_lines in attrs.items():
                for q in attributes:
                    q_id = str(page_id) + '_' + str(len(paragraphs)) + '_' + str(attributes.index(q))

                    if q in FLAG_ATTRS:
                        continue

                    answers = []
                    qas.append({"question": q, "id": q_id, "answers": answers})

                paragraphs.append({"context": paragraph, "start_line":para_start_line_num, "end_line":para_end_line_num, "qas": qas})
                paragraph = ''

        squad_json = {"title": title, 'WikipediaID': page_id, "ENE":ENE, "paragraphs": paragraphs}

        squad_data.append(squad_json)
        logger.info('---- {} / {} {} {} ----'.format(str(i), str(data_size), str(page_id), title))
        if len(squad_data) >= process_count:
            yield squad_data
            squad_data = []

    yield squad_data

if args.do_formal:
    FLAG_ATTRS = ['総称']
    mode="formal"
    processor = ShinraProcessor(tokenizer, tokenizer_name=args.tokenizer_name)

    print(args.dist_file)
    answer = get_annotation(args.dist_file)
    ene = get_ene(answer)

    _, _, _, attributes = liner2dict(answer, ene)
    logger.info('**{} {} {} {} **'.format(args.category, ene, 'attributes:', ' '.join(attributes)))

    model = model_class.from_pretrained(args.output_dir+args.best_model_dir)
    model.to(args.device)

    output_shinra_results_file = os.path.join(args.output_dir+args.best_model_dir, "shinra_{}_{}_results{}.json".format(args.category,mode,args.result_file_prefix))
    page_ids=set()

    for input_data in html_process(args, attributes, ene, process_count=args.num_process, page_ids=page_ids):
        examples = processor._create_examples(input_data, mode)

        examples_split = split_list(examples, args.num_examples_split)
        # print(len(examples), len(examples_split), [len(ex) for ex in examples_split])

        for ex in examples_split:
            features, dataset = squad_convert_examples_to_features(
                examples=ex,
                labels=labels,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False,
                return_dataset='pt',
                pad_token_label_id=pad_token_label_id
            )
            _, _, results = evaluate(args, model, tokenizer, labels, pad_token_label_id, dataset, ex, features, prefix=mode, output_shinra=True)
            with open(output_shinra_results_file, "a") as writer:
                for l in results:
                    writer.write(json.dumps(l, ensure_ascii=False)+'\n')

if args.do_predict and args.local_rank in [-1, 0]:
    # modes = ['test', 'dev']
    modes = ['test']
    # modes = ['dev']
    # modes = ['train']
    for c in args.categories:
        for mode in modes:
            args.category = c
            input_dir = args.data_dir if args.data_dir else "."
            cached_features_file = os.path.join(input_dir, 'cached_{}_{}_{}_{}_{}'.format(
                args.test_case_str,
                c,
                mode,
                list(filter(None, args.model_name_or_path.split('/'))).pop(),
                str(args.max_seq_length))
            )

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
                features_and_dataset = torch.load(cached_features_file)
                examples, features, dataset = features_and_dataset["examples"], features_and_dataset["features"], features_and_dataset["dataset"]
            else:
                # args.test_file = 'squad_{}-test.json'.format(c)
                args.test_file = 'squad_{}-test.json'.format(c)
                args.predict_file = 'squad_{}-dev.json'.format(c)

                examples = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, output_examples=True, mode=mode)
                examples = examples[args.category]
                # dataset, examples, features = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, output_examples=True, mode=prefix)
                features, dataset = squad_convert_examples_to_features(
                    examples=examples,
                    labels=labels,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=False,
                    return_dataset='pt',
                    pad_token_label_id=pad_token_label_id
                )
                if args.local_rank in [-1, 0]:
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save({"features": features, "dataset": dataset, "examples":examples}, cached_features_file)
            if args.make_cache: continue
            if args.best_model_dir:
                model = model_class.from_pretrained(args.output_dir+args.best_model_dir)
                model.to(args.device)

                scores, preds_list, results = evaluate(args, model, tokenizer, labels, pad_token_label_id, dataset, examples, features, prefix=mode, output_shinra=True)

                # Save results
                # output_shinra_results_file = os.path.join(args.output_dir+args.best_model_dir, "shinra_{}_{}_results.json".format(c,mode))
                output_shinra_results_file = os.path.join(args.output_dir+args.best_model_dir, "shinra_{}_{}_results{}.json".format(c,mode,args.result_file_prefix))
                with open(output_shinra_results_file, "w") as writer:
                    for l in results:
                        writer.write(json.dumps(l, ensure_ascii=False)+'\n')

                # output_test_results_file = os.path.join(args.output_dir+args.best_model_dir, "test_{}_{}_results.txt".format(c,mode))
                output_test_results_file = os.path.join(args.output_dir+args.best_model_dir, "test_{}_{}_results{}.txt".format(c,mode,args.result_file_prefix))
                with open(output_test_results_file, "w") as writer:
                    for key in sorted(scores.keys()):
                        writer.write("{} = {}\n".format(key, str(scores[key])))

            else:
                for epoch in range(10):
                    # if epoch in [0, 1, 2, 3]: continue
                    args.best_model_dir = '/epoch-'+str(epoch)
                    model = model_class.from_pretrained(args.output_dir+args.best_model_dir)
                    model.to(args.device)

                    scores, preds_list, results = evaluate(args, model, tokenizer, labels, pad_token_label_id, dataset, examples, features, prefix=mode, output_shinra=True)

                    # Save results
                    # output_shinra_results_file = os.path.join(args.output_dir+args.best_model_dir, "shinra_{}_{}_results.json".format(c,mode))
                    output_shinra_results_file = os.path.join(args.output_dir+args.best_model_dir, "shinra_{}_{}_results{}.json".format(c,mode,args.result_file_prefix))
                    with open(output_shinra_results_file, "w") as writer:
                        for l in results:
                            writer.write(json.dumps(l, ensure_ascii=False)+'\n')

                    # output_test_results_file = os.path.join(args.output_dir+args.best_model_dir, "test_{}_{}_results.txt".format(c,mode))
                    output_test_results_file = os.path.join(args.output_dir+args.best_model_dir, "test_{}_{}_results{}.txt".format(c,mode,args.result_file_prefix))
                    with open(output_test_results_file, "w") as writer:
                        for key in sorted(scores.keys()):
                            writer.write("{} = {}\n".format(key, str(scores[key])))
