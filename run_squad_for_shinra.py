#!/usr/bin/env python3
# Copyright 2021, Nihon Unisys, Ltd.
#
# This source code is licensed under the BSD license.

import os
import argparse
import logging
import random
import pathlib
import json
import numpy as np

from collections import defaultdict
from scipy.special import softmax
from seqeval.metrics import precision_score, recall_score

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from transformers import BertConfig, BertTokenizer

from transformers import AdamW, get_linear_schedule_with_warmup
from pathlib import Path
import html_util
from shinra_jp_scorer.scoring import get_annotation, get_ene, liner2dict
from bert_for_shinra_jp import (BertForShinraJP, ShinraProcessor, 
				load_and_cache_examples, squad_convert_examples_to_features,
				ShinraExample, ShinraFeatures, ShinraFeatures2) 

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
	logger.info("Creating features from dataset file at %s", args.data_dir)
	dev_examples = load_and_cache_examples(args, tokenizer, mode='dev')
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
			pad_token_label_id=pad_token_label_id,
			not_with_negative=args.not_with_negative
		)
		dev_features_all[c] = dev_features
		dev_dataset_all[c] = dev_dataset

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

	for epoch_num in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
		for step, batch in enumerate(epoch_iterator):
			model.train()
			batch = tuple(t.to(args.device) for t in batch)

			inputs = {
				'input_ids':	   batch[0],
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
				scores_list = defaultdict(list)
				for c in args.categories:
					results, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, dev_dataset_all[c], dev_examples[c], dev_features_all[c], prefix='dev', output_shinra=False)
					for key, value in results.items():
						eval_key = 'eval_{}_{}'.format(c, key)
						logs[eval_key] = value
						scores_list[key].append(value)
				if len(args.categories) > 1:
					for key, value_list in scores_list.items():
						logger.info('epoch-{} evaluate average {} : {}'.format(epoch_num, key, str(np.average(value_list))))

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

	return global_step, tr_loss / global_step

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
				'input_ids':	  batch[0],
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
	preds_argmax = np.argmax(preds, axis=2)
	label_map = {i: label for i, label in enumerate(labels)}

	out_label_list = [[] for _ in range(out_label_ids.shape[0])]
	preds_list = [[] for _ in range(out_label_ids.shape[0])]
	preds_score_list = [[] for _ in range(out_label_ids.shape[0])]

	for i in range(out_label_ids.shape[0]):
		for j in range(out_label_ids.shape[1]):
			if out_label_ids[i, j] != pad_token_label_id:
				out_label_list[i].append(label_map[out_label_ids[i][j]])
				preds_list[i].append(label_map[preds_argmax[i][j]])
				preds_score_list[i].append(preds[i][j])

	precision = precision_score(out_label_list, preds_list)
	recall = recall_score(out_label_list, preds_list)
	f1 =  2 * recall * precision / (recall + precision)

	scores = {
		 "loss": eval_loss,
		 "precision": precision,
		 "recall": recall,
		 "f1": f1
	}
	
	logger.info("***** Eval results %s *****", prefix)
	for key in sorted(scores.keys()):
		logger.info("  %s = %s", key, str(scores[key]))

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

	# counter = defaultdict(lambda:{"TP":0, "TPFP":0, "TPFN":0})
	# for ex_id,item in chunks_gold_page.items():
	# 	for attribute in attributes:
	# 		if chunks_page.get(ex_id) is None or chunks_page[ex_id].get(attribute) is None:
	# 			res = []
	# 		else:
	# 			res = chunks_page[ex_id][attribute]
	# 		if item.get(attribute) is None:
	# 			ans = []
	# 		else:
	# 			ans = item[attribute]
	# 		counter[attribute]["TP"] += len(set(ans) & set(res))
	# 		counter[attribute]["TPFP"] += len(set(res))
	# 		counter[attribute]["TPFN"] += len(set(ans))

	return scores, preds_list, results

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
parser.add_argument("--do_eval", action='store_true',
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
#					 help='only meke cache')
parser.add_argument('--num_examples_split', type=int, default=10000)
parser.add_argument('--num_process', type=int, default=1000)
args = parser.parse_args()

if args.categories and args.group:
	args.categories=args.categories.split(',')
	args.category=args.group
else:
	args.categories = [args.category]

if not args.train_file: args.train_file = 'squad_{}-train.json'.format(args.category)
if not args.predict_file: args.predict_file = 'squad_{}-dev.json'.format(args.category)
if not args.test_file: args.test_file =  'squad_{}-test.json'.format(args.category)

logger = logging.getLogger(__name__)

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
# else:
# 	tokenizer = BertJapaneseTokenizer.from_pretrained(args.base_model_name_or_path,
# 												do_lower_case=args.do_lower_case,
# 												word_tokenizer_type='mecab')

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
		logger.info("Creating features from dataset file at %s", args.data_dir)
		examples = load_and_cache_examples(args, tokenizer)
		features, train_dataset = squad_convert_examples_to_features(
			examples=examples,
			labels=labels,
			tokenizer=tokenizer,
			max_seq_length=args.max_seq_length,
			doc_stride=args.doc_stride,
			max_query_length=args.max_query_length,
			is_training=True,
			return_dataset='pt',
			pad_token_label_id=pad_token_label_id,
			not_with_negative=args.not_with_negative
		)
		del examples
		if args.local_rank in [-1, 0]:
			logger.info("Saving features into cached file %s", cached_features_file)
			torch.save({"features": features, "dataset": train_dataset}, cached_features_file)


	if args.local_rank == 0:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
	# train_dataset = dataset
	# global_step, tr_loss = train(args, train_dataset, model, tokenizer)
	global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)
	logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

	exit()


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
		#	 if str(page_id) in page_ids:  continue
		#	 else:
		#		 logger.info('NOT FOUND {}'.format(str(page_id)))

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
				pad_token_label_id=pad_token_label_id,
				not_with_negative=args.not_with_negative
			)
			_, _, results = evaluate(args, model, tokenizer, labels, pad_token_label_id, dataset, ex, features, prefix=mode, output_shinra=True)
			with open(output_shinra_results_file, "a") as writer:
				for l in results:
					writer.write(json.dumps(l, ensure_ascii=False)+'\n')

if args.do_eval or args.do_predict:
	modes = []
	if args.do_eval: modes.append('dev')
	if args.do_predict: modes.append('test')
	scores_list = defaultdict(dict)

	for c in args.categories:
		for mode in modes:
			if not mode in scores_list: scores_list[mode] = defaultdict(dict)
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
				logger.info("Creating features from dataset file at %s", args.data_dir)
				examples = load_and_cache_examples(args, tokenizer, mode=mode)
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
					pad_token_label_id=pad_token_label_id,
					not_with_negative=args.not_with_negative
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

				if not args.best_model_dir in scores_list[mode]: scores_list[mode][args.best_model_dir] = defaultdict(list)

				with open(output_test_results_file, "w") as writer:
					for key in sorted(scores.keys()):
						writer.write("{} = {}\n".format(key, str(scores[key])))
						scores_list[mode][args.best_model_dir][key].append(scores[key])

			else:
				for epoch in range(args.num_train_epochs):
					if epoch in [0, 1, 2, 3]: continue
					if epoch in [0, 1, 2, 3, 4, 5, 6, 7]: continue
					best_model_dir = '/epoch-'+str(epoch)
					model = model_class.from_pretrained(args.output_dir+best_model_dir)
					model.to(args.device)

					scores, preds_list, results = evaluate(args, model, tokenizer, labels, pad_token_label_id, dataset, examples, features, prefix=mode, output_shinra=True)

					# Save results
					# output_shinra_results_file = os.path.join(args.output_dir+args.best_model_dir, "shinra_{}_{}_results.json".format(c,mode))
					output_shinra_results_file = os.path.join(args.output_dir+best_model_dir, "shinra_{}_{}_results{}.json".format(c,mode,args.result_file_prefix))
					with open(output_shinra_results_file, "w") as writer:
						for l in results:
							writer.write(json.dumps(l, ensure_ascii=False)+'\n')

					# output_test_results_file = os.path.join(args.output_dir+args.best_model_dir, "test_{}_{}_results.txt".format(c,mode))
					output_test_results_file = os.path.join(args.output_dir+best_model_dir, "test_{}_{}_results{}.txt".format(c,mode,args.result_file_prefix))

					if not epoch in scores_list[mode]: scores_list[mode][epoch] = defaultdict(list)
					with open(output_test_results_file, "w") as writer:
						for key in sorted(scores.keys()):
							writer.write("{} = {}\n".format(key, str(scores[key])))
							scores_list[mode][epoch][key].append(scores[key])

	if len(args.categories) > 1:
		for m, epoch_scores in scores_list.items():
			for epoch, scores in epoch_scores.items():
				for key, values in scores.items():
					logger.info("{}:epoch {} average {} = {}".format(m, epoch, key, str(np.average(values))))
