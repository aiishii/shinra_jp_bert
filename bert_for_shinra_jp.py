#!/usr/bin/env python3
# Copyright 2021, Nihon Unisys, Ltd.
#
# This source code is licensed under the BSD license. 
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset
from tqdm import tqdm

from transformers import BertModel, BertPreTrainedModel
from transformers.data.processors.squad import SquadProcessor, _is_whitespace
from transformers.file_utils import is_torch_available
import MeCab
import mojimoji

def load_and_cache_examples(args, tokenizer, mode='train'):
	if mode != 'train': evaluate = True
	else: evaluate = False
	if args.local_rank not in [-1, 0] and not evaluate:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

	processor = ShinraProcessor(tokenizer, tokenizer_name=args.tokenizer_name)

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
		#	 Labels for computing the token classification loss.
		#	 Indices should be in ``[0, ..., config.num_labels - 1]``.

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
							doc_stride, max_query_length, is_training, pad_token_label_id=None, not_with_negative=False):
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

		if not_with_negative and not include_answer:
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
			#	 # For training, if our document chunk does not contain an annotation
			#	 # we throw it out, since there is nothing to predict.
			#	 doc_start = span["start"]
			#	 doc_end = span["start"] + span["length"] - 1
			#	 out_of_span = False
			#
			#	 if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
			#		 out_of_span = True
			#
			#	 if out_of_span:
			#		 start_position = cls_index
			#		 end_position = cls_index
			#		 span_is_impossible = True
			#	 else:
			#		 if tokenizer.padding_side == "left":
			#			 doc_offset = 0
			#		 else:
			#			 doc_offset = len(truncated_query) + sequence_added_tokens
			#
			#		 start_position = tok_start_position - doc_start + doc_offset
			#		 end_position = tok_end_position - doc_start + doc_offset

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
			#	 del examples[example_index]

def squad_convert_examples_to_features(examples, labels, tokenizer, max_seq_length,
									   doc_stride, max_query_length, is_training,
									   return_dataset=False, pad_token_label_id=None, not_with_negative=False):

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
									doc_stride, max_query_length, is_training, pad_token_label_id=pad_token_label_id, not_with_negative=not_with_negative):
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
		#	 self.start_position = char_to_word_offset[start_position_character]
		#	 self.end_position = char_to_word_offset[end_position_character]
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