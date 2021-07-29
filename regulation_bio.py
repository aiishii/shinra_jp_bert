#!/usr/bin/env python3
# Copyright 2021, Nihon Unisys, Ltd.
#
# This source code is licensed under the BSD license.

import time
import argparse
import logging
import json

import os, os.path
import html_util
import re
from pathlib import Path
import mojimoji
from shinra_jp_scorer.scoring import get_annotation, get_ene
import regex

not_kanji_pattern = regex.compile(r'[\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Latin}\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]+')
symbol_pattern = re.compile('[\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]+')
htmltag_pattern = re.compile(r'\s*<!--.*?-->\s*|\s*<[^>]+?>\s*')

def is_valid_answer_text(attr_name, ans_t, score):
    ans_t = ans_t.strip()
    if len(ans_t) == 0:
        print('skip len(0)', ans_t)
        return False
    if len(ans_t) == 1 and not_kanji_pattern.fullmatch(ans_t):
        print('skip len(1)', ans_t)
        return False #漢字以外で1文字は回答にしない
    if htmltag_pattern.fullmatch(ans_t) or len(symbol_pattern.sub('', htmltag_pattern.sub('', ans_t).strip()).strip()) == 0:
        print('skip html', ans_t)
        return False #HTMLタグだけだったら回答にしない
    if ans_t in ['φ', '?', '？', '不明'] or ans_t[0] == 'φ':
        if score < 0.8:  return False

    return True

def get_answer_text(attr_value, raw_content):
    # print('get_answer_text', attr_value)#, raw_content)
    raw_content = raw_content.split('\n')
    ans_text = ''
    for idx,line_id in enumerate(range(attr_value["start"]["line_id"],attr_value["end"]["line_id"]+1)):
        sol,eol = 0,len(raw_content[line_id])
        if idx == 0:
            sol = attr_value["start"]["offset"]
        else:
            ans_text += "\n"
        if idx == attr_value["end"]["line_id"] - attr_value["start"]["line_id"]:
            eol = attr_value["end"]["offset"]
        ans_text += raw_content[line_id][sol:eol]
    # print('get_answer_text', attr_value, ans_text)
    return ans_text

def regulation(attr_name, attr_value, raw_content):
    ans_text = get_answer_text(attr_value, raw_content)
    attr_value['text'] = ans_text
    if 'score' in attr_value:
        score = attr_value['score']
    else:
        score = 0.5

    plain_text = html_util.get_plain_text(raw_content)
    ans_plain_text = get_answer_text(attr_value, plain_text)
    attr_value['text'] = ans_plain_text
    if len(plain_text.split('\n')) != len(raw_content.split('\n')):
        print('ERROR line len', len(plain_text.split('\n')) , len(raw_content.split('\n')))
        print(raw_content)
        print('\n')
        print(plain_text)
        exit()

    for c in ans_plain_text:
        if c == ' ':
            attr_value['start']['offset'] += 1
        elif c == '\n' and attr_value['start']['line_id'] < len(plain_text.split('\n'))-1:
            attr_value['start']['line_id'] += 1
            attr_value['start']['offset'] = 0
        else:
            break
    ans_plain_text = get_answer_text(attr_value, plain_text)
    attr_value['text'] = ans_plain_text
    for c in reversed(ans_plain_text):
        if c == ' ':
            attr_value['end']['offset'] -= 1
        elif c == '\n' and attr_value['end']['line_id'] > 0:
            attr_value['end']['line_id'] -= 1
            attr_value['end']['offset'] = len(plain_text.split('\n')[attr_value['end']['line_id']])
        else:
            break

    attr_value['text'] = get_answer_text(attr_value, plain_text)
    #####
    if not is_valid_answer_text(attr_name, attr_value['text'], score):
        return None, None
    attr_value['text'] = get_answer_text(attr_value, raw_content)
    return attr_value, None

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--predicate_json', type=str, default=None,
                    help='Shinra predicted dataset')
parser.add_argument('--category', type=str, default=None,
                    help='Shinra category')
parser.add_argument('--ene', type=str, default=None,
                    help='Shinra category')
parser.add_argument('--SECTION_SEP', type=str, default='',
                    help='Shinra category')
parser.add_argument('--html_dir', type=str, default='',
                    help='')
parser.add_argument('--prefix', type=str, default='',
                    help='')
parser.add_argument('--dist_file', type=str, default=None)
args = parser.parse_args()
t0 = time.time()
#{"page_id": "2730730", "title": "グアヤキル湾", "attribute": "属する海域", "html_offset": {"start": {"line_id": 37, "offset": 192}, "end": {"line_id": 37, "offset": 195}, "text": "太平洋"}, "text_offset": {"start": {"line_id": 37, "offset": 36}, "end": {"line_id": 37, "offset": 39}, "text": "太平洋"}, "ENE": "1.5.3.7"}

answer = get_annotation(args.dist_file)
if args.category:
    category = args.category
    ENE = get_ene(answer)

ignore_pattern = re.compile(r'^[ \n\t]+|[ \n\t]+$')
def get_answer_line_and_offset(para, line_len_sum, start, end):
    # print('get_answer_line_and_offset0', para, start, end)
    ans_text = ''.join(para["context"])[start:end]
    # print('get_answer_line_and_offset', ans_text, start, end)
    iterator = ignore_pattern.finditer(ans_text)
    for match in iterator:
        if match.start() == 0:
            start += len(match.group())
            # print('del start space', ans_text, ''.join(para["context"])[start:end])
        if match.end() == len(ans_text):
            end -= len(match.group())
            # print('del end space', ans_text, ''.join(para["context"])[start:end])
    # print('get_answer_line_and_offset2', ''.join(para["context"])[start:end], start, end)
    start_line = -1
    end_line = -1
    # print('line_len_sum', len(line_len_sum), line_len_sum)
    for i, l_l in enumerate(line_len_sum):
        if start_line ==-1 and l_l >= start:
            start_line = i
        if l_l >= end:
            end_line = i
            break
    # print(start, start_line, line_len_sum)
    if start_line == 0: start_offset = start
    else: start_offset = start - line_len_sum[start_line-1]
    if end_line == 0: end_offset = end
    else: end_offset = end - line_len_sum[end_line-1]
    #print('get_answer_line_and_offset2', para["context"], len(para["context"]), start_line, end_line)
    start_section_idx = para["context"][start_line].find(args.SECTION_SEP)
    if start_section_idx >= 0:
        start_section_idx += len(args.SECTION_SEP)
    else:
        start_section_idx = 0
    end_section_idx = para["context"][end_line].find(args.SECTION_SEP)
    if end_section_idx >= 0:
        end_section_idx += len(args.SECTION_SEP)
    else:
        end_section_idx = 0
    start_offset = start_offset - start_section_idx
    end_offset = end_offset - end_section_idx
    # print(start_line, para["start_line"], start_offset, end_line, para["start_line"], end_offset)
    return (start_line+para["start_line"], start_offset, end_line+para["start_line"], end_offset)

results = []
answerd=set()
predicate_json_files = args.predicate_json.split(',')
for predicate_json_f in predicate_json_files:
    with open(predicate_json_f) as f:
        current_page_id = ''
        raw_content = None
        for line in f:
            line_json = json.loads(line)

            page_id = line_json['page_id']
            para_id = int(line_json['para_id']) if 'para_id' in line_json else 0
            q_id = int(line_json['q_id']) if 'q_id' in line_json else 0
            span_start = int(line_json['span_start']) if 'span_start' in line_json else 0
            #title = line_json['title']
            attribute = mojimoji.zen_to_han(line_json['attribute'], kana=False).replace('(','（').replace(')','）')

            html_offset = line_json['html_offset']
            if 'score' in line_json:
                score = line_json['score']
                # score2 = line_json['score2']
            elif 'score' in line_json['html_offset']:
                score = line_json['html_offset']['score']
                # score2 = line_json['html_offset']['score2']
            # print('line process', page_id, para_id, attribute, html_offset, score)
            # if score < 0.2:
            #     print('score < 0.2 continue')
            #     continue
            if 'ENE' in line_json:
                ENE = line_json['ENE']
                category = attr_list.get_category(ENE)
            if current_page_id != page_id:
                with Path(args.html_dir).joinpath(page_id+'.html').open() as f_html:
                    raw_content = f_html.read()
                    current_page_id = page_id
                # print(type(html_offset["start"]), type(html_offset["start"]) == dict)
                if type(html_offset["start"]) == dict:
                    title = html_util.get_title(raw_content)
                else:
                    clean_content, title = html_util.replace_html_tag(raw_content, SECTION_SEP=args.SECTION_SEP)
                    # print('line process2', page_id, title, para_id, attribute, args.SECTION_SEP)
                    paragraphs = []
                    para = []
                    para_start_line_num = 0
                    para_end_line_num = 0
                    for line_num, line in enumerate(clean_content):
                        if not para and len(line.replace(' ','').replace('\n','').strip()) == 0:
                            continue
                        if not para:
                            para_start_line_num = line_num
                        para.append(line)

                        if len(para) > 0 and len(line) > 0 and line[-1] == '\n':
                            para_end_line_num = line_num
                            paragraphs.append({"context": para, "start_line":para_start_line_num, "end_line":para_end_line_num})
                            para = []

                    para_line_len_sum = []
                    for p in paragraphs:
                        line_len = [len(line) for line in p["context"]]
                        para_line_len_sum.append([sum(line_len[:i+1]) for i, l in enumerate(line_len)])
            if type(html_offset["start"]) == dict:
                html_offset2 = html_offset
                # html_offset2['end']['offset'] += 1
            else:
                start_line, start_offset, end_line, end_offset = get_answer_line_and_offset(paragraphs[para_id], para_line_len_sum[para_id], html_offset["start"], html_offset["end"])

                html_offset2 = {"start": {"line_id": start_line, "offset": start_offset}, "end": {"line_id": end_line, "offset": end_offset}, "score": score}

            raw_answer = ""
            for l in raw_content.split('\n')[html_offset2['start']['line_id']:html_offset2['end']['line_id']+1]:
                if l == raw_content.split('\n')[html_offset2['start']['line_id']]:
                    if html_offset2['start']['line_id']==html_offset2['end']['line_id']:
                        raw_answer = l[html_offset2['start']['offset']:html_offset2['end']['offset']]
                        break
                    else:
                        raw_answer = l[html_offset2['start']['offset']:]
                elif l == raw_content.split('\n')[html_offset2['end']['line_id']]:
                    raw_answer += l[:html_offset2['end']['offset']]
                else:
                    raw_answer += l
            # print('raw answer:', raw_answer, raw_content.split('\n')[html_offset2['start']['line_id']][html_offset2['start']['offset']:html_offset2['end']['offset']])
            attr_value, extra_value = regulation(attribute, html_offset2, raw_content)
            for a_value in [attr_value, extra_value]:
                if not a_value: continue
                a_value_str = '-'.join([str(page_id),attribute,str(a_value['start']['line_id']), str(a_value['start']['offset']), str(a_value['end']['line_id']), str(a_value['end']['offset'])])
                if not a_value_str in answerd:
                    # results.append({"page_id": page_id, "title": title, "attribute":attribute, "html_offset": a_value, "ENE": ENE, "para_id": para_id, "q_id": q_id, "span_start": span_start, 'score': score, 'score2': score2})
                    # results.append({"page_id": page_id, "title": title, "attribute":attribute, "html_offset": a_value, "ENE": ENE, "para_id": para_id, "q_id": q_id, "span_start": span_start, 'score': score})
                    if '<div' in title:
                        title = title.split('<div')[0]
                    results.append({"page_id": page_id, "title": title, "attribute":attribute, "html_offset": a_value, "ENE": ENE, 'score': score})
                    answerd.add(a_value_str)

basename = os.path.basename(args.predicate_json).replace('.json', '')+'.reg'+args.prefix
outfile = os.path.join(os.path.dirname(args.predicate_json), basename)

logger.info('Writing results to %s' % outfile)

with open(outfile+'.json', 'w') as f:
    f.write('\n'.join([json.dumps(res, ensure_ascii=False) for res in results]))

logger.info('Total time: %.2f' % (time.time() - t0))
