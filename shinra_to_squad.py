#!/usr/bin/env python3
# Copyright 2021, Nihon Unisys, Ltd.
#
# This source code is licensed under the BSD license.

import argparse
import json
from pathlib import Path
import html_util
import sys
import os, os.path
# sys.path.append('../shinra_jp_scorer')
from shinra_jp_scorer.scoring import liner2dict, get_annotation, get_ene

FLAG_ATTRS = ['総称']

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

def make_split_data(data_set, split_nums=[0.9]):
    split_datasets = []
    start_idx = 0
    for split_num in split_nums:
        end_idx = int(split_num*len(data_set))
        split_datasets.append(data_set[start_idx:end_idx])
        start_idx = end_idx
    split_datasets.append(data_set[end_idx:])

    return split_datasets


def process(args, dataset, attributes):
    data_size = len(dataset.keys())
    squad_data = []

    for page_id, attrs in dataset.items():
        try:
            with Path(args.html_dir).joinpath(str(page_id)+'.html').open() as f:
                html_content = f.read()
        except:
            print('ERROR! No such file or directory:', filedir.joinpath(str(page_id)+'.html'))
            continue

        try:
            content, _ = html_util.replace_html_tag(html_content, html_tag=args.html_tag)
        except:
            print('ERROR! html_util')
            print(html_content)
            exit()

        line_len = []
        for line in content:
            if len(line) == 0 or (len(line) > 0 and line[-1] != '\n'):
                line_len.append(len(line)+1)
            else:
                line_len.append(len(line))

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

                for q, dist_lines in attrs.items():

                    q_id = str(page_id) + '_' + str(len(paragraphs)) + '_' + str(attributes.index(q))

                    if q in FLAG_ATTRS:
                        flags[q] = True
                        for ans in dist_lines:
                            ENE = ans['ENE']
                            title = ans['title']

                    else:
                        answers = []
                        for ans in dist_lines:
                            ENE = ans['ENE']
                            title = ans['title']
                            ans = ans['html_offset']

                            if para_start_line_num <= ans['start']['line_id'] and para_end_line_num >= ans['end']['line_id']:
                                start_section_idx = 0
                                end_section_idx = 0

                                if para_start_line_num == para_end_line_num:
                                    answer_start_position = ans['start']['offset'] + start_section_idx
                                    answer_end_position = ans['end']['offset'] + end_section_idx
                                else:
                                    if para_start_line_num < ans['start']['line_id']:
                                        answer_start_position = sum(line_len[para_start_line_num:ans['start']['line_id']]) + ans['start']['offset'] + start_section_idx
                                    else:
                                        answer_start_position = ans['start']['offset']  + start_section_idx
                                    if para_start_line_num < ans['end']['line_id']:
                                        answer_end_position = sum(line_len[para_start_line_num:ans['end']['line_id']]) + ans['end']['offset'] + end_section_idx
                                    else:
                                        answer_end_position = ans['end']['offset'] + end_section_idx
                                found_answers.add('-'.join([str(a) for a in [ans['start']['line_id'],ans['start']['offset'],ans['end']['line_id'],ans['end']['offset']]]))
                                if len(paragraph[answer_start_position:answer_end_position].replace(' ','').replace('\n','').strip()) == 0:
                                    print('WARNING! answer text is N/A', q, ans, paragraph[answer_start_position:answer_end_position], title, page_id)
                                    continue
                                answers.append({"answer_start": answer_start_position, "answer_end": answer_end_position, "text": paragraph[answer_start_position:answer_end_position]})

                        qas.append({"question": q, "id": q_id, "answers": answers})

                for q in set(attributes) - set(attrs.keys()):
                    if q in FLAG_ATTRS:
                        flags[q] = False
                    else:
                        qas.append({"question": q, "id": str(page_id) + '_' + str(len(paragraphs)) + '_' + str(attributes.index(q)), "answers": []})
                paragraphs.append({"context": paragraph, "start_line":para_start_line_num, "end_line":para_end_line_num, "qas": qas})
                paragraph = ''

        try:
            if flags.keys():
                squad_json = {"title": title, 'WikipediaID': page_id, "ENE":ENE, "paragraphs": paragraphs, "flags": flags}
            else:
                squad_json = {"title": title, 'WikipediaID': page_id, "ENE":ENE, "paragraphs": paragraphs}
        except Exception as e:
            print(e)
            print('ERROR', page_id, line_num, line)
            print(paragraphs)
            exit()

        squad_data.append(squad_json)
        print('-'*5, str(len(squad_data)) + '/' + str(data_size), str(page_id), title, '-'*5)

    return squad_data


def process_formal(args):
    ENE = attr_list.get_ENE(args.category)

    attr_names = attr_list.get_attr_list(category=args.category)
    attributes = {att:[] for att in attr_names}
    squad_data = []

    files = [f for f in iter_files(Path(args.html_dir))]
    data_size = len(files)

    for i, file in enumerate(files):
        page_id = Path(file).stem
        with open(file) as f:
            html_content = f.read()
        content, title = html_util.replace_html_tag(html_content, html_tag=args.html_tag)

        print('-'*5, str(i) + '/' + str(data_size), str(page_id), title, '-'*5)

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
                q_idx = 0
                qas = []

                for k,v in attributes.items():
                    q = k

                    q_idx += 1
                    q_id = str(page_id) + '_' + str(len(paragraphs)) + '_' + str(q_idx)
                    answers = []
                    qas.append({"answers": answers, "question": q, "id": q_id})

                paragraphs.append({"context": paragraph, "qas": qas, "start_line":para_start_line_num, "end_line":para_end_line_num})
                paragraph = ''

        squad_json = {"title": title, 'WikipediaID': page_id, "ENE":ENE, "paragraphs": paragraphs}
        #print(squad_json)
        squad_data.append(squad_json)

    return squad_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--split_dev', type=float, default=0.85,
                        help='start point of dev data')
    parser.add_argument('--split_test', type=float, default=0.90,
                        help='start point of test data')
    parser.add_argument('--formal', action='store_true',
                        help='formal mode')
    parser.add_argument('--html_dir', type=str, default=None)
    parser.add_argument('--html_tag', action='store_true',
                        help='')
    args = parser.parse_args()


    answer = get_annotation(args.input)
    ene = get_ene(answer)
    id_dict, html, plain, attributes = liner2dict(answer, ene)
    print('attributes:', attributes)

    squad_data = process(args, id_dict, attributes)

    split_dataset = make_split_data(squad_data, split_nums=[args.split_dev, args.split_test])

    with open(args.output.replace('.json', '-train.json'), 'w') as f:
        f.write(json.dumps({"data": split_dataset[0]}, ensure_ascii=False))

    with open(args.output.replace('.json', '-dev.json'), 'w') as f:
        f.write(json.dumps({"data": split_dataset[1]}, ensure_ascii=False))

    if not args.formal:
        with open(args.output.replace('.json', '-test.json'), 'w') as f:
            f.write(json.dumps({"data": split_dataset[2]}, ensure_ascii=False))

main()
