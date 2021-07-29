#!/usr/bin/env python3
# Copyright 2021, Nihon Unisys, Ltd.
#
# This source code is licensed under the BSD license.

import re
import html
from pathlib import Path

start_pattern = re.compile(r'^\s*<h1\s.+?>(.*?)</h1>\s*')
end_pattern = re.compile(r'^.*<div class="printfooter">.*')
htmltag_pattern = re.compile(r'<!--.*?-->|<[^>]+?>')
section_pattern = re.compile(r'<h(\d).*?><span class="mw-headline".*?>(.+?)</span></h\d>')
escape_pattern = re.compile(r'&[\w#][\w\d]?[\w\d]?[\w\d]?[\w\d]?[\w\d]?;')
htmltag_clean_pattern = re.compile(r'<(\w+)[ ]?[^>]*?>')
br_pattern = re.compile(r'<[/]?(\w+)[ ][/]?>')
table_start_pattern = re.compile(r'(<table)|(&lt;table&gt)')
infobox_start_pattern = re.compile(r'.*<table class="infobox.*|.*<div id="mw-content-text" .*?<table.*')
table_end_pattern = re.compile(r'(</table>)|(&lt;/table&gt)')
title_kakko_pattern = re.compile(r'([^ ]*)[ ]?\(.*\)\w*')
coordinates_pattern = re.compile(r'^(<p>|<span [^>]*>)<span id="coordinates".*')

def html_clean(line):
    return htmltag_clean_pattern.sub(r'<\1>', br_pattern.sub(r'<\1>', line))

def add_section(section_list, layer, midashi):
    if layer > len(section_list):
        section_list.append(midashi)
    elif layer == len(section_list):
        section_list[layer-1] = midashi
    else:
        for i in range(layer-1,len(section_list)):
            section_list.pop()
        section_list.append(midashi)
    return section_list

def get_title(html_data):
    html_data = html_data.replace('‐', '-')
    for line_id, line in enumerate(html_data.split('\n')):
        if start_pattern.match(line):
            title = start_pattern.sub(r'\1', line)
            if title_kakko_pattern.match(title):
                title = title_kakko_pattern.sub(r'\1', title)
            return title
    return ''

def get_plain_text(html_data):
    original_len = len(html_data.split('\n'))
    htmltag_pattern = re.compile(r'<[^>\n]+?>')
    html_data = list(html_data)

    iterator = escape_pattern.finditer(''.join(html_data))
    for match in iterator:
        unescaped_str = ''
        if html.unescape(match.group()) == '\n':
            unescaped_str = ' '
        else:
            unescaped_str = html.unescape(match.group())

        if len(match.group()) != len(unescaped_str):
            html_data[match.start():match.end()] = [' '] + list(unescaped_str) + [' '] + ['τ']*(match.end()-match.start()-len(unescaped_str)-2)
        else:
            html_data[match.start():match.end()] = list(unescaped_str)
    html_data = list(''.join(html_data).replace('τ', ' '))
    iterator = htmltag_pattern.finditer(''.join(html_data))
    for match in iterator:
        html_data[match.start():match.end()] = [' ']*(match.end()-match.start())
    return ''.join(html_data)

def replace_html_tag(html_data, html_tag=True):
    is_content = False
    end_content = False
    is_comment = False
    in_table_tag = 0
    end_table_tag = False
    is_section_start = False
    coordinates_tag = False
    in_infobox_tag = 0
    content_start_pattern = None
    content = []
    section = []
    row = 0
    pre_section_str = ''
    html_data = html_data.replace('‐', '-')
    for line_id, line in enumerate(html_data.split('\n')):
        if coordinates_tag: coordinates_tag = False
        if not is_content and start_pattern.match(line):
            title = start_pattern.sub(r'\1', line)
            if title_kakko_pattern.match(title):
                title = title_kakko_pattern.sub(r'\1', title)

            content_start_pattern = re.compile(r'(.*<p><b>[\w-]*'+re.escape(title.replace(' ', ''))[:1]+'.*)|(^<[pb]>[\w-]*'+re.escape(title.replace(' ', ''))[:1]+'.*)')
        if not is_content and not end_content and content_start_pattern and content_start_pattern.match(line) and \
            in_table_tag == 0:
            is_content = True
            section = add_section(section, 1, '概要')

            if in_infobox_tag > 0 and in_table_tag == 0:
                in_infobox_tag = 0

        if is_content and end_pattern.match(line):
            break
        if is_content and not is_comment and line.startswith('<!--') and not '-->' in line:
            is_comment = True
        res_table_start = table_start_pattern.findall(line)
        if len(res_table_start) > 0:
            in_table_tag += len(res_table_start)
            if infobox_start_pattern.match(line):
                in_infobox_tag = 1
                if len(res_table_start) > 0:
                    in_infobox_tag += (len(res_table_start)-1)
            elif in_infobox_tag > 0:
                in_infobox_tag += len(res_table_start)
        elif infobox_start_pattern.match(line):
            if not 'この記事は' in line:
                in_infobox_tag = 1
        res_table_end = table_end_pattern.findall(line)
        if len(res_table_end) > 0:
            end_table_tag = True

        if coordinates_pattern.match(line):
            coordinates_tag = True
        line = list(line)
        iterator = escape_pattern.finditer(''.join(line))
        for match in iterator:
            if len(match.group()) != len(html.unescape(match.group())):
                line[match.start():match.end()] = [' '] + list(html.unescape(match.group())) + [' '] + ['τ']*(match.end()-match.start()-len(html.unescape(match.group()))-2)
        line = ''.join(line)
        if in_table_tag > 0 or in_infobox_tag > 0:
            if is_comment:
                content.append('')
            else:
                line = list(line)
                iterator = htmltag_pattern.finditer(''.join(line))
                for match in iterator:
                    if html_tag:
                        cleantag = html_clean(match.group())
                        line[match.start():match.end()] = list(cleantag) + ['τ']*(match.end()-match.start()-len(cleantag))
                    else:
                        line[match.start():match.end()] = ['τ']*(match.end()-match.start())
                if (len(section) > 0 and section[-1] in ['参考文献', '出典', '外部リンク', '参考文献、脚注']) or \
                    '加筆訂正願います' in ''.join(line) or \
                    '執筆の途中です' in ''.join(line):
                    content.append('')
                    is_content = False
                    end_content = True
                elif (len(section) > 0 and section[-1] in ['目次', '脚注', '関連項目']) or \
                    '」は途中まで翻訳されたものです' in ''.join(line) or \
                    '曖昧さ回避' in ''.join(line) or \
                    'この項目では、' in ''.join(line) or \
                    'この項目は、' in ''.join(line) or \
                    '書きかけの項目' in ''.join(line) or \
                    'この項目を加筆・訂正' in ''.join(line):
                    content.append('')
                else:
                    if len(''.join(line).replace(' ', '').strip()) > 0:

                        if pre_section_str == ''.join(section):
                            if len(content[-1]) > 0 and ''.join(content)[-1] == '\n':
                                last_new_line_index = [i for i, s in enumerate(content) if '\n' in s][-1]
                                content[last_new_line_index] = content[last_new_line_index].replace('\n', ' ')

                        content.append(''.join(line).replace('τ', ' ')+' ')
                    else:
                        content.append('')
        elif (not is_content and not '<div class="pathnavbox">' in line and not coordinates_tag) or is_comment or \
            'ウィキメディア・コモンズには、' in line or \
            'この項目は、' in line or \
            '書きかけの項目' in line or \
            'この項目を加筆・訂正' in line or \
            'に関連するカテゴリがあります。 ' in line:
            content.append('')
        else:
            if '<h2>目次</h2>' in line:
                section = add_section(section, 1, '目次')
                is_section_start = True
                section_match_str='目次'
            elif section_pattern.match(line):
                section_match = section_pattern.sub(r'\1\t\2', line).split('\t')
                iterator = htmltag_pattern.finditer(section_match[1])
                section_clean = list(section_match[1])
                for match in iterator:
                    section_clean[match.start():match.end()] = ['τ']*(match.end()-match.start())
                section_clean = ''.join(section_clean).replace('τ', '')
                section = add_section(section, int(section_match[0])-1, section_clean)
                is_section_start = True
                section_match_str = section_clean

            line = list(line)
            iterator = htmltag_pattern.finditer(''.join(line))
            for match in iterator:
                cleantag = html_clean(match.group())
                if html_tag:
                    cleantag = html_clean(match.group())
                    line[match.start():match.end()] = list(cleantag) + ['τ']*(match.end()-match.start()-len(cleantag))
                else:
                    line[match.start():match.end()] = ['τ']*(match.end()-match.start())

            line = ''.join(line).replace('τ', ' ')
            if coordinates_tag:
                content.append(line+'\n')
            elif (not is_section_start and len(section) > 0 and section[-1] in ['参考文献', '出典', '外部リンク', '参考文献、脚注']) or \
                '加筆訂正願います' in ''.join(line) or \
                '執筆の途中です' in ''.join(line):
                content.append('')
                is_content = False
                end_content = True
            elif not is_section_start and len(section) > 0 and section[-1] in ['目次', '脚注', '関連項目'] or \
                '」は途中まで翻訳されたものです' in ''.join(line):
                content.append('')
            elif is_section_start and htmltag_pattern.sub('', line).replace(' ', '').strip() == section_match_str:
                if section_match_str in  ['参考文献', '出典', '外部リンク', '脚注', '参考文献、脚注', '目次', '関連項目']:
                    content.append('\n')
                else:
                    content.append(line+'\n')
                pre_section_str = ''.join(section)
                is_section_start = False
            else:
                if len(''.join(line).replace(' ', '').strip()) > 0:
                    if len(section) > 0 and pre_section_str == ''.join(section):
                        if len(content[-1]) > 0 and content[-1][-1] == '\n':
                            content[-1] = content[-1][:-1] + ' '
                    else:
                        if (len(content[-1]) > 0 and content[-1][-1] != '\n') or len(content[-1]) == 0:
                            content[-1] = content[-1] + '\n'

                if len(section) > 0 and len(''.join(section).replace(' ', '').strip()) > 0 and len(''.join(line).replace(' ', '').strip()) > 0:
                    content.append(line+'\n')
                    pre_section_str = ''.join(section)
                elif len(''.join(line).replace(' ', '').strip()) > 0:
                    content.append(line+'\n')
                else:
                    content.append('')

        if end_table_tag:
            end_table_tag = False
            in_table_tag -= len(res_table_end)
            if in_infobox_tag > 0:
                in_infobox_tag -= len(res_table_end)
            if in_table_tag <= 0 and ( (len(content[-1]) > 0 and content[-1][-1] != '\n') or len(content[-1]) == 0):
                content[-1] = content[-1]+'\n'

        if is_comment and '-->' in line:
            is_comment = False

    return content, title
