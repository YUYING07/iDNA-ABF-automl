# -*- coding=utf-8 -*-
import os
import re
import yaml
# import chardet
from config.load_constant import constant


def check_path(path):
    if not os.path.isdir(path):
        path = os.path.abspath(os.path.dirname(path))
    if not os.path.exists(path):
        os.makedirs(path)
        print('mkdir {}'.format(path))


def read_tsv_data(filename, skip_first=True):
    sequences = []
    labels = []
    with open(filename, 'r') as file:
        if skip_first:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(list[2])
            labels.append(int(list[1]))
    return [sequences, labels]


def read_txt_data(filename, skip_first=False):
    sequences = []
    with open(filename, 'r') as file:
        if skip_first:
            next(file)
        for line in file:
            if '>' not in line and '|' not in line:
                seq = re.match('[\w]+', line).group()
                sequences.append(seq)
    return sequences


def read_fasta_data(fasta_filename):
    with open(fasta_filename, 'r') as file:
        content = file.read()

    content = content.split('>')[1:]
    titles, attributes, seqs = [], [], []
    for item in content:
        head = item.split('\n')[0]
        seq_segs = item.split('\n')[1:-1]

        seq = ''
        for seg in seq_segs:
            seq += seg
        if seq != '':
            seqs.append(seq)

        if '|' in head:
            title = head.split('|')[0]
            attribute = head.split('|')[1]
        else:
            title = head
            attribute = ''

        titles.append(title)
        attributes.append(attribute)

    return titles, attributes, seqs


def write_tsv_data(tsv_filename, labels, sequences):
    if len(labels) == len(sequences):
        with open(tsv_filename, 'w') as file:
            file.write('index\tlabel\tsequence\n')
            for i in range(len(labels)):
                file.write('{}\t{}\t{}\n'.format(i, labels[i], sequences[i]))
        return True
    return False


def write_fasta_data(fasta_filename, titles, attributes, sequences):
    with open(fasta_filename, 'w') as file:
        for i in range(len(sequences)):
            file.write('>{} | {}\n{}\n'.format(titles[i], attributes[i], sequences[i]))
    return True


def save_dict_to_yaml(dict_value: dict, save_path: str):
    # dict保存为yaml
    with open(save_path, 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))


def read_yaml_to_dict(yaml_path: str):
    # 读取yaml并转为dict
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value
