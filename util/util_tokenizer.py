import re
from transformers import T5Tokenizer
import numpy as np
from config.load_constant import constant


def get_tokenizer(path_tokenizer=None):
    if path_tokenizer is None:
        path_tokenizer = constant['path_tokenizer']
    tokenizer = T5Tokenizer.from_pretrained(path_tokenizer, do_lower_case=False)
    return tokenizer


def get_map_dict(tokenizer):
    raw_map_dict = tokenizer.get_vocab()
    residue2id = {}
    for key, value in raw_map_dict.items():
        if 'extra_id' not in key:
            if '<' in key:
                residue2id[key] = value
            else:
                residue2id[key[-1]] = value
    id2residue = {}
    for key, value in residue2id.items():
        id2residue[value] = key
    return {'raw_map_dict': raw_map_dict, 'residue2id': residue2id, 'id2residue': id2residue}


def tokenize(data_list, tokenizer, **kwargs):
    pattern = re.compile('.{1}')
    seqs = [' '.join(pattern.findall(seq)) for seq in data_list]
    seqs_std = [re.sub(r"[UZOB]", "X", seq) for seq in seqs]
    tokenize_res = tokenizer.batch_encode_plus(seqs_std, **kwargs)
    return tokenize_res


def tokenize_residue(data_list, max_len, padding=False):
    pattern = re.compile('.{1}')
    seqs = [','.join(pattern.findall(seq)) for seq in data_list]
    # 分组
    seqs = [seq.split(',') for seq in seqs]
    residue_dicts = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
                     'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
    for seq in seqs:
        for i in range(len(seq)):
            seq[i] = residue_dicts[seq[i]]
    if padding == True:
        for seq in seqs:
            while len(seq) < max_len:
                seq.append('Z')
    print('after padding', np.array(seqs).shape)
    return seqs


def tokenize_feature(data_list, max_len, padding=False):
    data_list = [i.replace('[', '') for i in data_list]
    data_list = [i.replace(']', '') for i in data_list]
    data_list = [i.split(',') for i in data_list]
    tmp, avg_len = [], []
    for i in range(len(data_list)):
        tmp2 = []
        for j in data_list[i]:  # 没有padding
            tmp2.append(float(j))
        avg_len.append(len(data_list[i]))
        tmp.append(tmp2)
        # print(max(avg_len))
        # padding
    if padding == True:
        for i in tmp:
            while len(i) < int(max(avg_len) / 110) * max_len:
                i.append(0)
    print('after padding', np.array(tmp).shape)
    return tmp


# version 1
def get_sequence_from_id(id_list, tokenizer):
    id2residue = get_map_dict(tokenizer)['id2residue']
    raw_seq = [id2residue[id] for id in id_list]
    seq_list = [s for s in raw_seq if '<' not in s]
    seq = ''.join(seq_list)
    return seq


# version 2
def get_sequence_from_tokens(token_list, tokenizer):
    seqs = tokenizer.decode(token_list, skip_special_tokens=True).replace(' ', '')
    return seqs


def get_std_residue(tokenizer):
    id2residue = get_map_dict(tokenizer)['id2residue']
    non_std_list = ['<pad>', '</s>', '<unk>', '</eos>', 'U', 'Z', 'O', 'B', 'X', '<cls>', '<sep>', '<mask>']
    residue_list = id2residue.values()
    std_residue_ids = [i for i, residue in enumerate(residue_list) if residue not in non_std_list]
    std_residue_tokens = [id2residue[i] for i in std_residue_ids if i in id2residue]
    return std_residue_tokens, std_residue_ids


def cat_two_seq(seq1, seq2, special_token):
    cat_seq = ' '.join(str(i) for i in seq1) + ' ' + special_token + ' ' + ' '.join(str(i) for i in seq2)
    return cat_seq
