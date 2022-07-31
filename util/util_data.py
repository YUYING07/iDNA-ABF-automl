import re
from transformers import T5Tokenizer
import torch
import numpy as np


def normalization(matrix, col=False, scaling=2):
    res_matrix, tmp_matrix = [], []
    # 得到矩阵形式matrix
    for i in matrix:
        tmp_matrix.append(np.array(i))
    matrix = np.array(tmp_matrix)
    # 每列进行计算
    if col == True:
        for j in range(matrix.shape[1]):
            a = torch.tensor(matrix[:, j])
            # print(a, torch.mean(a), torch.std(a))
            if scaling == 1:  # z-score
                res_matrix.append(z_score_standardization(a).numpy().tolist())
            else:
                res_matrix.append(min_max_scaling(a).numpy().tolist())
        res_matrix = np.array(res_matrix).T
    else:
        # 对每行进行计算
        for i in matrix:
            a = torch.tensor(i)
            # print(a, torch.mean(a), torch.std(a))
            if scaling == 1:  # z-score
                res_matrix.append(z_score_standardization(a).numpy().tolist())
            else:
                res_matrix.append(min_max_scaling(a).numpy().tolist())
        res_matrix = np.array(res_matrix)
    # print('normlize_matrix', res_matrix.shape)
    #   调整矩阵格式 为每条 str
    final_matrix = []
    for i in range(res_matrix.shape[0]):
        final_matrix.append(str(res_matrix[i, :].tolist()))
    return final_matrix


def normalization_matrix(matrix):
    res_matrix,tmp_matrix = [], []
    # 得到矩阵形式matrix
    for i in matrix:
        tmp_matrix.append(np.array(i))
    matrix = torch.tensor(np.array(tmp_matrix))
    matrix = min_max_scaling_matrix(matrix).numpy()
    for i in range(matrix.shape[0]):
        res_matrix.append(str(matrix[i, :].tolist()))
    return res_matrix


def z_score_standardization(a):
    # Z-score standardization to list
    mean_a = torch.mean(a)
    std_a = torch.std(a)
    n1 = (a - mean_a) / std_a
    # print('z_score_standardization', torch.mean(n1), torch.std(n1))
    return n1


def min_max_scaling(a):
    # Min-Max scaling to list
    min_a = torch.min(a)
    max_a = torch.max(a)
    if min_a == 0 and max_a == 0:
        n2 = torch.zeros(a.shape)
    else:
        n2 = (a - min_a) / (max_a - min_a)
    # print('min_max_scaling', a, n2.numpy(), torch.mean(n2), torch.std(n2))
    return n2


def min_max_scaling_matrix(matrix):
    # Min-Max scaling to list
    min_a = torch.min(matrix)
    max_a = torch.max(matrix)
    # print(min_a, max_a)
    if min_a == 0 and max_a == 0:
        n2 = torch.zeros(matrix.shape)
    else:
        n2 = (matrix - min_a) / (max_a - min_a)
    # print('min_max_scaling', matrix, n2.numpy(), torch.mean(n2), torch.std(n2))
    return n2
