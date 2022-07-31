import os
import random
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningDataModule
from model import Pretrain_Feature_Extractor
from util import util_file, util_tokenizer


class SeqDataSet(Data.Dataset):
    def __init__(self, input_ids, attn_mask, labels, features=None):
        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.features is not None:
            return self.input_ids[idx], self.attn_mask[idx], self.labels[idx], self.features[idx]
        else:
            return self.input_ids[idx], self.attn_mask[idx], self.labels[idx]


class SeqDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_pretrain_data, self.valid_pretrain_data, self.test_pretrain_data = None, None, None
        self.k = None  # cross validation

    def construct_dataset(self, input_ids, attn_mask, label, features=None):
        return SeqDataSet(input_ids, attn_mask, label, features)

    def prepare_train_mode_data(self, mode):
        # load cooked data
        self.train_raw_data, self.train_label = util_file.read_tsv_data(self.args.path_train_data)
        self.valid_raw_data, self.valid_label = util_file.read_tsv_data(self.args.path_valid_data)
        self.test_raw_data, self.test_label = util_file.read_tsv_data(self.args.path_test_data)

        self.num_train = len(self.train_raw_data)
        self.num_valid = len(self.valid_raw_data)
        self.num_test = len(self.test_raw_data)
        self.all_raw_data = self.train_raw_data + self.valid_raw_data + self.test_raw_data
        print('self.train_raw_data', len(self.train_raw_data), self.num_train)
        print('self.valid_raw_data', len(self.valid_raw_data), self.num_valid)
        print('self.test_raw_data', len(self.test_raw_data), self.num_test)
        # print('self.all_raw_data', len(self.all_raw_data))
        self.sample_train_set()  # sample train set with arg.proportion
        self.tokenization(mode)  # tokenization
        self.load_pretrained_data(mode)  # load pretrained data

    def prepare_test_mode_data(self, mode):
        self.test_raw_data, self.test_label = util_file.read_tsv_data(self.args.path_test_data)
        # print('self.test_raw_data', len(self.test_raw_data))
        self.tokenization(mode)  # tokenization
        self.load_pretrained_data(mode)  # load pretrained data

    # TODO: 考虑使用StratifiedKFold实现KFold，更简洁
    def prepare_cross_validation_mode_data(self, mode, k):
        self.train_raw_data, self.train_label = util_file.read_tsv_data(self.args.path_train_data)
        # print('self.train_raw_data', len(self.train_raw_data))
        self.sample_train_set()  # sample train set with arg.proportion
        self.tokenization(mode)  # tokenization
        self.load_pretrained_data(mode)  # load pretrained data

        if self.k is None:
            self.k = k
            self.train_ids_CV = []
            self.valid_ids_CV = []
            self.train_attn_mask_CV = []
            self.valid_attn_mask_CV = []
            self.train_label_CV = []
            self.valid_label_CV = []
            self.train_pretrain_data_CV = []
            self.valid_pretrain_data_CV = []
            for iter_k in range(self.args.k_fold):
                train_ids_k = [x for i, x in enumerate(self.train_ids) if
                               i % self.args.k_fold != iter_k]
                valid_ids_k = [x for i, x in enumerate(self.train_ids) if
                               i % self.args.k_fold == iter_k]
                train_attn_mask_k = [x for i, x in enumerate(self.train_attn_mask) if
                                     i % self.args.k_fold != iter_k]
                valid_attn_mask_k = [x for i, x in enumerate(self.train_attn_mask) if
                                     i % self.args.k_fold == iter_k]
                train_label_k = [x for i, x in enumerate(self.train_label) if i % self.args.k_fold != iter_k]
                valid_label_k = [x for i, x in enumerate(self.train_label) if i % self.args.k_fold == iter_k]
                self.train_ids_CV.append(train_ids_k)
                self.valid_ids_CV.append(valid_ids_k)
                self.train_attn_mask_CV.append(train_attn_mask_k)
                self.valid_attn_mask_CV.append(valid_attn_mask_k)
                self.train_label_CV.append(train_label_k)
                self.valid_label_CV.append(valid_label_k)
                if self.train_pretrain_data:
                    train_pretrain_data_k = [x for i, x in enumerate(self.train_pretrain_data) if
                                             i % self.args.k_fold != iter_k]
                    valid_pretrain_data_k = [x for i, x in enumerate(self.train_pretrain_data) if
                                             i % self.args.k_fold == iter_k]
                    self.train_pretrain_data_CV.append(train_pretrain_data_k)
                    self.valid_pretrain_data_CV.append(valid_pretrain_data_k)

        if self.k < self.args.k_fold:
            self.train_ids, self.valid_ids = self.train_ids_CV[k], self.valid_ids_CV[k]
            self.train_attn_mask, self.valid_attn_mask = self.train_attn_mask_CV[k], self.valid_attn_mask_CV[k]
            self.train_label, self.valid_label = self.train_label_CV[k], self.valid_label_CV[k]
            if len(self.train_pretrain_data_CV) > 0:
                self.train_pretrain_data, self.valid_pretrain_data = self.train_pretrain_data_CV[k], \
                                                                     self.valid_pretrain_data_CV[k]

    def tokenization(self, mode):
        # data are auto padded by tokenizer
        tokenizer = util_tokenizer.get_tokenizer()
        # default tokenization settings
        tokenize_args = {'add_special_tokens': False, 'padding': 'max_length',
                         'max_length': 50, 'return_tensors': 'pt'}
        # update tokenization settings according to args
        tokenize_args['add_special_tokens'] = self.args.add_special_tokens
        tokenize_args['padding'] = self.args.padding
        tokenize_args['max_length'] = self.args.max_length
        tokenize_args['return_tensors'] = self.args.return_tensors

        if mode == 'train':
            if self.args.onehot == True:
                tokenize_res = util_tokenizer.tokenize(self.all_raw_data, tokenizer, **tokenize_args)
                self.all_input_ids = tokenize_res['input_ids']
                self.all_attn_mask = tokenize_res['attention_mask']
            else:
                self.all_input_ids = self.all_raw_data
                self.all_attn_mask = torch.zeros((len(self.all_raw_data), len(self.all_raw_data[0])))
                # print(self.all_input_ids, self.all_attn_mask)
            self.train_ids, self.train_attn_mask = self.all_input_ids[:self.num_train], self.all_attn_mask[
                                                                                        :self.num_train]
            self.valid_ids, self.valid_attn_mask = self.all_input_ids[self.num_train:self.num_train + self.num_valid], \
                                                   self.all_attn_mask[self.num_train:self.num_train + self.num_valid]
            self.test_ids, self.test_attn_mask = self.all_input_ids[self.num_train + self.num_valid:], \
                                                 self.all_attn_mask[self.num_train + self.num_valid:]
        elif mode == 'test':

            tokenize_res = util_tokenizer.tokenize(self.test_raw_data, tokenizer, **tokenize_args)
            self.test_ids = tokenize_res['input_ids']
            self.test_attn_mask = tokenize_res['attention_mask']
        elif mode == 'cross_validation':
            # 交叉验证时只用到train set
            tokenize_res = util_tokenizer.tokenize(self.train_raw_data, tokenizer, **tokenize_args)
            self.train_ids = tokenize_res['input_ids']
            self.train_attn_mask = tokenize_res['attention_mask']
        else:
            raise RuntimeError('No Such Mode')

        # check the unified length
        if self.args.padding != 'max_length':
            assert self.args.max_length == len(tokenize_res['input_ids'][0])
        # else:
        #     print('tokenization max_length: [{}]'.format(len(tokenize_res['input_ids'][0])))

    def load_pretrained_data(self, mode):
        # use pretrained embeddings
        if mode == 'train':
            raw_data_to_encode = self.all_raw_data
        elif mode == 'test':
            raw_data_to_encode = self.test_raw_data
        elif mode == 'cross_validation':
            raw_data_to_encode = self.train_raw_data
        else:
            raise RuntimeError('No Such Mode')

        if self.args.path_pretrain_model is not None:
            flag = False
            if os.path.exists(self.args.path_pretrained_data):
                flag = True
            else:
                feature_extractor = Pretrain_Feature_Extractor.Pretrain_Feature_Extractor(self.args)
                print('encoding pretrained data within [{}] mode ...'.format(mode))
                flag = feature_extractor(raw_data_to_encode)
            if flag:
                self.all_pretrain_data = torch.load(self.args.path_pretrained_data)

            if mode == 'train':
                self.train_pretrain_data = self.all_pretrain_data[:self.num_train]
                self.valid_pretrain_data = self.all_pretrain_data[self.num_train:self.num_train + self.num_valid]
                self.test_pretrain_data = self.all_pretrain_data[self.num_train + self.num_valid:]
                # print('self.train_pretrain_data', len(self.train_pretrain_data))
                # print('self.valid_pretrain_data', len(self.valid_pretrain_data))
                # print('self.test_pretrain_data', len(self.test_pretrain_data))
            elif mode == 'test':
                self.test_pretrain_data = self.all_pretrain_data
            elif mode == 'cross_validation':
                self.train_pretrain_data = self.all_pretrain_data
            else:
                raise RuntimeError('No Such Mode')

    def sample_train_set(self):
        # 按比例对训练集采样用于训练模型
        if self.args.proportion is not None:
            num_train_selected = int(self.args.proportion * len(self.train_raw_data))
            random.seed(self.args.seed)
            random.shuffle(self.train_raw_data)
            random.seed(self.args.seed)
            random.shuffle(self.train_label)
            self.train_raw_data = self.train_raw_data[:num_train_selected]
            self.train_label = self.train_label[:num_train_selected]
            print('sample train set with proportion: [{}%]'.format(self.args.proportion * 100))

    def prepare_data(self, mode='train'):
        # mode: train, test, inference, K-fold [fold=1, fold=2, ...]
        print('prepare_data [mode]:', mode)
        if mode == 'train':
            self.prepare_train_mode_data(mode)
        elif mode == 'test' or mode == 'inference':
            self.prepare_test_mode_data(mode)
        elif 'fold=' in mode:
            k = int(mode.replace('fold=', ''))
            self.prepare_cross_validation_mode_data('cross_validation', k)
        else:
            raise RuntimeError('No Such Mode')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            assert self.train_ids is not None
            assert self.train_attn_mask is not None
            assert self.train_label is not None
            assert self.valid_ids is not None
            assert self.valid_attn_mask is not None
            assert self.valid_label is not None
            self.train_dataset = self.construct_dataset(self.train_ids, self.train_attn_mask, self.train_label,
                                                        self.train_pretrain_data)
            self.val_dataset = self.construct_dataset(self.valid_ids, self.valid_attn_mask, self.valid_label,
                                                      self.valid_pretrain_data)
            print('self.train_dataset', len(self.train_dataset))
            print('self.val_dataset', len(self.val_dataset))
        elif stage == 'test' or stage is None:
            assert self.test_ids is not None
            assert self.test_attn_mask is not None
            assert self.test_label is not None
            self.test_dataset = self.construct_dataset(self.test_ids, self.test_attn_mask, self.test_label,
                                                       self.test_pretrain_data)
            print('self.test_dataset', len(self.test_dataset))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                          num_workers=self.args.num_workers, drop_last=True)

    # 经实测，训练集不变，随机种子固定，测试性能是一定的，batch_size大小不影响性能
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=True,
                          num_workers=self.args.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True,
                          num_workers=self.args.num_workers)
