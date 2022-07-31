# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        self.visualization = False

        vocab_size = args.vocab_size
        dim_embedding = args.dim_embedding
        filter_num = args.num_filter
        filter_sizes = [int(fsz) for fsz in args.filter_sizes.split(',')]
        num_class = args.num_class

        self.embedding = nn.Embedding(vocab_size, dim_embedding)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, dim_embedding)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(len(filter_sizes) * filter_num, num_class)

        # visualization
        self.filters = [conv.weight for conv in self.convs]
        self.feature_map = None

    def forward(self, x, atten_mask=None, embeddings=None):
        # print(x.shape, x[0])
        if embeddings is None:
            x = self.embedding(x)
            # one-hot
            # x = torch.nn.functional.one_hot(x, 28).float()
        else:
            x = embeddings.requires_grad_(False)

        # print('x.size()', x.size())
        x = x.view(x.size(0), 1, x.size(1), -1)
        x = [F.relu(conv(x)) for conv in self.convs]

        if self.visualization:
            self.feature_map = x

        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        embedding = torch.cat(x, 1)
        output = self.dropout(embedding)
        logits = self.linear(output)
        if self.visualization:
            return logits, embedding, self.feature_map
        else:
            return logits, embedding
