import torch
import torch.nn as nn

from transformers import BertTokenizer, BertConfig, BertModel

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")

'''DNA bert 模型'''


class BERT(nn.Module):
    def __init__(self, kmer):
        super(BERT, self).__init__()
        # 加载预训练模型参数
        self.kmer = kmer
        if self.kmer == 3:
            self.pretrainpath = '../data/pretrain_data/DNAbert_3mer'
        elif self.kmer == 4:
            self.pretrainpath = '../data/pretrain_data/DNAbert_4mer'
        elif self.kmer == 5:
            self.pretrainpath = '../data/pretrain_data/DNAbert_5mer'
        elif self.kmer == 6:
            self.pretrainpath = '../data/pretrain_data/DNAbert_6mer'
        self.setting = BertConfig.from_pretrained(
            self.pretrainpath,
            num_labels=2,
            finetuning_task="dnaprom",
            cache_dir=None,
        )

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath)
        self.bert = BertModel.from_pretrained(self.pretrainpath, config=self.setting)

    def forward(self, seqs):
        # print(seqs)
        seqs = list(seqs)
        kmer = [[seqs[i][x:x + self.kmer] for x in range(len(seqs[i]) + 1 - self.kmer)] for i in range(len(seqs))]
        # print(kmer)
        kmers = [" ".join(kmer[i]) for i in range(len(kmer))]
        # print(kmers)
        # print(len(kmers))
        token_seq = self.tokenizer(kmers, return_tensors='pt')
        # print(token_seq)
        input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
            'attention_mask']
        representation = self.bert(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda())["pooler_output"]
        # if self.config.cuda:
        #     representation = self.bert(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda())["pooler_output"]
        # else:
        #     representation = self.bert(input_ids, token_type_ids, attention_mask)["pooler_output"]

        return representation


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(description='common meta 8learning config')
    parse.add_argument('-kmer', type=int, default=6)
    config = parse.parse_args()

    model = BERT(config).cuda()

    # x = ('TTTAACGATATAACAATCCCAGATTCACAAAGAGATGACCT', 'TGTGTAAGGCGCGTGAACATAGGAAGGAGAAAGCTCGAAGG','TTGATTATACCATTTCAACCATTCAAAGAAGTGCAGATGAT')
    x = ('TTTAAC', 'TTTAAC')

    # encoded_input = tokenizer("AAAAAA", return_tensors='pt')
    # output = model(**encoded_input)
    # tokens = tokenizer.tokenize("CTTGTT")
    model.train()
    output = model(x)
    # print(encoded_input)
    # ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    # print(ids)
    # output  = model(**encoded_input)
    # print(output)
