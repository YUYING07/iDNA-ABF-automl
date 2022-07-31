import time
import torch
import torch.nn as nn
from transformers import T5Model
from util import util_tokenizer, util_file
from tqdm import tqdm


class Pretrain_Feature_Extractor(nn.Module):
    def __init__(self, args):
        super(Pretrain_Feature_Extractor, self).__init__()
        self.args = args
        self.tokenizer = util_tokenizer.get_tokenizer()

        print('loading pretrain model ...')
        time_start = time.time()
        self.pretrain_model = T5Model.from_pretrained(self.args.path_pretrain_model).cuda()
        time_end = time.time()
        print('pretrain model load complete (time cost: [{}] s)'.format(time_end - time_start))

    def forward(self, raw_seqs):
        ids = util_tokenizer.tokenize(raw_seqs, self.tokenizer)
        input_ids = torch.tensor(ids['input_ids']).cuda()
        attention_mask = torch.tensor(ids['attention_mask']).cuda()
        decoder_input_ids = torch.zeros_like(input_ids).cuda()
        decoder_attention_mask = torch.zeros_like(attention_mask).cuda()

        embeddings = []
        print('model inference ...')
        with torch.no_grad():
            nums = [i for i in range(len(input_ids))]
            pbar = tqdm(nums)
            for i in pbar:
                pbar.set_description('Processing')
                input_ids_i = input_ids[i].unsqueeze(0)
                attention_mask_i = attention_mask[i].unsqueeze(0)
                decoder_input_ids_i = decoder_input_ids[i].unsqueeze(0)
                decoder_attention_mask_i = decoder_attention_mask[i].unsqueeze(0)

                embedding = self.pretrain_model(input_ids=input_ids_i,
                                                attention_mask=attention_mask_i,
                                                decoder_input_ids=decoder_input_ids_i,
                                                decoder_attention_mask=decoder_attention_mask_i)
                embeddings.append(embedding['encoder_last_hidden_state'])
        print('model inference over')

        encode_embeddings = torch.cat(embeddings, dim=0)
        print('encode_embeddings', encode_embeddings.size())

        util_file.check_path(self.args.path_pretrained_data)
        torch.save(encode_embeddings.cpu(), self.args.path_pretrained_data)
        return True
