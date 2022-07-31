from util import util_tokenizer

'''
This file checks if T5 tokenizer is functioning properly.
'''


def test_tokenizer():
    tokenizer = util_tokenizer.get_tokenizer()
    print('tokenizer.cls_token', tokenizer.cls_token, tokenizer.cls_token_id)
    print('tokenizer.sep_token', tokenizer.sep_token, tokenizer.sep_token_id)
    print('tokenizer.mask_token', tokenizer.mask_token, tokenizer.mask_token_id)
    tokenizer.add_special_tokens({'cls_token': '<cls>'})
    tokenizer.add_special_tokens({'sep_token': '<sep>'})
    tokenizer.add_special_tokens({'sep_token': '<mask>'})
    print('tokenizer.cls_token', tokenizer.cls_token, tokenizer.cls_token_id)
    print('tokenizer.sep_token', tokenizer.sep_token, tokenizer.sep_token_id)
    print('tokenizer.mask_token', tokenizer.mask_token, tokenizer.mask_token_id)

    print('-' * 50, 'tokenizer info', '-' * 50)
    map_dict = util_tokenizer.get_map_dict(tokenizer)
    raw_map_dict = map_dict['raw_map_dict']
    residue2id = map_dict['residue2id']
    id2residue = map_dict['id2residue']
    print('raw_map_dict:', raw_map_dict)
    print('residue2id:', residue2id)
    print('id2residue:', id2residue)

    std_residue_tokens, std_residue_ids = util_tokenizer.get_std_residue(tokenizer)
    print('std_residue_tokens:', std_residue_tokens)
    print('std_residue_ids:', std_residue_ids)

    print('-' * 50, 'decode example', '-' * 50)
    example_tokens = [3, 4, 5, 6, 7, 8, 1, 0, 0, 0, 0]
    example_seq = util_tokenizer.get_sequence_from_tokens(example_tokens, tokenizer)
    print('example_tokens:', example_tokens)
    print('example_seq:', example_seq)

    print('-' * 50, 'tokenization example', '-' * 50)
    example_seqs = ['SRTVRKTSRLWSSLSLNTCNNVHSKS', 'SPNITVTLKKFPL']
    example_ids_1 = util_tokenizer.tokenize(example_seqs, tokenizer, add_special_tokens=False,
                                            padding='max_length', max_length=50)
    example_ids_2 = util_tokenizer.tokenize(example_seqs, tokenizer, add_special_tokens=True,
                                            padding=True)
    print('example_sequences:', example_seqs)
    print('example_ids [no special_tokens, padding to specific length]:')
    for ids in example_ids_1['input_ids']:
        print(len(ids), ids)
    print('example_ids [add special_tokens, padding to max length of all sentence]:')
    for ids in example_ids_2['input_ids']:
        print(len(ids), ids)

    print('-' * 50, 'tokenization with added special token example', '-' * 50)
    # cat_seq = ['S R T V R K T S R L W S S L S L N T C N N V H S K S <sep> S P N I T V T L K K F P L']
    cat_seq = [util_tokenizer.cat_two_seq(example_seqs[0], example_seqs[1], '<sep>')]
    print('cat_seq', cat_seq)
    tokenization = tokenizer.batch_encode_plus(cat_seq, add_special_tokens=True)
    print(tokenization['input_ids'])
    print(tokenization['attention_mask'])


if __name__ == '__main__':
    test_tokenizer()
