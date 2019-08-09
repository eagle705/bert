from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pprint import pprint
from model.bert import BERT
from tqdm import tqdm

class BERTDataset(tf.keras.Model):
    def __init__(self, vocab_word2idx, maxlen):
        super(BERTDataset, self).__init__(name='BERTDataset')
        self.vocab_word2idx = vocab_word2idx
        self.vocab_idx2word = {v:k for k,v in vocab_word2idx.items()}
        self.maxlen = maxlen

    def create_LM_mask(self, training_ids_batch, tokenizer, mask_ratio=0.15):
        """
        batch: batch 형태
        ids: sequence
        id or elm: units in sequence
        :param training_ids_batch:
        :param tokenizer:
        :param mask_ratio:
        :return:
        """
        import random
        import copy
        # replace_time = 10 #copy.deepcopy(training_ids_batch) * replace_time


        masked_training_ids_batch = []
        mask_LM_position_index_batch = []
        mask_LM_token_ids_batch = []

        for i, training_ids in enumerate(training_ids_batch): # sentence 단위
            # print("training_ids: ", training_ids)
            # extract random index
            mask_LM_position_index = sorted(random.sample(range(len(training_ids)), int(mask_ratio * len(training_ids))))
            # print("mask_LM_position_index: ", mask_LM_position_index)

            mask_LM_token_ids = []
            masked_training_ids = copy.deepcopy(training_ids)

            for mask_LM_position_index_elm in mask_LM_position_index: # [0, 3, 12, 19]
                mask_LM_token_ids.append(training_ids[mask_LM_position_index_elm])
                # print("mask_LM_position_index_elm: ", mask_LM_position_index_elm)
                # print("training_ids[mask_LM_position_index_elm]: ", training_ids[mask_LM_position_index_elm])

                # 80% of time, replace with [MASK]
                if random.random() < 0.8:
                    # for loop 에서 꺼낸건 call by value임
                    masked_training_ids[mask_LM_position_index_elm] = tokenizer.piece_to_id('[MASK]')

                else:
                    # 10% of time, keep original
                    if random.random() < 0.5:
                        pass
                    # 10% of time, replace with random word
                    else:
                        masked_training_ids[mask_LM_position_index_elm] = list(self.vocab_word2idx.values())[random.randint(0, len(self.vocab_word2idx) - 1)]

            masked_training_ids_batch.append(masked_training_ids)
            mask_LM_position_index_batch.append(mask_LM_position_index)
            mask_LM_token_ids_batch.append(mask_LM_token_ids)

        return masked_training_ids_batch, mask_LM_position_index_batch, mask_LM_token_ids_batch # masked input, masked seq index, LM y

    def add_padding(self, _masked_training_ids_batch, PAD_ID, maxlen):
        """
        return type : <class 'numpy.ndarray'>
        :param _masked_training_ids_batch:
        :param PAD_ID:
        :param maxlen:
        :return:
        """
        return keras.preprocessing.sequence.pad_sequences(_masked_training_ids_batch,
                                                   value=PAD_ID,
                                                   padding='post',
                                                   truncating='post', #이게 pre로 되있었나봄.. 매우중요!
                                                   maxlen=maxlen)

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions so that we can add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_masks(self, inp):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)
        return enc_padding_mask

    def choose_rand_except_num(self, num, total_size):
        import random
        assert num < total_size
        _range = list(range(0, num)) + list(range(num + 1, total_size))
        return random.choice(_range)

    def create_nsp_MLM_dataset(self, ids_batch, mask_LM_label_batch, mask_LM_position_index_batch, tokenizer):
        # Todo:
        # 보통 seq2seq에는 전에 넣는데, 얘는 문장을 붙여버리니 흠.. -> 차후 변경! 패딩 넣고 마지막칸에 넣으면됨

        nsp_input_ids_batch = []

        # NSP label
        nsp_sparse_label_batch = [] # nsp label 1,  not nsp label 0

        # new mask LM index
        integrated_mask_LM_position_index_batch = []

        # segment embedding token
        seg_embed_token_batch = []


        total_size = len(ids_batch)
        segment_to_id = {"<pad>": 0,
                         "[SEG_A]": 1,
                         "[SEG_B]": 2}

        # 그냥 셔플해서 구현해도됨
        for i in range(total_size-1):
            rand_num = self.choose_rand_except_num(i, total_size)

            # nsp input
            nsp_sent_ids = [tokenizer.piece_to_id('[CLS]')] + ids_batch[i] + [tokenizer.piece_to_id('[SEP]')] + ids_batch[i + 1] + [tokenizer.piece_to_id('[SEP]')]
            nsp_input_ids_batch.append(nsp_sent_ids)

            # seg_embed_token = [tokenizer.piece_to_id('[SEG_A]')] * (2 + len(ids_batch[i])) + [tokenizer.piece_to_id('[SEG_B]')] * (len(ids_batch[i+1]) + 1)
            # # 0: padding, 1: Segment_A, 2: Segment_B
            seg_embed_token = [ segment_to_id['[SEG_A]'] ] * (2 + len(ids_batch[i])) + [ segment_to_id['[SEG_B]'] ] * (len(ids_batch[i + 1]) + 1)
            seg_embed_token_batch.append(seg_embed_token)

            integrated_mask_LM_position_index = [pos_index for pos_index in mask_LM_position_index_batch[i]] + [len(ids_batch[i])] +  [pos_index+len(ids_batch[i]) + 1 for pos_index in mask_LM_position_index_batch[i+1]] + [len(ids_batch[i]) + len(ids_batch[i+1]) + 1]
            integrated_mask_LM_position_index = [pos_index for pos_index in integrated_mask_LM_position_index if pos_index < self.maxlen]
            integrated_mask_LM_position_index_batch.append(integrated_mask_LM_position_index)

            # nsp_sparse_label
            nsp_sparse_label = [1] + mask_LM_label_batch[i] + [tokenizer.piece_to_id('[SEP]')] + mask_LM_label_batch[i + 1] + [tokenizer.piece_to_id('[SEP]')]
            nsp_sparse_label_batch.append(nsp_sparse_label)

            # not nsp input
            not_nsp_sent_ids = [tokenizer.piece_to_id('[CLS]')] + ids_batch[i] + [tokenizer.piece_to_id('[SEP]')] + ids_batch[rand_num] + [tokenizer.piece_to_id('[SEP]')]
            nsp_input_ids_batch.append(not_nsp_sent_ids)

            integrated_mask_LM_position_index = [ pos_index for pos_index in mask_LM_position_index_batch[i]] + [len(ids_batch[i])] +  [ pos_index+len(ids_batch[i]) + 1 for pos_index in mask_LM_position_index_batch[rand_num]]+ [len(ids_batch[i]) + len(ids_batch[rand_num]) + 1]
            integrated_mask_LM_position_index = [pos_index for pos_index in integrated_mask_LM_position_index if pos_index < self.maxlen]
            integrated_mask_LM_position_index_batch.append(integrated_mask_LM_position_index)

            # seg_embed_token = [tokenizer.piece_to_id('[SEG_A]')] * (2 + len(ids_batch[i])) + [tokenizer.piece_to_id('[SEG_B]')] * (len(ids_batch[rand_num]) + 1)
            seg_embed_token = [ segment_to_id['[SEG_A]'] ] * (2 + len(ids_batch[i])) + [ segment_to_id['[SEG_B]'] ] * (len(ids_batch[rand_num]) + 1)
            seg_embed_token_batch.append(seg_embed_token)

            not_nsp_sparse_label = [0] + mask_LM_label_batch[i] + [tokenizer.piece_to_id('[SEP]')] + mask_LM_label_batch[rand_num] + [tokenizer.piece_to_id('[SEP]')]

            nsp_sparse_label_batch.append(not_nsp_sparse_label)

            # integrated_mask_LM_position_index = [pos_index for pos_index in mask_LM_position_index_batch[i]] +  [len(ids_batch[i])] + [pos_index + len(ids_batch[i]) + 1 for pos_index in mask_LM_position_index_batch[rand_num]]
            # integrated_mask_LM_position_index = [pos_index for pos_index in integrated_mask_LM_position_index if pos_index < self.maxlen]
            # integrated_mask_LM_position_index_batch.append(integrated_mask_LM_position_index)

        return nsp_input_ids_batch, nsp_sparse_label_batch, integrated_mask_LM_position_index_batch, seg_embed_token_batch