from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pprint import pprint
from konlpy.tag import Mecab

import sys
import pickle
import os
import codecs
import argparse
from collections import Counter
from threading import Thread
from tqdm import tqdm


PAD = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK = "<unk>"
NUM = "<num>"
NONE = "0"
CLS = "[CLS]"
MASK = "[MASK]"
SEP = "[SEP]"
mecab = Mecab()


import sentencepiece as spm

class SentencePieceTokenizer:
    def __init__(self):
        """
        # https://github.com/google/sentencepiece#redefine-special-meta-tokens
        # https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
        # https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
        """

        # training setup
        # CLS 토큰쪽이 0,1 뱉는데 이게 Padding, BOS랑 헷갈릴수있음.. 체크해야함
        self.templates = '--input={} --model_prefix={} --vocab_size={} --user_defined_symbols=[MASK],[CLS],[SEP],[SEG_A],[SEG_B] --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3'# --max_sentence_length={}'
        self.vocab_size = 780
        self.prefix = 'm'#'2016-10-20-news'
        self.input_file = './data_in/sentencepiece_train.txt'

    def train(self):
        cmd = self.templates.format(self.input_file, self.prefix, self.vocab_size)
        # Train model
        spm.SentencePieceTrainer.Train(cmd)

    def load(self):
        # Load model
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load('{}.model'.format(self.prefix))

        self.vocab_token2idx = {self.sp.id_to_piece(id): id for id in range(self.sp.get_piece_size())}
        self.vocab_idx2token = {id: self.sp.id_to_piece(id) for id in range(self.sp.get_piece_size())}

    def piece_to_id(self, piece):
        return self.sp.piece_to_id(piece)

    def pad_id(self):
        return self.sp.pad_id()

    def get_piece_size(self):
        return self.sp.get_piece_size()

    def encode_as_pieces(self, sent):
        return self.sp.EncodeAsPieces(sent)

    def encode_as_ids(self, sent):
        return self.sp.EncodeAsIds(sent)

    def id_to_piece(self, id):
        return self.sp.IdToPiece(id)

    def decode_pieces(self, encoded_pieces):
        return self.sp.decode_pieces(encoded_pieces)

    def decode_ids(self, ids):
        return self.sp.decode_ids(ids)


class Vocabulary(object):
    """Vocab Class"""

    def __init__(self, token2idx=None):

        self.token2idx = {}
        self.idx2token = {}
        self.idx = 0

        self.PAD = "<pad>"; self.START_TOKEN = "<s>"; self.END_TOKEN = "</s>"
        self.UNK = "<unk>"; self.CLS = "[CLS]"; self.MASK = "[MASK]"
        self.SEP = "[SEP]"; self.SEG_A = "[SEG_A]"; self.SEG_B = "[SEG_B]"
        self.NUM = "<num>"; self.NONE = "0"

        self.special_tokens = [self.PAD,
                                self.START_TOKEN,
                                self.END_TOKEN,
                                self.UNK,
                                self.CLS,
                                self.MASK,
                                self.SEP,
                                self.SEG_A,
                                self.SEG_B,
                                self.NUM,
                                self.NONE]
        self.init_vocab()


        if token2idx is not None:
            self.token2idx = token2idx
            self.idx2token = {v:k for k,v in token2idx.items()}
            self.idx = len(token2idx) - 1

    def init_vocab(self):
        for special_token in self.special_tokens:
            self.add_token(special_token)

    def to_indices(self, tokens):
        return [self.transform_token2idx(X_token) for X_token in tokens]

    def add_token(self, token):
        if not token in self.token2idx:
            self.token2idx[token] = self.idx
            self.idx2token[self.idx] = token
            self.idx += 1

    def transform_token2idx(self, token):
        try:
            return self.token2idx[token]
        except:
            print("key error: "+ str(token))
            token = UNK
            return self.token2idx[token]

    def transform_idx2token(self, idx):
        try:
            return self.idx2token[idx]
        except:
            print("key error: " + str(idx))
            idx = self.token2idx[UNK]
            return self.idx2token[idx]

    def __len__(self):
        return len(self.token2idx)

    def build_vocab(self, text_list, threshold=1, vocab_save_path="./data_in/token_vocab.json", split_fn=Mecab().pos):
        """Build a token vocab"""

        def do_concurrent_tagging(start, end, text_list, counter):
            for i, text in enumerate(text_list[start:end]):
                text = text.strip()
                text = text.lower()

                try:
                    tokens_ko = split_fn(text)
                    tokens_ko = [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]
                    counter.update(tokens_ko)

                    if i % 1000 == 0:
                        print("[%d/%d (total: %d)] Tokenized input text." % (
                            start + i, start + len(text_list[start:end]), len(text_list)))

                except Exception as e:  # OOM, Parsing Error
                    print(e)
                    continue

        counter = Counter()

        num_thread = 4
        thread_list = []
        n_x_text = len(text_list)
        for i in range(num_thread):
            thread_list.append(Thread(target=do_concurrent_tagging, args=(
                int(i * n_x_text / num_thread), int((i + 1) * n_x_text / num_thread), text_list, counter)))

        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()

        print(counter.most_common(10))  # print most common tokens
        tokens = [token for token, cnt in counter.items() if cnt >= threshold]


        for i, token in enumerate(tokens):
            self.add_token(str(token))

        import json
        # print(self.token2idx)
        with open(vocab_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=4)
        # with open(vocab_save_path, 'wb') as f:
        #     pickle.dump(self.token2idx, f)


def mecab_token_pos_flat_fn(string):
    tokens_ko = mecab.pos(string)
    return [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]

def mecab_token_pos_sep_fn(string):
    tokens_ko = mecab.pos(string)
    list_of_token = []
    list_of_pos = []
    for token, pos in tokens_ko:
        list_of_token.append(token)
        list_of_pos.append(pos)
    return list_of_token, list_of_pos

def mecab_token_fn(string):
    tokens_ko = mecab.pos(string)
    return [str(pos[0]) for pos in tokens_ko]

def mecab_pos_fn(string):
    tokens_ko = mecab.pos(string)
    return [str(pos[0]) for pos in tokens_ko]

def keras_pad_fn(token_ids_batch, maxlen, pad_id=0, padding='post', truncating='post'):
    padded_token_ids_batch = keras.preprocessing.sequence.pad_sequences(token_ids_batch,
                                                                    value=pad_id,#vocab.transform_token2idx(PAD),
                                                                    padding=padding,
                                                                    truncating=truncating,
                                                                    maxlen=maxlen)
    return np.array(padded_token_ids_batch)


class Tokenizer:
    """ Tokenizer class"""
    def __init__(self, vocab, split_fn, pad_fn, maxlen):
        self._vocab = vocab
        self._split = split_fn
        self._pad = pad_fn
        self._maxlen = maxlen

    # def split(self, string: str) -> list[str]:
    def split(self, string):
        tokens = self._split(string)
        return tokens

    # def transform(self, list_of_tokens: list[str]) -> list[int]:
    def transform(self, tokens):
        indices = self._vocab.to_indices(tokens)
        pad_indices = self._pad(indices, pad_id=0, maxlen=self._maxlen) if self._pad else indices
        return pad_indices

    # def split_and_transform(self, string: str) -> list[int]:
    def split_and_transform(self, string):
        return self.transform(self.split(string))

    @property
    def vocab(self):
        return self._vocab

    # def list_of_string_to_two_list_of_tokens_poses(self, X_str_batch):
    #     X_token_batch = []
    #     X_pos_batch = []
    #     for X_str in X_str_batch:
    #         x_token, x_pos = self._split(X_str)
    #         X_token_batch.append(x_token)
    #         X_pos_batch.append(x_pos)
    #     return X_token_batch, X_pos_batch

    def list_of_tokens_to_list_of_token_ids(self, X_token_batch):
        X_ids_batch = []
        for X_tokens in X_token_batch:
            X_ids_batch.append([self._vocab.transform_token2idx(X_token) for X_token in X_tokens])
        return X_ids_batch

    def list_of_string_to_list_of_tokens(self, X_str_batch):
        X_token_batch = [self._split(X_str) for X_str in X_str_batch]
        return X_token_batch

    def list_of_tokens_to_list_of_token_ids(self, X_token_batch):
        X_ids_batch = []
        for X_tokens in X_token_batch:
            X_ids_batch.append([self._vocab.transform_token2idx(X_token) for X_token in X_tokens])
        return X_ids_batch

    def list_of_string_to_list_token_ids(self, X_str_batch, add_start_end_token=False):
        X_token_batch = self.list_of_string_to_list_of_tokens(X_str_batch)
        X_ids_batch = self.list_of_tokens_to_list_of_token_ids(X_token_batch)

        return X_ids_batch

    def list_of_string_to_arr_of_pad_token_ids(self, X_str_batch, add_start_end_token=False):
        X_token_batch = self.list_of_string_to_list_of_tokens(X_str_batch)
        if add_start_end_token is True:
            return self.add_start_end_token_with_pad(X_token_batch)
        else:
            X_ids_batch = self.list_of_tokens_to_list_of_token_ids(X_token_batch)
            pad_X_ids_batch = self._pad(X_ids_batch, pad_id=0, maxlen=self._maxlen)
        return pad_X_ids_batch

    def add_start_end_token_with_pad(self, X_token_batch):
        X_start_end_token_batch = [[START_TOKEN] + X_token + [END_TOKEN] for X_token in X_token_batch]
        X_start_end_ids_batch = self.list_of_tokens_to_list_of_token_ids(X_start_end_token_batch)
        pad_X_start_end_ids_batch = self._pad(X_start_end_ids_batch, pad_id=0, maxlen=self._maxlen)

        target_input_token_batch = [[START_TOKEN] + X_token for X_token in X_token_batch]
        target_real_token_batch = [X_token + [END_TOKEN] for X_token in X_token_batch]

        target_input_ids_batch = self.list_of_tokens_to_list_of_token_ids(target_input_token_batch)
        pad_target_input_ids_batch = self._pad(target_input_ids_batch, pad_id=0, maxlen=self._maxlen)

        target_real_ids_batch = self.list_of_tokens_to_list_of_token_ids(target_real_token_batch)
        pad_target_real_ids_batch = self._pad(target_real_ids_batch, pad_id=0, maxlen=self._maxlen)
        return pad_X_start_end_ids_batch, pad_target_input_ids_batch, pad_target_real_ids_batch

    def decode_token_ids(self, token_ids_batch):
        token_token_batch = []
        for token_ids in token_ids_batch:
            token_token = [self._vocab.transform_idx2token(token_id) for token_id in token_ids]
            # token_token = [self._vocab[token_id] for token_id in token_ids]
            token_token_batch.append(token_token)
        return token_token_batch

        return ' '.join([reverse_token_index.get(i, '?') for i in text])



def build_vocab(text_list, threshold=1, vocab_path="./data_in/token_vocab.pkl", tokenizer_type="mecab"):
    """Build a token vocab"""

    def do_concurrent_tagging(start, end, text_list, counter):
        for i, text in enumerate(text_list[start:end]):
            text = text.strip()
            text = text.lower()

            try:
                if tokenizer_type == "mecab":
                    tokens_ko = mecab.pos(text)
                    tokens_ko = [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]

                counter.update(tokens_ko)

                if i % 1000 == 0:
                    print("[%d/%d (total: %d)] Tokenized input text." % (
                    start + i, start + len(text_list[start:end]), len(text_list)))

            except Exception as e:  # OOM, Parsing Error
                print(e)
                continue

    counter = Counter()

    num_thread = 4
    thread_list = []
    n_x_text = len(text_list)
    for i in range(num_thread):
        thread_list.append(Thread(target=do_concurrent_tagging, args=(
        int(i * n_x_text / num_thread), int((i + 1) * n_x_text / num_thread), text_list, counter)))

    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()

    print(counter.most_common(10))  # print most common tokens
    tokens = [token for token, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_token(PAD)
    vocab.add_token(START_TOKEN)
    vocab.add_token(END_TOKEN)
    vocab.add_token(UNK)
    vocab.add_token(CLS)
    vocab.add_token(MASK)
    vocab.add_token(SEP)

    for i, token in enumerate(tokens):
        vocab.add_token(str(token))

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    return vocab


def main():
    # print("vocab_tokenizer")
    #
    # corpus = """
    #     초기 인공지능 연구에 대한 대표적인 정의는 다트머스 회의에서 존 매카시가 제안한 것으로 "기계를 인간 행동의 지식에서와 같이 행동하게 만드는 것"이다.
    #     그러나 이 정의는 범용인공지능(AGI, 강한 인공지능)에 대한 고려를 하지 못한 것 같다.
    #     인공지능의 또다른 정의는 인공적인 장치들이 가지는 지능이다.
    #     대부분 정의들이 인간처럼 사고하는 시스템, 인간처럼 행동하는 시스템, 이성적으로 사고하는 시스템 그리고 이성적으로 행동하는 시스템이라는 4개의 분류로 분류된다.
    #     강한 인공지능은 어떤 문제를 실제로 사고하고 해결할 수 있는 컴퓨터 기반의 인공적인 지능을 만들어 내는 것에 관한 운터운터떤 면에서 보면 지능적인 행동을 보일 것이다.
    #     오늘날 이 분야의 연구는 주로 미리 정의된 규칙의 모음을 이용해서 지능을 흉내내는 컴퓨터 프로그램을 개발하는 것에 맞추어져 있다.
    #     강한 인공지능 분야의 발전은 무척이나 미약했지만, 목표를 무엇에 두느냐에 따라 약한 인공지능 분야에서는 꽤 많은 발전이 이루어졌다고 볼 수 있다."""
    # text_list = corpus.split('\n')
    #
    # vocab = Vocabulary()
    # vocab.build_vocab(text_list=text_list)
    # tokenizer = Tokenizer(vocab=vocab, split_fn=Mecab().pos, pad_fn=keras_pad_fn, maxlen=15)
    # print(tokenizer.list_of_string_to_arr_of_pad_token_ids([corpus]))

    print(mecab_token_pos_sep_fn("안녕하세요"))

if __name__ == '__main__':
    main()