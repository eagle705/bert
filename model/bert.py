from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
from pprint import pprint
import collections
import re
from model.transformer import TransformerBlock
from model.embedding.bert import BERTEmbedding

class BERTConfig():
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size #len(vocab_idx2word)
        self.maxlen = 100
        self.embed_dim = 100
        self.head_num = 5
        self.split_embed_dim = 20
        self.layer_num = 3
        self.feed_forward_dim = 100
        self.lr = 1e-3

class BERT(tf.keras.Model):
    def __init__(self, config):
        super(BERT, self).__init__(name='BERT')
        self.config = config
        self.transformer_encoder = TransformerBlock(config)
        self.embedding = BERTEmbedding(config['vocab_size'], config['embed_dim'])
        self.layer_num = config['layer_num']

    def call(self, x_token, pad_seg_embed_token, training, enc_padding_mask):
        x = self.embedding(x_token, pad_seg_embed_token=pad_seg_embed_token)
        self.attention_weights = {}
        for i in range(self.layer_num): # ToDo: 재사용이 아니라 새로 생성하는 방법으로 바꿔야함!
            # 이상하게 attention_weight를 저장하면 학습이 오래 걸림
            x, attention_weight = self.transformer_encoder(x, training, enc_padding_mask)
            # self.attention_weights['encoder_layer{}_block1'.format(i + 1)] = attention_weight

        self.encoder_final_output = x

        return self.encoder_final_output, attention_weight #self.attention_weights

    def get_sequence_output(self):
        return self.encoder_final_output

    def get_assignment_map_from_checkpoint(self, tvars, init_checkpoint):
        """Compute the union of the current variables and checkpoint variables.
            ref: https://github.com/google-research/bert/blob/master/modeling.py
            assignment_map: Dict, where keys are names of the variables in the
            checkpoint and values are current variables or names of current variables
            (in default graph)
        """
        assignment_map = {}
        initialized_variable_names = {}

        name_to_variable = collections.OrderedDict()
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var

        init_vars = tf.train.list_variables(init_checkpoint)

        assignment_map = collections.OrderedDict()
        for x in init_vars:
            (name, var) = (x[0], x[1])
            if "BERTLM/BERT" in name:
                new_scope_name = name.replace("BERTLM", "BERTWrapper")
                if new_scope_name not in name_to_variable:
                    print("out of model scope :", new_scope_name)
                    continue
            else:
                continue

            assignment_map[name] = name_to_variable[new_scope_name]
            initialized_variable_names[new_scope_name] = 1
            initialized_variable_names[new_scope_name + ":0"] = 1

        return (assignment_map, initialized_variable_names)

def main():
    print("BERT")
    print(tf.__version__)
    config = {}
    config['vocab_size'] = 780
    config['maxlen'] = 100
    config['embed_dim'] = 100
    config['head_num'] = 5
    config['split_embed_dim'] = 20
    config['layer_num'] = 3
    config['feed_forward_dim'] = 100
    lr = 1e-3
    bert = BERT(config=config)






    import sentencepiece as spm
    # https://github.com/google/sentencepiece#redefine-special-meta-tokens
    # https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
    # https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb

    # # CLS 토큰쪽이 0,1 뱉는데 이게 Padding, BOS랑 헷갈릴수있음.. 체크해야함
    # templates = '--input={} --model_prefix={} --vocab_size={} --user_defined_symbols=[MASK],[CLS],[SEP],[SEG_A],[SEG_B] --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3'# --max_sentence_length={}'
    # vocab_size = 780
    prefix = 'm'#'2016-10-20-news'
    # input_file = '../data_in/sentencepiece_train.txt'
    #
    # cmd = templates.format(input_file, prefix, vocab_size)
    #
    # # Train model
    # spm.SentencePieceTrainer.Train(cmd)

    # Load model
    sp = spm.SentencePieceProcessor()
    sp.Load('{}.model'.format(prefix))

    training_corpus = """
    초기 인공지능 연구에 대한 대표적인 정의는 다트머스 회의에서 존 매카시가 제안한 것으로 "기계를 인간 행동의 지식에서와 같이 행동하게 만드는 것"이다. 
    그러나 이 정의는 범용인공지능(AGI, 강한 인공지능)에 대한 고려를 하지 못한 것 같다. 
    인공지능의 또다른 정의는 인공적인 장치들이 가지는 지능이다. 
    대부분 정의들이 인간처럼 사고하는 시스템, 인간처럼 행동하는 시스템, 이성적으로 사고하는 시스템 그리고 이성적으로 행동하는 시스템이라는 4개의 분류로 분류된다.
    강한 인공지능은 어떤 문제를 실제로 사고하고 해결할 수 있는 컴퓨터 기반의 인공적인 지능을 만들어 내는 것에 관한 운터운터떤 면에서 보면 지능적인 행동을 보일 것이다.
    오늘날 이 분야의 연구는 주로 미리 정의된 규칙의 모음을 이용해서 지능을 흉내내는 컴퓨터 프로그램을 개발하는 것에 맞추어져 있다.
    강한 인공지능 분야의 발전은 무척이나 미약했지만, 목표를 무엇에 두느냐에 따라 약한 인공지능 분야에서는 꽤 많은 발전이 이루어졌다고 볼 수 있다.
    상당수 인공지능 연구의 (본래) 목적은 심리학에 대한 실험적인 접근이었고, 언어 지능(linguistic intelligence)이 무엇인지를 밝혀내는 것이 주목표였다(튜링 테스트가 대표적인 예이다).
    언어 지능을 제외한 인공지능에 대한 시도들은 로보틱스와 집합적 지식(?)을 포함한다.
    이들은 환경에 대한 처리, 의사 결정을 일치시키는 것에 중심을 두며 어떻게 지능적 행동이 구성되는 것인가를 찾을 때, 생물학과, 정치과학으로부터 이끌어 낸다.
    사회적 계획성과 인지성의 능력은 떨어지지만 인간과 유사한 유인원을 포함한, 복잡한 인식방법을 가진 동물뿐만 아니라 특히 곤충들(로봇들로 모방하기 쉬운)까지 포함한 동물학으로부터 인공지능 과학은 시작된다.
    여러가지 생명체들의 모든 논리구조를 가져온 다는 것은 이론적으로는 가능하지만 수치화, 기계화 한다는 것은 쉬운 일이 아니다.
    인공지능 학자는 동물들은 인간들보다 모방하기 쉽다고 주장한다.
    그러나 동물의 지능을 만족하는 계산 모델은 없다.
    매컬러가 쓴 신경 행동에서 내재적 사고의 논리적 계산[3], 튜링의 기계와 지능의 계산[4] 그리고 리클라이더의 인간과 컴퓨터의 공생[5]가 기계 지능의 개념에 관한 독창적인 논문들이다.
    """

    training_corpus = training_corpus.replace("\n", '').split('.')[:-1] # 개행문자제거, 문장 분리
    training_corpus = [_.strip() for _ in training_corpus] # 문장 앞 뒤의 불필요한 공백 제거
    print(training_corpus)
    print("training_corpus len: ", len(training_corpus))

    training_ids_batch = []
    for sent in training_corpus:
        encode_piece = sp.EncodeAsPieces(sent)
        training_ids_batch.append(sp.EncodeAsIds(sent))
        print("raw text: ", sent)
        print("enc text: ", encode_piece)
        print("dec text: ", sp.decode_pieces(encode_piece))
        print("enc ids: ", sp.EncodeAsIds(sent))
        print("")

    for i in range(10):
        print(str(i)+": "+sp.IdToPiece(i))

    maxlen = config['maxlen']
    from dataset.dataset import BERTDataset
    bert_dataset = BERTDataset(sp, maxlen)
    masked_training_ids_batch, mask_MLM_position_index_batch, mask_MLM_token_ids_batch = bert_dataset.create_LM_mask(training_ids_batch, sp)


    mask_MLM_label_batch = []

    for i, positions in enumerate(mask_MLM_position_index_batch):
        mask_MLM_label_batch.append([0]*len(masked_training_ids_batch[i]))
        for j, pos in enumerate(positions):
            if maxlen <= pos: # maxlen에 맞게 짜른거 교정
                continue
            mask_MLM_label_batch[i][pos] = mask_MLM_token_ids_batch[i][j]



    nsp_sent_ids_batch, nsp_sparse_label_batch, integrated_mask_LM_position_index_batch, seg_embed_token_batch = bert_dataset.create_nsp_MLM_dataset(masked_training_ids_batch, mask_MLM_label_batch, mask_MLM_position_index_batch, sp)

    pad_nsp_sent_ids_batch = bert_dataset.add_padding(nsp_sent_ids_batch, sp.pad_id(), maxlen=maxlen)
    pad_nsp_sparse_label_batch = bert_dataset.add_padding(nsp_sparse_label_batch, sp.pad_id(), maxlen=maxlen)
    pad_seg_embed_token_batch = bert_dataset.add_padding(seg_embed_token_batch, sp.pad_id(), maxlen=maxlen)

    X_train = np.array(pad_nsp_sent_ids_batch)
    tar_real = np.array(pad_nsp_sparse_label_batch)
    pad_seg_embed_token_batch = np.array(pad_seg_embed_token_batch)
    enc_padding_mask = bert_dataset.create_masks(X_train)

    encoder_output, attention_weights = bert(X_train, pad_seg_embed_token=pad_seg_embed_token_batch,
                                                                training=True, enc_padding_mask=enc_padding_mask)



    # bert.load_weights('../save_model/bert_model')
    print("before, bert.trainable_variables: ", bert.trainable_variables)
    # print("bert.transformer_encoder.trainable_weights: ", bert.transformer_encoder.trainable_variables)
    tvars = bert.trainable_variables
    # print("tvars: ", tvars)

    checkpoint_dir = '../save_model'
    init_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    (assignment_map, initialized_variable_names) = bert.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    # checkpoint = tf.train.Checkpoint(model=bert)
    # ckpt_init = tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map) # tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    # checkpoint.restore(ckpt_init)
    print("assignment_map: ", assignment_map)
    print("initialized_variable_names: ", initialized_variable_names)

    bert.load_weights('../save_model/bert_model', initialized_variable_names)

    # saver = tf.train.Saver()
    # saver.restore(ckpt_init)
    print("after, bert.trainable_variables: ", bert.trainable_variables)


if __name__ == '__main__':
    main()