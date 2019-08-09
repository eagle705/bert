from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from dataset.dataset import BERTDataset
from dataset.vocab_tokenizer import build_vocab, Tokenizer, Vocabulary, keras_pad_fn
from model.bert import BERT
from model.language_model import BERTLM
import numpy as np
from tqdm import tqdm

import random
import sys
from dataset.vocab_tokenizer import SentencePieceTokenizer, Tokenizer
import sentencepiece as spm
# https://github.com/google/sentencepiece#redefine-special-meta-tokens
# https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
# https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb

np.set_printoptions(suppress=False)
np.set_printoptions(threshold=sys.maxsize)

def decode_word_ids(word_ids_batch, vocab_idx2word):
    word_token_batch = []
    for word_ids in word_ids_batch:
        word_token = [vocab_idx2word[word_id] if word_id in vocab_idx2word else "<unk>" for word_id in word_ids]
        word_token_batch.append(word_token)
    return word_token_batch

def main():

    # CLS 토큰쪽이 0,1 뱉는데 이게 Padding, BOS랑 헷갈릴수있음.. 체크해야함
    templates = '--input={} --model_prefix={} --vocab_size={} --user_defined_symbols=[MASK],[CLS],[SEP],[SEG_A],[SEG_B] --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3'# --max_sentence_length={}'
    vocab_size = 780
    prefix = 'm'#'2016-10-20-news'
    input_file = './data_in/sentencepiece_train.txt'
    cmd = templates.format(input_file, prefix, vocab_size)

    # Train model
    spm.SentencePieceTrainer.Train(cmd)

    # Load model
    sp = spm.SentencePieceProcessor()
    sp.Load('{}.model'.format(prefix))

    sp_tokenizer = SentencePieceTokenizer()
    sp_tokenizer.load()

    training_corpus_batch = ["""
    그러나 이 정의는 범용인공지능(AGI, 강한 인공지능)에 대한 고려를 하지 못한 것 같다.
    인공지능의 또다른 정의는 인공적인 장치들이 가지는 지능이다. """,
    """
    인공지능 학자는 동물들은 인간들보다 모방하기 쉽다고 주장한다.
    그러나 동물의 지능을 만족하는 계산 모델은 없다.
    """]

    training_ids_corpus_batch = []
    for training_corpus in tqdm(training_corpus_batch):
        training_corpus = training_corpus.replace("\n", '').split('.')[:-1]  # 개행문자제거, 문장 분리
        training_corpus = [_.strip() for _ in training_corpus if _ != '']  # 문장 앞 뒤의 불필요한 공백 제거

        training_ids_corpus = []
        for sent in training_corpus:
            encode_piece = sp_tokenizer.encode_as_pieces(sent)
            training_ids_corpus.append(sp_tokenizer.encode_as_ids(sent))
            print("raw text: ", sent)
            print("enc text: ", encode_piece)
            print("dec text: ", sp_tokenizer.decode_pieces(encode_piece))
            print("enc ids: ", sp_tokenizer.encode_as_ids(sent))
            print("")
        training_ids_corpus_batch.append(training_ids_corpus)

    for i in range(10):
        print(str(i)+": "+sp_tokenizer.id_to_piece(i))

    def decode_batch(ids_batch, tokenizer):
        decode_result = []
        # for ids in ids_batch:
        #     _ids = [int(i) for i in ids]
        #     decode_result.append(tokenizer.decode_token_ids(_ids))
        decode_result.append(tokenizer.decode_token_ids(ids_batch))
        return decode_result


    # Data params
    maxlen = 100
    vocab_word2idx = {sp_tokenizer.id_to_piece(id): id for id in range(sp_tokenizer.get_piece_size())}
    vocab_idx2word = {id: sp_tokenizer.id_to_piece(id) for id in range(sp_tokenizer.get_piece_size())}
    print("vocab_word2idx: ", vocab_word2idx)

    # define vocab
    vocab = Vocabulary(vocab_word2idx)
    tokenizer = Tokenizer(vocab, sp_tokenizer.encode_as_ids, keras_pad_fn, maxlen)

    # Model Params
    bert_dataset = BERTDataset(vocab_word2idx, maxlen)
    config = {}
    config['vocab_size'] = len(bert_dataset.vocab_word2idx)
    config['maxlen'] = maxlen
    config['embed_dim'] = 100
    config['head_num'] = 5
    config['split_embed_dim'] = 20
    config['layer_num'] = 3
    config['feed_forward_dim'] = 100
    config['log_steps'] = 200

    # Train Params
    EPOCHS = 5000
    BATCH_SIZE = 30
    lr = 1e-3

    list_of_pad_nsp_sent_ids_batch = []
    list_of_pad_nsp_sparse_label_batch = []
    list_of_pad_seg_embed_token_batch = []
    for training_ids_batch in training_ids_corpus_batch:
        # preprocessing MLM
        masked_training_ids_batch, mask_MLM_position_index_batch, mask_MLM_token_ids_batch = bert_dataset.create_LM_mask(training_ids_batch, sp_tokenizer)
        print("training_ids_batch:", training_ids_batch)
        print("masked_training_ids_batch:",masked_training_ids_batch)
        print("mask_MLM_position_index_batch:", mask_MLM_position_index_batch)
        print("mask_MLM_token_ids_batch:", mask_MLM_token_ids_batch)

        mask_MLM_label_batch = []
        for i, positions in enumerate(mask_MLM_position_index_batch):
            mask_MLM_label_batch.append([0] * len(masked_training_ids_batch[i]))
            for j, pos in enumerate(positions):
                # print("pos: ", pos)
                # print("masked_training_ids_batch[i]: ", masked_training_ids_batch[i])
                if maxlen <= pos or len(masked_training_ids_batch[i]) <= pos:  # maxlen에 맞게 짜른거 교정
                    continue
                mask_MLM_label_batch[i][pos] = mask_MLM_token_ids_batch[i][j]



        # preprocessing NSP
        nsp_sent_ids_batch, nsp_sparse_label_batch, integrated_mask_LM_position_index_batch, seg_embed_token_batch = bert_dataset.create_nsp_MLM_dataset(masked_training_ids_batch,  mask_MLM_label_batch, mask_MLM_position_index_batch, sp_tokenizer)

        pad_nsp_sent_ids_batch = bert_dataset.add_padding(nsp_sent_ids_batch, sp_tokenizer.pad_id(), maxlen=maxlen)
        pad_nsp_sparse_label_batch = bert_dataset.add_padding(nsp_sparse_label_batch, sp_tokenizer.pad_id(), maxlen=maxlen)
        pad_seg_embed_token_batch = bert_dataset.add_padding(seg_embed_token_batch, sp_tokenizer.pad_id(), maxlen=maxlen)

        list_of_pad_nsp_sent_ids_batch.append(pad_nsp_sent_ids_batch)
        list_of_pad_nsp_sparse_label_batch.append(pad_nsp_sparse_label_batch)
        list_of_pad_seg_embed_token_batch.append(pad_seg_embed_token_batch)

    # concat
    pad_nsp_sent_ids_batch = np.concatenate((list_of_pad_nsp_sent_ids_batch))
    pad_nsp_sparse_label_batch = np.concatenate((list_of_pad_nsp_sparse_label_batch))
    pad_seg_embed_token_batch = np.concatenate((list_of_pad_seg_embed_token_batch))

    # data
    X_train = np.array(pad_nsp_sent_ids_batch)
    tar_real = np.array(pad_nsp_sparse_label_batch)
    pad_seg_embed_token_batch = np.array(pad_seg_embed_token_batch)

    # model
    bert = BERT(config = config)
    model = BERTLM(bert, config)
    MLM_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')  # input label == index of class (index, logits)
    NSP_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    # optim
    optimizer = tf.compat.v2.keras.optimizers.Adam(lr)

    # loss
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_MLM_loss = tf.keras.metrics.Mean(name='train_MLM_loss')
    train_NSP_loss = tf.keras.metrics.Mean(name='train_NSP_loss')

    # metric
    train_MLM_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_MLM_accuracy')
    train_NSP_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_NSP_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    def loss_function(real, pred, loss_type="MLM"):

        if loss_type == "MLM":
            loss_ = MLM_loss_object(real, pred)
            mask = tf.math.logical_not(tf.math.equal(real, 0)) # padding 아닌건 1
            # print("LM mask[0]:", mask[0])
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask # 패딩이 아닌 1인 값은 살리고, 패딩인 값인 0인 값은 없앰
        else:
            loss_ = NSP_loss_object(real, pred)

        return tf.reduce_mean(loss_)


    # 세션 대신 tf.function() decorator로 파이썬 함수를 감싸면, 이 함수를 하나의 그래프로 실행하기 위해 JIT 컴파일함
    # tf.function()을 쓰면 eager mode -> graph mode 되는 것임
    # @tf.function
    def train_step(enc_input, tar_real, MLM_position_index, pad_seg_embed_token):
        # tar_inp = label[:, :-1] # remove </s>
        # tar_real = label[:, 1:] # remove <s>
        enc_padding_mask = bert_dataset.create_masks(enc_input)

        with tf.GradientTape() as tape:
            NSP_predictions, MLM_predictions, attention_weights = model(enc_input, pad_seg_embed_token=pad_seg_embed_token, training=True, enc_padding_mask=enc_padding_mask)
            # MLM_loss = loss_function(tar_real[:,1:][MLM_position_index], MLM_predictions[MLM_position_index], loss_type="MLM") # 이렇게 뽑아서 Loss 계산해도 되지만.. 그냥 마스킹하는게 편함
            MLM_loss = loss_function(tar_real[:, 1:], MLM_predictions, loss_type="MLM")  # masking losses for padding, 나머지 ids은 어떻하지?
            NSP_loss = loss_function(tar_real[:, 0], NSP_predictions, loss_type="NSP")
            loss = MLM_loss + NSP_loss
            predicted_id = tf.cast(tf.argmax(MLM_predictions, axis=2), tf.int32) # 단어 사전에서 가장 확률 큰 단어 argmax로 꺼내기

            print("X_train: ", decode_batch(enc_input.numpy(), tokenizer))
            print("MLM_tar_real_all: ", decode_batch([tar_real[i, 1:].numpy() for i, MLM_position_index_item in enumerate(MLM_position_index)], tokenizer))
            print("MLM_tar_real: ", decode_batch([tar_real[i, 1:][MLM_position_index_item].numpy() for i, MLM_position_index_item in enumerate(MLM_position_index)], tokenizer))
            print("MLM_pred: ", decode_batch([predicted_id[i][MLM_position_index_item].numpy() for i, MLM_position_index_item in enumerate(MLM_position_index)], tokenizer))
            print("NSP_tar_real: ", tar_real[:, 0].numpy())
            print("NSP_pred: ", np.argmax(NSP_predictions, axis=-1))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_MLM_loss(MLM_loss)
        train_NSP_loss(NSP_loss)

        # MLM_position_index = np.array(MLM_position_index)
        MLM_real = tar_real[:,1:][MLM_position_index]
        MLM_pred = MLM_predictions[MLM_position_index]

        train_MLM_accuracy(MLM_real, MLM_pred) # argmax, logits
        train_NSP_accuracy(tar_real[:, 0], NSP_predictions)

    # @tf.function
    # def test_step(Y_test, label):
    #     predictions = model(Y_test)
    #     t_loss = loss_object(label, predictions)
    #
    #     test_loss(t_loss)
    #     test_accuracy(label, predictions)



    train_ds = tf.data.Dataset.from_tensor_slices((X_train, tar_real, pad_seg_embed_token_batch))
    train_ds = train_ds.repeat(EPOCHS).shuffle(1024).batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)


    for step, (X_train_batch, tar_real, pad_seg_embed_token_batch) in enumerate(train_ds):
        LM_position_index = np.not_equal(tar_real[:,1:], np.zeros_like(tar_real[:,1:]))
        train_step(X_train_batch, tar_real, LM_position_index, pad_seg_embed_token_batch)

        template = 'Step {}, Total Loss: {}, MLM Loss: {}, NSP Loss: {}, NSP_Accuracy: {}, MLM_Accuracy: {}'
        print(template.format(step + 1,
                              train_loss.result(),
                              train_MLM_loss.result(),
                              train_NSP_loss.result(),
                              train_NSP_accuracy.result() * 100,
                              train_MLM_accuracy.result() * 100))

        # template = 'Step {}, Loss: {}, NSP_Accuracy: {}, MLM_Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        # print(template.format(step + 1,
        #                       train_loss.result(),
        #                       train_NSP_accuracy.result() * 100,
        #                       train_MLM_accuracy.result() * 100,
        #                       test_loss.result(),
        #                       test_accuracy.result() * 100))

        # import tempfile
        if step == 20:
            model.save_weights('./save_model/bert_model')

            # checkpoint_dir = './save_model/bert_model'#tempfile.mkdtemp()
            # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            #
            # root = tf.train.Checkpoint(model=model)
            #
            # print("save!")
            # root.save(checkpoint_prefix)
            # print("restore!")
            # root.restore(tf.train.latest_checkpoint(checkpoint_dir))


if __name__ == '__main__':
    print(tf.__version__)
    main()