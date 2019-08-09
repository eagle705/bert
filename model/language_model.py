from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
from pprint import pprint
from model.bert import BERT

class BERTLM(tf.keras.Model):
    def __init__(self, bert: BERT, config):
        super(BERTLM, self).__init__(name='BERTLM')
        self.bert = bert
        self.next_sentence_prediction = NextSentencePrediction()
        self.masked_language_model = MaskedLanguageModel(config['vocab_size'])


    def call(self, input, pad_seg_embed_token, training, enc_padding_mask):
        self.encoder_output, self.attention_weight_in_encoder = self.bert(input, pad_seg_embed_token, training, enc_padding_mask)

        NSP_prob = self.next_sentence_prediction(self.encoder_output)
        MLM_prob = self.masked_language_model(self.encoder_output)

        return NSP_prob, MLM_prob, self.attention_weight_in_encoder


class NextSentencePrediction(tf.keras.Model):
    def __init__(self):
        super(NextSentencePrediction, self).__init__(name='NextSentencePrediction')
        self.NSP_linear = tf.keras.layers.Dense(2)
        self.NSP_softmax = tf.keras.layers.Softmax()

    def call(self, x):
        return self.NSP_softmax(self.NSP_linear(x[:, 0])) # index 0 : CLS token

class MaskedLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size):
        super(MaskedLanguageModel, self).__init__(name='MaskedLanguageModel')
        self.MLM_linear = tf.keras.layers.Dense(vocab_size)
        self.MLM_softmax = tf.keras.layers.Softmax()

    def call(self, x):
        return self.MLM_softmax(self.MLM_linear(x[:, 1:])) # index > 0 : id token



def main():
    print("BERTLM")


if __name__ == '__main__':
    main()