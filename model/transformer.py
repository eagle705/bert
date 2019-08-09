from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
from pprint import pprint
from model.ops import MultiHeadAttention
from model.ops import gelu

class TransformerBlock(tf.keras.Model):

    def __init__(self, config):
        super(TransformerBlock, self).__init__(name='TransformerBlock')

        self.config = config
        self.embed_dim = self.config['embed_dim'] # d_model
        self.head_num = self.config['head_num'] # h # split_embed_dim * head_num == embed_dim
        self.split_embed_dim = self.config['split_embed_dim'] # dim_k, dim_v # self-attention에는 context vector에 쓰였던 context vector를 위한 attention dim 개념이 없음, 자기 차원끼리 attention을 구하니까 attention을 위한 벡터가 따로 필요없거든
        self.feed_forward_dim = config['feed_forward_dim'] # dim_ffc

        # Multi Head Attention
        self.mha = MultiHeadAttention(self.config)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="last_layer")
        self.position_wise_fc_1 = tf.keras.layers.Dense(units=self.feed_forward_dim, activation=gelu) #'relu'
        self.position_wise_fc_2 = tf.keras.layers.Dense(units=self.embed_dim)

    def position_wise_fc(self, vector):
        out = self.position_wise_fc_1(vector) # (batch, seq, dim_ffc)
        out = self.position_wise_fc_2(out) # (batch, seq, model_dim)
        return out


    def sub_layer(self, x, training=False, padding_mask=None):
        out_1, attention_weight = self.mha(x, K = x, V = x, mask=padding_mask) # 첫번째 인자 Q를 input으로 인식함
        out_1 = self.dropout1(out_1, training=training)
        out_2 = self.layer_norm_1(out_1 + x)
        out_3 = self.position_wise_fc(out_2)
        out_3 = self.dropout2(out_3, training=training)
        out_4 = self.layer_norm_2(out_2 + out_3)

        return out_4, attention_weight


    def call(self, x_embed, training=False, mask=None):
        x_embed, attention_weight = self.sub_layer(x_embed, training, mask)
        return x_embed, attention_weight

