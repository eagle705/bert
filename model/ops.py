from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
from pprint import pprint

def gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  # ref: https://github.com/google-research/bert/blob/master/modeling.py#L264

  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

class MultiHeadAttention(tf.keras.Model): # tf.keras.layers.Layer vs tf.keras.Model

    def __init__(self, config):
        super(MultiHeadAttention, self).__init__(name='MultiHeadAttention')

        self.embed_dim = config['embed_dim'] # 100  # model_dim
        self.head_num = config['head_num'] # 4  # h # split_embed_dim * head_num == embed_dim
        self.split_embed_dim = config['split_embed_dim'] # 25 # dim_k, dim_v # self-attention에는 attention dim 개념이 없음, 자기 차원끼리 attention을 구하니까 attention을 위한 벡터가 따로 필요없거든

        # Multi Head Attention
        self.Wq = tf.keras.layers.Dense(units=self.embed_dim, activation=gelu) #'relu'
        self.Wk = tf.keras.layers.Dense(units=self.embed_dim, activation=gelu)
        self.Wv = tf.keras.layers.Dense(units=self.embed_dim, activation=gelu)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        matmul_qk = tf.matmul(Q, K, transpose_b=True)  # (batch, head_num, seq, split_embed_dim) * (batch, head_num, split_embed_dim, seq) = (batch, head_num, seq, seq) # 곱하면 가장 마지막 차원 끝에서 두개가 계산 되는 듯
        dk = tf.cast(tf.shape(K)[-1], tf.float32) # dk dim
        scaled_dot_product_qk = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            minus_infinity = -1e9
            scaled_dot_product_qk += mask * minus_infinity  # broadcasting, masking에서 seq은 마지막자리
            # mask와 scaled_dot_product_qk의 차원은 다르지만, 마지막 차원이 같기 때문에 broadcasting이 가능함

        attention_weight = tf.nn.softmax(scaled_dot_product_qk, axis=-1)
        scaled_attention_output = tf.matmul(attention_weight, V) # (batch, head_num, seq, seq) * (batch, head_num, seq, split_embed_dim) = (batch, head_num, seq, split_embed_dim)

        return scaled_attention_output, attention_weight


    def split_head(self, vector):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        batch_size = tf.shape(vector)[0]

        # (batch, seq, embed_dim) -> (batch, seq, head_num, split_embed_dim)
        x = tf.reshape(vector, (batch_size, -1, self.head_num, self.split_embed_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # (batch, head_num, seq, split_embed_dim)


    def call(self, Q, K, V, mask=None):
        # Query, Key 꺼낼 필요 없이 3개 복사해서 쓰면 됨
        # 쪼갠다음에 weight 선언 후 매트릭스 곱? -> 쪼갠 다음에 Dense -> 쪼개면 for loop 때문에 병렬처리 안되잖아 -> 다 계산후에 쪼개자 -> 쪼개지말고 reshape으로 하면 더 깔끔하다
        multi_head_Q = self.split_head(self.Wq(Q))
        multi_head_K = self.split_head(self.Wk(K))
        multi_head_V = self.split_head(self.Wv(V))

        self.scaled_attention_output, self.attention_weight = self.scaled_dot_product_attention(multi_head_Q, multi_head_K, multi_head_V, mask)

        # (batch, head_num, seq, split_embed_dim) -> (batch, seq, split_embed_dim)
        self.concat_scaled_attention = tf.reshape(self.scaled_attention_output, (tf.shape(Q)[0], -1, self.embed_dim))

        return self.concat_scaled_attention, self.attention_weight