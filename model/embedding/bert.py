from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np


class BERTEmbedding(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim):
        super(BERTEmbedding, self).__init__(name='BERTEmbedding')
        self.embed_dim = embed_dim
        self.token_embed = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_shape=(None,))
        self.seg_embed = tf.keras.layers.Embedding(input_dim=3,
                                                   output_dim=embed_dim)  # 0: padding, 1: Segment_A, 2: Segment_B

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, max_len_position, embed_dim):
        angle_rads = self.get_angles(np.arange(max_len_position)[:, np.newaxis],
                                     np.arange(embed_dim)[np.newaxis, :], embed_dim)

        # apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, pad_seg_embed_token):
        seq_len = tf.shape(x)[1]
        x_embed = self.token_embed(x)
        # x_embed *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32)) # legacy of transformer
        x_embed += self.positional_encoding(max_len_position=512, embed_dim=self.embed_dim)[:, :seq_len, :]
        x_embed += self.seg_embed(pad_seg_embed_token)  # add segment embedding

        return x_embed


def main():
    print("BERTEmbedding")


if __name__ == '__main__':
    main()