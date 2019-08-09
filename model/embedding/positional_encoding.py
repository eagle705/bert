from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.Model):
    """
    reference: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/text/transformer.ipynb#scrollTo=1Rz82wEs5biZ

    """
    def __init__(self, max_len_position, embed_dim):
        super(PositionalEncoding, self).__init__(name='PositionalEncoding')
        self.pos_encoding = self.positional_encoding(max_len_position, embed_dim)

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


    def call(self, x):
        seq_len = tf.shape(x)[1]
        return self.pos_encoding[:, :seq_len, :]


def main():
    print("PositionalEncoding")


if __name__ == '__main__':
    main()