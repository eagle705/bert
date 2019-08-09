from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

class TokenEmbedding(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim):
        super(TokenEmbedding, self).__init__(name='TokenEmbedding')
        self.embed = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_shape=(None,))

    def call(self, x):
        x_embed = self.embed(x)
        return x_embed

def main():
    print("TokenEmbedding")

if __name__ == '__main__':
    main()