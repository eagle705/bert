from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

class SegmentEmbedding(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim):
        super(SegmentEmbedding, self).__init__(name='SegmentEmbedding')
        self.seg_embed = tf.keras.layers.Embedding(input_dim=3, output_dim=embed_dim) # 0: padding, 1: Segment_A, 2: Segment_B

    def call(self, x):
        seg_embed = self.seg_embed(x)
        return seg_embed

def main():
    print("SegmentEmbedding")

if __name__ == '__main__':
    main()