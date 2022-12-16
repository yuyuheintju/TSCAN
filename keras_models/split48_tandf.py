from __future__ import absolute_import
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.models import Model
# from keras.layers import Input, Dense, LSTM, Masking, Dropout
# from keras.layers.wrappers import Bidirectional, TimeDistributed
from mimic3models.keras_utils import LastTimestep
from mimic3models.keras_utils import ExtendMask
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.keras import layers


def masking(sequence, task='Padding'):
    """
    :param sequence: 输入tensor
    :param task: 分为"Padding"和"Sequence"(Look-ahead),默认为Padding
    :return:
    """

    if task == "Padding":
        zeroT = tf.cast(tf.math.equal(sequence, 0), tf.float32)  # 元素为0的位置标记为1，其余位置标记为0
        return zeroT[:, tf.newaxis, tf.newaxis, :]  # 构造四个维度，为了应该用时映射[batch_size, num_head, seq_len, seq_len]

    elif task == "Sequence":
        size = sequence.shape[1]
        triMatrix = tf.linalg.band_part(tf.ones((size, size)), -1, 0)  # 只保留对角线及下三角阵内容
        return 1 - triMatrix  # 上三角阵（不包括对角线）内容为1，代表Attention时候看不到的内容

    else:
        raise ValueError("任务名称只能是“Padding”或“Sequence”")


def make_masking(inp, tar):
    enc_padding_mask = masking(inp, task='Padding')
    dec_padding_mask = masking(inp, task='Padding')

    look_ahead_mask = masking(tar, task='Sequence')
    dec_tar_padding_mask = masking(tar, task='Padding')

    combined_mask = tf.maximum(look_ahead_mask, dec_tar_padding_mask)  # 取对应元素较大值。numpy的broadcast原理

    # print(combined_mask.shape)
    # print(dec_padding_mask.shape)

    return enc_padding_mask, combined_mask, dec_padding_mask


def get_angles(pos, i, d_model):
    # 这里的i等价与上面公式中的2i和2i+1
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # 第2i项使用sin
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1项使用cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


'''多头Attention'''


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, out_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(out_dim)

    def attention(self, query, key, value, mask):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # 掩码
        if mask is not None:
            scaled_score += (mask * -1e9)

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(q)[0]
        query = self.query_dense(q)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(k)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(v)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value, mask)

        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, out_dim)
        return output, weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate, batch_size):
        super(EncoderLayer, self).__init__()
        self.att = MultiHeadAttention(embed_dim, out_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(out_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)
        self.dropout3 = layers.Dropout(drop_rate)

        self.out_dim = out_dim

    def call(self, inputs):
        attn_output, _ = self.att(inputs, inputs, inputs, mask=None)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout3(ffn_output)
        output = self.layernorm3(out1 + ffn_output)
        return output


'''Transformer的Encoder部分'''


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate, batch_size):
        super(DecoderLayer, self).__init__()
        self.sel_att = MultiHeadAttention(embed_dim, out_dim, num_heads)
        self.att = MultiHeadAttention(embed_dim, out_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(out_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)
        self.out_dim = out_dim

    def call(self, inputs, en_outputs, look_ahead_mask):
        sel_output, _ = self.sel_att(inputs, inputs, inputs, look_ahead_mask)
        sel_output = self.dropout1(sel_output)
        out1 = self.layernorm1(inputs + sel_output)

        att_output, weights = self.att(out1, en_outputs, en_outputs, look_ahead_mask)
        att_output = self.dropout1(att_output)
        out2 = self.layernorm1(out1 + att_output)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output)
        output = self.layernorm2(out2 + ffn_output)

        return output, weights


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, n_layers, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate, batch_size):
        super(TransformerEncoderBlock, self).__init__()

        self.time_dim = time_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.pos_embedding = positional_encoding(time_dim, out_dim)
        self.encode_layer = [EncoderLayer(time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate, batch_size)
                             for _ in range(n_layers)]

    def call(self, inputs):
        # weight =  attn_output/inputs
        # weight = tf.reduce_mean(weight, axis=2, keep_dims=False)

        x = inputs + self.pos_embedding[:, :self.time_dim, :]
        for i in range(self.n_layers):
            x = self.encode_layer[i](x)
        return x


class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, n_layers, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate, batch_size):
        super(TransformerDecoderBlock, self).__init__()
        self.time_dim = time_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.pos_embedding = positional_encoding(time_dim, out_dim)
        self.decoder_layer = [DecoderLayer(time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate, batch_size)
                             for _ in range(n_layers)]

    def call(self, inputs, encoder_out, look_ahead_mask):
        # inputs = inputs[:, :, 0:self.out_dim]

        h = inputs + self.pos_embedding[:, :self.time_dim, :]
        for i in range(self.n_layers):
            h, weights = self.decoder_layer[i](h, encoder_out, look_ahead_mask)
        return h, weights


# class Transformer(tf.keras.layers.Layer):
#     def __init__(self, n_layers_en, n_layers_de, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
#                  batch_size):
#         super(Transformer, self).__init__()
#         self.time_dim = time_dim
#         self.out_dim = out_dim
#         self.embed_dim = embed_dim
#
#         self.encoder = TransformerEncoderBlock(n_layers_en, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
#                                                batch_size)
#         self.decoder = TransformerDecoderBlock(n_layers_de, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
#                                                batch_size)
#
#     def call(self, inputs, look_ahead_mask):
#         encoder_in = inputs[:, 0:12, 0:self.out_dim]
#         encoder_out = self.encoder(encoder_in)
#
#         decoder_in = inputs[:, 12:24, 0:self.out_dim]
#         decode_out = self.decoder(decoder_in, encoder_out, look_ahead_mask)
#         return decode_out

class MultiDecoder_f(tf.keras.layers.Layer):
    '''
    注意力放在feature上的一层transformer
    '''

    def __init__(self, n_layers_en, n_layers_de, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
                 batch_size):
        super(MultiDecoder_f, self).__init__()
        self.time_dim = time_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim

        self.encoderf = TransformerEncoderBlock(n_layers_en, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
                                               batch_size)
        self.decoderf1 = TransformerDecoderBlock(n_layers_de, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
                                               batch_size)
        self.decoderf2 = TransformerDecoderBlock(n_layers_de, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
                                               batch_size)
        self.decoderf3 = TransformerDecoderBlock(n_layers_de, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
                                               batch_size)

    def call(self, inputs, look_ahead_mask):

        inputs = tf.transpose(inputs, perm=[0, 2, 1])  # [B, outdim, timedim]
        encoder_in = inputs[:, 0:self.out_dim, 0:12]
        encoder_out = self.encoderf(encoder_in)
        # 第一层decoder
        decoder_in1 = inputs[:, 0:self.out_dim, 12:24]
        decode_out1, _ = self.decoderf1(decoder_in1, encoder_out, look_ahead_mask)
        # 第2层decoder
        last_in = decode_out1
        decoder_in2 = inputs[:, 0:self.out_dim, 24:36]
        decode_out2, _ = self.decoderf2(decoder_in2, last_in, look_ahead_mask)
        # 第3层decoder
        last_in = decode_out2
        decoder_in3 = inputs[:, 0:self.out_dim, 36:48]
        decode_out3, _ = self.decoderf3(decoder_in3, last_in, look_ahead_mask)

        decoder_outf = tf.transpose(decoder_out3, perm=[0, 2, 1])

        return decoder_out


class MultiDecoder(tf.keras.layers.Layer):
    def __init__(self, n_layers_en, n_layers_de, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
                 batch_size):
        super(MultiDecoder, self).__init__()
        self.time_dim = time_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size

        self.encoder = TransformerEncoderBlock(n_layers_en, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
                                               batch_size)
        self.decoder1 = TransformerDecoderBlock(n_layers_de, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
                                                batch_size)
        self.decoder2 = TransformerDecoderBlock(n_layers_de, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
                                                batch_size)
        self.decoder3 = TransformerDecoderBlock(n_layers_de, time_dim, embed_dim, out_dim, num_heads, ff_dim, drop_rate,
                                                batch_size)

        self.encoderf = TransformerEncoderBlock(n_layers_en, out_dim, embed_dim, 12, num_heads, ff_dim, drop_rate,
                                               batch_size)
        self.decoderf1 = TransformerDecoderBlock(n_layers_de, out_dim, embed_dim, 12, num_heads, ff_dim, drop_rate,
                                               batch_size)
        self.decoderf2 = TransformerDecoderBlock(n_layers_de, out_dim, embed_dim, 12, num_heads, ff_dim, drop_rate,
                                               batch_size)
        self.decoderf3 = TransformerDecoderBlock(n_layers_de, out_dim, embed_dim, 12, num_heads, ff_dim, drop_rate,
                                               batch_size)
        # self.decoder_f = TransformerDecoderBlock(n_layers_en, out_dim, embed_dim, 12, num_heads, ff_dim, drop_rate, batch_size)

    def call(self, inputs, look_ahead_mask):

        encoder_in = inputs[:, 0:12, 0:self.out_dim]
        encoder_out = self.encoder(encoder_in)
        # 第一层decoder
        decoder_in1 = inputs[:, 12:24, 0:self.out_dim]
        decoder_out1, w1 = self.decoder1(decoder_in1, encoder_out, look_ahead_mask)
        # 第2层decoder
        last_in = decoder_out1
        decoder_in2 = inputs[:, 24:36, 0:self.out_dim]
        decoder_out2, w2 = self.decoder2(decoder_in2, last_in, look_ahead_mask)
        # 第3层decoder
        last_in = decoder_out2
        decoder_in3 = inputs[:, 36:48, 0:self.out_dim]
        decoder_out3, w3 = self.decoder3(decoder_in3, last_in, look_ahead_mask)

        #特征注意力
        mask_input2 = tf.random_uniform((self.batch_size, self.out_dim))
        mask2 = make_masking(mask_input2, mask_input2)

        inputs = tf.transpose(inputs, perm=[0, 2, 1])  # [B, outdim, timedim]
        encoderf_in = inputs[:, 0:self.out_dim, 0:12]
        encoderf_out = self.encoderf(encoderf_in)
        # 第一层decoder
        decoderf_in1 = inputs[:, 0:self.out_dim, 12:24]
        decoderf_out1, _ = self.decoderf1(decoderf_in1, encoderf_out, look_ahead_mask=mask2[1])
        # 第2层decoder
        last_in = decoderf_out1
        decoderf_in2 = inputs[:, 0:self.out_dim, 24:36]
        decoderf_out2, _ = self.decoderf2(decoderf_in2, last_in, look_ahead_mask=mask2[1])
        # 第3层decoder
        last_in = decoderf_out2
        decoderf_in3 = inputs[:, 0:self.out_dim, 36:48]
        decoderf_out3, _ = self.decoderf3(decoderf_in3, last_in, look_ahead_mask=mask2[1])

        decoderf_out = tf.transpose(decoderf_out3, perm=[0, 2, 1])
        # x2= self.decoder_f(inputs, look_ahead_mask=mask2[1])

        output = tf.concat([decoder_out3, decoderf_out], 1)

        return output, w1, w2, w3


'''Transformer的输入编码层'''


class Network(tf.keras.Model):
    def __init__(self, batch_size, dim, batch_norm, dropout, rec_dropout, task,
                 target_repl=False, deep_supervision=False, num_classes=1,
                 depth=1, input_dim=76, **kwargs):
        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
        self.batch_size = batch_size

        super(Network, self).__init__()

        input_dim = 76
        out_dim = 59
        time_dim = 48
        split_time = 12

        embed_dim = 512  # Embedding size for each token
        num_heads = 8  # Number of attention heads
        ff_dim = 512  # Hidden layer size in feed forward network inside transformer
        n_layers_en = 1
        n_layers_de = 1

        drop_rate = 0.3

        if task in ['decomp', 'ihm', 'ph']:
            final_activation = 'sigmoid'
        elif task in ['los']:
            if num_classes == 1:
                final_activation = 'relu'
            else:
                final_activation = 'softmax'
        else:
            raise ValueError("Wrong value for task")



        self.decoder = MultiDecoder(n_layers_en, n_layers_de, split_time, embed_dim, out_dim, num_heads, ff_dim,
                                    drop_rate, batch_size)

        self.GlobalAveragePooling1D = layers.GlobalAveragePooling1D()
        # self.Dropout1 = layers.Dropout(drop_rate)
        self.Dropout2 = layers.Dropout(drop_rate)
        # self.Dense1 = layers.Dense(20, activation=final_activation)
        self.Dense2 = layers.Dense(num_classes, activation=final_activation)
        self.batchnorm = layers.BatchNormalization(epsilon=1e-6)

        X = layers.Input(shape=(time_dim, input_dim), name='X')
        inputs = X

        mask_input1 = tf.random_uniform((batch_size, split_time))
        mask_input2 = tf.random_uniform((batch_size, out_dim))
        mask1 = make_masking(mask_input1, mask_input1)
        mask2 = make_masking(mask_input2, mask_input2)

        # out = self.transformer(X, look_ahead_mask=mask[1])

        x = self.decoder(X, look_ahead_mask=mask1[1])
        x = x[0]
        x = self.GlobalAveragePooling1D(x)
        x = self.Dropout1(x)
        x = self.Dense1(x)
        x = self.Dropout2(x)
        outputs = self.Dense2(x)
        super(Network, self).__init__(inputs=inputs, outputs=outputs)

    def say_name(self):
        return "{}.n{}{}{}{}.dep{}".format('split48_tandf',
                                           self.dim,
                                           ".bn" if self.batch_norm else "",
                                           ".d{}".format(self.dropout) if self.dropout > 0 else "",
                                           ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                                           self.depth)


'''
时间上
分成4份，依次经过1个encoder和3个deccoder
特征上
分成4份，依次经过1个encoder和3个deccoder
最后简单拼接
'''

