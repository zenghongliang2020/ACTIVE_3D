import math

import torch
import torch.nn as nn
import os


def transpose_qkv(x, num_heads):
    """

    :param x:
    :param num_heads:
    :return:
    """
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x, num_heads):
    """

    :param x:
    :param num_heads:
    :return:
    """
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)


class Dotproduct_Attention(nn.Module):
    """

    """
    def __init__(self, dropout, **kwargs):
        super(Dotproduct_Attention, self).__init__(**kwargs)
        self.dropout = dropout

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHead_Attention(nn.Module):
    """

    """
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHead_Attention, self).__init__()
        self.num_heads = num_heads
        self.attention = Dotproduct_Attention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class Position_Ecoding(nn.Module):
    def __init__(self, num_imput, num_hiddens, num_outputs, **kwargs):
        super(Position_Ecoding, self).__init__(**kwargs)
        self.dense1 = nn.Linear(num_imput, num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHead_Attention(key_size, query_size, value_size,
                                             num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = Position_Ecoding(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

