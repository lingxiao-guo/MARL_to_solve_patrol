import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

FILE_ABSOLUTE_PATH = os.path.abspath(__file__)  # get absolute filepath
CURRENT_DIR = os.path.dirname(FILE_ABSOLUTE_PATH)  # get directory path of file
PARENT_DIR = os.path.dirname(CURRENT_DIR)  # get parent directory path
BASE_DIR = os.path.dirname(PARENT_DIR)  # get grandparent directory path

sys.path.append(BASE_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], num_heads, -1)

    # num_hiddens/num_heads)
    X = X.permute(0, 3, 1, 2, 4)

    return X.reshape(-1, X.shape[2], X.shape[3], X.shape[4])


# @save
def transpose_output(X, num_heads):
    X = X.permute(2, 0, 1, 3)  # (num_agent,graph_size,batch_size*num_heads,hidden_dim//num_heads)
    return X.reshape(X.shape[2] // num_heads, X.shape[0], X.shape[1], num_heads * X.shape[3])
    # (batch_size,num_agent,graph_size,hidden_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, env, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.adj = env.adj
        self.num_heads = num_heads
        self.attention = DotProductGraphAttention(dropout)

        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        # （batch_size,num_agent,graph_size,hidden_dim)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        output = self.attention(queries, keys, values, self.adj)

        output_concat = transpose_output(output, self.num_heads)
        output_concat = output_concat.to(device)
        return F.relu(self.W_o(output_concat))


# @save
class DotProductGraphAttention(nn.Module):

    def __init__(self, dropout, **kwargs):
        super(DotProductGraphAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, adj):
        dim = queries.size(-1)

        e = torch.matmul(queries, keys.transpose(2, 3)) / 8
        zero_vec = -9e15 * torch.ones_like(e)
        np.expand_dims(adj, 0).repeat(queries.shape[1], axis=0)
        np.expand_dims(adj, 0).repeat(queries.shape[0], axis=0)
        adj = torch.Tensor(adj).to(device)
        attention = torch.where(adj > 0, e, zero_vec)
        # (batch_size*num_heads,num_agent,graph_size,graph_size)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, values)
        # (batch_size*num_heads,num_agent,graph_size,feature_dim)
        return h_prime.view(h_prime.shape[2], h_prime.shape[0], h_prime.shape[1], h_prime.shape[3])
        ## (graph_size,batch_size*num_heads,num_agent,feature_dim)


class AddNorm(nn.Module):

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class EncoderBlock(nn.Module):

    def __init__(self, env, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__()
        self.graph_attention = MultiHeadAttention(env,
                                                  key_size, query_size, value_size, num_hiddens, num_heads, dropout,
                                                  use_bias
                                                  )

        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)

        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.graph_attention(X, X, X))
        return self.addnorm2(Y, self.ffn(Y))


class AgentAttention(nn.Module):
    def __init__(self, env, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, norm_shape=32, bias=False):
        super(AgentAttention, self).__init__()

        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)

        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.adj = env.adj
        self.ln1 = nn.LayerNorm(32)
        self.ln2 = nn.LayerNorm(32)
        self.ln3 = nn.LayerNorm(32)

    def forward(self, queries, keys, values):
        # （batch_size,graph_size,num_agent,hidden_dim)
        queries = transpose_qkv(self.ln1(self.W_q(queries) + queries), self.num_heads)
        keys = transpose_qkv(self.ln2(self.W_k(keys) + keys), self.num_heads)
        values = transpose_qkv(self.ln3(self.W_v(values) + values), self.num_heads)

        # output (num_agent,batch_size*num_heads，graph_size，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, self.adj)

        output_concat = transpose_output(output, self.num_heads)
        output_concat = output_concat.to(device)
        return self.ln1(self.W_o(output_concat) + output_concat)


class DotProductAttention(nn.Module):

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, adj):
        dim = queries.size(-1)

        attention = torch.matmul(queries, keys.transpose(2, 3)) / 16

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, values)

        return h_prime.view(h_prime.shape[2], h_prime.shape[0], h_prime.shape[1], h_prime.shape[3])
        # (num_agent,batch_size*num_heads,graph_size,feature_dim)


class GraphEncoder(nn.Module):
    def __init__(self, env, args, dropout=0.0, use_bias=False, **kwargs):
        input_size = args.num_input
        key_size = args.num_hidden
        query_size = args.num_hidden
        value_size = args.num_hidden
        num_hiddens = args.num_hidden
        norm_shape = args.num_hidden
        ffn_num_input = args.num_hidden
        ffn_num_hiddens = args.num_hidden
        num_heads = args.num_head
        num_layers = args.num_layers
        super(GraphEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Linear(input_size, num_hiddens)
        self.blks = nn.Sequential()
        self.agent_attention = AgentAttention(env, key_size, query_size, value_size, num_hiddens, num_heads, dropout,
                                              use_bias)
        self.state_value = nn.Linear(num_hiddens * len(env.patrol_points[0]), 1)

        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(env, key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))

    def forward(self, X, *args):

        X = torch.tensor(X, dtype=torch.float32).to(device)

        if X.ndimension() < 4:
            X = X.unsqueeze(0)

        X = self.embedding(X)
        # (batch_size,num_agent,graph_size,hidden_dim)
        for i, blk in enumerate(self.blks):
            X = blk(X)
        Y = X.contiguous().reshape(X.shape[0], X.shape[1], -1)
        state_value = self.state_value(Y)

        return X, state_value
        # encoder_output : X: (batch_size,num_agents,graph_size,hidden_dim)
        # value: (batch_Size,num_agents,1)
