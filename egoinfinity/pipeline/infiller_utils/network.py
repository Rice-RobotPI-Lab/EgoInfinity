"""
Motion infiller Transformer network.
Adapted from HaWoR (infiller/lib/model/network.py). Self-contained.
"""
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SinPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout=0.1, pre_lnorm=True, bias=False):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.atten_scale = 1 / math.sqrt(d_model)
        self.pre_lnorm = pre_lnorm

        self.q_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.k_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.v_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.out_linear = nn.Linear(n_head * d_head, d_model, bias=bias)
        self.droput_layer = nn.Dropout(dropout)
        self.atten_dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, hidden, memory=None, mask=None, extra_atten_score=None):
        combined = hidden
        if self.pre_lnorm:
            hidden = self.layer_norm(hidden)
            combined = self.layer_norm(combined)

        q = self.q_linear(hidden)
        k = self.k_linear(combined)
        v = self.v_linear(combined)

        q = q.reshape(q.shape[0], q.shape[1], self.n_head, self.d_head).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], self.n_head, self.d_head).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self.n_head, self.d_head).transpose(1, 2)

        atten_score = torch.matmul(q, k.transpose(-1, -2)) * self.atten_scale
        if mask is not None:
            atten_score = atten_score.masked_fill(mask, float("-inf"))
        atten_score = atten_score.softmax(dim=-1)
        atten_score = self.atten_dropout_layer(atten_score)

        atten_vec = torch.matmul(atten_score, v)
        atten_vec = atten_vec.transpose(1, 2).flatten(start_dim=-2)
        output = self.droput_layer(self.out_linear(atten_vec))

        if self.pre_lnorm:
            return hidden + output
        else:
            return self.layer_norm(hidden + output)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1, pre_lnorm=True):
        super().__init__()
        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(d_model)
        self.network = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_inner, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        if self.pre_lnorm:
            return x + self.network(self.layer_norm(x))
        else:
            return self.layer_norm(x + self.network(x))


class TransformerModel(nn.Module):
    def __init__(self, seq_len, input_dim, d_model, nhead, d_hid, nlayers,
                 dropout=0.5, out_dim=91, masked_attention_stage=False):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.pos_embedding = SinPositionalEncoding(d_model=d_model, dropout=0.1, max_len=seq_len)

        if masked_attention_stage:
            self.input_layer = nn.Linear(input_dim + 1, d_model)
            self.att_layers = nn.ModuleList()
            self.pff_layers = nn.ModuleList()
            self.pre_lnorm = True
            self.layer_norm = nn.LayerNorm(d_model)
            for _ in range(nlayers):
                self.att_layers.append(
                    MultiHeadedAttention(nhead, d_model, d_model // nhead,
                                         dropout=dropout, pre_lnorm=True, bias=False))
                self.pff_layers.append(
                    FeedForward(d_model, d_hid, dropout=dropout, pre_lnorm=True))
        else:
            self.att_layers = None
            self.input_layer = nn.Linear(input_dim, d_model)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation="gelu")
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(d_model, out_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor,
                data_mask=None, atten_mask=None) -> Tensor:
        if data_mask is not None:
            src = torch.cat([src, data_mask.expand(*src.shape[:-1], data_mask.shape[-1])], dim=-1)
        src = self.input_layer(src)
        output = self.pos_embedding(src)
        if self.att_layers:
            assert atten_mask is not None
            output = output.permute(1, 0, 2)
            for i in range(self.nlayers):
                output = self.att_layers[i](output, mask=atten_mask)
                output = self.pff_layers[i](output)
            if self.pre_lnorm:
                output = self.layer_norm(output)
            output = output.permute(1, 0, 2)
        output = self.transformer_encoder(output)
        output = self.decoder(output)
        return output
