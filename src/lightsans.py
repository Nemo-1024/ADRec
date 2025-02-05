# -*- coding: utf-8 -*-
# @Time    : 2021/05/01
# @Author  : Xinyan Fan
# @Email   : xinyan.fan@ruc.edu.cn

"""
LightSANs
################################################
Reference:
    Xin-Yan Fan et al. "Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation." in SIGIR 2021.
Reference:
    https://github.com/BELIEVEfxy/LightSANs
"""

import copy
import torch
from torch import nn
from sasrec import SASRec
from common import PositionwiseFeedForward
import math
class LightSANs(SASRec):
    def __init__(self, args):
        super().__init__(args)

        self.trm_encoder = LightTransformerEncoder(args)
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def embedding_layer(self, item_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        return item_emb, position_embedding

    def forward(self, item_seq, tgt_seq, train_flag=True):
        item_emb, position_embedding = self.embedding_layer(item_seq)
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)

        output = self.trm_encoder(
            item_emb, position_embedding, output_all_encoded_layers=True
        )
        output = output[-1]
        return output,output[:,-1,:]


class ItemToInterestAggregation(nn.Module):
    def __init__(self, seq_len, hidden_size, k_interests=5):
        super().__init__()
        self.k_interests = k_interests  # k latent interests
        self.theta = nn.Parameter(torch.randn([hidden_size, k_interests]))

    def forward(self, input_tensor):  # [B, L, d] -> [B, k, d]
        D_matrix = torch.matmul(input_tensor, self.theta)  # [B, L, k]
        D_matrix = nn.Softmax(dim=-2)(D_matrix)
        result = torch.einsum("nij, nik -> nkj", input_tensor, D_matrix)  # #[B, k, d]

        return result

class LightMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        k_interests,
        hidden_size,
        seq_len,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(LightMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialization for low-rank decomposed self-attention
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attpooling_key = ItemToInterestAggregation(
            seq_len, hidden_size, k_interests
        )
        self.attpooling_value = ItemToInterestAggregation(
            seq_len, hidden_size, k_interests
        )

        # initialization for decoupled position encoding
        self.attn_scale_factor = 2
        self.pos_q_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_k_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_scaling = (
            float(self.attention_head_size * self.attn_scale_factor) ** -0.5
        )
        self.pos_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):  # transfor to multihead
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, pos_emb):
        # linear map
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # low-rank decomposed self-attention: relation of items
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(self.attpooling_key(mixed_key_layer))
        value_layer = self.transpose_for_scores(
            self.attpooling_value(mixed_value_layer)
        )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-2)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer_item = torch.matmul(attention_probs, value_layer)

        # decoupled position encoding: relation of positions
        value_layer_pos = self.transpose_for_scores(mixed_value_layer)
        pos_emb = self.pos_ln(pos_emb).unsqueeze(0)
        pos_query_layer = (
            self.transpose_for_scores(self.pos_q_linear(pos_emb)) * self.pos_scaling
        )
        pos_key_layer = self.transpose_for_scores(self.pos_k_linear(pos_emb))

        abs_pos_bias = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))
        abs_pos_bias = abs_pos_bias / math.sqrt(self.attention_head_size)
        abs_pos_bias = nn.Softmax(dim=-2)(abs_pos_bias)

        context_layer_pos = torch.matmul(abs_pos_bias, value_layer_pos)

        context_layer = context_layer_item + context_layer_pos

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class LightTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): the output of the point-wise feed-forward sublayer, is the output of the transformer layer
    """

    def __init__(
        self,
        n_heads,
        k_interests,
        hidden_size,
        seq_len,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(LightTransformerLayer, self).__init__()
        self.multi_head_attention = LightMultiHeadAttention(
            n_heads,
            k_interests,
            hidden_size,
            seq_len,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
        )
        self.feed_forward = PositionwiseFeedForward(hidden_size,0.1)

    def forward(self, hidden_states, pos_emb):
        attention_output = self.multi_head_attention(hidden_states, pos_emb)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class LightTransformerEncoder(nn.Module):
    r"""One LightTransformerEncoder consists of several LightTransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'.
            candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        args,
    ):
        super(LightTransformerEncoder, self).__init__()
        hidden_size = args.hidden_size
        seq_len = args.max_len
        inner_size = 2*hidden_size
        hidden_dropout_prob = args.dropout
        attn_dropout_prob = args.dropout
        hidden_act = args.hidden_act
        layer_norm_eps = 1e-12
        n_layers=4
        n_heads = 4
        k_interests = 5
        layer = LightTransformerLayer(
            n_heads,
            k_interests,
            hidden_size,
            seq_len,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, pos_emb, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TrandformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer layers' output,
            otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, pos_emb)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers