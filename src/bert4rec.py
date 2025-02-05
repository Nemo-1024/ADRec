# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 12:08
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

# UPDATE
# @Time   : 2023/9/4
# @Author : Enze Liu
# @Email  : enzeeliu@foxmail.com

r"""
BERT4Rec
################################################

Reference:
    Fei Sun et al. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer."
    In CIKM 2019.

Reference code:
    The authors' tensorflow implementation https://github.com/FeiSun/BERT4Rec

"""

import random

import torch
from torch import nn
from sasrec import SASRec
from common import TransformerEncoder

class BERT4Rec(SASRec):
    def __init__(self, args):
        super().__init__(args)
        self.mask_ratio = 0.5
        # define layers and loss
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.hidden_size)
        self.output_bias = nn.Parameter(torch.zeros(args.item_num))
        self.trm_encoder = TransformerEncoder(args, num_blocks=2, norm_first=False,is_causal=False)
        # parameters initialization
        self.apply(self._init_weights)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reconstruct_test_data(self, item_seq, item_seq_len):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(
            item_seq.size(0), dtype=torch.long, device=item_seq.device
        )  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        item_seq = item_seq[:, 1:]
        return item_seq

    def forward(self, item_seq, tgt_seq, train_flag=True):
        mask_seq = (item_seq > 0).float()  # Padding 掩码

        # Mask Generation
        if train_flag:
            bert_mask = (torch.rand_like(item_seq.float()) <= self.mask_ratio) * mask_seq  # 被掩盖位置
        else:
            bert_mask = torch.zeros_like(item_seq)  # 测试阶段不掩盖

        # Apply Mask to Input
        item_seq = torch.where(bert_mask > 0, torch.zeros_like(item_seq), item_seq)

        # Embedding
        item_emb, position_emb = self.embedding_layer(item_seq)
        input_emb = item_emb + position_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # Transformer Encoder
        trm_output = self.trm_encoder(input_emb, mask_seq)  # [B, L, K]

        # Output Layer
        ffn_output = self.output_ffn(trm_output)  # [B, L, K]
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)  # [B, L, K]
        if train_flag:
            # Apply Mask to Output
            output = output * bert_mask.unsqueeze(-1)  # 仅保留掩盖位置输出
        else:
            output = output

        # Return Output
        return output, output[:, -1, :]  # [B, L, K], [B, K]

    def calculate_loss(self, seq_output,tgt_seq):
        index = seq_output.sum(-1) != 0
        seq_output = seq_output[index]
        tgt_seq = tgt_seq[index]
        # loss_type = 'CE'
        logits = torch.matmul(seq_output, self.item_embedding.weight.t())
        loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), tgt_seq.reshape(-1))
        return loss

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(
            masked_index.size(0), max_length, device=masked_index.device
        )
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot