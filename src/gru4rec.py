
r"""
GRU4Rec
################################################

Reference:
    Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.

"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from sasrec import SASRec


class GRU4Rec(SASRec):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, args):
        # load parameters info
        super().__init__(args)
        self.gru_layers = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=4,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq, tag_seq,train_flag=True):
        if isinstance(self.gru_layers, torch.nn.GRU):
            self.gru_layers.flatten_parameters()
        item_emb,position_emb = self.embedding_layer(item_seq)
        item_seq_emb = item_emb + position_emb
        item_seq_emb_dropout = self.dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        last_item = gru_output[:,-1,:]
        return gru_output,last_item
