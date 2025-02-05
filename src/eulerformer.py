import torch
import torch.nn as nn
from common import EulerFormerBlock
from sasrec import SASRec
from common import RotaryPositionalEmbeddings
# from torchtune.modules import RotaryPositionalEmbeddings
class EulerFormerNet(SASRec):
    def __init__(self, args):
        super(EulerFormerNet, self).__init__(args)
        self.position_embedding =RotaryPositionalEmbeddings(args.hidden_size)
        self.trm_encoder = EulerFormerBlock(args.hidden_size,4,args.dropout,args.is_causal,norm_first=False)

    def embedding_layer(self, item_seq):
        item_emb = self.item_embedding(item_seq)
        position_embedding = self.position_embedding(item_emb)

        return item_emb, position_embedding