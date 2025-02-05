import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sasrec import SASRec

class CORE(SASRec):
    def __init__(self, args):
        super(CORE, self).__init__(args)

        # load parameters info
        self.hidden_size = args.hidden_size

        self.sess_dropout = nn.Dropout(args.dropout)
        self.item_dropout = nn.Dropout(args.emb_dropout)
        self.temperature = 1

        # parameters initialization
        self._reset_parameters()
        self.net =TransNet(args,self.trm_encoder)
    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def ave_net(self, item_seq):
        mask = item_seq.gt(0)
        alpha = mask.to(torch.float) / mask.sum(dim=-1, keepdim=True)
        return alpha.unsqueeze(-1)

    def forward(self, item_seq,tgt_seq,train_flag=True):
        x, _ = self.embedding_layer(item_seq)
        x = self.dropout(x)
        # Representation-Consistent Encoder (RCE)
        alpha = self.net(item_seq,x)
        seq_output = torch.sum(alpha * x, dim=1,keepdim=True).repeat(1, x.shape[1], 1)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output, seq_output[:,-1]

    def calculate_loss(self,seq_output, tgt_seq):
        index = tgt_seq > 0
        seq_output = seq_output[index]
        tgt_seq = tgt_seq[index]
        all_item_emb = self.item_embedding.weight
        # Robust Distance Measuring (RDM)
        all_item_emb = F.normalize(all_item_emb, dim=-1)
        logits = torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature
        loss = self.loss_fct(logits.reshape(-1,logits.shape[-1]), tgt_seq.reshape(-1))
        return loss

class TransNet(nn.Module):
    def __init__(self, args,att):
        super().__init__()
        self.layer_norm_eps = 1e-9
        self.initializer_range = 0.1
        self.hidden_size = args.hidden_size
        self.position_embedding = nn.Embedding(args.item_num, self.hidden_size)
        self.trm_encoder = att

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(args.dropout)
        self.fn = nn.Linear(self.hidden_size, 1)

        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        pad_mask = item_seq>0

        output = self.trm_encoder(input_emb, pad_mask)

        alpha = self.fn(output).to(torch.double)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()