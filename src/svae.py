import torch.nn as nn
import torch
import numpy as np
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(
            args.hidden_size, args.hidden_size
        )
        nn.init.xavier_normal(self.linear1.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(16, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size,args.hidden_size)
        nn.init.xavier_normal(self.linear1.weight)
        nn.init.xavier_normal(self.linear2.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class SVAE(nn.Module):
    def __init__(self, args):
        super(SVAE, self).__init__()
        self.args = args

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        # Since we don't need padding, our vocabulary size = "hyper_params['total_items']" and not "hyper_params['total_items'] + 1"
        self.item_embedding = nn.Embedding(args.item_num+1, args.hidden_size, padding_idx=0)

        self.gru = nn.GRU(
            args.hidden_size, args.hidden_size,
            batch_first=True, num_layers=1
        )

        self.linear1 = nn.Linear(args.hidden_size, 32)
        nn.init.xavier_normal(self.linear1.weight)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self.tanh = nn.Tanh()

    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        temp_out = self.linear1(h_enc)

        self.z_mean = temp_out[:, :16]
        self.z_log_sigma = temp_out[:, 16:]


        std = torch.exp(0.5 * self.z_log_sigma)
        eps = torch.randn_like(std)
        return self.z_mean + eps * std

    def forward(self, x,tgt_seq,train_flag=True):
        if isinstance(self.gru, nn.GRU):
            self.gru.flatten_parameters()
        in_shape = x.shape  # [bsz x seq_len] = [1 x seq_len]
        # x = x.view(-1)  # [seq_len]

        x = self.item_embedding(x)  # [seq_len x embed_size]
        # x = x.view(in_shape[0], in_shape[1], -1)  # [1 x seq_len x embed_size]

        rnn_out, _ = self.gru(x)  # [1 x seq_len x rnn_size]
        rnn_out = rnn_out.reshape(in_shape[0] * in_shape[1], -1)  # [seq_len x rnn_size]

        enc_out = self.encoder(rnn_out)  # [seq_len x hidden_size]
        sampled_z = self.sample_latent(enc_out)  # [seq_len x latent_size]

        dec_out = self.decoder(sampled_z)  # [seq_len x total_items]
        dec_out = dec_out.view(in_shape[0], in_shape[1], -1)  # [1 x seq_len x total_items]
        return dec_out, dec_out[:,-1]

    def calculate_loss(self, seq_output,tgt_seq):
        index = tgt_seq > 0
        seq_output = seq_output[index]
        tgt_seq = tgt_seq[index]
        # loss_type = 'CE'
        logits = torch.matmul(seq_output, self.item_embedding.weight.t())
        loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), tgt_seq.reshape(-1))
        kld = torch.mean(torch.sum(0.5 * (-self.z_log_sigma + torch.exp(self.z_log_sigma) + self.z_mean ** 2 - 1), -1))
        return loss+kld

    def calculate_score(self, item):
        scores = torch.matmul(item.reshape(-1, item.shape[-1]), self.item_embedding.weight.t())
        return scores