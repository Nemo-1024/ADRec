import torch.nn as nn
import torch as th
import math
import torch
import torch.nn.functional as F
from einops import rearrange

def exists(v):
    return v is not None
def default(v, d):
    return v if exists(v) else d
def divisible_by(num, den):
    return (num % den) == 0
def generate_square_subsequent_mask(sz: int, device):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=torch.float32, device=device),
        diagonal=1,
    )

class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout,norm_first=False):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if self.norm_first:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return self.norm(x + self.dropout(sublayer(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (
                    1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))



class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)
        for l in self.linear_layers:
            nn.init.xavier_normal_(l.weight)
    def forward(self, q, k, v, padding_mask=None,is_causal=False):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in
                   zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if padding_mask is not None:
            # assert padding_mask.shape == [corr.shape[0],corr.shape[-1]]
            padding_mask = padding_mask.view(batch_size, 1, -1, 1).repeat([1, corr.shape[1], 1, corr.shape[-1]])
            corr = corr.masked_fill(padding_mask == 0, -1e9)
        if is_causal:
            causal_mask = generate_square_subsequent_mask(corr.shape[-1],device=corr.device)
            corr += causal_mask.unsqueeze(0).unsqueeze(0).repeat([corr.shape[0],corr.shape[1],1,1])
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden

class EulerFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = None

        self.b = nn.Parameter(torch.zeros(1))
        # Note: This bias is the global wise, or you can use the positional wise to adjust each query: nn.Parameter(torch.zeros(1, 1, config.MAX_ITEM_LIST_LENGTH, config.hidden_size // 2))


        self.delta = nn.Parameter(torch.ones(1) * 1)
        # Note: You can also use its vector wise: nn.Parameter(torch.ones(config.hidden_size // 2) * config.init_factor)

        # self.get_alpha(1, 1, config.max_len, config.hidden_size)

    def forward(self, v):
        # v = v.unsqueeze(1)
        r = v[..., ::2]
        p = v[..., 1::2]
        batch_size = v.shape[0]
        nums_head = v.shape[1]
        max_len = v.shape[2]
        output_dim = v.shape[-1]

        # Euler Trans
        lam = torch.sqrt(r ** 2 + p ** 2)
        theta = torch.atan2(p, r)
        type = ['ro',]
        if 'ro' in type:
            alpha = self.get_alpha(batch_size, nums_head, max_len, output_dim)
            theta = theta * self.delta + alpha.to(theta).data
            if 'query' in type:
                theta = theta + self.b

        r, p = lam * torch.cos(theta), lam * torch.sin(theta)
        embeddings = torch.stack([r, p], dim=-1)
        embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
        return embeddings.squeeze(1)

    def get_alpha(self, batch_size, nums_head, max_len, output_dim):
        if self.alpha is None:
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
            ids = torch.arange(0, output_dim // 2, dtype=torch.float)
            theta = torch.pow(10000, -2 * ids / output_dim)
            embeddings = position * theta
            embeddings = torch.stack([embeddings, embeddings], dim=-1)
            self.alpha = nn.Parameter(embeddings)

        embeddings = self.alpha.repeat((batch_size, nums_head, *([1] * len(self.alpha.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
        return embeddings[..., ::2]

class MultiHeadedAttentionwithEulerFormer(MultiHeadedAttention):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__(heads, hidden_size, dropout)
        self.euler_q = EulerFormer()
        self.euler_k = EulerFormer()
    def forward(self, q, k, v, padding_mask=None,is_causal=False):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in
                   zip(self.linear_layers, (q, k, v))]
        q = self.euler_q(q)
        k = self.euler_k(k)
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if padding_mask is not None:
            # assert padding_mask.shape == [corr.shape[0],corr.shape[-1]]
            padding_mask = padding_mask.view(batch_size, 1, -1, 1).repeat([1, corr.shape[1], 1, corr.shape[-1]])
            corr = corr.masked_fill(padding_mask == 0, -1e9)
        if is_causal:
            causal_mask = generate_square_subsequent_mask(corr.shape[-1], device=corr.device)
            corr += causal_mask.unsqueeze(0).unsqueeze(0).repeat([corr.shape[0], corr.shape[1], 1, 1])
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout,is_causal,norm_first=False):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout,norm_first=norm_first)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout,norm_first=norm_first)
        self.is_causal = is_causal
    def forward(self, hidden, padding_mask):
        hidden = self.input_sublayer(hidden,
                                     lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, padding_mask=padding_mask, is_causal=self.is_causal))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return hidden

class EulerFormerBlock(TransformerEncoderBlock):
    def __init__(self, hidden_size, attn_heads, dropout,is_causal,norm_first=False):
        super().__init__(hidden_size, attn_heads, dropout,is_causal,norm_first=False)
        self.attention = MultiHeadedAttentionwithEulerFormer(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout, is_causal,norm_first=False):
        super(TransformerDecoderBlock, self).__init__()
        # Self-attention
        self.self_attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        # Cross-attention
        self.cross_attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        # Sublayer connections
        self.input_sublayer_1 = SublayerConnection(hidden_size=hidden_size, dropout=dropout,norm_first=norm_first)
        self.input_sublayer_2 = SublayerConnection(hidden_size=hidden_size, dropout=dropout,norm_first=norm_first)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout,norm_first=norm_first)
        self.is_causal = is_causal

    def forward(self, hidden, memory, padding_mask=None, memory_padding_mask=None):
        # Self-attention: Apply self-attention on the target sequence
        hidden = self.input_sublayer_1(hidden,
                                       lambda _hidden: self.self_attention.forward(_hidden,_hidden, _hidden, padding_mask=padding_mask, is_causal=self.is_causal))
        # Cross-attention: Apply cross-attention with memory (from encoder)
        hidden = self.input_sublayer_2(hidden,
                                       lambda _hidden: self.cross_attention.forward(_hidden, memory, memory, padding_mask=memory_padding_mask, is_causal=self.is_causal))
        # Feed-forward network
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return hidden


class TransformerDecoder(nn.Module):
    def __init__(self, args,num_blocks,norm_first=False):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(args.hidden_size, 4, args.dropout, args.is_causal,norm_first=norm_first) for _ in range(num_blocks)
        ])

    def forward(self, tgt, memory, padding_mask=None, memory_padding_mask=None):
        hidden = tgt
        for layer in self.layers:
            hidden = layer(hidden, memory, padding_mask, memory_padding_mask)
        return hidden



class TransformerEncoder(nn.Module):
    def __init__(self, args,num_blocks,norm_first=False,hidden_size=None,is_causal=None):
        super(TransformerEncoder, self).__init__()
        if hidden_size is not None:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        if is_causal is not None:
            self.is_causal = is_causal
        else:
            self.is_causal = args.is_causal
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.hidden_size, self.heads, self.dropout,self.is_causal,norm_first=norm_first) for _ in range(num_blocks)])
    def forward(self, hidden, padding_mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, padding_mask)
        return hidden
    def make_causal(self,is_causal):
        self.is_causal = is_causal



class AdaptiveLayerNorm(nn.Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.to_gamma = nn.Linear(dim_condition, dim, bias = False)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.)

class RotaryPositionalEmbeddings(torch.nn.Module):
  def __init__(self, d: int, max_len=1000):

    super().__init__()
    self.d = d
    self.cos_cached = None
    self.sin_cached = None
    self.max_len = max_len
  def _build_cache(self, x: torch.Tensor):

    if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
      return
    theta = 1. / (1000 ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

    seq_idx = torch.arange(self.max_len, device=x.device).float().to(x.device) #Position Index -> [0,1,2...seq-1]

    idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)


    self.cos_cached = idx_theta2.cos()[:, None, :] #Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
    self.sin_cached = idx_theta2.sin()[:, None,  :] #cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

  def _neg_half(self, x: torch.Tensor):

    d_2 = self.d // 2 #

    return torch.cat([-x[:, :, d_2:], x[:, :, :d_2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]


  def forward(self, x: torch.Tensor):

    self._build_cache(x)

    neg_half_x = self._neg_half(x)

    x_rope = (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]]) # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]

    return x_rope



class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, t):
        t= t.unsqueeze(-1)
        freqs = t * rearrange(self.weights, 'd -> 1 d') * 2 * torch.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((t, fouriered), dim = -1)
        return fouriered    # B,*,D+1



def modulate(x, shift, scale):
    return x * (1 + scale) + shift
class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.unsqueeze(-1).float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        layers = nn.ModuleList([])
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        for _ in range(args.dif_blocks):
            adaptive_layernorm = AdaptiveLayerNorm(self.hidden_size, )
            block = nn.Sequential(
                nn.Linear(self.hidden_size, 4*self.hidden_size),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(4*self.hidden_size, self.hidden_size)
            )

            block_out_gamma = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            nn.init.zeros_(block_out_gamma.weight)

            layers.append(nn.ModuleList([
                adaptive_layernorm,
                block,
                block_out_gamma
            ]))

        self.layers = layers

    def forward(
            self,
            noised,
            cond,
    ):
        # assert noised.ndim == 2

        denoised = noised

        for adaln, block, block_out_gamma in self.layers:
            residual = denoised
            denoised = adaln(denoised, condition=cond)

            block_out = block(denoised) * (block_out_gamma(cond) + 1.)
            denoised = block_out + residual

        return denoised

class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,args,
        num_blocks
    ):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.model_channels = self.hidden_size*2
        self.num_res_blocks = num_blocks
        self.time_embed = TimestepEmbedder(self.model_channels)
        self.cond_embed = nn.Linear(self.hidden_size, self.model_channels)

        self.input_proj = nn.Linear(self.hidden_size, self.model_channels)

        res_blocks = []
        for i in range(self.num_res_blocks):
            res_blocks.append(ResBlock(
                self.model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(self.model_channels, self.hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c
        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.hidden_size], model_out[:, self.hidden_size:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)