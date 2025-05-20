import torch.nn as nn

import torch.nn.functional as F
from common import SiLU, TransformerEncoder
from utils import _extract_into_tensor,exponential_mapping
from step_sample import *

class DenoisedModel(nn.Module):
    def __init__(self, args):
        super(DenoisedModel, self).__init__()
        self.hidden_size = args.hidden_size
        if args.dif_decoder =='mlp':
            self.decoder = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4),
                                        SiLU(),
                                        nn.Linear(self.hidden_size * 4, self.hidden_size),
                                        nn.LayerNorm(self.hidden_size),
                                        )
        else:
            self.decoder = TransformerEncoder(args,num_blocks=2,norm_first=False,hidden_size=self.hidden_size)

        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4),
                                        SiLU(),
                                        nn.Linear(self.hidden_size * 4, self.hidden_size)
                                        )

        self.lambda_uncertainty = args.lambda_uncertainty


    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        assert dim % 2 == 0
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps.unsqueeze(-1).float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward_cfg(self,c, x, t, mask_seq,mask_tgt,cfg_scale=1.0):
        cond_eps = self.forward(c,x, t,mask_seq,mask_tgt)
        uncond_eps = self.forward(c,x, t,mask_seq,mask_tgt,condition=False)
        eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return eps


    def forward(self, rep_item, x_t, t, mask_seq,mask_tgt,condition=True):
        if condition is not True:  #CFG
            rep_item = torch.zeros_like(rep_item)
            # mask = torch.rand_like(mask_seq) > 0.5
            # rep_item = torch.where(mask.unsqueeze(-1), torch.zeros_like(rep_item), rep_item)
        t=t.reshape(x_t.shape[0],-1)
        time_emb = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        lambda_uncertainty = self.lambda_uncertainty  ### fixed

        rep_diffu = rep_item + lambda_uncertainty * (x_t + time_emb)

        if isinstance(self.decoder, nn.Sequential):
            # 如果是 MLP，直接应用
            rep_diffu = self.decoder(rep_diffu)
        else:
            rep_diffu = self.decoder(rep_diffu, mask_seq)

        return rep_diffu

class AdRec(nn.Module):
    def __init__(self, args,):
        super(AdRec, self).__init__()

        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])
        betas = get_named_beta_schedule(args)
         # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        # self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        # self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        # print(self.posterior_mean_coef1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])
       
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)  ## lossaware (schedule_sample)
        self.timestep_map = self.time_map()
        self.rescale_timesteps = args.rescale_timesteps
        self.original_num_steps = len(betas)

        # self.xstart_model = self.dif_model(args)
        self.net = DenoisedModel(args)
        self.independent_diffusion = args.independent
        self.cfg_scale = args.cfg_scale
        self.geodesic = args.geodesic
        self.ag_encoder = TransformerEncoder(args, num_blocks=2, norm_first=False)

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        if self.geodesic:
            x_start = F.normalize(x_start, p=2, dim=-1)
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick
        if self.geodesic:
            # exp_x[v] = cos(||v||) * x + sin(||v||) * (v / ||v||)
            x_t = exponential_mapping(x_start, x_t)
        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return th.where(mask==0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t

    def time_map(self):
        timestep_map = []
        for i in range(len(self.alphas_cumprod)):
            if i in self.use_timesteps:
                timestep_map.append(i)
        return timestep_map

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        # print(t)
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        # print(t[0,-10:])
        # print(posterior_mean[0,-1,0])
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq,mask_tag):
        # print("func p_mean_variance", rep_item.shape,x_t.shape)
        if self.cfg_scale==1.:
            x_0 = self.net(rep_item, x_t, self._scale_timesteps(t), mask_seq,mask_tag)
        else:
            x_0 = self.net.forward_cfg(rep_item, x_t, self._scale_timesteps(t), mask_seq, mask_tag,self.cfg_scale)
        # x_0 = model_output.unsqueeze(1)  ##output predict
        # x_0 = self._predict_xstart_from_eps(x_t, t, model_output)  ## eps predict
        # x_0 = x_0.clamp_(-1., 1.)
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)  ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq,mask_tag):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq,mask_tag)
        noise = th.randn_like(noise_x_t)
        # print("noise shape in func p_sample",noise.shape)
        nonzero_mask = (t != 0).float().unsqueeze(-1)  # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick
        if self.geodesic:
            sample_xt = F.normalize(sample_xt,p=2,dim=-1)
        return sample_xt

    def denoise_sample(self, seq, tgt, mask_seq, mask_tag):
        seq = self.ag_encoder(seq, mask_seq)
        # return self.xstart_model(item_rep, noise_x_t, th.tensor([1] * item_rep.shape[0], device=item_rep.device), mask_seq)[0]
        noise_x_t = th.randn_like(tgt)
        indices = list(range(self.num_timesteps))[::-1]
        for i in indices: # from T to 0, reversion iteration  
            t = th.tensor([0]*(seq.shape[1]-1) + [i], device=seq.device).unsqueeze(0).repeat(seq.shape[0],1)
            noise_x_t = torch.concat([tgt[:, :-1], noise_x_t[:, -1:]], dim=1)
            # noise_x_t = torch.concat([torch.zeros_like(tgt[:, :-1]),noise_x_t[:, -1:]], dim=1)
            noise_x_t = self.p_sample(seq, noise_x_t, t, mask_seq, mask_tag)
        # print(noise_x_t[0,-1,:10])
        return noise_x_t

    def independent_diffuse(self, tgt, mask, is_independent=False):
        if is_independent:
            t, weights = self.schedule_sampler.sample(tgt.shape[0] * tgt.shape[1], tgt.device)
            t = t * mask.reshape(-1).long()
            x_t = self.q_sample(tgt.reshape(-1, tgt.shape[-1]), t, mask=mask.reshape(-1)).reshape(*tgt.shape)
        else:
            t, weights = self.schedule_sampler.sample(tgt.shape[0], tgt.device)
            x_t = self.q_sample(tgt, t, mask=mask)
        return x_t,t
    def forward(self, item_rep, item_tag, mask_seq,mask_tag):
        item_rep = self.ag_encoder(item_rep, mask_seq)
        x_t,t = self.independent_diffuse(item_tag, mask_tag, self.independent_diffusion)
        if self.cfg_scale != 1:
            mask = torch.rand([mask_seq.shape[0],1,1],device=item_rep.device) > 0.7
            item_rep = torch.where(mask,torch.zeros_like(item_rep),item_rep)
        denoised_seq = self.net(item_rep, x_t, self._scale_timesteps(t), mask_seq,mask_tag)  ##output predict
        # print(denoised_seq.shape,item_tag.shape,mask_tag.shape)
        losses = F.mse_loss(denoised_seq,item_tag, reduction='none')* (mask_tag / mask_tag.sum(1,keepdim=True)).unsqueeze(-1)
        losses = losses.sum(1).mean()
        return denoised_seq, losses





