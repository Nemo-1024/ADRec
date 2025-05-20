import torch.utils.data as data_utils
import torch
from tqdm import tqdm
from einops import pack,unpack
import torch.backends.cudnn as cudnn
import numpy as np
from collections import Counter
import random
from scipy.stats import beta

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, id2seq, max_len,parallel_ag=False):
        self.id2seq = id2seq
        self.max_len = max_len
        self.parallel = parallel_ag
    def __len__(self):
        return len(self.id2seq)

    def __getitem__(self, index):
        seq = self._getseq(index)
        hist = seq[:-1]
        hist = hist[-self.max_len:]
        mask_len = self.max_len - len(hist)
        hist_pad = [0] * mask_len + hist
        if self.parallel is True:
            # mask_len = self.max_len - len(target)
            target = [0] * mask_len + seq[-len(hist):]
            assert sum([i>0 for i in hist_pad]) == sum([i>0 for i in target])
        else:
            target = [0] * (self.max_len-1) + [seq[-1]]

        return torch.LongTensor(hist_pad), torch.LongTensor(target)

    def _getseq(self, idx):
        return self.id2seq[idx]


class Data_Train():
    def __init__(self, data_train, args):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.id_seq = data_train
        self.split = args.split_onebyone
        self.parallel_ag = args.parallel_ag
        if self.split:
            print('splitting data onebyone ...')
            self.split_onebyone()

    def split_onebyone(self):
        self.id_seq = {}
        idx = 0
        for seq_temp in self.u2seq:
            seq_temp = seq_temp[-self.max_len-1:]
            # 只能从预截取长度后进行子序列切分
            # 加一是为了包含tgt
            for star in range(len(seq_temp) - 1):
                self.id_seq[idx] = seq_temp[:star + 2]
                idx += 1

    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.id_seq, self.max_len,self.parallel_ag)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len):
        self.u2seq = u2seq
        # self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, index):
        # user = self.users[index]
        seq = self.u2seq[index]
        hist = seq[-self.max_len:]
        padding_len = self.max_len - len(hist)
        hist_pad = [0] * padding_len + hist
        answer_pad = [0] * padding_len + seq[-(len(hist)-1):] + self.u2answer[index]
        assert sum([i>0 for i in hist_pad]) == sum([i>0 for i in answer_pad])
        return torch.LongTensor(hist_pad), torch.LongTensor(answer_pad)


class Data_Val():
    def __init__(self, data_train, data_val, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len
        # self.parallel_ag = True if args.model == 'adrec' else False

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer, max_len):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        # self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, index):
        # user = self.users[index]
        seq = self.u2seq[index] + self.u2seq_add[index]
        # seq = self.u2seq[user]
        hist = seq[-self.max_len:]
        padding_len = self.max_len - len(hist)
        hist_pad = [0] * padding_len + hist
        answer_pad = [0] * padding_len + seq[-(len(hist)-1):] + self.u2answer[index]
        assert sum([i>0 for i in hist_pad]) == sum([i>0 for i in answer_pad])
        return torch.LongTensor(hist_pad), torch.LongTensor(answer_pad)


class Data_Test():
    def __init__(self, data_train, data_val, data_test, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader



def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """

    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)



def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
def exists(v):
    return v is not None
def identity(t, *args, **kwargs):
    return t
def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def safe_div(num, den, eps = 1e-5):
    return num / den.clamp(min = eps)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim

    if padding_dims <= 0:
        return t

    return t.view(*t.shape, *((1,) * padding_dims))

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one


def exponential_mapping(x, v):
    # exp_x[v] = cos(||v||) * x + sin(||v||) * (v / ||v||)
    norm_v = v.norm(p=2, dim=-1, keepdim=True)  # L2 norm of v
    v_unit = v / (norm_v + 1e-8)  # Normalize v to unit vector
    cos_v = torch.cos(norm_v)  # Cosine of the norm of v
    sin_v = torch.sin(norm_v)  # Sine of the norm of v
    return cos_v * x + sin_v * v_unit  # Geodesic exponential map


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


