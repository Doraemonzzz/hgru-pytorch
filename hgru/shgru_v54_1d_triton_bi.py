import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module, get_norm_fn

from .hgru_real_cuda import HgruRealFunction

from .gla.intra_chunk_contribution.fn import intra_chunk_onc
from .gla.inter_chunk_contribution.fn import inter_chunk_onc

class SHgruV54_2d_triton_bi(nn.Module):
    def __init__(
        self,
        embed_dim,
        expand_ratio=2,
        act_fun="silu",
        uv_act_fun="sigmoid",
        use_norm=True,
        bias=True,
        norm_type="layernorm",
        chunk_size=16,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.expand_ratio = expand_ratio
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        self.out_act = get_activation_fn(uv_act_fun)
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fn(norm_type)(embed_dim)

        self.chunk_size = chunk_size
        
    def pad_chunk(self, x):
        n, b, h, d = x.shape
        d = self.chunk_size - n % self.chunk_size
        if d > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, d))
            
        return x
        
    def scan(self, V, Q, log_lambda_, K):
        n, b, h, d = Q.shape
        
        V, Q, G_K, K = map(
            lambda x: rearrange(self.pad_chunk(x), "(n c) b h d -> b h n c d", c = self.chunk_size).contiguous(),
            [V, Q, log_lambda_, K],
        )

        G_V = None
        G_K, G_V, o1 = inter_chunk_onc(Q, K, V, G_K, G_V)        
        o2 = intra_chunk_onc(Q, K, V, G_K, G_V)
        o = (o1 + o2)        
        o = rearrange(o, "b h n c d -> (n c) b h d")[:n]
        
        return o
        
    def reverse_scan(self, V, Q, log_lambda_, K):
        output = self.scan(
            torch.flip(V, dims=[0]),
            torch.flip(Q, dims=[0]),
            torch.flip(log_lambda_, dims=[0]),
            torch.flip(K, dims=[0]),
        )

        return torch.flip(output, dims=[0])

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        feature = self.in_proj(x)
        V, Q, F_ = feature.chunk(3, dim=-1)
        V = self.act(V)
        Q = self.out_act(Q)
        F_ = F.sigmoid(F_)
        if type(lower_bound) == int:
            lower_bound = torch.zeros_like(x).to(x)

        # reshape
        # h is num_head, d is head dimension
        V, Q, F_, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [V, Q, F_, lower_bound],
        )
        # # 限制一下最小的log_decay, 保证数值稳定。
        # G_K = (lower_bound[None, None, :, :] * F).clam_min(-3)
        lambda_ = lower_bound + (1 - lower_bound) * F_

        log_lambda_ = torch.log(lambda_)

        K = 1 - lambda_
        output_state_forward = self.scan(V, Q, log_lambda_, K)
        output_state_reverse = self.reverse_scan(V, Q, log_lambda_, K)
        
        # fusion
        output_state = output_state_forward + output_state_reverse
        
        output_state = rearrange(output_state, '... h d -> ... (h d)')
        
        # output gate
        if self.use_norm:
            output_state = self.norm(output_state)

        # out proj
        output = self.out_proj(output_state)

        return output

    def extra_repr(self):
        return print_module(self)