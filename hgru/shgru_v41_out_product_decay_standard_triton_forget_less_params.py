import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module

from .hgru_real_cuda import HgruRealFunction

from .gla.intra_chunk_contribution.fn import intra_chunk_onc
from .gla.inter_chunk_contribution.fn import inter_chunk_onc

class SHgruV41(nn.Module):
    def __init__(
        self,
        embed_dim,
        gate_dim=128,
        expand_ratio=2,
        act_fun="silu",
        causal=True,
        bias=True,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.in_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.forget_gate_proj = nn.Sequential(
            nn.Linear(embed_dim, gate_dim, bias=bias),
            nn.Linear(gate_dim, embed_dim, bias=bias),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.expand_ratio = expand_ratio
        self.norm = nn.LayerNorm(embed_dim)
        self.act = get_activation_fn(act_fun)
        self.causal = causal

        self.chunk_size = 128

    def forward(self, x, lower_bound=0):
        ## x: n b d
        n, b, d = x.shape
        feature = self.in_proj(x)
        V, Q = feature.chunk(2, dim=-1)
        F_= self.forget_gate_proj(x)
        V = self.act(V)
        Q = F.sigmoid(Q)
        F_ = F.sigmoid(F_)
        
        # reshape
        # h is num_head, d is head dimension
        V, Q, F_, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [V, Q, F_, lower_bound],
        )
        # # 限制一下最小的log_decay, 保证数值稳定。
        # G_K = (lower_bound[None, None, :, :] * F).clam_min(-3)
        log_lambda_ = (lower_bound * F_)#.clamp_min(-3)
        lambda_ = torch.exp(log_lambda_)

        K = 1 - lambda_

        V, Q, G_K, K = map(
            lambda x: rearrange(x, "(n c) b h d -> b h n c d", c = self.chunk_size).contiguous(),
            [V, Q, log_lambda_, K],
        )

        G_V = None
        G_K, G_V, o1 = inter_chunk_onc(Q, K, V, G_K, G_V)        
        o2 = intra_chunk_onc(Q, K, V, G_K, G_V)
        o = (o1 + o2)        
        o = rearrange(o, "b h n c d -> (n c) b (h d)")
        
        o = self.norm(o)


        # out proj
        output = self.out_proj(o)
        return output
    
    def extra_repr(self):
        return print_module(self)