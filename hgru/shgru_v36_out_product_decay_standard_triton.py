import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module

from .hgru_real_cuda import HgruRealFunction

from ..gla.intra_chunk_contribution.fn import intra_chunk_onc
from ..gla.inter_chunk_contribution.fn import inter_chunk_onc

class SHgruV36_Triton(nn.Module):
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

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.expand_ratio = expand_ratio
        self.norm = nn.LayerNorm(embed_dim)
        self.act = get_activation_fn(act_fun)
        self.causal = causal

        # self.scan = HgruRealFunction.apply

        self.chunk_size = 128

    def forward(self, x, lower_bound=0):
        ## x: (b, n, d)
        b, n, d = x.shape
        feature = self.in_proj(x)
        V, Q, F = feature.chunk(3, dim=-1)
        V = self.act(V)
        Q = F.sigmoid(Q)
        F = F.sigmoid(F)
        
        # reshape
        # h is num_head, d is head dimension
        V, Q, F, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [ V, Q, F, lower_bound],
        )
        
        # # 限制一下最小的log_decay, 保证数值稳定。
        # G_K = (lower_bound[None, None, :, :] * F).clam_min(-3)

        K = 1 - lambda_.exp()

        V, Q, G_K, K = map(
            lambda x: rearrange(x, "b (n c) h d -> b h n c d", c = chunk_size).contiguous(),
            [V, Q, G_K, K],
        )

        G_V = None         
        G_K, G_V, o1 = inter_chunk_onc(Q, K, B, G_K, G_V)        
        o2 = intra_chunk_onc(Q, K, V, G_K, G_V)
        o = (o1 + o2)        
        o = rearrange(o, "b h n c d -> b (n c) (h d)")
        
        o = self.norm(o)


        # out proj
        output = self.out_proj(o)
        return output

    def extra_repr(self):
        return print_module(self)