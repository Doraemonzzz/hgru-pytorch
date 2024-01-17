import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module, get_norm_fn

from .hgru_real_cuda import HgruRealFunction

from .ops.triton.gla import fused_chunk_gla

# bug

class SHgruV46(nn.Module):
    def __init__(
        self,
        embed_dim,
        expand_ratio=2,
        act_fun="silu",
        uv_act_fun="sigmoid",
        use_norm=True,
        bias=True,
        norm_type="layernorm",
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

        self.chunk_size = 128

    def forward(self, x, lower_bound=0):
        ## x: n b d
        n, b, d = x.shape
        feature = self.in_proj(x)
        V, Q, F_ = feature.chunk(3, dim=-1)
        V = self.act(V)
        Q = self.out_act(Q)
        F_ = F.sigmoid(F_)
        
        # reshape
        # h is num_head, d is head dimension
        V, Q, F_ = map(
            lambda x: rearrange(x, "n b (h d) -> b h n d", d=self.expand_ratio),
            [V, Q, F_],
        )
        lower_bound = rearrange(lower_bound, "(h d) -> h d", d=self.expand_ratio).unsqueeze(1)
        # # 限制一下最小的log_decay, 保证数值稳定。
        # G_K = (lower_bound[None, None, :, :] * F).clam_min(-3)

        log_lambda_ = (lower_bound * F_)#.clamp_min(-3)
        lambda_ = torch.exp(log_lambda_)

        K = 1 - lambda_

        G_K = log_lambda_

        o = fused_chunk_gla(Q, K, V, G_K)   
        
        o = rearrange(o, "b h n d -> n b (h d)")  
        
        if self.use_norm:
            o = self.norm(o)

        # out proj
        output = self.out_proj(o)
        return output
    
    def forward_naive(self, x, lower_bound=0):
        from .gla.intra_chunk_contribution.fn import intra_chunk_onc
        from .gla.inter_chunk_contribution.fn import inter_chunk_onc

        ## x: n b d
        n, b, d = x.shape
        feature = self.in_proj(x)
        V, Q, F_ = feature.chunk(3, dim=-1)
        V = self.act(V)
        Q = self.out_act(Q)
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
        
        if self.use_norm:
            o = self.norm(o)

        # out proj
        output = self.out_proj(o)
        return output

    def extra_repr(self):
        return print_module(self)