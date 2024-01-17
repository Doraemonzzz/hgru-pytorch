import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module, get_norm_fn

from .hgru_real_cuda import HgruRealFunction

from .gla.intra_chunk_contribution.fn import intra_chunk_onc
from .gla.inter_chunk_contribution.fn import inter_chunk_onc

class SHgruV43(nn.Module):
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
        self.mask = torch.empty(0)

    def forward(self, x, lower_bound=0):
        ## x: n b d
        n, b, d = x.shape
        feature = self.in_proj(x)
        input, output_gate, forget_gate = feature.chunk(3, dim=-1)
        input = self.act(input)
        output_gate = self.out_act(output_gate)
        forget_gate = F.sigmoid(forget_gate)
        
        # reshape
        input, output_gate, forget_gate, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [input, output_gate, lower_bound],
        )
        log_lambda_ = (lower_bound * forget_gate)
        
        o = self.compute(input, output_gate, log_lambda_)
        
        if self.use_norm:
            o = self.norm(o)

        # out proj
        output = self.out_proj(o)
        return output
    
    def compute(self, input, output_gate, log_lambda_):
        if self.mask.shape[0] == 0:
            self.mask = torch.triu(torch.ones(self.chunk_size, self.chunk_size), diagonal=1).bool().to(input.device)
        
        input, output_gate, log_lambda_ = map(
            lambda x: rearrange(x, "(n c) b h d -> b h n c d", c=self.chunk_size).contiguous(),
            [input, output_gate, log_lambda_],
        )

        log_lambda_, o1 = inter_chunk_onc(input, output_gate, log_lambda_)        
        o2 = intra_chunk_onc(input, output_gate, log_lambda_, self.mask)
        o = (o1 + o2)        
        o = rearrange(o, "b h n c d -> (n c) b (h d)")
        
        return o

    def extra_repr(self):
        return print_module(self)