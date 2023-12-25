import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module

from .hgru_real_cuda import HgruRealFunction
from .glt_no_share import GroupLinearNoShare

class SHgruV20(nn.Module):
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
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.down = nn.Linear(expand_ratio, 1, bias=bias)
        
        self.in_glt = GroupLinearNoShare(embed_dim // expand_ratio, embed_dim, expand_ratio)
        self.forget_glt = GroupLinearNoShare(embed_dim // expand_ratio, embed_dim, expand_ratio)

        self.expand_ratio = expand_ratio
        self.norm = nn.LayerNorm(embed_dim)
        self.act = get_activation_fn(act_fun)
        self.causal = causal

        self.scan = HgruRealFunction.apply

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        feature = self.in_proj(x)
        state, output_gate = feature.chunk(2, dim=-1)
        
        # group linear: n, b, d -> n, b, kd
        input = self.in_glt(state)
        forget_gate = self.forget_glt(state)
        
        # mix
        input = self.act(input)
        output_gate = F.sigmoid(output_gate)
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(forget_gate)
        input = (1 - lambda_) * input

        output_state = self.scan(input, lambda_)

        # down
        output_state = rearrange(output_state, 'n b (k d) -> n b d k', k=self.expand_ratio)
        output_state = self.down(output_state).squeeze(-1)
        
        # output gate
        output_state = self.norm(output_state * output_gate)

        # out proj
        output = self.out_proj(output_state)

        return output

    def extra_repr(self):
        return print_module(self)