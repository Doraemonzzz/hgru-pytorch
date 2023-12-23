import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module

from .hgru_real_cuda import HgruRealFunction
from .glt_v4 import GroupLinearGroupSoftmax

class SHgruV16GroupSoftmax(nn.Module):
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

        self.in_proj = GroupLinearGroupSoftmax(embed_dim // expand_ratio, embed_dim * 3, expand_ratio)
        self.out_proj = GroupLinearGroupSoftmax(
            embed_dim, embed_dim // expand_ratio, expand_ratio
        )
        
        self.expand_ratio = expand_ratio
        self.norm = nn.LayerNorm(expand_ratio * embed_dim)
        self.act = get_activation_fn(act_fun)
        self.causal = causal

        self.scan = HgruRealFunction.apply

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        feature = self.in_proj(x)
        input, output_gate, forget_gate = feature.chunk(3, dim=-1)
        
        input = self.act(input)
        output_gate = F.sigmoid(output_gate)
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(forget_gate)
        input = (1 - lambda_) * input

        output_state = self.scan(input, lambda_)
        
        output_state = self.norm(output_state * output_gate)

        # out proj
        output = self.out_proj(output_state)

        return output

    def extra_repr(self):
        return print_module(self)