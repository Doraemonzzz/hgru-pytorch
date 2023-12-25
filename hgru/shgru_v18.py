import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module

from .hgru_real_cuda import HgruRealFunction
from .kronecker import KroneckerUp, KroneckerDown

class SHgruV18(nn.Module):
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

        self.input_proj = KroneckerUp(embed_dim, expand_ratio)
        self.output_proj = KroneckerUp(embed_dim, expand_ratio)
        self.forget_proj = KroneckerUp(embed_dim, expand_ratio)
        self.out_proj = KroneckerDown(embed_dim, expand_ratio)
        
        self.expand_ratio = expand_ratio
        self.norm = nn.LayerNorm(expand_ratio * embed_dim)
        self.act = get_activation_fn(act_fun)
        self.causal = causal

        self.scan = HgruRealFunction.apply

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        # d -> d k
        input = self.input_proj(x)
        output_gate = self.output_proj(x)
        forget_gate = self.forget_proj(x)
        
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