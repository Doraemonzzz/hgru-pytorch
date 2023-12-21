import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module

from .hgru_real_cuda import HgruRealFunction

class SHgruV8(nn.Module):
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

        # d -> k, d / k, -> k, d
        # N(0, 2 / (fin + fanout))
        c = (2 / (embed_dim // expand_ratio + embed_dim)) ** 0.5
        proj_weight = torch.randn(expand_ratio, embed_dim // expand_ratio, 3 * embed_dim) * c
        self.proj_weight = nn.Parameter(
            proj_weight, requires_grad=True,
        )
        out_proj = torch.randn(expand_ratio, embed_dim, embed_dim // expand_ratio) * c
        # k, d -> k, d / k
        self.out_proj = nn.Parameter(
            out_proj, requires_grad=True,
        )
        self.expand_ratio = expand_ratio
        self.norm = nn.LayerNorm(expand_ratio * embed_dim)
        self.act = get_activation_fn(act_fun)
        self.causal = causal

        self.scan = HgruRealFunction.apply

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        x = rearrange(x, 'n b (k d) -> n b k d', k=self.expand_ratio)
        feature = torch.einsum('n b k d, k d e -> n b k e', x, self.proj_weight)
        feature = rearrange(feature, 'n b k e -> n b (k e)')
        input, output_gate, forget_gate = feature.chunk(3, dim=-1)
        
        input = self.act(input)
        output_gate = F.sigmoid(output_gate)
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(forget_gate)
        input = (1 - lambda_) * input

        output_state = self.scan(input, lambda_)
        
        output_state = self.norm(output_state * output_gate)

        # out proj
        output_state = rearrange(output_state, 'n b (k d) -> n b k d', k=self.expand_ratio)
        output = torch.einsum('n b k d, k d e -> n b k e', output_state, self.out_proj)
        output = rearrange(output, 'n b k e -> n b (k e)')

        return output

    def extra_repr(self):
        return print_module(self)