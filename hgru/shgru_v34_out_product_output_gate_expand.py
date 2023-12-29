import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module

from .hgru_real_cuda import HgruRealFunction

class SHgruV34(nn.Module):
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
        self.output_gate_expand = nn.Parameter(torch.ones(embed_dim * expand_ratio), requires_grad=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.expand_ratio = expand_ratio
        self.norm = nn.LayerNorm(embed_dim)
        self.act = get_activation_fn(act_fun)
        self.causal = causal

        self.scan = HgruRealFunction.apply

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        feature = self.in_proj(x)
        input, output_gate, forget_gate = feature.chunk(3, dim=-1)
        input = self.act(input)
        forget_gate = F.sigmoid(forget_gate)
        
        # up
        lower_bound = rearrange(lower_bound, '(d k) -> d k', k=self.expand_ratio)

        # mix
        log_lambda = forget_gate.unsqueeze(-1) * lower_bound
        lambda_ = torch.exp(log_lambda)
        input = (1 - lambda_) * input.unsqueeze(-1)
        # mix
        input, lambda_ = map(
            lambda x: rearrange(x, '... d k -> ... (d k)'),
            [input, lambda_]
        )
        output_state = self.scan(input, lambda_)
    
        # down
        output_gate_expand, output_state = map(
            lambda x: rearrange(x, '... (d k) -> ... d k', k=self.expand_ratio),
            [self.output_gate_expand, output_state]
        )
        output_gate = F.sigmoid(output_gate.unsqueeze(-1) * output_gate_expand)
        output_state = torch.einsum('... d k, ... d k -> ... d', output_state, output_gate)

        # norm
        output_state = self.norm(output_state)

        # out proj
        output = self.out_proj(output_state)

        return output

    def extra_repr(self):
        return print_module(self)