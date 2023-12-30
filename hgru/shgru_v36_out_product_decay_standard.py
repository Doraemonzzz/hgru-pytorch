import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module

from .hgru_real_cuda import HgruRealFunction

class SHgruV36(nn.Module):
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

        self.scan = HgruRealFunction.apply

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        feature = self.in_proj(x)
        input, output_gate, forget_gate = feature.chunk(3, dim=-1)
        input = self.act(input)
        output_gate = F.sigmoid(output_gate)
        forget_gate = F.sigmoid(forget_gate)
        
        # reshape
        input, output_gate, forget_gate, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [input, output_gate, forget_gate, lower_bound],
        )
        
        # mix
        lambda_ = torch.exp(lower_bound * forget_gate)
        input = torch.einsum('... h d, ... h e -> ... h d e', 1 - lambda_, input)
        lambda_ = repeat(lambda_, '... h d -> ... h d e', e=self.expand_ratio)
        # reshape
        input, lambda_ = map(
            lambda x: rearrange(x, '... h d e -> ... (h d e)'),
            [input, lambda_]
        )
        # mix
        output_state = self.scan(input, lambda_)

        # down
        output_state = rearrange(output_state, '... (h d e) -> ... h d e', d=self.expand_ratio, e=self.expand_ratio)
        output_state = torch.einsum('... h d e, ... h d -> ... h e', output_state, output_gate)
        output_state = rearrange(output_state, '... h e -> ... (h e)')
        
        # output gate
        output_state = self.norm(output_state)

        # out proj
        output = self.out_proj(output_state)

        return output

    def extra_repr(self):
        return print_module(self)