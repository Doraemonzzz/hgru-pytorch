import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module

from .hgru_real_cuda import HgruRealFunction

class SHgruV28(nn.Module):
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
        self.input_up_proj = nn.Linear(1, expand_ratio, bias=bias)
        self.forget_up_proj = nn.Linear(1, expand_ratio, bias=bias)
        self.down_proj = nn.Linear(expand_ratio, 1, bias=bias)
        
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
        output_gate = F.sigmoid(output_gate)

        # d -> (d, k)
        input = self.act(self.input_up_proj(input.unsqueeze(-1)))
        forget_gate = F.sigmoid(self.forget_up_proj(forget_gate.unsqueeze(-1)))
        input, forget_gate = map(
            lambda x: rearrange(x, "... d k -> ... (d k)"),
            [input, forget_gate],
        )
        
        # mix
        lambda_ = torch.exp(lower_bound * forget_gate)
        input = (1 - lambda_) * input
        output_state = self.scan(input, lambda_)
        
        # reshape
        output_state = rearrange(output_state, "... (d k) -> ... d k", k=self.expand_ratio)
        output_state = self.down_proj(output_state).squeeze(-1)
        
        # output gate
        output_state = self.norm(output_gate * output_state)

        # out proj
        output = self.out_proj(output_state)

        return output

    def extra_repr(self):
        return print_module(self)