import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module

from .hgru_real_cuda import HgruRealFunction

class SHgruV26(nn.Module):
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
        self.input_up_proj = nn.Parameter(torch.randn(embed_dim) * 0.1, requires_grad=True)
        self.forget_up_proj = nn.Parameter(torch.randn(embed_dim) * 0.1, requires_grad=True)
        self.down_proj = nn.Parameter(torch.randn(embed_dim) * 0.1, requires_grad=True)
        
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
        
        # up
        input, input_up_proj, forget_gate, forget_up_proj, down_proj, lower_bound = map(
            lambda x: rearrange(x, "... (k d) -> ... k d", d=self.expand_ratio),
            [input, self.input_up_proj, forget_gate, self.forget_up_proj, self.down_proj, lower_bound],
        )
        # k d, m d -> k d m
        input = torch.einsum('... k d, k e -> ... k d e', input, input_up_proj)
        forget_gate = torch.einsum('... k d, k e -> ... k d e', torch.exp(lower_bound * F.sigmoid(forget_gate)), \
                                                                torch.exp(lower_bound * F.sigmoid(forget_up_proj)))
        input, forget_gate = map(
            lambda x: rearrange(x, "... k d e -> ... (k d e)"),
            [input, forget_gate],
        )
        
        # mix
        # exp(log_slope * (sigmoid(x) + sigmoid(y))
        lambda_ = forget_gate

        input = (1 - lambda_) * input
        output_state = self.scan(input, lambda_)

        # down
        output_state = rearrange(output_state, '... (k d e) -> ... k d e', d=self.expand_ratio, e=self.expand_ratio)
        output_state = torch.einsum('... k d e, k d -> ... k e', output_state, down_proj)
        output_state = rearrange(output_state, '... k d -> ... (k d)')
        
        # output gate
        output_gate = self.act(output_gate)
        output_state = self.norm(output_state * output_gate)

        # out proj
        output = self.out_proj(output_state)

        return output

    def extra_repr(self):
        return print_module(self)