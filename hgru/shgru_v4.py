import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params

from .hgru_real_cuda import HgruRealFunction

class SHgruV4(nn.Module):
    def __init__(
        self,
        embed_dim,
        gate_dim=128,
        act_fun="silu",
        causal=True,
        bias=True,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.input_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.forget_gate = nn.Sequential(
            nn.Linear(embed_dim, gate_dim, bias=bias),
            nn.Linear(gate_dim, 2 * embed_dim, bias=bias),
        )
        self.output_gate = nn.Sequential(
            nn.Linear(embed_dim, gate_dim, bias=bias),
            nn.Linear(gate_dim, 2 * embed_dim, bias=bias),
        )
        self.out_proj = nn.Linear(2 * embed_dim, embed_dim, bias=bias)
        self.norm = nn.LayerNorm(2 * embed_dim)
        self.act = get_activation_fn(act_fun)
        self.causal = causal

        self.scan = HgruRealFunction.apply

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        input = self.act(self.input_proj(x))
        output_gate = F.sigmoid(self.output_gate(x))
        lambda_ = torch.exp(lower_bound * F.sigmoid(self.forget_gate(x)))
        input = (1 - lambda_) * input

        output_state = self.scan(input, lambda_)
        
        output_state = self.norm(output_state * output_gate)

        output = self.out_proj(output_state)

        return output
