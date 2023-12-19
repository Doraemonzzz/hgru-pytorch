import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params

from .luru_fuse_cuda import LuruFuseFunction

class LuruV5(nn.Module):
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
        
        theta = 10000 ** (-1 / embed_dim * torch.arange(embed_dim))
        self.register_buffer('theta', theta)
        
        self.input_gate = nn.Sequential(
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

        self.scan = LuruFuseFunction.apply

    def forward(self, x, **kwargs):
        # x: n, b, d
        input_state = F.sigmoid(self.input_gate(x)) * self.act(self.input_proj(x))
        output_gate = F.sigmoid(self.output_gate(x))
        output_state = self.scan(input_state, self.theta)
        
        output_state = self.norm(output_state * output_gate)

        output = self.out_proj(output_state)

        return output