import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params

from .luru_cuda import LuruFunction

class Luru(nn.Module):
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
        
        theta = nn.Parameter(
            10000 ** (-1 / embed_dim * torch.arange(embed_dim)).reshape(1, 1, -1),
            requires_grad=False,
        )
        self.register_buffer('theta', theta)
        self.index = torch.empty(0)
        
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, gate_dim, bias=bias),
            nn.Linear(gate_dim, 2 * embed_dim, bias=bias),
        )
        self.out_proj = nn.Linear(2 * embed_dim, embed_dim, bias=bias)
        self.norm = nn.LayerNorm(2 * embed_dim)
        self.act = get_activation_fn(act_fun)
        self.causal = causal

        self.scan = LuruFunction.apply

    def rotate(self, x):
        n, b, d = x.shape
        if self.index.shape[0] == 0:
            self.index = torch.arange(n - 1, -1, -1).reshape(-1, 1, 1).to(x)
            theta = self.index * self.theta
            self.theta_ = torch.polar(torch.ones_like(theta).to(torch.float32), theta.to(torch.float32))
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x_ * self.theta_).flatten(2)
        x_out = x_out.type_as(x)

        return x_out

    def forward(self, x, **kwargs):
        # x: n, b, d
        input_state = self.act(self.input_proj(x))
        gate = F.sigmoid(self.gate(x))
        input_state = self.rotate(input_state)
        output_state = self.scan(input_state)

        output_state = self.norm(output_state * gate)

        output = self.out_proj(output_state)

        return output

    def forward_naive(self, x, **kwargs):
        # x: n, b, d
        input_state = self.act(self.input_proj(x))
        gate = F.sigmoid(self.gate(x))
        input_state = self.rotate(input_state)
        output_state = torch.cumsum(input_state, dim=0)

        output_state = self.norm(output_state * gate)

        output = self.out_proj(output_state)

        return output