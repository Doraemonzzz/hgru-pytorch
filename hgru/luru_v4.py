import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params

from .luru_fuse_cuda import LuruFuseFunction

class LuruV4(nn.Module):
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
        
        self.gate = nn.Sequential(
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
        input_state = self.act(self.input_proj(x))
        gate = F.sigmoid(self.gate(x))
        output_state = self.scan(input_state, self.theta)
        
        output_state = self.norm(output_state * gate)

        output = self.out_proj(output_state)

        return output
    
    def rotate1(self, x):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        theta_ = torch.polar(torch.ones_like(self.theta).to(torch.float32), self.theta.to(torch.float32))
        x_out = torch.view_as_real(x_ * theta_).flatten(2)
        x_out = x_out.type_as(x)
        
        return x_out
    
    def rotate2(self, x):
        dtype = x.dtype
        theta = torch.stack([self.theta, self.theta], dim=-1).reshape(1, 1, -1).to(torch.float32)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        # (-q1, -q3), (q0, q2) -> (-q1, q0, -q3, q2)
        x_half = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).to(torch.float32)
        x_out = cos * x.to(torch.float32) + sin * x_half
        
        return x_out.to(dtype)

    def forward_naive(self, x, **kwargs):
        # x: n, b, d
        input_state = self.act(self.input_proj(x))
        gate = F.sigmoid(self.gate(x))
        output_state = []
        n, b, d = input_state.shape
        memory = torch.zeros(1, b, d).to(x).float()

        for i in range(n):
            memory_next = self.rotate1(memory.float()) + input_state[i:i+1].float()
            output_state.append(memory_next.to(x.dtype))
            memory = memory_next
        output_state = torch.cat(output_state, dim=0)
        
        output_state = self.norm((output_state * gate.float()).to(x.dtype))

        output = self.out_proj(output_state)

        return output