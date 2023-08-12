import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params

from .hgru_real_cuda import HgruRealFunction

triton_parallel_scan = HgruRealFunction.apply


class HgruRealV6(nn.Module):
    def __init__(
        self,
        embed_dim,
        act_fun="silu",
        causal=True,
        use_triton=False,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.lambda_proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = get_activation_fn(act_fun)
        self.causal = causal

        self.scan = HgruRealFunction.apply if not use_triton else triton_parallel_scan

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        input = self.act(self.input_proj(x))
        gate = self.act(self.gate(x))
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(self.lambda_proj(x))
        input = (1 - lambda_) * input

        if self.causal:
            hiddens = self.scan(input, lambda_)
        else:
            hiddens_forward = self.scan(input, lambda_)
            hiddens_backward = self.reverse_scan(input, lambda_)
            hiddens = hiddens_forward + hiddens_backward
        hiddens = self.norm(hiddens)

        output = self.out_proj(hiddens * gate)

        return output

    def reverse_scan(self, input, lambda_):
        hiddens_reverse = self.scan(
            torch.flip(input, dims=[0]),
            torch.flip(lambda_, dims=[0]),
        )

        return torch.flip(hiddens_reverse, dims=[0])

    def forward_naive(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        input = self.act(self.input_proj(x))
        gate = self.act(self.gate(x))
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(self.lambda_proj(x))
        input = (1 - lambda_) * input

        hidden = torch.zeros(1, b, d).to(x)
        hiddens = []
        for i in range(n):
            hidden = lambda_[i] * hidden + input[i]
            hiddens.append(hidden)
        hiddens = torch.cat(hiddens, dim=0)

        hiddens = self.norm(hiddens)
        output = self.out_proj(hiddens * gate)

        return output
