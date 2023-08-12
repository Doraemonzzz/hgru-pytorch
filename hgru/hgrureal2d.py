import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params

from .hgru_real_cuda import HgruRealFunction

triton_parallel_scan = HgruRealFunction.apply


class HgruReal2d(nn.Module):
    def __init__(self, embed_dim, act_fun="silu", causal=True, use_triton=False):
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
        h, w, b, d = x.shape
        input = self.act(self.input_proj(x))
        gate = self.act(self.gate(x))
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(self.lambda_proj(x))
        input = (1 - lambda_) * input

        input1, lambda1 = map(
            lambda x: rearrange(x, "h w b d -> h (w b) d"),
            [input, lambda_],
        )
        input2, lambda2 = map(
            lambda x: rearrange(x, "h w b d -> w (h b) d"),
            [input, lambda_],
        )

        hiddens1_forward = self.scan(input1, lambda1)
        hiddens1_reverse = self.reverse_scan(input1, lambda1)
        hiddens2_forward = self.scan(input2, lambda2)
        hiddens2_reverse = self.reverse_scan(input2, lambda2)

        hiddens1 = rearrange(hiddens1_forward, "h (w b) d -> h w b d", w=w) + rearrange(
            hiddens1_reverse, "h (w b) d -> h w b d", w=w
        )
        hiddens2 = rearrange(hiddens2_forward, "w (h b) d -> h w b d", h=h) + rearrange(
            hiddens2_reverse, "w (h b) d -> h w b d", h=h
        )

        hiddens = hiddens1 + hiddens2

        feature = self.norm(hiddens)

        output = self.out_proj(feature * gate)

        return output

    def reverse_scan(self, input, lambda_):
        hiddens_reverse = self.scan(
            torch.flip(input, dims=[0]),
            torch.flip(lambda_, dims=[0]),
        )

        return torch.flip(hiddens_reverse, dims=[0])
