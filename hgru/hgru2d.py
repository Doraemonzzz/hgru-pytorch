import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params

from .hgru_cuda import HgruFunction

triton_parallel_scan = HgruFunction.apply


class Hgru2d(nn.Module):
    def __init__(self, embed_dim, act_fun="silu", causal=True, use_triton=False):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.input_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.lambda_proj = nn.Linear(embed_dim, embed_dim)
        self.theta = nn.Parameter(
            10000 ** (-2 / embed_dim * torch.arange(embed_dim)).reshape(1, 1, -1),
            requires_grad=True,
        )
        self.gate = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(2 * embed_dim)
        self.act = get_activation_fn(act_fun)
        self.causal = causal
        self.scan = HgruFunction.apply if not use_triton else triton_parallel_scan

    def forward(self, x, lower_bound=0):
        h, w, b, d = x.shape
        input_state = self.act(self.input_proj(x))
        index = torch.ones(h, w, 1, 1).to(x)
        theta = self.theta
        theta = index * theta
        gate = self.act(self.gate(x))
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(self.lambda_proj(x))

        input_state = rearrange(input_state, "... (e k) -> ... e k", k=2)

        gamma_real = lambda_ * torch.cos(theta)
        gamma_imag = lambda_ * torch.sin(theta)

        # 1 - lambda
        input_real = (1 - lambda_) * input_state[..., 0]
        input_imag = (1 - lambda_) * input_state[..., 1]

        if self.causal:
            input_real1, input_imag1, gamma_real1, gamma_imag1 = map(
                lambda x: rearrange(x, "h w b d -> h (w b) d"),
                [input_real, input_imag, gamma_real, gamma_imag],
            )
            input_real2, input_imag2, gamma_real2, gamma_imag2 = map(
                lambda x: rearrange(x, "h w b d -> w (h b) d"),
                [input_real, input_imag, gamma_real, gamma_imag],
            )
            hiddens_real1, hiddens_imag1 = self.scan(
                input_real1, input_imag1, gamma_real1, gamma_imag1
            )
            hiddens_real2, hiddens_imag2 = self.scan(
                input_real2, input_imag2, gamma_real2, gamma_imag2
            )
            hiddens_real = rearrange(
                hiddens_real1, "h (w b) d -> h w b d", w=w
            ) + rearrange(hiddens_real2, "w (h b) d -> h w b d", h=h)
            hiddens_imag = rearrange(
                hiddens_imag1, "h (w b) d -> h w b d", w=w
            ) + rearrange(hiddens_imag2, "w (h b) d -> h w b d", h=h)
        else:
            input_real1, input_imag1, gamma_real1, gamma_imag1 = map(
                lambda x: rearrange(x, "h w b d -> h (w b) d"),
                [input_real, input_imag, gamma_real, gamma_imag],
            )
            input_real2, input_imag2, gamma_real2, gamma_imag2 = map(
                lambda x: rearrange(x, "h w b d -> w (h b) d"),
                [input_real, input_imag, gamma_real, gamma_imag],
            )
            hiddens_real1_forward, hiddens_imag1_forward = self.scan(
                input_real1, input_imag1, gamma_real1, gamma_imag1
            )
            hiddens_real1_reverse, hiddens_imag1_reverse = self.reverse_scan(
                input_real1, input_imag1, gamma_real1, gamma_imag1
            )
            hiddens_real1_reverse, hiddens_imag1_reverse = self.reverse_scan(
                input_real1, input_imag1, gamma_real1, gamma_imag1
            )
            hiddens_real2_forward, hiddens_imag2_forward = self.scan(
                input_real2, input_imag2, gamma_real2, gamma_imag2
            )
            hiddens_real2_reverse, hiddens_imag2_reverse = self.reverse_scan(
                input_real2, input_imag2, gamma_real2, gamma_imag2
            )

            hiddens_real = (
                rearrange(hiddens_real1_forward, "h (w b) d -> h w b d", w=w)
                + rearrange(hiddens_real1_reverse, "h (w b) d -> h w b d", w=w)
                + rearrange(hiddens_real2_forward, "w (h b) d -> h w b d", h=h)
                + rearrange(hiddens_real2_reverse, "w (h b) d -> h w b d", h=h)
            )
            hiddens_imag = (
                rearrange(hiddens_imag1_forward, "h (w b) d -> h w b d", w=w)
                + rearrange(hiddens_imag1_reverse, "h (w b) d -> h w b d", w=w)
                + rearrange(hiddens_imag2_forward, "w (h b) d -> h w b d", h=h)
                + rearrange(hiddens_imag2_reverse, "w (h b) d -> h w b d", h=h)
            )

        feature = torch.cat([hiddens_real, hiddens_imag], dim=-1)

        feature = self.norm(feature)

        output = self.out_proj(feature * gate)

        return output

    def reverse_scan(self, input_real, input_imag, gamma_real, gamma_imag):
        hiddens_real_reverse, hiddens_imag_reverse = self.scan(
            torch.flip(input_real, dims=[0]),
            torch.flip(input_imag, dims=[0]),
            torch.flip(gamma_real, dims=[0]),
            torch.flip(gamma_imag, dims=[0]),
        )

        return torch.flip(hiddens_real_reverse, dims=[0]), torch.flip(
            hiddens_imag_reverse, dims=[0]
        )
