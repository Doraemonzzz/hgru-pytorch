import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params

from .hgru_cuda import HgruFunction

triton_parallel_scan = HgruFunction.apply


class BiHgru1d(nn.Module):
    def __init__(
        self,
        embed_dim,
        act_fun="silu",
        use_triton=False,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.input_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.lambda_proj = nn.Linear(embed_dim, embed_dim)
        self.theta_forward = nn.Parameter(
            10000 ** (-2 / embed_dim * torch.arange(embed_dim)).reshape(1, 1, -1),
            requires_grad=True,
        )
        self.theta_reverse = nn.Parameter(
            10000 ** (-2 / embed_dim * torch.arange(embed_dim)).reshape(1, 1, -1),
            requires_grad=True,
        )
        self.gate = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(2 * embed_dim)
        self.act = get_activation_fn(act_fun)

        self.scan = HgruFunction.apply if not use_triton else triton_parallel_scan

    def forward(self, x, lower_bound=0):
        n, b, d = x.shape
        input_state = self.act(self.input_proj(x))
        index = torch.ones(n, 1, 1).to(x)
        # theta forward and reverse
        theta_forward = self.theta_forward
        theta_forward = index * theta_forward
        theta_reverse = self.theta_reverse
        theta_reverse = index * theta_reverse
        gate = self.act(self.gate(x))
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(self.lambda_proj(x))

        input_state = rearrange(input_state, "... (e k) -> ... e k", k=2)

        gamma_real_forward = lambda_ * torch.cos(theta_forward)
        gamma_imag_forward = lambda_ * torch.sin(theta_forward)
        gamma_real_reverse = lambda_ * torch.cos(theta_reverse)
        gamma_imag_reverse = lambda_ * torch.sin(theta_reverse)

        # 1 - lambda
        input_real = (1 - lambda_) * input_state[..., 0]
        input_imag = (1 - lambda_) * input_state[..., 1]

        hiddens_real_forward, hiddens_imag_forward = self.scan(
            input_real, input_imag, gamma_real_forward, gamma_imag_forward
        )
        hiddens_real_reverse, hiddens_imag_reverse = self.reverse_scan(
            input_real, input_imag, gamma_real_reverse, gamma_imag_reverse
        )
        hiddens_real = hiddens_real_forward + hiddens_real_reverse
        hiddens_imag = hiddens_imag_forward + hiddens_imag_reverse

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
