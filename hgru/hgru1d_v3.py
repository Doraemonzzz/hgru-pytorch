import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params

from .hgru_cuda import HgruFunction

triton_parallel_scan = HgruFunction.apply

# use parameters efficient gate

class Hgru1dV3(nn.Module):
    def __init__(
        self,
        embed_dim,
        gate_dim=128,
        act_fun="silu",
        causal=True,
        use_triton=False,
        bias=True,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.input_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.lambda_proj = nn.Sequential(
                nn.Linear(embed_dim, gate_dim, bias=bias),
                nn.Linear(gate_dim, embed_dim, bias=bias),
            )
        self.theta = nn.Parameter(
            10000 ** (-2 / embed_dim * torch.arange(embed_dim)).reshape(1, 1, -1),
            requires_grad=True,
        )
        self.gate = nn.Sequential(
                nn.Linear(embed_dim, gate_dim, bias=bias),
                nn.Linear(gate_dim, 2 * embed_dim, bias=bias),
            )
        self.out_proj = nn.Linear(2 * embed_dim, embed_dim, bias=bias)
        self.norm = nn.LayerNorm(2 * embed_dim)
        self.act = get_activation_fn(act_fun)
        self.causal = causal

        self.scan = HgruFunction.apply if not use_triton else triton_parallel_scan

    def forward(self, x, lower_bound=0):
        n, b, d = x.shape
        input_state = self.act(self.input_proj(x))
        index = torch.ones(n, 1, 1).to(x)
        theta = self.theta
        theta = index * theta
        gate = F.sigmoid(self.gate(x))
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(self.lambda_proj(x))

        input_state = rearrange(input_state, "... (e k) -> ... e k", k=2)

        gamma_real = lambda_ * torch.cos(theta)
        gamma_imag = lambda_ * torch.sin(theta)

        # 1 - lambda
        input_real = (1 - lambda_) * input_state[..., 0]
        input_imag = (1 - lambda_) * input_state[..., 1]

        if self.causal:
            hiddens_real, hiddens_imag = self.scan(
                input_real, input_imag, gamma_real, gamma_imag
            )
        else:
            hiddens_real_forward, hiddens_imag_forward = self.scan(
                input_real, input_imag, gamma_real, gamma_imag
            )
            hiddens_real_reverse, hiddens_imag_reverse = self.reverse_scan(
                input_real, input_imag, gamma_real, gamma_imag
            )
            hiddens_real = hiddens_real_forward + hiddens_real_reverse
            hiddens_imag = hiddens_imag_forward + hiddens_imag_reverse

        feature = torch.cat([hiddens_real, hiddens_imag], dim=-1)
        feature = self.norm(feature * gate)

        output = self.out_proj(feature)

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

    # for test
    def forward_triton(self, x, lower_bound=0):
        n, b, d = x.shape
        input_state = self.act(self.input_proj(x))
        index = torch.ones(n, 1, 1).to(x)
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
            hiddens_real, hiddens_imag = triton_parallel_scan(
                input_real, input_imag, gamma_real, gamma_imag
            )
        else:
            hiddens_real_forward, hiddens_imag_forward = triton_parallel_scan(
                input_real, input_imag, gamma_real, gamma_imag
            )
            hiddens_real_reverse, hiddens_imag_reverse = triton_parallel_scan(
                torch.flip(input_real, dims=[0]),
                torch.flip(input_imag, dims=[0]),
                torch.flip(gamma_real, dims=[0]),
                torch.flip(gamma_imag, dims=[0]),
            )
            hiddens_real = hiddens_real_forward + hiddens_real_reverse
            hiddens_imag = hiddens_imag_forward + hiddens_imag_reverse

        feature = torch.cat([hiddens_real, hiddens_imag], dim=-1)

        feature = self.norm(feature * gate)

        output = self.out_proj(feature)

        return output

    def forward_naive(self, x, lower_bound=0):
        n, b, d = x.shape
        input_state = self.act(self.input_proj(x))
        index = torch.ones(n, 1, 1).to(x)
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

        hidden_real = torch.zeros(1, b, d).to(x)
        hidden_imag = torch.zeros(1, b, d).to(x)
        hiddens_real = []
        hiddens_imag = []
        for i in range(n):
            hidden_real_next = (
                gamma_real[i] * hidden_real
                - gamma_imag[i] * hidden_imag
                + input_real[i]
            )
            hidden_imag_next = (
                gamma_real[i] * hidden_imag
                + gamma_imag[i] * hidden_real
                + input_imag[i]
            )
            # print(hidden_real_next.shape)
            hiddens_real.append(hidden_real_next)
            hiddens_imag.append(hidden_imag_next)
            hidden_real = hidden_real_next
            hidden_imag = hidden_imag_next
        hiddens_real = torch.cat(hiddens_real, dim=0)
        hiddens_imag = torch.cat(hiddens_imag, dim=0)
        feature = torch.cat([hiddens_real, hiddens_imag], dim=-1)

        feature = self.norm(feature * gate)

        output = self.out_proj(feature)

        return output

    def inference(
        self,
        x,
        state,
        lower_bound=0,
    ):
        b, d = x.shape
        hidden_real = state[0]
        hidden_imag = state[1]
        input_state = self.act(self.input_proj(x))
        theta = self.theta.squeeze(0)
        gate = self.act(self.gate(x))
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(self.lambda_proj(x))

        input_state = rearrange(input_state, "... (e k) -> ... e k", k=2)

        gamma_real = lambda_ * torch.cos(theta)
        gamma_imag = lambda_ * torch.sin(theta)

        # 1 - lambda
        input_real = (1 - lambda_) * input_state[..., 0]
        input_imag = (1 - lambda_) * input_state[..., 1]

        hidden_real_next = (
            gamma_real[i] * hidden_real - gamma_imag[i] * hidden_imag + input_real[i]
        )
        hidden_imag_next = (
            gamma_real[i] * hidden_imag + gamma_imag[i] * hidden_real + input_imag[i]
        )
        hidden_real = hidden_real_next
        hidden_imag = hidden_imag_next

        feature = torch.cat([hidden_real, hidden_imag], dim=-1)

        feature = self.norm(feature * gate)

        output = self.out_proj(feature)

        return output, [hidden_real, hidden_imag]
