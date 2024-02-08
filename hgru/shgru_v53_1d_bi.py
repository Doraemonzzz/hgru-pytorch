import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module, get_norm_fn

from .hgru_real_cuda import HgruRealFunction

from .gla.intra_chunk_contribution.fn import intra_chunk_onc
from .gla.inter_chunk_contribution.fn import inter_chunk_onc

class SHgruV53_1d_bi(nn.Module):
    def __init__(
        self,
        embed_dim,
        expand_ratio=2,
        act_fun="silu",
        uv_act_fun="sigmoid",
        use_norm=True,
        bias=True,
        norm_type="layernorm",
        **kwargs,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.expand_ratio = expand_ratio
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        self.out_act = get_activation_fn(uv_act_fun)
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fn(norm_type)(embed_dim)

        self.chunk_size = 128
        self.scan = HgruRealFunction.apply
        
    def reverse_scan(self, input, lambda_):
        output_state = self.scan(
            torch.flip(input, dims=[0]),
            torch.flip(lambda_, dims=[0]),
        )

        return torch.flip(output_state, dims=[0])

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        feature = self.in_proj(x)
        input, output_gate, forget_gate = feature.chunk(3, dim=-1)
        input = self.act(input)
        output_gate = self.out_act(output_gate)
        forget_gate = F.sigmoid(forget_gate)

        # reshape
        input, output_gate, forget_gate, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [input, output_gate, forget_gate, lower_bound],
        )

        # mix
        lambda_ = lower_bound + (1 - lower_bound) * forget_gate
        input = torch.einsum('... h d, ... h e -> ... h d e', 1 - lambda_, input)

        lambda_ = repeat(lambda_, '... h d -> ... h d e', e=self.expand_ratio)

        # reshape
        input, lambda_ = map(
            lambda x: rearrange(x, '... h d e -> ... (h d e)'),
            [input, lambda_]
        )
        lambda_ = lambda_.to(input.dtype)
        
        # mix
        output_state_forward = self.scan(input, lambda_)
        output_state_reverse = self.reverse_scan(input, lambda_)
        output_state = output_state_forward + output_state_reverse
        # down
        output_state = rearrange(output_state, '... (h d e) -> ... h d e', d=self.expand_ratio, e=self.expand_ratio)
        output_state = torch.einsum('... h d e, ... h d -> ... h e', output_state, output_gate)
        output_state = rearrange(output_state, '... h e -> ... (h e)')
        
        # output gate
        if self.use_norm:
            output_state = self.norm(output_state)

        # out proj
        output = self.out_proj(output_state)

        return output

    def extra_repr(self):
        return print_module(self)