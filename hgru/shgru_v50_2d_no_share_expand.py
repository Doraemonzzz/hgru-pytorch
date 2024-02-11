import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .helpers import get_activation_fn, print_params, print_module, get_norm_fn

from .hgru_real_cuda import HgruRealFunction

from .gla.intra_chunk_contribution.fn import intra_chunk_onc
from .gla.inter_chunk_contribution.fn import inter_chunk_onc

class SHgruV50_2d_no_share_expand(nn.Module):
    def __init__(
        self,
        embed_dim,
        expand_ratio=2,
        act_fun="silu",
        uv_act_fun="sigmoid",
        use_norm=True,
        bias=True,
        norm_type="layernorm",
        feature_expand_ratio=1,
        **kwargs,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.expand_ratio = expand_ratio
        d = int(embed_dim * feature_expand_ratio)
        self.in_proj = nn.Linear(embed_dim, 3 * d, bias=bias)
        self.out_proj = nn.Linear(d, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        self.out_act = get_activation_fn(uv_act_fun)
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fn(norm_type)(d)

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
        H, W, B, D = x.shape
        feature = self.in_proj(x)
        input, output_gate, forget_gate = feature.chunk(3, dim=-1)
        input = self.act(input)
        output_gate = self.out_act(output_gate)
        forget_gate = F.sigmoid(forget_gate)
        
        lower_bound_forward, lower_bound_reverse = lower_bound.chunk(2, dim=-1)

        # reshape
        input, output_gate, forget_gate, lower_bound_forward, lower_bound_reverse = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [input, output_gate, forget_gate, lower_bound_forward, lower_bound_reverse],
        )

        # mix forward
        lambda_forward = lower_bound_forward + (1 - lower_bound_forward) * forget_gate
        input_forward = torch.einsum('... h d, ... h e -> ... h d e', 1 - lambda_forward, input)
        lambda_forward = repeat(lambda_forward, '... h d -> ... h d e', e=self.expand_ratio)
        # reshape
        input_forward, lambda_forward = map(
            lambda x: rearrange(x, '... h d e -> ... (h d e)'),
            [input_forward, lambda_forward]
        )
        lambda_forward = lambda_forward.to(input.dtype)
        
        # mix reverse
        lambda_reverse = lower_bound_reverse + (1 - lower_bound_reverse) * forget_gate
        input_reverse = torch.einsum('... h d, ... h e -> ... h d e', 1 - lambda_reverse, input)
        lambda_reverse = repeat(lambda_reverse, '... h d -> ... h d e', e=self.expand_ratio)
        # reshape
        input_reverse, lambda_reverse = map(
            lambda x: rearrange(x, '... h d e -> ... (h d e)'),
            [input_reverse, lambda_reverse]
        )
        lambda_reverse = lambda_reverse.to(input.dtype)
        
        # mix
        input_h_forward, lambda_h_forward, input_h_reverse, lambda_h_reverse = map(
            lambda x: rearrange(x, "h w b d -> h (w b) d"),
            [input_forward, lambda_forward, input_reverse, lambda_reverse],
        )
        input_w_forward, lambda_w_forward, input_w_reverse, lambda_w_reverse = map(
            lambda x: rearrange(x, "h w b d -> w (h b) d"),
            [input_forward, lambda_forward, input_reverse, lambda_reverse],
        )
        output_state_h_forward = self.scan(input_h_forward, lambda_h_forward)
        output_state_h_reverse = self.reverse_scan(input_h_reverse, lambda_h_reverse)
        output_state_w_forward = self.scan(input_w_forward, lambda_w_forward)
        output_state_w_reverse = self.reverse_scan(input_w_reverse, lambda_w_reverse)
        output_state = rearrange(output_state_h_forward, "h (w b) d -> h w b d", w=W) \
                     + rearrange(output_state_h_reverse, "h (w b) d -> h w b d", w=W) \
                     + rearrange(output_state_w_forward, "w (h b) d -> h w b d", h=H) \
                     + rearrange(output_state_w_reverse, "w (h b) d -> h w b d", h=H)

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