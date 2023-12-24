import torch
import torch.nn as nn
from einops import rearrange

from .helpers import get_activation_fn, print_params, print_module

class KRLinear(nn.Module):
    def __init__(
        self,
        in_dim,
        expand_ratio=2,
    ):
        super().__init__()
        self.kr_linear = nn.Parameter(
            torch.Tensor(in_dim, expand_ratio), 
            requires_grad=True
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kr_linear.data)
        
    def extra_repr(self):
        return print_module(self)
    
    def forward(self, x):
        # x: n, b, d
        x = torch.einsum('n b d, d k -> n b d k', x, self.kr_linear)
        output = rearrange(x, 'n b d k -> n b (d k)')
        
        return output