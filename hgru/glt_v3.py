import torch
import torch.nn as nn
from einops import rearrange

from .helpers import get_activation_fn, print_params, print_module

class GroupLinearV3(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        expand_ratio=2,
    ):
        super().__init__()
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features))
        c1 = expand_ratio ** -0.5
        self.group_weight = nn.Parameter(
            torch.empty(expand_ratio, expand_ratio).uniform_(-c1, c1), 
            requires_grad=True
        )
        c2 = in_dim ** -0.5
        self.feature_weight = nn.Parameter(
            torch.empty(expand_ratio, in_dim, out_dim).uniform_(-c2, c2), 
            requires_grad=True
        )
        self.expand_ratio = expand_ratio
        
    def extra_repr(self):
        return print_module(self)
    
    def forward(self, x):
        # x: n, b, d
        x = rearrange(x, 'n b (k d) -> n b k d', k=self.expand_ratio)
        x = torch.einsum('n b k d, k e -> n b e d', x, self.group_weight)
        # n b k d, k d e -> n b k e
        x = torch.einsum('n b k d, k d e -> n b k e', x, self.feature_weight)
        output = rearrange(x, 'n b k e -> n b (k e)')
        
        return output