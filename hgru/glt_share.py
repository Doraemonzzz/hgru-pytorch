import torch
import torch.nn as nn
from einops import rearrange

from .helpers import get_activation_fn, print_params, print_module

class GroupLinearShare(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        expand_ratio=2,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.expand_ratio = expand_ratio
        
    def extra_repr(self):
        return print_module(self)
    
    def forward(self, x):
        # x: n, b, d
        x = rearrange(x, 'n b (k d) -> n b k d', k=self.expand_ratio)
        x = self.proj(x)
        output = rearrange(x, 'n b k e -> n b (k e)')
        
        return output