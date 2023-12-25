import torch
import torch.nn as nn
from einops import rearrange

from .helpers import get_activation_fn, print_params, print_module

class KroneckerUp(nn.Module):
    def __init__(
        self,
        in_dim,
        expand_ratio=2,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.up = nn.Parameter(
            torch.randn(1, expand_ratio) * 0.1, 
            requires_grad=True
        )
        
    def extra_repr(self):
        return print_module(self)
    
    def forward(self, x):
        # x: n, b, d
        x = self.in_proj(x)
        # (n, b, d, 1), (1, k) -> (n, b, d, k)
        x = x.unsqueeze(-1) * self.up
        x = rearrange(x, '... d k -> ... (d k)')
        
        return x
    
class KroneckerDown(nn.Module):
    def __init__(
        self,
        in_dim,
        expand_ratio=2,
    ):
        super().__init__()
        self.expand_ratio = expand_ratio
        self.out_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.down = nn.Linear(expand_ratio, 1, bias=False)
        
    def extra_repr(self):
        return print_module(self)
    
    def forward(self, x):
        # x: n, b, d
        x = rearrange(x, '... (d k) -> ... d k', k=self.expand_ratio)
        x = self.down(x).squeeze(-1)
        x = self.out_proj(x)
        
        return x