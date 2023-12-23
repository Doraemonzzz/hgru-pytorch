import torch
import torch.nn as nn
from einops import rearrange

from .helpers import get_activation_fn, print_params, print_module

class GroupLinearPost(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        expand_ratio=2,
    ):
        super().__init__()
        self.group_weight = nn.Parameter(
            torch.Tensor(expand_ratio, expand_ratio), 
            requires_grad=True
        )
        self.feature_weight = nn.Parameter(
            torch.Tensor(expand_ratio, in_dim, out_dim), 
            requires_grad=True
        )
        self.expand_ratio = expand_ratio
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.group_weight.data)
        nn.init.xavier_uniform_(self.feature_weight.data)
        
    def extra_repr(self):
        return print_module(self)
    
    def forward(self, x):
        # x: n, b, d
        x = rearrange(x, 'n b (k d) -> n b k d', k=self.expand_ratio)
        # n b k d, k d e -> n b k e
        x = torch.einsum('n b k d, k d e -> n b k e', x, self.feature_weight)
        x = torch.einsum('n b k d, k e -> n b e d', x, self.group_weight)
        output = rearrange(x, 'n b k e -> n b (k e)')
        
        return output