import torch 
import time
import math
from typing import Tuple, Union, Optional
import torch
import torch.nn.functional as F
from einops import rearrange
import torch
import triton
import triton.language as tl
import numpy as np
import math


from .fn_only_gk import FlashGRet

def intra_chunk_onc(input, output_gate, log_lambda_, mask=None):
    assert input.is_contiguous()
    assert output_gate.is_contiguous()
    assert log_lambda_.is_contiguous()
    
    origin_chunk_size = input.shape[-2]
    assert input.shape[-2] % 16 == 0
    
    A = FlashGRet.apply(output_gate, log_lambda_)

    if mask is None:
        mask = torch.triu(torch.ones(A.shape[-2], A.shape[-2]), diagonal=1).bool().to(A.device)
    A.masked_fill_(mask, 0)

    O = A.to(input) @ input        

    return O
