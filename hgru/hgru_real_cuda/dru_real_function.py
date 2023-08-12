import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
import os
import hgru_real_cuda


class HgruRealFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        output = hgru_real_cuda.forward(x, lambda_)
        ctx.save_for_backward(x, lambda_, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, lambda_, hidden_states = ctx.saved_tensors
        grad_output = grad_output.to(x)
        grad_x, grad_lambda = hgru_real_cuda.backward(
            x, lambda_, hidden_states, grad_output
        )
        return grad_x, grad_lambda
