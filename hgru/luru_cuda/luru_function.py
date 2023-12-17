import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
import os
import luru_cuda


class LuruFunction(Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        output = luru_cuda.forward(x)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        grad_x = luru_cuda.backward(grad_output)

        return grad_x
