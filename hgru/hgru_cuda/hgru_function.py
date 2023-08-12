import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
import os
import hgru_cuda


class HgruFunction(Function):
    @staticmethod
    def forward(ctx, x_real, x_imag, lambda_real, lambda_imag):
        x_real = x_real.contiguous()
        x_imag = x_imag.contiguous()
        lambda_real = lambda_real.contiguous()
        lambda_imag = lambda_imag.contiguous()
        output_real, output_imag = hgru_cuda.forward(
            x_real, x_imag, lambda_real, lambda_imag
        )
        ctx.save_for_backward(
            x_real, x_imag, lambda_real, lambda_imag, output_real, output_imag
        )

        return output_real.contiguous(), output_imag.contiguous()

    @staticmethod
    def backward(ctx, grad_output_real, grad_output_imag):
        (
            x_real,
            x_imag,
            lambda_real,
            lambda_imag,
            output_real,
            output_imag,
        ) = ctx.saved_tensors
        grad_output_real = grad_output_real.contiguous()
        grad_output_imag = grad_output_imag.contiguous()
        (
            grad_x_real,
            grad_x_imag,
            grad_lambda_real,
            grad_lambda_imag,
        ) = hgru_cuda.backward(
            x_real,
            x_imag,
            lambda_real,
            lambda_imag,
            output_real,
            output_imag,
            grad_output_real,
            grad_output_imag,
        )

        return (
            grad_x_real.contiguous(),
            grad_x_imag.contiguous(),
            grad_lambda_real.contiguous(),
            grad_lambda_imag.contiguous(),
        )
