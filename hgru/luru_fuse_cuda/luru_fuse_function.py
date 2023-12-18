from torch.autograd import Function
import torch
import luru_fuse_cuda

class LuruFuseFunction(Function):
    @staticmethod
    def forward(ctx, x, theta):
        n, b, d = x.shape
        output = luru_fuse_cuda.forward(x, theta)
        
        ctx.save_for_backward(theta)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        n, b, d = grad_output.shape
        theta, = ctx.saved_tensors
        theta = theta.contiguous()
        grad_output = grad_output.contiguous()
        grad_x = luru_fuse_cuda.backward(grad_output, theta)

        return grad_x, None