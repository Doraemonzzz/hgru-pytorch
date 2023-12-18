
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF_AND_BF16(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BF16(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                        \
      TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF_AND_BF16(__VA_ARGS__))


// CUDA kernel for forward pass
template <typename scalar_t>
__global__ void luru_fuse_forward_kernel(
    const scalar_t* x,
    const scalar_t* theta,
    scalar_t* output, int64_t n, int64_t b, int64_t d, int64_t e) {
    // Compute the global indices of the current thread
    // batch
    int64_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    // feature
    int64_t idz = blockIdx.x * blockDim.x + threadIdx.x;

    if ((idy < b) && (idz < e)) {
        scalar_t hidden_state1 = 0;
        scalar_t hidden_state2 = 0;
        scalar_t hidden_state1_next, hidden_state2_next;
        float theta_ = float(theta[idz]);
        float cos_ = cos(theta_);
        float sin_ = sin(theta_);

        for (int64_t idx = 0; idx < n; ++idx) {
            int64_t index1 = idx * b * d + idy * d + 2 * idz;
            int64_t index2 = index1 + 1;
            scalar_t x1 = x[index1];
            scalar_t x2 = x[index2];

            hidden_state1_next = float(cos_) * float(hidden_state1) - float(sin_) * float(hidden_state2) + float(x1);
            hidden_state2_next = float(sin_) * float(hidden_state1) + float(cos_) * float(hidden_state2) + float(x2);

            output[index1] = hidden_state1_next;
            output[index2] = hidden_state2_next;

            hidden_state1 = hidden_state1_next;
            hidden_state2 = hidden_state2_next;
        }
    }
}

// CUDA kernel for backward pass
template <typename scalar_t>
__global__ void luru_fuse_backward_kernel(
    const scalar_t* grad_output, 
    const scalar_t* theta, 
    scalar_t* grad_x,
    int64_t n, int64_t b, int64_t d, int64_t e) {
    // batch
    int64_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    // feature
    int64_t idz = blockIdx.x * blockDim.x + threadIdx.x;

    if ((idy < b) && (idz < e)) {
        scalar_t grad_hidden_state1 = 0;
        scalar_t grad_hidden_state2 = 0;
        scalar_t grad_hidden_state1_next, grad_hidden_state2_next;
        float theta_ = float(theta[idz]);
        float cos_ = cos(theta_);
        float sin_ = sin(theta_);

        for (int64_t idx = n - 1; idx >= 0; --idx) {
            int64_t index1 = idx * b * d + idy * d + 2 * idz;
            int64_t index2 = index1 + 1;
            scalar_t g1 = grad_output[index1];
            scalar_t g2 = grad_output[index2];

            grad_hidden_state1_next = float(cos_) * float(grad_hidden_state1) + float(sin_) * float(grad_hidden_state2) + float(g1);
            grad_hidden_state2_next = -float(sin_) * float(grad_hidden_state1) + float(cos_) * float(grad_hidden_state2) + float(g2);

            grad_x[index1] = grad_hidden_state1_next;
            grad_x[index2] = grad_hidden_state2_next;

            grad_hidden_state1 = grad_hidden_state1_next;
            grad_hidden_state2 = grad_hidden_state2_next;
        }
    }
}

torch::Tensor luru_fuse_forward_cuda(torch::Tensor &x, torch::Tensor &theta) {
    auto output = torch::zeros_like(x);
    const int64_t n = x.size(0);
    const int64_t b = x.size(1);
    const int64_t d = x.size(2);
    const int64_t e = d / 2;

    dim3 threads(128, 8);
    dim3 blocks((e + threads.x - 1) / threads.x, (b + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BF16(x.scalar_type(), "luru_fuse_forward_cuda", ([&] {
        luru_fuse_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            theta.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), 
            n, b, d, e
        );
    }));

    return output;
}

torch::Tensor luru_fuse_backward_cuda(torch::Tensor &grad_output, torch::Tensor &theta) {
    auto grad_x = torch::zeros_like(grad_output);

    const int64_t n = grad_output.size(0);
    const int64_t b = grad_output.size(1);
    const int64_t d = grad_output.size(2);
    const int64_t e = d / 2;

    dim3 threads(128, 8);
    dim3 blocks((e + threads.x - 1) / threads.x, (b + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BF16(grad_output.scalar_type(), "luru_fuse_backward_cuda", ([&] {
        luru_fuse_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(), 
            theta.data_ptr<scalar_t>(), 
            grad_x.data_ptr<scalar_t>(), 
            n, b, d, e
        );
    }));

    return {grad_x};
}