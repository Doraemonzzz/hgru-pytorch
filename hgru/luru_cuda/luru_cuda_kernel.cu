
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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
__global__ void luru_forward_kernel(
    const scalar_t* x,
    scalar_t* output, int64_t n, int64_t b, int64_t d) {
    // Compute the global indices of the current thread
    // batch
    int64_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    // feature
    int64_t idz = blockIdx.x * blockDim.x + threadIdx.x;

    if (idy < b && idz < d) {
        scalar_t hidden_state = 0;
        for (int64_t idx = 0; idx < n; ++idx) {
            int64_t index = idx * b * d + idy * d + idz;
            hidden_state += x[index];
            output[index] = hidden_state;
        }
    }
}

// CUDA kernel for backward pass
template <typename scalar_t>
__global__ void luru_backward_kernel(
    const scalar_t* grad_output, 
    scalar_t* grad_x,
    int64_t n, int64_t b, int64_t d) {
    // batch
    int64_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    // feature
    int64_t idz = blockIdx.x * blockDim.x + threadIdx.x;

    if (idy < b && idz < d) {
        scalar_t grad_hidden_state = 0;
        for (int64_t idx = n - 1; idx >= 0; --idx) {
            int64_t index = idx * b * d + idy * d + idz;
            grad_hidden_state += grad_output[index];
            grad_x[index] = grad_hidden_state;
        }
    }
}

torch::Tensor luru_forward_cuda(torch::Tensor &x) {
    auto output = torch::zeros_like(x);
    const int64_t n = x.size(0);
    const int64_t b = x.size(1);
    const int64_t d = x.size(2);

    dim3 threads(128, 8);
    dim3 blocks((d + threads.x - 1) / threads.x, (b + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BF16(x.scalar_type(), "luru_forward_cuda", ([&] {
        luru_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), n, b, d);
    }));

    return output;
}

torch::Tensor luru_backward_cuda(torch::Tensor &grad_output) {
    auto grad_x = torch::zeros_like(grad_output);

    const int64_t n = grad_output.size(0);
    const int64_t b = grad_output.size(1);
    const int64_t d = grad_output.size(2);

    dim3 threads(128, 8);
    dim3 blocks((d + threads.x - 1) / threads.x, (b + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BF16(grad_output.scalar_type(), "luru_backward_cuda", ([&] {
        luru_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(), grad_x.data_ptr<scalar_t>(), n, b, d);
    }));

    return {grad_x};
}