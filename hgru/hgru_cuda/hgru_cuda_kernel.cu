#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>


template <typename scalar_t>
__global__ void hgru_forward_kernel(
    const scalar_t* x_real, const scalar_t* x_imag,
    const scalar_t* lambda_real, const scalar_t* lambda_imag,
    scalar_t* output_real, scalar_t* output_imag,
    int64_t n, int64_t b, int64_t d
) {
    // Compute the global indices of the current thread
    // batch
    int64_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    // feature
    int64_t idz = blockIdx.x * blockDim.x + threadIdx.x;

    if (idy < b && idz < d) {
        scalar_t hidden_state_real = 0;
        scalar_t hidden_state_imag = 0;
        scalar_t hidden_state_real_next, hidden_state_imag_next;
        for (int64_t idx = 0; idx < n; ++idx) {
            int64_t index = idx * b * d + idy * d + idz;
            scalar_t x_r = x_real[index];
            scalar_t x_i = x_imag[index];
            scalar_t lambda_r = lambda_real[index];
            scalar_t lambda_i = lambda_imag[index];

            hidden_state_real_next = lambda_r * hidden_state_real - lambda_i * hidden_state_imag + x_r;
            hidden_state_imag_next = lambda_r * hidden_state_imag + lambda_i * hidden_state_real + x_i;

            output_real[index] = hidden_state_real_next;
            output_imag[index] = hidden_state_imag_next;

            hidden_state_real = hidden_state_real_next;
            hidden_state_imag = hidden_state_imag_next;
        }
    }
}

template <typename scalar_t>
__global__ void hgru_backward_kernel(
    const scalar_t* x_real, const scalar_t* x_imag,
    const scalar_t* lambda_real, const scalar_t* lambda_imag,
    const scalar_t* hidden_states_real, const scalar_t* hidden_states_imag,
    const scalar_t* grad_output_real, const scalar_t* grad_output_imag,
    scalar_t* grad_x_real, scalar_t* grad_x_imag,
    scalar_t* grad_lambda_real, scalar_t* grad_lambda_imag,
    int64_t n, int64_t b, int64_t d
) {
    // Compute the global indices of the current thread
    // batch
    int64_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    // feature
    int64_t idz = blockIdx.x * blockDim.x + threadIdx.x;

    if (idy < b && idz < d) {
        scalar_t grad_hidden_state_real = 0;
        scalar_t grad_hidden_state_imag = 0;
        for (int64_t idx = n - 1; idx >= 0; --idx) {
            int64_t index = idx * b * d + idy * d + idz;
            int64_t j = ((idx == n - 1) ? 0 : index + b * d);

            scalar_t grad_hidden_state_real_next = grad_output_real[index] + lambda_real[j] * grad_hidden_state_real + lambda_imag[j] * grad_hidden_state_imag;
            scalar_t grad_hidden_state_imag_next = grad_output_imag[index] + lambda_real[j] * grad_hidden_state_imag - lambda_imag[j] * grad_hidden_state_real;

            if (idx == 0) {
                grad_lambda_real[index] = 0;
                grad_lambda_imag[index] = 0;
            } else {
                scalar_t hidden_state_real_prev = hidden_states_real[index - b * d];
                scalar_t hidden_state_imag_prev = hidden_states_imag[index - b * d];
                grad_lambda_real[index] =  grad_hidden_state_real_next * hidden_state_real_prev + \
                                           grad_hidden_state_imag_next * hidden_state_imag_prev;
                grad_lambda_imag[index] = -grad_hidden_state_real_next * hidden_state_imag_prev + \
                                           grad_hidden_state_imag_next * hidden_state_real_prev;
            }
            grad_x_real[index] = grad_hidden_state_real_next;
            grad_x_imag[index] = grad_hidden_state_imag_next;

            grad_hidden_state_real = grad_hidden_state_real_next;
            grad_hidden_state_imag = grad_hidden_state_imag_next;
        }
    }
}

std::vector<torch::Tensor> hgru_forward_cuda(
    torch::Tensor& x_real,
    torch::Tensor& x_imag,
    torch::Tensor& lambda_real,
    torch::Tensor& lambda_imag
) {
    const auto n = x_real.size(0);
    const auto b = x_real.size(1);
    const auto d = x_real.size(2);

    auto output_real = torch::zeros_like(x_real);
    auto output_imag = torch::zeros_like(x_imag);

    dim3 threads(128, 4);
    dim3 blocks((d + threads.x - 1) / threads.x, (b + threads.y - 1) / threads.y);

    switch (x_real.type().scalarType()) {
        case torch::ScalarType::BFloat16:
            hgru_forward_kernel<at::BFloat16><<<blocks, threads>>>(
                x_real.data_ptr<at::BFloat16>(),
                x_imag.data_ptr<at::BFloat16>(),
                lambda_real.data_ptr<at::BFloat16>(),
                lambda_imag.data_ptr<at::BFloat16>(),
                output_real.data_ptr<at::BFloat16>(),
                output_imag.data_ptr<at::BFloat16>(),
                n, b, d
            );
            break;
        default:
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_real.scalar_type(), "hgru_forward_cuda", ([&] {
                hgru_forward_kernel<scalar_t><<<blocks, threads>>>(
                    x_real.data_ptr<scalar_t>(),
                    x_imag.data_ptr<scalar_t>(),
                    lambda_real.data_ptr<scalar_t>(),
                    lambda_imag.data_ptr<scalar_t>(),
                    output_real.data_ptr<scalar_t>(),
                    output_imag.data_ptr<scalar_t>(),
                    n, b, d
                );
            }));
    }


    return {output_real, output_imag};
}

std::vector<torch::Tensor> hgru_backward_cuda(
    torch::Tensor& x_real,
    torch::Tensor& x_imag,
    torch::Tensor& lambda_real,
    torch::Tensor& lambda_imag,
    torch::Tensor& hidden_states_real,
    torch::Tensor& hidden_states_imag,
    torch::Tensor& grad_output_real,
    torch::Tensor& grad_output_imag
) {
    const auto n = x_real.size(0);
    const auto b = x_real.size(1);
    const auto d = x_real.size(2);

    auto grad_x_real = torch::zeros_like(x_real);
    auto grad_x_imag = torch::zeros_like(x_imag);
    auto grad_lambda_real = torch::zeros_like(lambda_real);
    auto grad_lambda_imag = torch::zeros_like(lambda_imag);

    dim3 threads(128, 4);
    dim3 blocks((d + threads.x - 1) / threads.x, (b + threads.y - 1) / threads.y);

    switch (x_real.type().scalarType()) {
        case torch::ScalarType::BFloat16:
            hgru_backward_kernel<at::BFloat16><<<blocks, threads>>>(
                x_real.data_ptr<at::BFloat16>(), x_imag.data_ptr<at::BFloat16>(),
                lambda_real.data_ptr<at::BFloat16>(), lambda_imag.data_ptr<at::BFloat16>(),
                hidden_states_real.data_ptr<at::BFloat16>(), hidden_states_imag.data_ptr<at::BFloat16>(),
                grad_output_real.data_ptr<at::BFloat16>(), grad_output_imag.data_ptr<at::BFloat16>(),
                grad_x_real.data_ptr<at::BFloat16>(), grad_x_imag.data_ptr<at::BFloat16>(),
                grad_lambda_real.data_ptr<at::BFloat16>(), grad_lambda_imag.data_ptr<at::BFloat16>(),
                n, b, d
            );
            break;
        default:
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_real.scalar_type(), "hgru_backward_cuda", ([&] {
                hgru_backward_kernel<scalar_t><<<blocks, threads>>>(
                    x_real.data_ptr<scalar_t>(), x_imag.data_ptr<scalar_t>(),
                    lambda_real.data_ptr<scalar_t>(), lambda_imag.data_ptr<scalar_t>(),
                    hidden_states_real.data_ptr<scalar_t>(), hidden_states_imag.data_ptr<scalar_t>(),
                    grad_output_real.data_ptr<scalar_t>(), grad_output_imag.data_ptr<scalar_t>(),
                    grad_x_real.data_ptr<scalar_t>(), grad_x_imag.data_ptr<scalar_t>(),
                    grad_lambda_real.data_ptr<scalar_t>(), grad_lambda_imag.data_ptr<scalar_t>(),
                    n, b, d
                );
            }));
    }

    return {grad_x_real, grad_x_imag, grad_lambda_real, grad_lambda_imag};
}