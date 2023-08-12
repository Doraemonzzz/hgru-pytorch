#include <torch/extension.h>

std::vector<torch::Tensor> hgru_forward_cuda(
    torch::Tensor& x_real,
    torch::Tensor& x_imag,
    torch::Tensor& lambda_real,
    torch::Tensor& lambda_imag
);

std::vector<torch::Tensor> hgru_backward_cuda(
    torch::Tensor& x_real,
    torch::Tensor& x_imag,
    torch::Tensor& lambda_real,
    torch::Tensor& lambda_imag,
    torch::Tensor& hidden_states_real,
    torch::Tensor& hidden_states_imag,
    torch::Tensor& grad_output_real,
    torch::Tensor& grad_output_imag
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hgru_forward_cuda, "Hgru forward (CUDA)");
  m.def("backward", &hgru_backward_cuda, "Hgru backward (CUDA)");
}
