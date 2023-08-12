#include <torch/extension.h>

torch::Tensor hgru_real_forward_cuda(
    torch::Tensor &x,
    torch::Tensor &lambda);

std::vector<torch::Tensor> hgru_real_backward_cuda(
    torch::Tensor &x,
    torch::Tensor &lambda,
    torch::Tensor &hidden_states,
    torch::Tensor &grad_output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hgru_real_forward_cuda, "HgruReal forward (CUDA)");
  m.def("backward", &hgru_real_backward_cuda, "HgruReal backward (CUDA)");
}

// TORCH_LIBRARY(hgru_real_cuda, m) {
//     m.def("forward", hgru_real_forward_cuda);
//     m.def("backward", hgru_real_backward_cuda);
// }