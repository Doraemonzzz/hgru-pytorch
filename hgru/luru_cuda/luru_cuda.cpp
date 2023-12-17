#include <torch/extension.h>

torch::Tensor luru_forward_cuda(torch::Tensor &x);

torch::Tensor luru_backward_cuda(torch::Tensor &grad_output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &luru_forward_cuda, "Luru forward (CUDA)");
  m.def("backward", &luru_backward_cuda, "Luru backward (CUDA)");
}