#include <torch/extension.h>

torch::Tensor luru_fuse_forward_cuda(torch::Tensor &x, torch::Tensor &theta);

torch::Tensor luru_fuse_backward_cuda(torch::Tensor &grad_output, torch::Tensor &theta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &luru_fuse_forward_cuda, "Luru fuse forward (CUDA)");
  m.def("backward", &luru_fuse_backward_cuda, "Luru fuse backward (CUDA)");
}