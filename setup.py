import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_gpu_arch_flags():
    try:
        major = torch.cuda.get_device_capability()[0]
        return [f"-gencode=arch=compute_{major}0,code=sm_{major}0"]
    except Exception as e:
        print(f"Error while detecting GPU architecture: {e}")
        return []


arch_flags = get_gpu_arch_flags()

setup(
    name="hgru",
    version="0.0.0",
    url="https://github.com/Doraemonzzz/hgru-pytorch",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "hgru_cuda",
            sources=[
                "hgru/hgru_cuda/hgru_cuda_kernel.cu",
                "hgru/hgru_cuda/hgru_cuda.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++14", "-D_GLIBCXX_USE_CXX11_ABI=0"],
                "nvcc": ["-O2", "-std=c++14", "-D_GLIBCXX_USE_CXX11_ABI=0"]
                + arch_flags,
            },
        ),
        CUDAExtension(
            "hgru_real_cuda",
            sources=[
                "hgru/hgru_real_cuda/hgru_real_cuda_kernel.cu",
                "hgru/hgru_real_cuda/hgru_real_cuda.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++14", "-D_GLIBCXX_USE_CXX11_ABI=0"],
                "nvcc": ["-O2", "-std=c++14", "-D_GLIBCXX_USE_CXX11_ABI=0"]
                + arch_flags,
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=False),
    },
    install_requires=[
        "torch",
        "einops",
    ],
    keywords=[
        "artificial intelligence",
        "sequential model",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
