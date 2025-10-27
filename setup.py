from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='qkv_fusion',
    ext_modules=[
        CUDAExtension(
            name='qkv_fusion_cuda',
            sources=[
                'csrc/qkv_fused_api.cpp',
                # 'csrc/kernels/qkv_fused_fp16.cu',  # Not used
                'csrc/kernels/qkv_fused_optimized.cu',  # Phase 2: Optimized with cuBLAS
                'csrc/kernels/qkv_fused_lightweight.cu',  # Phase 3: Lightweight GEMM+Bias
                'csrc/kernels/qkv_fused_int4.cu',       # Phase 4: INT4 quantization
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--use_fast_math',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-gencode', 'arch=compute_90,code=sm_90',  # H800 (Hopper)
                    '-gencode', 'arch=compute_80,code=sm_80',  # A100 (Ampere) for compatibility
                ]
            },
            include_dirs=[
                f'{this_dir}/csrc',
                # Include flash-attn headers for CUTLASS
                f'{this_dir}/../flash-attention/csrc/flash_attn',
                f'{this_dir}/../flash-attention/csrc/cutlass/include',
            ],
            libraries=['cublas'],  # Link cuBLAS library
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    packages=['qkv_fusion'],
)