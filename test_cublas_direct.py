#!/usr/bin/env python3
"""
Test cuBLAS directly with the same parameters as our kernel
to verify if the GEMM call is correct
"""

import torch
import time
import ctypes

# Load cuBLAS library
try:
    cublas = ctypes.CDLL('libcublas.so.12')
except:
    try:
        cublas = ctypes.CDLL('libcublas.so.11')
    except:
        cublas = ctypes.CDLL('libcublas.so')

def test_cublas_gemm_parameters():
    """
    Test if our cuBLAS parameters are correct by comparing with torch.matmul
    """
    batch_size = 4
    seqlen = 512
    M = batch_size * seqlen  # 2048
    K = 3584
    N = 5120
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # Create test matrices (row-major, same as our kernel)
    A = torch.randn(M, K, dtype=dtype, device=device)  # hidden_states [M, K]
    B = torch.randn(K, N, dtype=dtype, device=device)  # qkv_weight [K, N]
    
    # Expected result using PyTorch
    C_expected = torch.matmul(A, B)  # [M, N]
    
    print("=" * 80)
    print("Testing cuBLAS GEMM Parameters")
    print("=" * 80)
    print(f"Matrix sizes: A=[{M}, {K}], B=[{K}, {N}], C=[{M}, {N}]")
    print(f"A shape: {A.shape}, stride: {A.stride()}")
    print(f"B shape: {B.shape}, stride: {B.stride()}")
    print(f"C shape: {C_expected.shape}, stride: {C_expected.stride()}")
    print()
    
    # Analyze memory layout
    print("Memory layout analysis:")
    print(f"  A is {'row-major' if A.stride(1) == 1 else 'column-major'}")
    print(f"  B is {'row-major' if B.stride(1) == 1 else 'column-major'}")
    print(f"  A stride: {A.stride()} (row_stride={A.stride(0)}, col_stride={A.stride(1)})")
    print(f"  B stride: {B.stride()} (row_stride={B.stride(0)}, col_stride={B.stride(1)})")
    print()
    
    # What our kernel does (WRONG leading dimensions):
    print("Current kernel parameters (INCORRECT):")
    print(f"  cublasGemmEx(")
    print(f"    CUBLAS_OP_N, CUBLAS_OP_N,")
    print(f"    N={N}, M={M}, K={K},")
    print(f"    qkv_weight, lda={N},  ← WRONG! Should be K")
    print(f"    hidden_states, ldb={K},  ← WRONG! Should be K") 
    print(f"    qkv_buf, ldc={N}  ← WRONG! Should be N")
    print(f"  )")
    print()
    
    # Correct parameters for row-major C = A @ B:
    print("CORRECT parameters for row-major matrices:")
    print("  For C[M,N] = A[M,K] @ B[K,N] in row-major:")
    print("  We compute C^T[N,M] = B^T[N,K] @ A^T[K,M] in column-major")
    print()
    print(f"  cublasGemmEx(")
    print(f"    CUBLAS_OP_N, CUBLAS_OP_N,")
    print(f"    N={N}, M={M}, K={K},")
    print(f"    B (as B^T), lda={N},  ← Leading dim of B^T in col-major = N")
    print(f"    A (as A^T), ldb={K},  ← Leading dim of A^T in col-major = K")
    print(f"    C (as C^T), ldc={N}   ← Leading dim of C^T in col-major = N")
    print(f"  )")
    print()
    
    # Benchmark torch.matmul
    for _ in range(10):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / num_iters * 1000
    
    flops = 2 * M * N * K
    gflops = flops / (torch_time * 1e6)
    
    print(f"PyTorch torch.matmul performance:")
    print(f"  Time: {torch_time:.3f} ms")
    print(f"  Performance: {gflops:.2f} GFLOPS")
    print()
    
    # Recommendation
    print("=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    print("The kernel's cuBLAS call has INCORRECT leading dimensions!")
    print()
    print("For row-major matrices [M,K] @ [K,N] = [M,N]:")
    print("  - Leading dimension is the stride to next row")
    print("  - For A[M,K] row-major: lda = K (elements per row)")
    print("  - For B[K,N] row-major: ldb = N (elements per row)")
    print("  - For C[M,N] row-major: ldc = N (elements per row)")
    print()
    print("But when we interpret as column-major (transposed):")
    print("  - A[M,K] row-major = A^T[K,M] column-major, lda = K")
    print("  - B[K,N] row-major = B^T[N,K] column-major, ldb = N")
    print("  - C[M,N] row-major = C^T[N,M] column-major, ldc = N")
    print()
    print("FIX: Change line 78 in qkv_fused_optimized.cu:")
    print("  FROM: N,  // leading dimension")
    print("  TO:   K,  // leading dimension (for B^T[N,K], ld = K)")
    print("=" * 80)

if __name__ == "__main__":
    test_cublas_gemm_parameters()

