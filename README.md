# QKV Fusion - Optimized Attention Projections

High-performance CUDA kernels for fused Q, K, V projections in transformer attention layers with FlashAttention-2 integration.

## Features

- ✅ **2x Speedup** over PyTorch baseline (3 separate `nn.Linear`)
- ✅ **Grouped Query Attention (GQA)** support (e.g., 32 Q heads, 4 KV heads)
- ✅ **cuBLAS + Tensor Cores** for optimal GEMM performance
- ✅ **FlashAttention-2** compatible outputs
- ✅ **FP16** optimized for modern GPUs (Ampere/Hopper)

## Quick Start

### Installation

```bash
# Requirements: CUDA 11.8+, PyTorch 2.1+, GCC 9+
pip install -e . --no-build-isolation
```

### Basic Usage

```python
from qkv_fusion import qkv_fused_forward_optimized
from qkv_fusion.weight_utils import prepare_fused_qkv_weights

# Prepare fused weights
fused_weight, fused_bias = prepare_fused_qkv_weights(q_weight, k_weight, v_weight)

# Run optimized kernel
q, k, v = qkv_fused_forward_optimized(
    hidden_states,      # [batch, seq, hidden_dim]
    fused_weight,       # [hidden_dim, qkv_out_dim]
    fused_bias,
    num_q_heads=32,
    num_kv_heads=4,
    head_dim=128
)
# Output: Q [batch, 32, seq, 128], K [batch, 4, seq, 128], V [batch, 4, seq, 128]
```

### FlashAttention Integration

```python
from qkv_fusion import qkv_projection_for_flash_attention
from flash_attn import flash_attn_func

# Get Q, K, V ready for FlashAttention
q, k, v = qkv_projection_for_flash_attention(
    hidden_states, fused_weight, fused_bias,
    num_q_heads=32, num_kv_heads=4, head_dim=128
)

# Run FlashAttention
attn_output = flash_attn_func(q, k, v, causal=True)
```

## Architecture

**Optimization Strategy:**
1. **Concatenate weights offline:** `W_qkv = [W_q | W_k | W_v]`
2. **Single cuBLAS GEMM:** `QKV_buf = hidden_states @ W_qkv` (uses tensor cores)
3. **Custom split kernel:** Split + bias + transpose in one pass

**Performance:**
- Qwen3-7B config (batch=4, seq=512): **2.08x faster** than PyTorch
- Reduces 3 GEMM launches → 1 GEMM + 1 lightweight kernel
- Better memory reuse and tensor core utilization

## Testing

```bash
# Compile
cd qkv_fusion
pip install -e . --no-build-isolation -v

# Run tests
python test_optimized.py
```

**Expected output:**
```
✓ PASS: Optimized kernel matches PyTorch baseline!
✓ SUCCESS: Optimized kernel is 2.08x faster!
```

## Requirements

- **GPU:** NVIDIA Ampere (SM 80) or Hopper (SM 90)
- **CUDA:** 11.8 or 12.x
- **PyTorch:** 2.1+
- **Python:** 3.9-3.11
- **FlashAttention:** 2.x (optional, for integration)

## API Reference

### Low-Level Kernel

```python
qkv_fused_forward_optimized(
    hidden_states,      # [batch, seq, hidden_dim]
    qkv_fused_weight,   # [hidden_dim, qkv_out_dim]
    qkv_fused_bias,     # [qkv_out_dim] or None
    num_q_heads,
    num_kv_heads,
    head_dim
) -> (Q, K, V)
```

### FlashAttention Helper

```python
qkv_projection_for_flash_attention(
    hidden_states,
    qkv_fused_weight,
    qkv_fused_bias,
    num_q_heads,
    num_kv_heads,
    head_dim
) -> (Q, K, V)  # Transposed for FlashAttention
```

### Complete Attention

```python
fused_qkv_attention(
    hidden_states,
    qkv_fused_weight,
    qkv_fused_bias,
    num_q_heads,
    num_kv_heads,
    head_dim,
    causal=True
) -> attn_output
```

### PyTorch Module

```python
attn = FusedQKVAttention(
    hidden_dim=3584,
    num_q_heads=32,
    num_kv_heads=4,
    head_dim=128,
    bias=True,
    causal=True
)

output = attn(hidden_states)
```

## Project Structure

```
qkv_fusion/
├── csrc/
│   ├── qkv_fused_api.cpp           # PyTorch bindings
│   ├── qkv_fused_params.h          # Parameter structures
│   └── kernels/
│       ├── qkv_fused_fp16.cu       # Baseline kernel
│       ├── qkv_fused_optimized.cu  # Optimized cuBLAS kernel
│       └── qkv_fused_int4.cu       # INT4 support (future)
├── qkv_fusion/
│   ├── __init__.py                 # Main interface
│   ├── qkv_interface.py            # High-level API
│   └── weight_utils.py             # Weight preparation
├── examples/
│   └── flash_attention_integration.py
├── setup.py                        # Build configuration
└── test_optimized.py               # Tests & benchmarks
```

## Performance Benchmarks

Tested on NVIDIA H800 GPU (Qwen3-7B configuration):

| Batch | Seq Len | PyTorch | Optimized | Speedup |
|-------|---------|---------|-----------|---------|
| 1     | 128     | 0.85 ms | 0.52 ms   | 1.6x    |
| 2     | 256     | 1.45 ms | 0.78 ms   | 1.9x    |
| 4     | 512     | 2.50 ms | 1.20 ms   | 2.1x    |
| 8     | 1024    | 4.80 ms | 2.15 ms   | 2.2x    |

## Citation

If you use this code in your research, please cite:

```bibtex
@software{qkv_fusion,
  author = {Chen, Hilary},
  title = {QKV Fusion: Optimized Attention Projections for Transformers},
  year = {2025},
  url = {https://github.com/hilaryKChen/qkv_fusion}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)'s fused QKV approach
- Built on [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) architecture
- Uses NVIDIA cuBLAS for optimal GEMM performance

## Contact

- GitHub: [@hilaryKChen](https://github.com/hilaryKChen)
- Issues: [GitHub Issues](https://github.com/hilaryKChen/qkv_fusion/issues)
