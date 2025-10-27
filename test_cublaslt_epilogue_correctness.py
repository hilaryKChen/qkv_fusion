#!/usr/bin/env python3
"""
Unit test: cuBLASLt fused QKV (bias epilogue + correctness)

Compares the optimized cuBLASLt path (single GEMM + bias epilogue if available)
against a PyTorch nn.Linear baseline for Q/K/V. Uses strict tolerances and
prints a concise log excerpt to verify layout/orders, ld's, M/K/N, chosen algo,
and whether epilogue was fused or fallback bias kernel was used.
"""

import torch
from qkv_fusion import qkv_fused_forward_optimized
from qkv_fusion.weight_utils import prepare_fused_qkv_weights


def run_test(batch_size=2, seqlen=128, hidden_dim=2048, num_q_heads=32, num_kv_heads=4, head_dim=128):
    device = torch.device("cuda:0")
    dtype = torch.float16

    torch.manual_seed(123)
    hidden_states = torch.randn(batch_size, seqlen, hidden_dim, device=device, dtype=dtype)

    # Baseline projections
    q_proj = torch.nn.Linear(hidden_dim, num_q_heads * head_dim, device=device, dtype=dtype)
    k_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, device=device, dtype=dtype)
    v_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, device=device, dtype=dtype)

    with torch.no_grad():
        q_ref = q_proj(hidden_states)
        k_ref = k_proj(hidden_states)
        v_ref = v_proj(hidden_states)

    # Reshape/transpose to [B, H, T, D]
    q_ref = q_ref.view(batch_size, seqlen, num_q_heads, head_dim).transpose(1, 2).contiguous()
    k_ref = k_ref.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    v_ref = v_ref.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2).contiguous()

    # Optimized fused path
    q_w = q_proj.weight.t().contiguous()
    k_w = k_proj.weight.t().contiguous()
    v_w = v_proj.weight.t().contiguous()
    fused_w, fused_b = prepare_fused_qkv_weights(q_w, k_w, v_w, q_proj.bias, k_proj.bias, v_proj.bias)

    q_opt, k_opt, v_opt = qkv_fused_forward_optimized(
        hidden_states, fused_w, fused_b,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )

    # Tolerances
    max_abs_tol = 1e-2
    mean_rel_tol = 1e-3

    # Metrics
    q_max_abs = (q_opt - q_ref).abs().max().item()
    k_max_abs = (k_opt - k_ref).abs().max().item()
    v_max_abs = (v_opt - v_ref).abs().max().item()

    q_mean_rel = ((q_opt - q_ref).abs() / (q_ref.abs() + 1e-5)).mean().item()
    k_mean_rel = ((k_opt - k_ref).abs() / (k_ref.abs() + 1e-5)).mean().item()
    v_mean_rel = ((v_opt - v_ref).abs() / (v_ref.abs() + 1e-5)).mean().item()

    # Non-zero guard: sample Q row
    sample_nonzero = q_opt[0, 0, 0, :].abs().sum().item() > 0

    print("=" * 80)
    print("cuBLASLt QKV epilogue correctness test")
    print("=" * 80)
    print(f"Shapes: A=[{batch_size*seqlen},{hidden_dim}], B=[{hidden_dim},{(num_q_heads+2*num_kv_heads)*head_dim}], D=[{batch_size*seqlen},{(num_q_heads+2*num_kv_heads)*head_dim}]")
    print(f"Max abs diffs: Q={q_max_abs:.6f} K={k_max_abs:.6f} V={v_max_abs:.6f}")
    print(f"Mean rel errs: Q={q_mean_rel:.6f} K={k_mean_rel:.6f} V={v_mean_rel:.6f}")
    print(f"Sample Q[0,0,0,:] non-zero: {sample_nonzero}")
    print("-" * 80)
    print("Note: Below, the C++ kernel prints layout orders, ld's, M/K/N, heuristic counts, workspace,\n"
          "      selected algo, and whether bias epilogue fused or fallback bias kernel was used.")
    print("-" * 80)

    ok = (
        q_max_abs <= max_abs_tol and k_max_abs <= max_abs_tol and v_max_abs <= max_abs_tol and
        q_mean_rel < mean_rel_tol and k_mean_rel < mean_rel_tol and v_mean_rel < mean_rel_tol and
        sample_nonzero
    )

    if ok:
        print("PASS: Lt path matches baseline within strict tolerances.")
    else:
        print("FAIL: Differences exceed tolerance or sample is zeroed.")

    return ok


if __name__ == "__main__":
    success = run_test()
    if not success:
        raise SystemExit(1)


