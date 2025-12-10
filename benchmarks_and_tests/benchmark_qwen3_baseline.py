#!/usr/bin/env python3
"""
Baseline Benchmark for Qwen3-30B-A3B-GPTQ-Int4 Model

This script measures the baseline performance of Qwen3-30B-A3B-GPTQ-Int4:
1. Prefill latency (processing initial prompt)
2. Per-token generation latency (autoregressive generation)
3. Attention layer breakdown (QKV projection, output projection)

Target Model: https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4
- Type: MoE (Mixture of Experts) model
- Total Parameters: 30.5B (3.3B activated per token)
- Layers: 48
- Attention Heads: 32 Q heads, 4 KV heads (GQA)
- Experts: 128 total, 8 activated
- Context Length: 32,768 native (131,072 with YaRN)
- Quantization: GPTQ 4-bit

Usage:
    # Default: benchmark Qwen3-30B-A3B-GPTQ-Int4
    python benchmark_qwen3_baseline.py
    
    # With custom model path
    python benchmark_qwen3_baseline.py --model-path /local/path/to/model
    
    # Custom configuration
    python benchmark_qwen3_baseline.py --seq 1024 --gen-len 256
    
    # Save results
    python benchmark_qwen3_baseline.py --output results.json

Prerequisites:
    pip install transformers>=4.51.0 accelerate auto-gptq
    
    Note: transformers>=4.51.0 is REQUIRED for Qwen3-MoE models!
    With older versions you'll get: KeyError: 'qwen3_moe'
    
Output:
    - Prefill latency (ms) and throughput (tokens/sec)
    - Per-token generation latency (ms/token)
    - QKV projection timing breakdown
    - Memory usage statistics
"""

import argparse
import time
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import gc
import numpy as np


# Default model: Qwen3-30B-A3B-GPTQ-Int4
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    model_path: str = DEFAULT_MODEL
    batch_size: int = 1
    seq_len: int = 512          # Input sequence length (prefill)
    gen_len: int = 128          # Number of tokens to generate
    warmup_iters: int = 3
    bench_iters: int = 10
    quantization: str = "gptq"  # Qwen3-30B-A3B-GPTQ-Int4 uses GPTQ
    
    # Model dimensions for Qwen3-30B-A3B
    # These will be auto-populated from model config, but defaults match the model
    hidden_dim: int = 2048      # Qwen3-30B-A3B hidden size
    num_q_heads: int = 32       # 32 Q heads
    num_kv_heads: int = 4       # 4 KV heads (GQA)
    head_dim: int = 128
    num_layers: int = 48        # 48 layers
    num_experts: int = 128      # MoE: 128 total experts
    num_activated_experts: int = 8  # MoE: 8 activated per token


# =============================================================================
# GPU Utilities
# =============================================================================

def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated
        }
    return {}


# =============================================================================
# Model Loading
# =============================================================================

def check_transformers_version():
    """Check that transformers version is >= 4.51.0 for Qwen3-MoE support"""
    import transformers
    version = transformers.__version__
    major, minor, *_ = version.split('.')
    if int(major) < 4 or (int(major) == 4 and int(minor) < 51):
        print(f"WARNING: transformers version {version} detected!")
        print("Qwen3-MoE models require transformers>=4.51.0")
        print("Please upgrade: pip install transformers>=4.51.0")
        print("Otherwise you'll get: KeyError: 'qwen3_moe'")
        return False
    return True


def load_model(model_path: str, quantization: str = "gptq"):
    """
    Load Qwen3-30B-A3B-GPTQ-Int4 model.
    
    Args:
        model_path: Path to model or HuggingFace model ID
                   Default: Qwen/Qwen3-30B-A3B-GPTQ-Int4
        quantization: "gptq" (default for this model), "awq", or "none"
    
    Returns:
        model, tokenizer, model_config
    """
    from transformers import AutoTokenizer, AutoConfig
    
    # Check transformers version
    check_transformers_version()
    
    print(f"\n{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"Quantization: {quantization}")
    print(f"{'='*60}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"Model config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    
    # Detect quantization type if auto
    if quantization == "auto":
        quantization = detect_quantization(model_path, config)
        print(f"Detected quantization: {quantization}")
    
    # Load model based on quantization type
    model = None
    
    if quantization == "gptq":
        try:
            from transformers import AutoModelForCausalLM
            print("Loading GPTQ model via transformers...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        except Exception as e:
            print(f"Transformers GPTQ failed: {e}")
            try:
                from auto_gptq import AutoGPTQForCausalLM
                print("Loading GPTQ model via auto-gptq...")
                model = AutoGPTQForCausalLM.from_quantized(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    use_safetensors=True,
                )
            except ImportError:
                print("auto-gptq not installed. Install with: pip install auto-gptq")
                raise
    
    elif quantization == "awq":
        try:
            from awq import AutoAWQForCausalLM
            print("Loading AWQ model...")
            model = AutoAWQForCausalLM.from_quantized(
                model_path,
                fuse_layers=True,
                trust_remote_code=True,
            )
        except ImportError:
            print("autoawq not installed. Install with: pip install autoawq")
            raise
    
    else:  # none or fp16
        from transformers import AutoModelForCausalLM
        print("Loading FP16 model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    
    model.eval()
    
    # Print memory usage
    mem_info = get_gpu_memory_info()
    print(f"GPU Memory after loading: {mem_info.get('allocated_gb', 0):.2f} GB")
    
    return model, tokenizer, config


def detect_quantization(model_path: str, config) -> str:
    """Detect quantization type from model config or files"""
    import os
    
    # Check config for quantization info
    if hasattr(config, 'quantization_config'):
        qconfig = config.quantization_config
        if hasattr(qconfig, 'quant_method'):
            return qconfig.quant_method.lower()
    
    # Check for GPTQ/AWQ in model name or path
    path_lower = model_path.lower()
    if 'gptq' in path_lower or 'int4' in path_lower:
        return 'gptq'
    if 'awq' in path_lower:
        return 'awq'
    
    # Check for quantize_config.json
    if os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, 'quantize_config.json')):
            return 'gptq'
        if os.path.exists(os.path.join(model_path, 'quant_config.json')):
            return 'awq'
    
    return 'none'


# =============================================================================
# Benchmarking Functions
# =============================================================================

def benchmark_prefill(
    model, 
    tokenizer, 
    config: BenchmarkConfig,
    prompt: str = None
) -> Dict[str, Any]:
    """
    Benchmark prefill latency (initial prompt processing).
    
    This measures the time to process the entire input sequence in parallel,
    which is the first phase of inference before generation begins.
    """
    print("\n" + "=" * 60)
    print("PREFILL BENCHMARK")
    print("=" * 60)
    
    device = next(model.parameters()).device
    
    # Create input of desired length
    if prompt is None:
        prompt = "Hello, I am a helpful AI assistant. " * (config.seq_len // 10 + 1)
    
    # Tokenize and truncate/pad to exact length
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=config.seq_len,
        truncation=True,
        padding="max_length",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    actual_seq_len = input_ids.shape[1]
    print(f"Input shape: {input_ids.shape}")
    print(f"Actual sequence length: {actual_seq_len}")
    
    # Warmup
    print(f"\nWarming up ({config.warmup_iters} iterations)...")
    for _ in range(config.warmup_iters):
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask, use_cache=False)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking prefill ({config.bench_iters} iterations)...")
    times = []
    for _ in range(config.bench_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    times = np.array(times)
    
    result = {
        "name": "prefill",
        "seq_len": actual_seq_len,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "samples": times.tolist(),
        "tokens_per_second": actual_seq_len / (np.mean(times) / 1000),
    }
    
    print(f"\nPrefill Results:")
    print(f"  Latency: {result['mean_ms']:.2f} ± {result['std_ms']:.2f} ms")
    print(f"  Throughput: {result['tokens_per_second']:.0f} tokens/sec")
    print(f"  Per-token: {result['mean_ms']/actual_seq_len*1000:.2f} µs/token")
    
    return result


def benchmark_generation(
    model,
    tokenizer,
    config: BenchmarkConfig,
    prompt: str = None
) -> Dict[str, Any]:
    """
    Benchmark per-token generation latency (autoregressive decoding).
    
    This measures the actual generation speed including KV cache usage,
    which is the main latency users experience during text generation.
    """
    print("\n" + "=" * 60)
    print("GENERATION BENCHMARK (Per-Token Latency)")
    print("=" * 60)
    
    device = next(model.parameters()).device
    
    # Use a shorter prompt for generation
    if prompt is None:
        prompt = "Write a detailed explanation of"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    
    print(f"Prompt length: {prompt_len} tokens")
    print(f"Generating: {config.gen_len} tokens")
    
    # Warmup
    print(f"\nWarming up...")
    with torch.no_grad():
        _ = model.generate(
            input_ids,
            max_new_tokens=min(10, config.gen_len),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking generation ({config.bench_iters} iterations)...")
    times = []
    tokens_generated = []
    
    for i in range(config.bench_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=config.gen_len,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        
        num_new_tokens = outputs.shape[1] - prompt_len
        times.append(elapsed)
        tokens_generated.append(num_new_tokens)
    
    times = np.array(times)
    tokens_generated = np.array(tokens_generated)
    
    # Calculate per-token latency
    avg_tokens = np.mean(tokens_generated)
    avg_time = np.mean(times)
    per_token_ms = avg_time / avg_tokens if avg_tokens > 0 else 0
    
    result = {
        "name": "generation",
        "gen_len": config.gen_len,
        "actual_tokens_generated": float(np.mean(tokens_generated)),
        "total_time_ms": float(avg_time),
        "per_token_ms": per_token_ms,
        "tokens_per_second": avg_tokens / (avg_time / 1000) if avg_time > 0 else 0,
        "std_ms": float(np.std(times)),
        "samples": times.tolist(),
    }
    
    print(f"\nGeneration Results:")
    print(f"  Total time: {result['total_time_ms']:.2f} ± {result['std_ms']:.2f} ms")
    print(f"  Per-token latency: {result['per_token_ms']:.2f} ms/token")
    print(f"  Throughput: {result['tokens_per_second']:.1f} tokens/sec")
    
    return result


def benchmark_layer_breakdown(
    model,
    tokenizer, 
    config: BenchmarkConfig
) -> Dict[str, Any]:
    """
    Benchmark individual attention layer components.
    
    Measures QKV projection and output projection timing separately
    to identify optimization targets.
    """
    print("\n" + "=" * 60)
    print("LAYER BREAKDOWN (Attention Components)")
    print("=" * 60)
    
    device = next(model.parameters()).device
    
    # Find attention modules
    attention_modules = find_attention_modules(model)
    
    if not attention_modules:
        print("Warning: Could not find attention modules for detailed profiling")
        return {}
    
    print(f"Found {len(attention_modules)} attention layers")
    
    # Create input
    prompt = "Hello " * (config.seq_len // 2)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=config.seq_len,
        truncation=True,
        padding="max_length",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Profile
    results = profile_attention_layers(model, attention_modules, input_ids, attention_mask, config)
    
    return results


def find_attention_modules(model) -> List[Tuple[str, Any]]:
    """Find attention modules in the model"""
    attention_modules = []
    
    for name, module in model.named_modules():
        module_name = module.__class__.__name__.lower()
        # Common attention class names
        if any(x in module_name for x in ['attention', 'attn']) and 'self' in name.lower():
            attention_modules.append((name, module))
        elif 'qwen' in module_name and 'attention' in module_name:
            attention_modules.append((name, module))
    
    # Fallback: look for modules with q_proj, k_proj, v_proj
    if not attention_modules:
        for name, module in model.named_modules():
            if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                attention_modules.append((name, module))
    
    return attention_modules


def profile_attention_layers(
    model, 
    attention_modules: List[Tuple[str, Any]], 
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: BenchmarkConfig
) -> Dict[str, Any]:
    """Profile attention layers with timing"""
    
    layer_times = {}
    
    if not attention_modules:
        return {}
    
    first_attn_name, first_attn = attention_modules[0]
    print(f"\nProfiling attention layer: {first_attn_name}")
    
    # Check what components exist
    has_q_proj = hasattr(first_attn, 'q_proj')
    
    print(f"\nMeasuring layer timing ({config.bench_iters} iterations)...")
    
    # Get hidden states by running through embedding
    with torch.no_grad():
        if hasattr(model, 'model'):
            embed = model.model.embed_tokens(input_ids)
        elif hasattr(model, 'transformer'):
            embed = model.transformer.wte(input_ids)
        else:
            # Fallback: measure full model forward
            print("Using full model timing (cannot isolate layers)")
            times = []
            for _ in range(config.warmup_iters):
                with torch.no_grad():
                    _ = model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            
            for _ in range(config.bench_iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(input_ids, attention_mask=attention_mask)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            times = np.array(times)
            num_layers = len(attention_modules)
            return {
                'total_forward_ms': float(np.mean(times)),
                'per_layer_estimate_ms': float(np.mean(times) / num_layers) if num_layers > 0 else 0,
                'num_layers': num_layers,
            }
    
    hidden_states = embed
    
    # Profile QKV projection
    if has_q_proj:
        print("  Profiling Q, K, V projections...")
        qkv_times = []
        
        for _ in range(config.warmup_iters):
            with torch.no_grad():
                _ = first_attn.q_proj(hidden_states)
                _ = first_attn.k_proj(hidden_states)
                _ = first_attn.v_proj(hidden_states)
        torch.cuda.synchronize()
        
        for _ in range(config.bench_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                q = first_attn.q_proj(hidden_states)
                k = first_attn.k_proj(hidden_states)
                v = first_attn.v_proj(hidden_states)
            torch.cuda.synchronize()
            qkv_times.append((time.perf_counter() - start) * 1000)
        
        qkv_times = np.array(qkv_times)
        layer_times['qkv_proj'] = {
            'mean_ms': float(np.mean(qkv_times)),
            'std_ms': float(np.std(qkv_times)),
            'min_ms': float(np.min(qkv_times)),
            'samples': qkv_times.tolist(),
        }
        print(f"    QKV projection: {layer_times['qkv_proj']['mean_ms']:.3f} ± {layer_times['qkv_proj']['std_ms']:.3f} ms")
    
    # Profile output projection
    if hasattr(first_attn, 'o_proj'):
        print("  Profiling output projection...")
        batch, seq, _ = hidden_states.shape
        
        if hasattr(first_attn, 'num_heads'):
            num_heads = first_attn.num_heads
        else:
            num_heads = config.num_q_heads if config.num_q_heads > 0 else 28
        
        head_dim = hidden_states.shape[-1] // num_heads if num_heads > 0 else config.head_dim
        
        dummy_attn_out = torch.randn(
            batch, seq, num_heads * head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        o_times = []
        for _ in range(config.warmup_iters):
            with torch.no_grad():
                _ = first_attn.o_proj(dummy_attn_out)
        torch.cuda.synchronize()
        
        for _ in range(config.bench_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = first_attn.o_proj(dummy_attn_out)
            torch.cuda.synchronize()
            o_times.append((time.perf_counter() - start) * 1000)
        
        o_times = np.array(o_times)
        layer_times['o_proj'] = {
            'mean_ms': float(np.mean(o_times)),
            'std_ms': float(np.std(o_times)),
            'samples': o_times.tolist(),
        }
        print(f"    Output projection: {layer_times['o_proj']['mean_ms']:.3f} ± {layer_times['o_proj']['std_ms']:.3f} ms")
    
    return layer_times


# =============================================================================
# Summary and Output
# =============================================================================

def print_summary(
    prefill_result: Dict[str, Any],
    gen_result: Dict[str, Any],
    layer_result: Dict[str, Any],
    config: BenchmarkConfig
):
    """Print comprehensive benchmark summary"""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_path}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Q heads: {config.num_q_heads}, KV heads: {config.num_kv_heads}")
    print(f"  Layers: {config.num_layers}")
    if config.num_experts > 0:
        print(f"  MoE: {config.num_experts} experts, {config.num_activated_experts} activated per token")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  Generation length: {config.gen_len}")
    
    print(f"\n{'Prefill Performance':-^60}")
    if prefill_result:
        print(f"  Latency: {prefill_result['mean_ms']:.2f} ± {prefill_result['std_ms']:.2f} ms")
        print(f"  Throughput: {prefill_result['tokens_per_second']:.0f} tokens/sec")
        print(f"  Per-token: {prefill_result['mean_ms']/prefill_result['seq_len']*1000:.2f} µs/token")
    
    print(f"\n{'Generation Performance (Per-Token Latency)':-^60}")
    if gen_result:
        print(f"  Total time ({gen_result['actual_tokens_generated']:.0f} tokens): {gen_result['total_time_ms']:.2f} ms")
        print(f"  Per-token latency: {gen_result['per_token_ms']:.2f} ms/token")
        print(f"  Throughput: {gen_result['tokens_per_second']:.1f} tokens/sec")
    
    print(f"\n{'Attention Layer Breakdown':-^60}")
    if layer_result:
        if 'qkv_proj' in layer_result:
            print(f"  QKV projection (single layer): {layer_result['qkv_proj']['mean_ms']:.3f} ms")
        if 'o_proj' in layer_result:
            print(f"  Output projection (single layer): {layer_result['o_proj']['mean_ms']:.3f} ms")
        if 'per_layer_estimate_ms' in layer_result:
            print(f"  Per-layer estimate: {layer_result['per_layer_estimate_ms']:.3f} ms")
            print(f"  Total layers: {layer_result['num_layers']}")
    
    print(f"\n{'Key Metrics for Optimization':-^60}")
    if gen_result:
        print(f"  ⭐ Per-token latency: {gen_result['per_token_ms']:.2f} ms")
        print(f"     (This is your main optimization target for generation)")
    if prefill_result:
        print(f"  ⭐ Prefill latency: {prefill_result['mean_ms']:.2f} ms for {prefill_result['seq_len']} tokens")
        print(f"     (This is your optimization target for prompt processing)")
    
    if layer_result and 'qkv_proj' in layer_result:
        qkv_time = layer_result['qkv_proj']['mean_ms']
        total_qkv = qkv_time * config.num_layers
        print(f"\n  QKV projection total ({config.num_layers} layers): {total_qkv:.2f} ms")
        if prefill_result:
            pct = total_qkv / prefill_result['mean_ms'] * 100
            print(f"     ({pct:.1f}% of prefill time)")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-30B-A3B-GPTQ-Int4 Baseline Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Target Model: {DEFAULT_MODEL}
- MoE model: 30.5B total params, 3.3B activated
- 48 layers, 32 Q heads, 4 KV heads (GQA)
- 128 experts, 8 activated per token
- GPTQ 4-bit quantization

Examples:
  # Default: benchmark Qwen3-30B-A3B-GPTQ-Int4 from HuggingFace
  python benchmark_qwen3_baseline.py
  
  # With local model path:
  python benchmark_qwen3_baseline.py --model-path /local/path/to/model
  
  # Custom configuration:
  python benchmark_qwen3_baseline.py --seq 1024 --gen-len 256
  
  # Save results:
  python benchmark_qwen3_baseline.py --output results.json

Prerequisites:
  pip install transformers>=4.51.0 accelerate auto-gptq
        """
    )
    
    # Model (optional, defaults to Qwen3-30B-A3B-GPTQ-Int4)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL,
                        help=f"Path to model or HuggingFace ID (default: {DEFAULT_MODEL})")
    
    # Model options
    parser.add_argument("--quantization", type=str, default="gptq",
                        choices=["auto", "gptq", "awq", "none"],
                        help="Quantization type (default: gptq for Qwen3-30B-A3B-GPTQ-Int4)")
    
    # Benchmark options
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq", type=int, default=512, help="Sequence length for prefill")
    parser.add_argument("--gen-len", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    
    # Output
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt for benchmarking")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("QWEN3-30B-A3B-GPTQ-Int4 BASELINE BENCHMARK")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Check transformers version
    import transformers
    print(f"Transformers: {transformers.__version__}")
    
    # Build config
    config = BenchmarkConfig(
        model_path=args.model_path,
        batch_size=args.batch,
        seq_len=args.seq,
        gen_len=args.gen_len,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        quantization=args.quantization,
    )
    
    # Load model
    model, tokenizer, model_config = load_model(args.model_path, args.quantization)
    
    # Update config with actual model dimensions
    if hasattr(model_config, 'hidden_size'):
        config.hidden_dim = model_config.hidden_size
    if hasattr(model_config, 'num_attention_heads'):
        config.num_q_heads = model_config.num_attention_heads
    if hasattr(model_config, 'num_key_value_heads'):
        config.num_kv_heads = model_config.num_key_value_heads
    if hasattr(model_config, 'num_hidden_layers'):
        config.num_layers = model_config.num_hidden_layers
    
    # MoE-specific config
    if hasattr(model_config, 'num_experts'):
        config.num_experts = model_config.num_experts
    if hasattr(model_config, 'num_experts_per_tok'):
        config.num_activated_experts = model_config.num_experts_per_tok
    
    print(f"\nModel dimensions:")
    print(f"  Hidden: {config.hidden_dim}")
    print(f"  Q heads: {config.num_q_heads}, KV heads: {config.num_kv_heads}")
    print(f"  Layers: {config.num_layers}")
    if config.num_experts > 0:
        print(f"  MoE: {config.num_experts} experts, {config.num_activated_experts} activated")
    
    # Run benchmarks
    prefill_result = benchmark_prefill(model, tokenizer, config, args.prompt)
    gen_result = benchmark_generation(model, tokenizer, config, args.prompt)
    layer_result = benchmark_layer_breakdown(model, tokenizer, config)
    
    # Print summary
    print_summary(prefill_result, gen_result, layer_result, config)
    
    # Save results if requested
    if args.output:
        all_results = {
            'config': {
                'model_path': config.model_path,
                'batch_size': config.batch_size,
                'seq_len': config.seq_len,
                'gen_len': config.gen_len,
                'hidden_dim': config.hidden_dim,
                'num_q_heads': config.num_q_heads,
                'num_kv_heads': config.num_kv_heads,
                'num_layers': config.num_layers,
                'num_experts': config.num_experts,
                'num_activated_experts': config.num_activated_experts,
            },
            'prefill': prefill_result,
            'generation': gen_result,
            'layer_breakdown': layer_result,
        }
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
