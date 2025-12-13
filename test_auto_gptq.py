from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

model_path = "/home/kchenbx/attention_optimization/data/models/qwen3-int4"

print("Testing config load...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("✓ Tokenizer OK")

print("Testing model init (no weights)...")
model = AutoGPTQForCausalLM.from_quantized(
    model_path,
    device="cpu",  # no GPU yet
    trust_remote_code=True,
    use_safetensors=True,
    warmup_triton=False,  # disable extra init
)
print("✓ Model structure loaded")