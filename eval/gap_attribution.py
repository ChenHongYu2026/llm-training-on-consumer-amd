#!/usr/bin/env python3
"""
Phase B2: 鸿沟归因实验 — 组件叠加法
====================================
从 GEMM 实践峰值 (101.53 TFLOPS) 出发，逐步叠加训练组件，
量化各组件对 MFU 的拖累，产出瀑布图数据。

叠加层级：
  L0: 纯 GEMM 峰值 (已有: 101.53 TFLOPS)
  L1: 模型前向推理 (generate) — 自回归 decode，带宽受限
  L2: SFT 训练步 (无 gradient checkpointing) — 前向+反向
  L3: SFT 训练步 (有 gradient checkpointing) — 重计算开销
  L4: GRPO 训练 (含生成+训练) — 完整 RL 循环

每个层级测量：achieved TFLOPS, tokens/s, 显存, GPU busy%
鸿沟 = L0 - L4 = 各组件损失之和

用法：
  sg render -c "cd {REPO_ROOT} && python3 eval/gap_attribution.py"
"""

import os, sys, json, time
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


MODEL_PATH = os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct")
OUTPUT_DIR = REPO_ROOT + "/results/efficiency/B2_gap_attribution"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 从 config 计算参数量
cfg = AutoConfig.from_pretrained(MODEL_PATH)
h = cfg.hidden_size
n = cfg.num_hidden_layers
nh = cfg.num_attention_heads
nkv = getattr(cfg, 'num_key_value_heads', nh)
inter = cfg.intermediate_size
vocab = cfg.vocab_size
hd = h // nh
attn_per_layer = h * (nh * hd) + 2 * h * (nkv * hd) + (nh * hd) * h
mlp_per_layer = h * inter * 3
NUM_PARAMS = vocab * h + n * (attn_per_layer + mlp_per_layer)
print(f"模型参数量: {NUM_PARAMS/1e9:.3f}B")

# GEMM 实践峰值
GEMM_PEAK = 101.53  # TFLOPS (已有)
THEO_PEAK = 122.8

results = {"model_params": NUM_PARAMS, "gemm_peak_tflops": GEMM_PEAK, "levels": {}}


def measure_tflops(num_tokens: int, elapsed_s: float, mode: str = "train") -> float:
    """计算 achieved TFLOPS。train: 6ND, inference: 2ND。"""
    coeff = 6.0 if mode == "train" else 2.0
    flops = coeff * NUM_PARAMS * num_tokens
    return flops / elapsed_s / 1e12


# ═══════════════════════════════════════════════════════════
# L1: 前向推理 (generate)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("L1: 前向推理 (autoregressive generate)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager",
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 生成固定长度序列测吞吐
prompt = "Solve step by step: What is 123 * 456 + 789?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
prompt_len = inputs["input_ids"].shape[1]

GEN_TOKENS = 256
WARMUP = 3
ITERS = 10

# warmup
for _ in range(WARMUP):
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=GEN_TOKENS, do_sample=False)
torch.cuda.synchronize()

# 测量
t0 = time.perf_counter()
for _ in range(ITERS):
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=GEN_TOKENS, do_sample=False)
torch.cuda.synchronize()
gen_elapsed = time.perf_counter() - t0

gen_tokens_total = GEN_TOKENS * ITERS
gen_tps = gen_tokens_total / gen_elapsed
gen_tflops = measure_tflops(gen_tokens_total, gen_elapsed, mode="inference")
gen_mem = torch.cuda.max_memory_allocated() / 1e9

results["levels"]["L1_inference"] = {
    "desc": "Autoregressive generate (batch=1, 256 new tokens)",
    "tokens": gen_tokens_total,
    "elapsed_s": round(gen_elapsed, 2),
    "tokens_per_s": round(gen_tps, 1),
    "achieved_tflops": round(gen_tflops, 3),
    "mfu_practical_pct": round(gen_tflops / GEMM_PEAK * 100, 3),
    "peak_mem_gb": round(gen_mem, 2),
    "note": "2ND (forward only, bandwidth-bound)",
}
print(f"  生成吞吐: {gen_tps:.1f} tok/s | TFLOPS={gen_tflops:.3f} | MFU={gen_tflops/GEMM_PEAK*100:.2f}%")
print(f"  显存: {gen_mem:.1f} GB")

# 释放推理模型
del model
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# ═══════════════════════════════════════════════════════════
# L2: SFT 训练步 (无 gradient checkpointing)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("L2: SFT 训练 (无 gradient checkpointing)")
print("=" * 60)

from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager",
)
lora_config = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_disable()
model.train()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-5
)

# 构造固定训练 batch
# 注意：RX 7900 XTX 24GB 显存约束，batch=4+seq=512 会 OOM
# 安全配置：batch=2, seq=256（约 12-15GB 显存）
SEQ_LEN = 256
BATCH = 2
dummy_ids = torch.randint(0, vocab, (BATCH, SEQ_LEN), device="cuda")
dummy_labels = dummy_ids.clone()

# warmup
for _ in range(3):
    out = model(input_ids=dummy_ids, labels=dummy_labels)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()

# 测量
STEPS = 20
t0 = time.perf_counter()
for _ in range(STEPS):
    out = model(input_ids=dummy_ids, labels=dummy_labels)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
l2_elapsed = time.perf_counter() - t0

l2_tokens = BATCH * SEQ_LEN * STEPS
l2_tflops = measure_tflops(l2_tokens, l2_elapsed, mode="train")
l2_mem = torch.cuda.max_memory_allocated() / 1e9

results["levels"]["L2_sft_no_ckpt"] = {
    "desc": f"SFT train (batch={BATCH}, seq={SEQ_LEN}, no grad_ckpt)",
    "tokens": l2_tokens,
    "elapsed_s": round(l2_elapsed, 2),
    "tokens_per_s": round(l2_tokens / l2_elapsed, 1),
    "achieved_tflops": round(l2_tflops, 3),
    "mfu_practical_pct": round(l2_tflops / GEMM_PEAK * 100, 3),
    "peak_mem_gb": round(l2_mem, 2),
    "note": "6ND (forward+backward, no recompute)",
}
print(f"  训练吞吐: {l2_tokens/l2_elapsed:.1f} tok/s | TFLOPS={l2_tflops:.2f} | MFU={l2_tflops/GEMM_PEAK*100:.2f}%")
print(f"  显存: {l2_mem:.1f} GB")

del model, optimizer
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# ═══════════════════════════════════════════════════════════
# L3: SFT 训练步 (有 gradient checkpointing)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("L3: SFT 训练 (有 gradient checkpointing)")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager",
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.train()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-5
)

# warmup
for _ in range(3):
    out = model(input_ids=dummy_ids, labels=dummy_labels)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()

# 测量
t0 = time.perf_counter()
for _ in range(STEPS):
    out = model(input_ids=dummy_ids, labels=dummy_labels)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
l3_elapsed = time.perf_counter() - t0

l3_tokens = BATCH * SEQ_LEN * STEPS
l3_tflops = measure_tflops(l3_tokens, l3_elapsed, mode="train")
l3_mem = torch.cuda.max_memory_allocated() / 1e9

results["levels"]["L3_sft_with_ckpt"] = {
    "desc": f"SFT train (batch={BATCH}, seq={SEQ_LEN}, grad_ckpt=True)",
    "tokens": l3_tokens,
    "elapsed_s": round(l3_elapsed, 2),
    "tokens_per_s": round(l3_tokens / l3_elapsed, 1),
    "achieved_tflops": round(l3_tflops, 3),
    "mfu_practical_pct": round(l3_tflops / GEMM_PEAK * 100, 3),
    "peak_mem_gb": round(l3_mem, 2),
    "note": "6ND + recompute overhead",
}
print(f"  训练吞吐: {l3_tokens/l3_elapsed:.1f} tok/s | TFLOPS={l3_tflops:.2f} | MFU={l3_tflops/GEMM_PEAK*100:.2f}%")
print(f"  显存: {l3_mem:.1f} GB")

del model, optimizer
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════
# L4: GRPO (引用已有 B1 数据)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("L4: GRPO 完整训练 (引用 B1 Unsloth BF16 数据)")
print("=" * 60)

b1_path = REPO_ROOT + "/results/efficiency/B1_unsloth_bf16_100steps_seed42/efficiency_report.json"
with open(b1_path) as f:
    b1 = json.load(f)
    b1s = b1.get("summary", b1)

l4_tflops = b1s["achieved_tflops"]
results["levels"]["L4_grpo_full"] = {
    "desc": "GRPO full (G=4, gen+train, Unsloth BF16)",
    "tokens": b1s["total_tokens"],
    "elapsed_s": round(b1s["total_time_s"], 2),
    "tokens_per_s": round(b1s["tokens_per_s"], 1),
    "achieved_tflops": round(l4_tflops, 3),
    "mfu_practical_pct": round(l4_tflops / GEMM_PEAK * 100, 3),
    "peak_mem_gb": b1s["peak_mem_gb"],
    "note": "6ND for train steps; generation amortized",
}
print(f"  GRPO 效率: {b1s['tokens_per_s']:.1f} tok/s | TFLOPS={l4_tflops:.2f} | MFU={l4_tflops/GEMM_PEAK*100:.2f}%")

# ═══════════════════════════════════════════════════════════
# 汇总：瀑布图数据
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("鸿沟归因瀑布图")
print("=" * 60)

levels_ordered = ["L1_inference", "L2_sft_no_ckpt", "L3_sft_with_ckpt", "L4_grpo_full"]
print(f"\n  L0 GEMM 峰值      : {GEMM_PEAK:.2f} TFLOPS (100%)")
prev = GEMM_PEAK
waterfall = [{"level": "L0_gemm_peak", "tflops": GEMM_PEAK, "delta": 0, "mfu_pct": 100.0}]

for lv in levels_ordered:
    d = results["levels"][lv]
    tf = d["achieved_tflops"]
    delta = prev - tf
    mfu = tf / GEMM_PEAK * 100
    waterfall.append({"level": lv, "tflops": tf, "delta": round(delta, 3), "mfu_pct": round(mfu, 2)})
    print(f"  {lv:<22}: {tf:>7.2f} TFLOPS | Δ=-{delta:.2f} | MFU={mfu:.1f}%")
    prev = tf

total_gap = GEMM_PEAK - results["levels"]["L4_grpo_full"]["achieved_tflops"]
print(f"\n  总鸿沟: {GEMM_PEAK:.2f} → {results['levels']['L4_grpo_full']['achieved_tflops']:.2f} = -{total_gap:.2f} TFLOPS ({total_gap/GEMM_PEAK*100:.1f}%)")

results["waterfall"] = waterfall
results["total_gap_tflops"] = round(total_gap, 3)
results["total_gap_pct"] = round(total_gap / GEMM_PEAK * 100, 1)

# 保存
out_path = os.path.join(OUTPUT_DIR, "gap_attribution.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n✅ 结果已保存: {out_path}")
