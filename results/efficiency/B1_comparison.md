# Phase B1 框架横向对比结果

> 生成时间: 2026-07-22 17:35
> 硬件: AMD Radeon RX 7900 XTX 24GB (RDNA3/gfx1100)
> 模型: Qwen2.5-3B-Instruct (3.086B params)
> Workload: GSM8K GRPO (G=4, batch=1×8, completion≤512)

| 配置 | tokens/s | s/step | MFU(理论)% | MFU(实践)% | VRAM(GB) | GPU busy% | 有效步/总步 |
|---|---|---|---|---|---|---|---|
| B1_hf_bf16_100steps_seed42 | 1305.5 | 144.9 | 19.68 | 23.81 | 8.0 | 0.0 | 20/20 (inv=0) |
| B1_unsloth_4bit_100steps_seed42 | 809.4 | 227.8 | 12.20 | 14.76 | 10.2 | 0.0 | 20/20 (inv=0) |
| B1_unsloth_bf16_100steps_seed42 | 1334.4 | 140.5 | 20.12 | 24.33 | 13.9 | 0.0 | 20/20 (inv=0) |

## 加速比分析

基准: B1_hf_bf16_100steps_seed42 (1305.5 tokens/s)

| 配置 | 相对吞吐比 | 解读 |
|---|---|---|
| B1_hf_bf16_100steps_seed42 | 1.00× | 持平 |
| B1_unsloth_4bit_100steps_seed42 | 0.62× | 慢 38% |
| B1_unsloth_bf16_100steps_seed42 | 1.02× | 持平 |

## 测量有效性审计

| 配置 | 有效步 | 无效步 | 有效率 | 判定 |
|---|---|---|---|---|
| B1_hf_bf16_100steps_seed42 | 20 | 0 | 100% | ✅ PASS |
| B1_unsloth_4bit_100steps_seed42 | 20 | 0 | 100% | ✅ PASS |
| B1_unsloth_bf16_100steps_seed42 | 20 | 0 | 100% | ✅ PASS |