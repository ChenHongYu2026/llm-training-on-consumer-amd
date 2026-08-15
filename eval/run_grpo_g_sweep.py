#!/usr/bin/env python3
"""
实验②: GRPO G 路生成加速 sweep (Arithmetic Intensity in the Real Training Loop)
==============================================================================
目的: 把 batch 吞吐发现接回**真实 GRPO 训练循环**——GRPO 的 num_generations G
      就是生成阶段的 batch 维度 (每个 prompt 复制 G 份批量生成)。验证:
      随 G 增大, 生成阶段 batch↑ → 算术强度↑ → 训练吞吐(tokens/s)/MFU 上升、
      s/step 亚线性增长 (与纯解码 batch_throughput 的机理一致)。

方法: 顺序调用已验证的 train/train_gsm8k_grpo.py, 固定除 G 外的全部参数,
      每档 ~25 步, HF+Qwen2.5-3B, 由 EfficiencyProfiler 采集 tokens/s / MFU / s/step。
      逐档 try/except 隔离 (G=16 若 OOM 不影响其余档)。

用法:
  sg render -c "cd {REPO_ROOT} && python3 -u eval/run_grpo_g_sweep.py"
"""

import os
import sys
import json
import time
import subprocess
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


BASE_DIR = REPO_ROOT
MODEL = os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct")
RESULTS = os.path.join(BASE_DIR, "results/efficiency/grpo_g_sweep")
os.makedirs(RESULTS, exist_ok=True)

# 扫 G (生成阶段 batch 维度); 其余参数固定
G_VALUES = [2, 4, 8, 16]
MAX_STEPS = 25
COMMON_ARGS = [
    "--framework", "hf",
    "--model-path", MODEL,
    "--no-sft-adapter",
    "--max-steps", str(MAX_STEPS),
    "--max-completion-length", "256",
    "--max-prompt-length", "256",
    "--per-device-batch", "1",
    "--grad-accum", "8",
    "--seed", "42",
]


def run_one(g: int) -> dict:
    """跑单个 G 档, 返回其 efficiency_report + timing 摘要。"""
    out_dir = os.path.join(RESULTS, f"G{g}")
    report_path = os.path.join(out_dir, "efficiency_report.json")

    if os.path.exists(report_path):
        print(f"⏭️  跳过 G={g}（已有 efficiency_report.json）")
    else:
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            sys.executable, os.path.join(BASE_DIR, "train/train_gsm8k_grpo.py"),
            "--num-generations", str(g),
            "--output", out_dir,
            *COMMON_ARGS,
        ]
        env = os.environ.copy()
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"\n{'='*60}\n🚀 G={g} | {MAX_STEPS} 步 | completion=256\n{'='*60}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, cwd=BASE_DIR)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"  ✗ G={g} 退出码 {proc.returncode}")
            return {"G": g, "error": f"returncode={proc.returncode}"}
        print(f"  ✓ G={g} 完成, 墙钟 {elapsed:.0f}s")

    # 读取效率报告
    if not os.path.exists(report_path):
        return {"G": g, "error": "no efficiency_report.json"}
    with open(report_path) as f:
        rep = json.load(f)
    timing_path = os.path.join(out_dir, "timing.json")
    s_per_step = None
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            s_per_step = json.load(f).get("s_per_step")
    return {
        "G": g,
        "tokens_per_s": rep.get("tokens_per_s"),
        "mfu_practical_pct": rep.get("mfu_practical_pct"),
        "mfu_theoretical_pct": rep.get("mfu_theoretical_pct"),
        "achieved_tflops": rep.get("achieved_tflops"),
        "s_per_step": rep.get("s_per_step") or s_per_step,
        "peak_mem_gb": rep.get("peak_mem_gb"),
        "valid_steps": rep.get("valid_steps"),
        "invalid_steps": rep.get("invalid_steps"),
        "gpu_busy_mean": (rep.get("gpu_busy", {}) or {}).get("gpu_busy_pct", {}).get("mean"),
    }


def main():
    print("=" * 60)
    print("实验②: GRPO G 路生成加速 sweep (真实训练循环)")
    print("=" * 60)
    print(f"  G 档: {G_VALUES} | 每档 {MAX_STEPS} 步 | HF + Qwen2.5-3B | completion=256")

    summary = []
    for g in G_VALUES:
        try:
            summary.append(run_one(g))
        except Exception as e:
            print(f"  ✗ G={g} 异常: {e}")
            summary.append({"G": g, "error": str(e)})
        time.sleep(10)  # 档间冷却

    # scaling: tokens/s 相对 G=2 基线
    base = next((r for r in summary if r.get("G") == G_VALUES[0] and r.get("tokens_per_s")), None)
    base_tps = base.get("tokens_per_s") if base else None
    for r in summary:
        if r.get("tokens_per_s") and base_tps:
            r["tps_vs_G2"] = round(r["tokens_per_s"] / base_tps, 2)

    out = {
        "metadata": {
            "experiment": "GRPO G-way Generation Scaling (real training loop)",
            "gpu": "RX 7900 XTX (RDNA3, 960 GB/s)",
            "model": "Qwen2.5-3B-Instruct",
            "framework": "hf",
            "max_steps": MAX_STEPS,
            "max_completion_length": 256,
            "note": "num_generations G = 生成阶段 batch 维度; tokens/s 由 EfficiencyProfiler(6ND) 采集",
        },
        "results": summary,
    }
    path = os.path.join(RESULTS, "summary.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 汇总表
    print(f"\n{'═'*72}\n  GRPO G 路生成加速结果\n{'═'*72}")
    print(f"  {'G':>4}{'tokens/s':>12}{'MFU实践%':>10}{'s/step':>10}{'峰值GB':>9}{'tps/G2':>9}")
    for r in summary:
        if r.get("error"):
            print(f"  {r['G']:>4}{'ERROR':>12}  {r['error']}")
        else:
            print(f"  {r['G']:>4}{r.get('tokens_per_s', 0):>12}{r.get('mfu_practical_pct', 0):>10}"
                  f"{r.get('s_per_step', 0):>10}{r.get('peak_mem_gb', 0):>9}{r.get('tps_vs_G2', ''):>9}")
    print(f"\n  结果: {path}")


if __name__ == "__main__":
    main()
