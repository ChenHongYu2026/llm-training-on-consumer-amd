#!/usr/bin/env python3
"""
Phase B1 批量实验运行器
=====================
顺序运行三组配置（每组 100 步），生成完整对比数据。

用法（需要 GPU 权限）：
  sg render -c "cd {REPO_ROOT} && python3 eval/run_b1_experiments.py"

预计总时长：~7h（Unsloth BF16 ~2h + HF BF16 ~3h + Unsloth 4bit ~3.5h）
"""

import subprocess
import sys
import os
import time
import json
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


BASE_DIR = REPO_ROOT
MODEL = os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct")
RESULTS = os.path.join(BASE_DIR, "results/efficiency")

# 三组实验配置
EXPERIMENTS = [
    {
        "name": "B1_unsloth_bf16_100steps_seed42",
        "args": ["--framework", "unsloth", "--no-4bit"],
        "desc": "Unsloth BF16 (无量化)",
    },
    {
        "name": "B1_hf_bf16_100steps_seed42",
        "args": ["--framework", "hf"],
        "desc": "HF 原生 BF16",
    },
    {
        "name": "B1_unsloth_4bit_100steps_seed42",
        "args": ["--framework", "unsloth"],
        "desc": "Unsloth 4-bit (默认模式)",
    },
]

COMMON_ARGS = [
    "--model-path", MODEL,
    "--no-sft-adapter",
    "--max-steps", "100",
    "--num-generations", "4",
    "--per-device-batch", "1",
    "--grad-accum", "8",
    "--seed", "42",
]


def run_experiment(exp: dict) -> bool:
    """运行单个实验，返回是否成功。"""
    output_dir = os.path.join(RESULTS, exp["name"])

    # 跳过已完成的
    report_path = os.path.join(output_dir, "efficiency_report.json")
    if os.path.exists(report_path):
        print(f"⏭️  跳过 {exp['name']}（已有结果）")
        return True

    print(f"\n{'='*60}")
    print(f"🚀 开始: {exp['desc']}")
    print(f"   输出: {output_dir}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, os.path.join(BASE_DIR, "train/train_gsm8k_grpo.py"),
        *exp["args"], *COMMON_ARGS,
        "--output", output_dir,
    ]

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"

    t0 = time.perf_counter()
    result = subprocess.run(cmd, env=env, cwd=BASE_DIR,
                           capture_output=False, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        print(f"✅ {exp['name']} 完成 | 耗时 {elapsed/60:.1f} min")
        return True
    else:
        print(f"❌ {exp['name']} 失败 | returncode={result.returncode}")
        return False


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 60)
    print("Phase B1 框架横向对比 — 批量实验")
    print(f"模型: {MODEL}")
    print(f"配置: 100 steps × 3 组")
    print("=" * 60)

    results = []
    total_t0 = time.perf_counter()

    for exp in EXPERIMENTS:
        ok = run_experiment(exp)
        results.append((exp["name"], ok))

    total_elapsed = time.perf_counter() - total_t0

    # 汇总
    print(f"\n{'='*60}")
    print(f"全部完成 | 总耗时 {total_elapsed/3600:.1f}h")
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")

    # 运行对比分析
    print(f"\n📊 运行对比分析...")
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "eval/compare_efficiency.py"),
                    "--dir", RESULTS], cwd=BASE_DIR)


if __name__ == "__main__":
    main()
