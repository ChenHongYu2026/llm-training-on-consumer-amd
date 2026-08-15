#!/usr/bin/env python3
"""
Phase B1 多 seed 补充实验
========================
补 seed123 + seed7（跳过 4-bit，结论已确定）。
预计耗时：4 runs × ~50min ≈ 3.3h

用法：
  sg render -c "cd {REPO_ROOT} && python3 eval/run_multiseed.py"
"""

import subprocess, sys, os, time
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


BASE_DIR = REPO_ROOT
MODEL = os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct")
RESULTS = os.path.join(BASE_DIR, "results/efficiency")

EXPERIMENTS = [
    # (name, framework_args, seed)
    ("B1_unsloth_bf16_100steps_seed123", ["--framework", "unsloth", "--no-4bit"], 123),
    ("B1_hf_bf16_100steps_seed123",      ["--framework", "hf"],                   123),
    ("B1_unsloth_bf16_100steps_seed7",   ["--framework", "unsloth", "--no-4bit"], 7),
    ("B1_hf_bf16_100steps_seed7",        ["--framework", "hf"],                   7),
]

COMMON = [
    "--model-path", MODEL,
    "--no-sft-adapter",
    "--max-steps", "100",
    "--num-generations", "4",
    "--per-device-batch", "1",
    "--grad-accum", "8",
]

def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 60)
    print("Phase B1 多 seed 补充 (seed123 + seed7)")
    print("=" * 60)

    for name, fw_args, seed in EXPERIMENTS:
        out_dir = os.path.join(RESULTS, name)
        report = os.path.join(out_dir, "efficiency_report.json")
        if os.path.exists(report):
            print(f"⏭️  跳过 {name}（已有结果）")
            continue

        print(f"\n🚀 {name} ...")
        cmd = [
            sys.executable, os.path.join(BASE_DIR, "train/train_gsm8k_grpo.py"),
            *fw_args, *COMMON,
            "--seed", str(seed),
            "--output", out_dir,
        ]
        env = os.environ.copy()
        env["HF_ENDPOINT"] = "https://hf-mirror.com"

        t0 = time.perf_counter()
        r = subprocess.run(cmd, env=env, cwd=BASE_DIR)
        elapsed = time.perf_counter() - t0
        status = "✅" if r.returncode == 0 else "❌"
        print(f"{status} {name} | {elapsed/60:.1f} min")

    print("\n" + "=" * 60)
    print("多 seed 实验全部完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
