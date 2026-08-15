#!/usr/bin/env python3
"""
Phase C: 调优 Sweep — 系统旋钮扫描
====================================
在 Unsloth BF16 上系统 sweep 关键旋钮，每个配置跑 50 步（效率指标已稳定）。

Sweep 维度：
  1. G (num_generations): 2, 4, 8  — 生成占比对 MFU 的影响
  2. 有效 batch: 1×4, 1×8, 2×4    — GEMM 尺寸对计算强度的影响
  3. gradient_checkpointing: on/off — 重计算 vs 显存 trade-off
  4. max_completion_length: 256, 512 — 序列长度对 attention 的影响

基准配置: G=4, batch=1×8, grad_ckpt=on, completion=512 (即 B1 配置)
总配置数: 10 (含基准重复验证)
预计耗时: 10 × ~25min ≈ 4h

用法：
  sg render -c "cd {REPO_ROOT} && python3 eval/run_c_sweep.py"
"""

import subprocess, sys, os, time, json
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


BASE_DIR = REPO_ROOT
MODEL = os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct")
RESULTS = os.path.join(BASE_DIR, "results/efficiency")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train/train_gsm8k_grpo.py")

# ═══════════════════════════════════════════════════════════
# Sweep 配置定义
# ═══════════════════════════════════════════════════════════
# 格式: (name, extra_args_list)
# 所有配置共用: --framework unsloth --no-4bit --max-steps 50 --seed 42

SWEEP_CONFIGS = [
    # --- 维度 1: G (num_generations) ---
    ("C_G2_b1x8_ckpt_512",
     ["--num-generations", "2", "--per-device-batch", "1", "--grad-accum", "8",
      "--max-completion-length", "512"]),
    # G=4 b1x8 即 B1 基准，已有 100 步数据，这里跑 50 步做对齐验证
    ("C_G4_b1x8_ckpt_512",
     ["--num-generations", "4", "--per-device-batch", "1", "--grad-accum", "8",
      "--max-completion-length", "512"]),
    ("C_G8_b1x8_ckpt_512",
     ["--num-generations", "8", "--per-device-batch", "1", "--grad-accum", "8",
      "--max-completion-length", "512"]),

    # --- 维度 2: 有效 batch ---
    ("C_G4_b1x4_ckpt_512",
     ["--num-generations", "4", "--per-device-batch", "1", "--grad-accum", "4",
      "--max-completion-length", "512"]),
    ("C_G4_b2x4_ckpt_512",
     ["--num-generations", "4", "--per-device-batch", "2", "--grad-accum", "4",
      "--max-completion-length", "512"]),

    # --- 维度 3: gradient checkpointing ---
    # 注：Unsloth 默认启用 grad_ckpt；此处对比关闭
    ("C_G4_b1x8_noCkpt_512",
     ["--num-generations", "4", "--per-device-batch", "1", "--grad-accum", "8",
      "--max-completion-length", "512", "--no-gradient-checkpointing"]),

    # --- 维度 4: max_completion_length ---
    ("C_G4_b1x8_ckpt_256",
     ["--num-generations", "4", "--per-device-batch", "1", "--grad-accum", "8",
      "--max-completion-length", "256"]),
    ("C_G4_b1x8_ckpt_1024",
     ["--num-generations", "4", "--per-device-batch", "1", "--grad-accum", "8",
      "--max-completion-length", "1024"]),

    # --- 组合最优候选 (基于前面结果选择) ---
    ("C_G2_b2x4_ckpt_256",
     ["--num-generations", "2", "--per-device-batch", "2", "--grad-accum", "4",
      "--max-completion-length", "256"]),
    ("C_G2_b1x8_ckpt_256",
     ["--num-generations", "2", "--per-device-batch", "1", "--grad-accum", "8",
      "--max-completion-length", "256"]),
]

COMMON_ARGS = [
    "--model-path", MODEL,
    "--no-sft-adapter",
    "--framework", "unsloth",
    "--no-4bit",
    "--max-steps", "50",
    "--seed", "42",
]


def run_experiment(name: str, extra_args: list) -> bool:
    """运行单个 sweep 实验，返回是否成功。"""
    out_dir = os.path.join(RESULTS, name)
    log_path = os.path.join(RESULTS, f"{name}.log")

    # 跳过已完成
    report = os.path.join(out_dir, "efficiency_report.json")
    if os.path.exists(report):
        print(f"  ⏭️  跳过 {name}（已完成）")
        return True

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        *COMMON_ARGS, *extra_args,
        "--output", out_dir,
    ]
    print(f"\n{'═' * 60}")
    print(f"  🚀 开始: {name}")
    print(f"  命令: {' '.join(cmd[-12:])}")
    print(f"{'═' * 60}")

    t0 = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=BASE_DIR)
    elapsed = time.time() - t0

    if proc.returncode == 0 and os.path.exists(report):
        print(f"  ✅ 完成: {name} ({elapsed/60:.1f} min)")
        return True
    else:
        print(f"  ❌ 失败: {name} (rc={proc.returncode}, {elapsed/60:.1f} min)")
        # 打印最后几行日志辅助诊断
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.read().replace('\r', '\n').strip().split('\n')
            print("  最后 5 行:")
            for l in lines[-5:]:
                print(f"    {l[:120]}")
        return False


def main():
    print("=" * 60)
    print("  Phase C: 调优 Sweep | Unsloth BF16 × Qwen2.5-3B × GSM8K")
    print(f"  配置数: {len(SWEEP_CONFIGS)} | 每组 50 步")
    print("=" * 60)

    results = []
    total_t0 = time.time()

    for i, (name, args) in enumerate(SWEEP_CONFIGS, 1):
        print(f"\n[{i}/{len(SWEEP_CONFIGS)}]", end="")
        ok = run_experiment(name, args)
        results.append((name, ok))

    total_elapsed = time.time() - total_t0
    print(f"\n{'═' * 60}")
    print(f"  Phase C Sweep 完成 | 总耗时: {total_elapsed/60:.1f} min")
    print(f"{'═' * 60}")
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")

    # 生成汇总
    summary_path = os.path.join(RESULTS, "C_sweep_summary.json")
    sweep_results = []
    for name, ok in results:
        report_path = os.path.join(RESULTS, name, "efficiency_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                r = json.load(f)
                s = r.get("summary", r)
            sweep_results.append({
                "name": name,
                "tokens_per_s": s.get("tokens_per_s"),
                "mfu_practical_pct": s.get("mfu_practical_pct"),
                "peak_mem_gb": s.get("peak_mem_gb"),
                "s_per_step": s.get("s_per_step"),
                "valid_steps": s.get("valid_steps"),
            })
    with open(summary_path, "w") as f:
        json.dump(sweep_results, f, ensure_ascii=False, indent=2)
    print(f"\n  汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
