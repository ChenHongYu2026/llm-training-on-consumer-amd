#!/usr/bin/env python3
"""
A.T2.1: G=4 吞吐重标定 — B_g 扫描 (DGBB 三参数重拟合)
============================================================
docs/33 §3 A.T2.1 + docs/34 预注册 (P-A2a: 函数形式不变, R²>0.95)

B_g 语义 (docs/19): 生成 batch = pdb × grad_accum
G=4 固定, B_b=4 固定 (pdb=4), 扫 B_g ∈ {32, 64, 128} (ga ∈ {8, 16, 32})
每档 25 步 (参考 run_grpo_g_sweep 口径), 顺序执行, 逐档 try/except 隔离

用法:
  sg render -c "cd {REPO_ROOT} && python3 -u eval/run_grpo_bg_sweep_g4.py"
产出: results/efficiency/grpo_bg_sweep_g4/{Bg32,Bg64,Bg128}/ + summary.json
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
RESULTS = os.path.join(BASE_DIR, "results/efficiency/grpo_bg_sweep_g4")
os.makedirs(RESULTS, exist_ok=True)

# G=4 固定; B_g 档 = (pdb=4, ga) 组合; B_b = pdb = 4
B_G_VALUES = [32, 64, 128]
GA_VALUES = {32: 8, 64: 16, 128: 32}   # B_g / pdb(4)
MAX_STEPS = 25


def run_one(bg: int) -> dict:
    ga = GA_VALUES[bg]
    out_dir = os.path.join(RESULTS, f"Bg{bg}")
    report_path = os.path.join(out_dir, "efficiency_report.json")

    if os.path.exists(report_path):
        print(f"⏭️  跳过 Bg={bg}（已有 efficiency_report.json）")
    else:
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            sys.executable, os.path.join(BASE_DIR, "train/train_gsm8k_grpo.py"),
            "--framework", "hf",
            "--model-path", MODEL,
            "--num-generations", "4",          # G=4
            "--per-device-batch", "4",          # B_b = pdb = 4
            "--grad-accum", str(ga),            # B_g = pdb × ga
            "--max-steps", str(MAX_STEPS),
            "--max-completion-length", "256",
            "--max-prompt-length", "256",
            "--seed", "42",
            "--output", out_dir,
        ]
        env = os.environ.copy()
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"\n{'='*60}\n🚀 G=4 Bg={bg} (pdb=4, ga={ga}) | {MAX_STEPS} 步\n{'='*60}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, cwd=BASE_DIR)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"  ✗ Bg={bg} 退出码 {proc.returncode}")
            return {"Bg": bg, "error": f"returncode={proc.returncode}"}
        print(f"  ✓ Bg={bg} 完成, 墙钟 {elapsed:.0f}s")

    if not os.path.exists(report_path):
        return {"Bg": bg, "error": "no efficiency_report.json"}
    with open(report_path) as f:
        rep = json.load(f)
    return {
        "Bg": bg,
        "B_b": 4,
        "G": 4,
        "tokens_per_s": rep.get("tokens_per_s"),
        "s_per_step": rep.get("s_per_step"),
        "mfu_practical_pct": rep.get("mfu_practical_pct"),
        "peak_mem_gb": rep.get("peak_mem_gb"),
        "valid_steps": rep.get("valid_steps"),
        "invalid_steps": rep.get("invalid_steps"),
    }


def main():
    print("=" * 60)
    print("A.T2.1: G=4 吞吐重标定 — B_g 扫描 (DGBB 三参数)")
    print("=" * 60)
    print(f"  B_g 档: {B_G_VALUES} (pdb=4, B_b=4) | 每档 {MAX_STEPS} 步 | G=4")

    summary = []
    for bg in B_G_VALUES:
        try:
            summary.append(run_one(bg))
        except Exception as e:
            print(f"  ✗ Bg={bg} 异常: {e}")
            summary.append({"Bg": bg, "error": str(e)})
        time.sleep(10)

    # DGBB 三参数拟合: T_step = a1 + a2·Bg + a3·(Bg/Bb)
    points = [(r["Bg"], r.get("s_per_step")) for r in summary if r.get("s_per_step")]
    fit = None
    if len(points) == 3:
        import numpy as np
        Bg = np.array([p[0] for p in points], dtype=float)
        T = np.array([p[1] for p in points], dtype=float)
        X = np.stack([np.ones_like(Bg), Bg, Bg / 4.0], axis=1)  # B_b=4
        try:
            coef, res, *_ = np.linalg.lstsq(X, T, rcond=None)
            T_hat = X @ coef
            ss_res = float(np.sum((T - T_hat) ** 2))
            ss_tot = float(np.sum((T - T.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            fit = {"a1": float(coef[0]), "a2": float(coef[1]),
                   "a3": float(coef[2]), "R2": r2,
                   "note": "3 点拟合(自由度 0); R² 仅形式参考, P-A2a 判定以 7 点扩展为准"}
        except Exception as e:
            print(f"  拟合失败: {e}")

    out = {
        "metadata": {
            "experiment": "A.T2.1 G=4 B_g sweep (DGBB refit)",
            "gpu": "RX 7900 XTX (RDNA3, 960 GB/s)",
            "model": "Qwen2.5-3B-Instruct",
            "framework": "hf",
            "G": 4, "B_b": 4,
            "max_steps": MAX_STEPS,
            "max_completion_length": 256,
            "note": "B_g = pdb×ga (docs/19); chunk=32 分批生成补丁生效中",
        },
        "results": summary,
        "dgbg_fit": fit,
    }
    path = os.path.join(RESULTS, "summary.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n✅ summary: {path}")
    if fit:
        print(f"  DGBB 拟合: a1={fit['a1']:.2f} a2={fit['a2']:.4f} a3={fit['a3']:.4f} R²={fit['R2']:.4f}")
    for r in summary:
        print(f"  Bg={r.get('Bg')}: {r.get('tokens_per_s')} tok/s, {r.get('s_per_step')} s/step, "
              f"{r.get('peak_mem_gb')} GB")


if __name__ == "__main__":
    main()
