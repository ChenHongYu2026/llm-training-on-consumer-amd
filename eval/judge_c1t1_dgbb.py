#!/usr/bin/env python3
"""WS-C C.T1 判定: DGBB 函数形式在 E2B 上的重标定拟合 (docs/36 §2 P-C1)。

从 grpo_2d_sweep 各配置的 timing.json 读 s_per_step, 拟合
  T_step = a1 + a2*B_g + a3*(B_g/B_b)   (B_g=gen_batch, B_b=pdb)
最小二乘 → R²。判据: R² >= 0.95 → PASS (函数形式不变, 系数不同)。
同时报告显存墙位置(各配置 peak_mem_gb 最大可用档)。

用法: python3 eval/judge_c1t1_dgbb.py [summary_json]
"""
import json
import os
import sys

import numpy as np
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


SWEEP_DIR = REPO_ROOT + "/results/efficiency/grpo_2d_sweep"


def main():
    rows = []
    summary_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SWEEP_DIR, "summary.json")
    if not os.path.exists(summary_path):
        print(f"summary.json 不存在: {summary_path} (扫描未完成?)")
        sys.exit(1)
    sweep_dir = os.path.dirname(os.path.abspath(summary_path))
    summary = json.load(open(summary_path))
    if isinstance(summary, dict) and "results" in summary:
        summary = summary["results"]  # sweep 脚本输出 {metadata, results} 结构

    for cfg in summary:
        if cfg.get("error") or cfg.get("steady_tps") is None:
            continue
        pdb, ga = cfg["pdb"], cfg["grad_accum"]
        timing_path = os.path.join(sweep_dir, f"pdb{pdb}_ga{ga}", "timing.json")
        s_per_step = None
        if os.path.exists(timing_path):
            s_per_step = json.load(open(timing_path)).get("s_per_step")
        if s_per_step is None:
            continue
        rows.append({
            "B_g": cfg["gen_batch"], "B_b": pdb,
            "T_step": s_per_step, "peak_mem_gb": cfg.get("peak_mem_gb"),
        })

    if len(rows) < 4:
        print(f"可用配置点不足: {len(rows)} (<4), 无法拟合")
        sys.exit(1)

    B_g = np.array([r["B_g"] for r in rows], dtype=float)
    B_b = np.array([r["B_b"] for r in rows], dtype=float)
    T = np.array([r["T_step"] for r in rows], dtype=float)
    X = np.stack([np.ones_like(B_g), B_g, B_g / B_b], axis=1)
    coef, res, *_ = np.linalg.lstsq(X, T, rcond=None)
    T_hat = X @ coef
    ss_res = float(((T - T_hat) ** 2).sum())
    ss_tot = float(((T - T.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot

    print("=== P-C1 DGBB 重标定拟合 (E2B) ===")
    print(f"配置点数: {len(rows)}")
    print(f"T_step = {coef[0]:.2f} + {coef[1]:.4f}·B_g + {coef[2]:.4f}·(B_g/B_b)")
    print(f"R² = {r2:.4f}  (阈值 0.95)")
    verdict = "PASS" if r2 >= 0.95 else "FAIL"
    print(f"判定 P-C1: {verdict}  (函数形式{'不变' if verdict == 'PASS' else '不成立'})")

    # 显存墙
    ok_mem = [r for r in rows if r["peak_mem_gb"]]
    if ok_mem:
        ok_mem.sort(key=lambda r: r["B_g"])
        print("\n=== 显存档位 ===")
        for r in ok_mem:
            print(f"  B_g={r['B_g']:>4} B_b={r['B_b']:>2}: {r['peak_mem_gb']:.2f} GB")
        # 甜点: 最大 B_g 且 <21GB (留 3GB 余量)
        sweet = [r for r in ok_mem if r["peak_mem_gb"] < 21.0]
        if sweet:
            best = max(sweet, key=lambda r: r["T_step"] and 1.0 / r["T_step"])
            print(f"\n甜点候选(显存<21GB 中每步最快): B_g={best['B_g']}, B_b={best['B_b']}, "
                  f"{best['T_step']:.1f}s/step, {best['peak_mem_gb']:.2f}GB")

    out = {"coef": coef.tolist(), "r2": round(r2, 4),
           "verdict": verdict, "n_points": len(rows),
           "rows": rows}
    out_path = os.path.join(sweep_dir, "dgbb_fit.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果: {out_path}")


if __name__ == "__main__":
    main()
