#!/usr/bin/env python3
"""
A.T1 判定脚本: fzs 轨迹分段拟合 (P-A1a/P-A1c)
====================================================
从 decool run 的 reward_log.jsonl 计算每步 fzs (frac_reward_zero_std),
分段(前 2/3 / 末 1/3)拟合, 判定:
  - P-A1a: p500 nocool 末段回升 ≥ +3pp (相对轨迹最低点) → 耗竭态解锁
  - P-A1c: p200 nocool 的 fzs 平台(对照实测)被破坏 (回升/下降 ≥3pp)
判定落盘 JSON 含 verdict (docs/33 §10.2 协议)

用法: python3 eval/judge_at1_fzs.py --runs decool/v1_500_p500_cool_s42,decool/v1_500_p500_nocool_s42,decool/v1_500_p200_nocool_s42
"""

import argparse
import json
import os
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


BASE = REPO_ROOT + "/results/efficiency"


def load_fzs(run: str):
    """从训练日志解析 TRL 计算的 frac_reward_zero_std (权威口径)。

    2026-08-12 口径审计: reward_log 重算(相邻/按 idx 分组)与 TRL 日志值偏差达 8pp
    (TRL 的组结构含非相邻同 idx 条目), 判定以 TRL 日志值为唯一权威。
    日志位置: logs/run_<run>.log (pipeline 重定向)。"""
    log_path = os.path.join(REPO_ROOT + "/logs", f"run_{run}.log")
    if not os.path.exists(log_path):
        return {}
    import re
    fzs = {}
    # 日志行: {'loss': ..., 'frac_reward_zero_std': '0.7188', ...} — 每 logging_steps(5) 一步
    for m in re.finditer(r"'frac_reward_zero_std': '([\d.]+)'", open(log_path).read()):
        fzs[5 * (len(fzs) + 1)] = float(m.group(1))
    return fzs


def seg_fit(fzs: dict):
    """前 2/3 / 末 1/3 分段线性拟合, 返回末段斜率与相对最低点回升。"""
    steps = sorted(fzs)
    n = len(steps)
    if n < 30:
        return None
    cut = int(n * 2 / 3)
    early = [fzs[s] for s in steps[:cut]]
    late = [fzs[s] for s in steps[cut:]]
    # 末段回升 = 末段均值 - 全局最低点(最低点取前 2/3 窗口, 排除冷启动)
    min_early = min(early[5:]) if len(early) > 5 else min(early)
    late_mean = sum(late) / len(late)
    rebound = late_mean - min_early
    # 末段斜率(线性)
    xs = list(range(len(late)))
    n2 = len(late)
    sx = sum(xs); sy = sum(late); sxy = sum(x * y for x, y in zip(xs, late)); sxx = sum(x * x for x in xs)
    slope = (n2 * sxy - sx * sy) / max(n2 * sxx - sx * sx, 1e-9)
    return {"n_steps": n, "cut": cut, "early_mean": round(sum(early) / len(early), 4),
            "late_mean": round(late_mean, 4), "min_early": round(min_early, 4),
            "rebound_pp": round(rebound * 100, 2), "late_slope": round(slope * 100, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, required=True, help="逗号分隔 run 路径")
    ap.add_argument("--output", type=str, default="results/efficiency/decool/fzs_judge.json")
    args = ap.parse_args()
    runs = args.runs.split(",")
    out = {"meta": {"purpose": "A.T1 fzs 轨迹分段拟合判定 (P-A1a/P-A1c)",
                    "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%S")},
           "results": {}}
    for run in runs:
        name = run.split("/")[-1]
        fzs = load_fzs(run)
        fit = seg_fit(fzs)
        verdict = "INVALID"
        note = ""
        if fit:
            if "nocool" in name and "p500" in run:
                # P-A1a: 末段回升 ≥ +3pp
                verdict = "PASS" if fit["rebound_pp"] >= 3.0 else "FAIL"
                note = f"P-A1a: rebound={fit['rebound_pp']}pp (阈值 ≥3pp)"
            elif "nocool" in name and "p200" in run:
                # P-A1c: 平台破坏(对照 cool 待重跑; 暂用绝对平台判断: |末段-早期|≥3pp)
                drift = abs(fit["late_mean"] - fit["early_mean"]) * 100
                verdict = "PASS" if drift >= 3.0 else "FAIL"
                note = f"P-A1c(暂): drift={drift:.2f}pp (cool 对照重跑后复判)"
            else:
                verdict = "REFERENCE"
                note = "对照 run, 提供平台基准"
        out["results"][name] = {"fit": fit, "verdict": verdict, "note": note,
                                "fzs_sample": {str(s): fzs[s] for s in sorted(fzs)[::50]}}
        print(f"{name}: verdict={verdict} | {note} | fit={fit}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"✅ 判定落盘: {args.output}")


if __name__ == "__main__":
    main()
