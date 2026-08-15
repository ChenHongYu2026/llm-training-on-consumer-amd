#!/usr/bin/env python3
"""
E-A: ZSBR 校准分析 (纯 CPU, docs/20 §7.1)
==========================================
从 uniform run 的 reward_log.jsonl 拟合每题经验 reward 分布 π_q →
1) 预测 frac_reward_zero_std 并与 trainer_state 实测对照 (预言 P4: 差<5pp)
2) 输出经验 p 分布直方图 + top-M 可达 S 上界修正 (docs/20 §1.4 声明 2)

用法: python3 eval/analyze_zsbr_calibration.py \
        --run results/efficiency/zsbr/uniform_s42 --max-steps 100
"""

import os
import json
import argparse
from collections import defaultdict
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


BASE_DIR = REPO_ROOT
OUT_DIR = os.path.join(BASE_DIR, "results/efficiency/zsbr")

REWARD_BINS = [0.0, 0.1, 0.9, 1.0]


def rbin(r: float) -> float:
    """归入 4 值支撑(容差)。"""
    return min(REWARD_BINS, key=lambda b: abs(b - r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="含 reward_log.jsonl 的 run 目录")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--G", type=int, default=2)
    ap.add_argument("--M", type=int, default=64, help="每步组预算(用于 top-M 可达上界)")
    args = ap.parse_args()

    log_path = os.path.join(args.run, "reward_log.jsonl")
    per_prompt = defaultdict(list)          # idx -> [rewards...]
    n_records = 0
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            per_prompt[rec["idx"]].append(rbin(rec["reward"]))
            n_records += 1

    # ── 经验 π 分布 → 每题 P_sig(精确 4 值式 1.1) 与 p(二值) ──
    # 小样本无偏修正(docs/20 §7.1): 插件估计 E[1−Σπ̂²]=P_sig·(n−1)/n → 乘 n/(n−1) 去偏
    G = args.G
    psig_exact, psig_binary, p_list = [], [], []
    for idx, rs in per_prompt.items():
        n = len(rs)
        probs = {b: rs.count(b) / n for b in set(rs)}
        plug = 1.0 - sum(pr ** G for pr in probs.values())
        psig_exact.append(plug * (n / (n - 1)) if n > 1 else plug)   # 无偏修正
        p = sum(1 for r in rs if r >= 0.9) / n
        p_list.append(p)
        psig_binary.append(1.0 - p ** G - (1 - p) ** G)

    S_exact = sum(psig_exact) / len(psig_exact)
    S_binary = sum(psig_binary) / len(psig_binary)
    pred_fzs_exact = 1.0 - S_exact
    pred_fzs_binary = 1.0 - S_binary

    # ── 实测 fzs (trainer_state) ──
    ts_path = os.path.join(args.run, f"checkpoint-{args.max_steps}", "trainer_state.json")
    fzs_meas = None
    if os.path.exists(ts_path):
        hist = json.load(open(ts_path)).get("log_history", [])
        vals = [h["frac_reward_zero_std"] for h in hist if h.get("frac_reward_zero_std") is not None]
        fzs_meas = sum(vals) / len(vals) if vals else None

    # ── p 分布直方图 (0.1 桶) ──
    hist_p = defaultdict(int)
    for p in p_list:
        hist_p[round(min(p, 0.999) // 0.1 * 0.1, 1)] += 1
    hist_p = {f"{k:.1f}-{k+0.1:.1f}": v for k, v in sorted(hist_p.items())}

    # ── top-M 可达上界修正 (docs/20 §1.4): 经验分布上 top-M 的 P_sig 均值 ──
    # 注: 单次观测(n=1)的 p∈{0,1} 会低估边界题占比, 用 n≥2 的子集另报一版
    top_all = sorted(psig_binary, reverse=True)[: args.M]
    S_top_all = sum(top_all) / len(top_all)
    multi_obs = [(1.0 - p ** G - (1 - p) ** G)
                 for idx, rs in per_prompt.items() if len(rs) >= 2 * G
                 for p in [sum(1 for r in rs if r >= 0.9) / len(rs)]]
    S_top_multi = None
    if len(multi_obs) >= args.M:
        tm = sorted(multi_obs, reverse=True)[: args.M]
        S_top_multi = sum(tm) / len(tm)

    p4_pass = (fzs_meas is not None and abs(pred_fzs_exact - fzs_meas) < 0.05)
    out = {
        "run": args.run, "n_records": n_records, "n_unique_prompts": len(per_prompt),
        "obs_per_prompt_mean": round(n_records / len(per_prompt), 2),
        "S_exact_4value_biascorrected": round(S_exact, 4), "S_binary_plugin": round(S_binary, 4),
        "pred_frac_zero_std_exact": round(pred_fzs_exact, 4),
        "pred_frac_zero_std_binary_plugin": round(pred_fzs_binary, 4),
        "measured_frac_zero_std": round(fzs_meas, 4) if fzs_meas is not None else None,
        "P4_calibration_pass(<5pp)": p4_pass,
        "p_histogram": hist_p,
        "attainable_S_topM": {
            "all_prompts(n>=1, 含单观测偏差)": round(S_top_all, 4),
            "multi_obs_prompts(n>=2G)": round(S_top_multi, 4) if S_top_multi else "样本不足",
            "note": "V1 真实可达上界≈topM 均值×(1−ε)+S0×ε; 单观测 p∈{0,1} 使 binary 口径低估",
        },
    }
    path = os.path.join(OUT_DIR, "calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n✅ 校准结果: {path}")


if __name__ == "__main__":
    main()
