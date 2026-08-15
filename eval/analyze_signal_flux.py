#!/usr/bin/env python3
"""
E-F0: 信号通量数据考古 (纯 CPU, docs/21 §7.1)
================================================
1) 真实 ρ(p) 反演: uniform 三 seed reward_log 合并 → 观测三池 → Beta-Binomial 反卷积(§1.2)
2) λ 初估: zsbr_v1_s{42,123,7}/zsbr_state.json 的复选题 p̂ 演化 → λ_MS(收割学会率);
   uniform 跨 seed 同题差异 → μ_0 旁证(不同轨迹下同题观测差)
3) F-b 预注册: 三池模型 (2.1) 用初估参数外推 v1_500 的 fzs(t) 曲线, E-F1 前落盘(时间戳为证)

用法: python3 eval/analyze_signal_flux.py
"""

import os
import json
import time
from collections import defaultdict
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


BASE_DIR = REPO_ROOT
ZDIR = os.path.join(BASE_DIR, "results/efficiency/zsbr")

G = 2
M_BUDGET = 64
EPSILON = 0.2
DELTA = 0.15          # 三池分界 (docs/21 §1.1)
S_M = 0.45            # M 池平均 P_sig(边界题非正好 p=0.5)
FZS_V1_100 = 0.575    # v1 100 步稳态(docs/20 §7.2)


def load_uniform_obs():
    """合并 uniform 三 seed 的每题观测(各 run 内该题的 2 条 reward)。"""
    per = defaultdict(list)   # idx -> list of per-run (passes, n)
    for s in [42, 123, 7]:
        p = os.path.join(ZDIR, f"uniform_s{s}", "reward_log.jsonl")
        if not os.path.exists(p):
            continue
        runp = defaultdict(lambda: [0, 0])
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                runp[r["idx"]][0] += 1 if r["reward"] >= 0.9 else 0
                runp[r["idx"]][1] += 1
        for i, (s_, n_) in runp.items():
            per[i].append((s_, n_))
    return per


def invert_rho(per):
    """Beta-Binomial 反卷积: 观测(通过数直方图) → 真实 p 三池占比。
    p 网格 11 点; 观测模型 = 合并该题全部观测的二项似然; 用 EM(混合权重)。"""
    import math
    grid = [j / 10 for j in range(11)]
    grid = [min(max(g, 0.02), 0.98) for g in grid]      # 避 0/1 退化
    w = [1.0 / len(grid)] * len(grid)
    # 每题合并观测 (S, N)
    obs = [(sum(s for s, _ in v), sum(n for _, n in v)) for v in per.values()]

    def binom(S, N, p):
        return math.comb(N, S) * (p ** S) * ((1 - p) ** (N - S))

    for _ in range(200):                                 # EM
        neww = [0.0] * len(grid)
        for S, N in obs:
            liks = [w[j] * binom(S, N, grid[j]) for j in range(len(grid))]
            tot = sum(liks) or 1e-12
            for j in range(len(grid)):
                neww[j] += liks[j] / tot
        w = [x / len(obs) for x in neww]
    pools = {"D": sum(w[j] for j, g in enumerate(grid) if g < DELTA),
             "M": sum(w[j] for j, g in enumerate(grid) if DELTA <= g <= 1 - DELTA),
             "S": sum(w[j] for j, g in enumerate(grid) if g > 1 - DELTA)}
    S_true = sum(w[j] * 2 * grid[j] * (1 - grid[j]) for j in range(len(grid)))
    return {"grid": grid, "weights": [round(x, 4) for x in w],
            "pools_true": {k: round(v, 4) for k, v in pools.items()},
            "signal_rate_true_rho": round(S_true, 4)}


def estimate_lambda_ms():
    """λ_MS 初估: v1 三 seed zsbr_state 的复选混合题——
    以 p̂ 上漂为'被收割学会'代理: 曾混合(0<s<n 且早期)且末态 p̂>0.7 的比率/人均复选次数。"""
    est = []
    detail = {}
    for s in [42, 123, 7]:
        p = os.path.join(ZDIR, f"zsbr_v1_s{s}", "zsbr_state.json")
        if not os.path.exists(p):
            continue
        z = json.load(open(p))
        # 复选题: n > G(被观测 >1 组)
        multi = [e for e in z["p_hat_seen"] if e["n"] > G + 0.5]
        mixed_multi = [e for e in multi if 0.2 < e["p_hat"] < 0.95 or e["s"] > 0.3]
        # 学会代理: 末态 p̂ 高(>0.7 → 近期多为通过)
        learned = [e for e in mixed_multi if e["p_hat"] > 0.7]
        n_obs_groups = sum(e["n"] for e in mixed_multi) / G
        rate = len(learned) / max(n_obs_groups, 1)     # 每组观测的学会概率 μ_h
        est.append(rate)
        detail[f"s{s}"] = {"n_multi": len(multi), "n_mixed_multi": len(mixed_multi),
                           "n_learned_proxy": len(learned), "mu_h_per_group": round(rate, 4)}
    mu_h = sum(est) / len(est) if est else 0.02
    return mu_h, detail


def simulate_fzs(mu_h, M0_pool, D0_pool, T=500, cycles_per_step=1):
    """三池模型 (2.1) 短视版(a_D=0)外推 fzs(t)。
    v1 每 cycle 收割槽 ~ (1-ε)·M_BUDGET=51 组, 冷却=3 → 有效轮换收割。
    μ_0(自然溢出)按 relay 弱假设=每 cycle 0.0005·D(保守, E-F1 后校准)。"""
    mu0 = 0.0005
    Mp, Dp = M0_pool, D0_pool            # 池内题数(绝对)
    harvest_groups = (1 - EPSILON) * M_BUDGET
    curve = []
    for t in range(T):
        lam_ms = mu_h * harvest_groups / max(Mp, 1)      # 人均被收割强度×学会率
        dM = mu0 * Dp - lam_ms * Mp
        Dp = max(Dp - mu0 * Dp, 0)
        Mp = max(Mp + dM, 1)
        # fzs ≈ 1 − [贪婪槽命中M池的信号 + ε 本底]
        greedy_sig = S_M * min(1.0, Mp / harvest_groups)  # M 池不足时贪婪槽掺入零信号题
        fzs = 1 - ((1 - EPSILON) * greedy_sig + EPSILON * 0.25)
        curve.append(round(fzs, 4))
    return curve, mu0


def main():
    out = {"meta": {"purpose": "E-F0 flux calibration + PREREGISTERED F-b extrapolation "
                               "(written BEFORE E-F1)",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}}

    per = load_uniform_obs()
    out["n_prompts_obs"] = len(per)
    out["rho_inversion"] = invert_rho(per)

    mu_h, detail = estimate_lambda_ms()
    out["lambda_ms_estimate"] = {"mu_h_per_group_obs": round(mu_h, 4), "per_seed": detail}

    # 池绝对量: 真实占比 × N
    N = 7473
    Mp0 = out["rho_inversion"]["pools_true"]["M"] * N
    Dp0 = out["rho_inversion"]["pools_true"]["D"] * N
    tau = Mp0 / max(mu_h * (1 - EPSILON) * M_BUDGET, 1e-9)
    out["tau_cycles"] = round(tau, 1)
    out["tau_verdict"] = ("F-a 500步内可见" if tau < 400 else
                          "τ≥400: F-a 可能不可判定(诚实条款, docs/21 §6)")

    curve, mu0 = simulate_fzs(mu_h, Mp0, Dp0, T=500)
    out["preregistered_fzs_curve_v1_500"] = {
        "mu0_assumed": mu0,
        "fzs_at": {str(t): curve[t - 1] for t in [50, 100, 200, 300, 400, 500]},
        "fzs_seg_late_300_500_pred": round(sum(curve[299:500]) / 201, 4),
        "hit_criteria": "F-b: 与实测逐 log 点均方误差 <5pp",
    }

    path = os.path.join(ZDIR, "flux_calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n✅ E-F0 校准+预注册: {path}")


if __name__ == "__main__":
    main()
