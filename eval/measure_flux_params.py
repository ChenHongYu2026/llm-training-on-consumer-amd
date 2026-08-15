#!/usr/bin/env python3
"""
E-G0: 相图参数补测 (纯 CPU, 计划: 相图补全与数据充分性)
========================================================
a) n_eff 直接测量: sfc_500_s42 水填组(同题同 cycle k·G 条 rollout)的组内相关 ρ_corr
   → n_eff = kG/(1+(kG−1)ρ)。方法: 按题 p̂ 分箱后池化矩估计(减轻 p 异质性混淆),
   二值 pass 与 4 值 reward 双口径; 另拟合运营口径标量 c: pred=1−(1−p̂)^{c·kG}
   使总体校准误差最小(直接给水填公式的修正系数)。
b) μ₀ 独立验证: v1 三 seed + v1_500 reward_log 的"曾全错题在间隔后重逢转出全错"
   频率/间隔 cycle 数(docs/21 §4 预注册路径首次执行; 全错组 advantage≈0 → 无自训练,
   转化即 relay 驱动)。与 fzs 斜率反推值 0.002 交叉验证, bootstrap CI。
c) 小池构成预估: uniform 三 seed 限 idx<500 子集 → EM 反演三池 → τ₅₀₀
   (E-G1 的 go/no-go 闸门: τ₅₀₀>400 → 停止阶段 2)。

用法: python3 eval/measure_flux_params.py
输出: results/efficiency/zsbr/flux_params_v2.json (E-G1 前落盘, 时间戳为证)
"""

import os
import sys
import json
import time
import random
from collections import defaultdict
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


BASE_DIR = REPO_ROOT
ZDIR = os.path.join(BASE_DIR, "results/efficiency/zsbr")
sys.path.insert(0, os.path.join(BASE_DIR, "eval"))   # analyze_signal_flux 复用(顶部统一注入)

G = 2
MU_H = 0.0268          # E-F0 三 seed 标定
EPSILON = 0.2
M_BUDGET = 64
DELTA = 0.15


def load_cells(arm):
    """reward_log → {step: {idx: [rewards]}}, 按行序两两配组已经审计验证(异常=0),
    这里直接按 (step,idx) 聚合(同 cycle 水填 k 组自然合并为 k·G 条)。"""
    cells = defaultdict(lambda: defaultdict(list))
    p = os.path.join(ZDIR, arm, "reward_log.jsonl")
    with open(p) as f:
        for line in f:
            e = json.loads(line)
            cells[e["step"]][e["idx"]].append(e["reward"])
    return cells


# ──────────────────────────── a) n_eff ────────────────────────────

def measure_n_eff():
    cells = load_cells("zsbr_sfc_500_s42")
    # 每题全程通过率(分箱用; 含自身 cell, 偏差写入 notes)
    per_prompt = defaultdict(lambda: [0, 0])
    for g in cells.values():
        for idx, rs in g.items():
            per_prompt[idx][0] += sum(1 for r in rs if r >= 0.9)
            per_prompt[idx][1] += len(rs)
    p_all = {i: s / n for i, (s, n) in per_prompt.items()}

    # 水填 cell(n>G) 与普通 cell(n=G) 分开; 组内两两对按 p̂ 分箱池化矩估计。
    # binary=True: pass 口径; binary=False: 4 值 reward 口径(审计补齐双口径承诺)。
    # 已知局限(写死): p_all 含自身 cell(题均 ~70+ 观测, 单 cell 占比小, 自引用偏差有限);
    # 箱内 p 异质残留会上拉 ρ̂(方向性: 真 ρ ≤ 报告值, 故 ρ≈0 的"独立"结论保守可靠)。
    def pooled_rho(cell_list, tag, binary=True):
        bins = defaultdict(lambda: {"pairs": [], "vals": []})
        for idx, rs in cell_list:
            b = min(4, int(p_all[idx] * 5))          # 5 箱
            xs = [1.0 if r >= 0.9 else 0.0 for r in rs] if binary else list(rs)
            bins[b]["vals"].extend(xs)
            for a in range(len(xs)):
                for c in range(a + 1, len(xs)):
                    bins[b]["pairs"].append((xs[a], xs[c]))
        out = {}
        num = den = 0.0
        for b, d in sorted(bins.items()):
            if len(d["pairs"]) < 30:
                continue
            m = sum(d["vals"]) / len(d["vals"])
            var = (sum(v * v for v in d["vals"]) / len(d["vals"])) - m * m \
                if not binary else m * (1 - m)
            if var < 1e-6:
                continue
            exy = sum(x * y for x, y in d["pairs"]) / len(d["pairs"])
            rho_b = (exy - m * m) / var
            w = len(d["pairs"])
            out[f"bin_{b}(p in [{b/5:.1f},{(b+1)/5:.1f}))"] = {
                "rho": round(rho_b, 4), "n_pairs": w, "mean": round(m, 3)}
            num += w * rho_b
            den += w
        rho = num / den if den else None
        return {"tag": tag, "rho_pooled": round(rho, 4) if rho is not None else None,
                "per_bin": out}

    wf_cells = []      # 水填 cell: 同 cycle n>G
    g2_cells = []      # 普通 G=2 cell
    for g in cells.values():
        for idx, rs in g.items():
            (wf_cells if len(rs) > G else g2_cells).append((idx, rs))
    res_wf = pooled_rho(wf_cells, f"waterfill cells n>{G} (k=2/3)")
    res_g2 = pooled_rho(g2_cells, "G=2 cells")
    res_wf4 = pooled_rho(wf_cells, "waterfill 4-value reward", binary=False)
    res_g24 = pooled_rho(g2_cells, "G=2 4-value reward", binary=False)

    def n_eff(kg, rho):
        return kg / (1 + (kg - 1) * rho)

    rho_wf = res_wf["rho_pooled"]
    summary = {}
    if rho_wf is not None:
        summary = {"n_eff_kG4": round(n_eff(4, rho_wf), 2),
                   "n_eff_kG6": round(n_eff(6, rho_wf), 2),
                   "n_eff_ratio": round(n_eff(4, rho_wf) / 4, 3),
                   "note_neg_rho": "ρ<0 时 n_eff>kG 仅为公式延伸; 结合箱内异质上拁方向, "
                                   "读作 'ρ≈0, rollout 基本独立, n_eff≈kG'"}

    # 运营口径: 复算 F-e 投资事件, 拟合 c 使 mean(1−(1−p̂)^{c·kG}) = 实测转化率
    z = json.load(open(os.path.join(ZDIR, "zsbr_sfc_500_s42", "zsbr_state.json")))
    hist = defaultdict(list)
    for st_, g in sorted(cells.items()):
        for idx, rs in g.items():
            hist[idx].append((st_, rs))
    events = []
    for e in z["conversion_log"]:
        idx, ic = e["idx"], e["invest_cycle"]
        obs = [r for st_, rs in hist[idx] if st_ < ic - 1 for r in rs]
        ns = len(obs)
        succ = sum(1 for r in obs if r >= 0.9)
        p_hat = (succ + 1) / (ns + 2)
        # k = 水填组数 = 该 cycle 该题的 rollout 数 / G (load_cells 已合并同 cell)
        k = 1
        for st_, rs in hist[idx]:
            if st_ == ic - 1:
                k = max(1, len(rs) // G)
                break
        events.append((p_hat, k * G, 1 if e["converted"] else 0))
    obs_rate = sum(y for _, _, y in events) / len(events)

    def mean_pred(c):
        return sum(1 - (1 - p) ** (c * kg) for p, kg, _ in events) / len(events)
    lo, hi = 0.05, 1.0
    # 边界检查(审计整改): 真值在区间外则报警而非静默夹住
    boundary_warn = None
    if mean_pred(lo) > obs_rate:
        boundary_warn = f"WARN: mean_pred({lo})={mean_pred(lo):.3f} > obs {obs_rate:.3f}, c 真值<{lo}"
    elif mean_pred(hi) < obs_rate:
        boundary_warn = f"WARN: mean_pred({hi})={mean_pred(hi):.3f} < obs {obs_rate:.3f}, c 真值>{hi}"
    for _ in range(60):                              # 二分(mean_pred 对 c 严格单调增: 逐项 (1-p)^{c·kG} 递减)
        mid = (lo + hi) / 2
        if mean_pred(mid) > obs_rate:
            hi = mid
        else:
            lo = mid
    c_fit = round((lo + hi) / 2, 3)
    return {"rho_waterfill": res_wf, "rho_G2": res_g2,
            "rho_waterfill_4value": res_wf4, "rho_G2_4value": res_g24,
            "n_eff_from_rho": summary,
            "operational_fit": {"n_events": len(events), "obs_conv_rate": round(obs_rate, 4),
                                "c_factor": c_fit, "boundary_warn": boundary_warn,
                                "n_eff_operational_kG4": round(c_fit * 4, 2),
                                "note": "pred=1−(1−p̂)^{c·kG}; c=有效试验数比例"}}


# ──────────────────────────── b) μ₀ 独立验证 ────────────────────────────

def measure_mu0():
    """全错题在间隔 Δt cycle 后重逢: 转出全错(≥1 pass)频率/Δt。
    两层混淆控制(审计 v2 修订):
    1) "未被训练"严格化: 仅纯零 cell(all r<0.05)入样——{0,0.1} 混合组 reward_std>0
       产生梯度=自训练, 其转化非 relay, 首版 all(r<0.5) 定义污染 μ₀ 向上;
    2) 观测噪声: 按转出前连续纯零 cell 数 c0 分层(cell 数非观测数, 避免水填 4-6 条/cell
       的语义漂移); c0 越大越接近真 D, 递减收敛=混淆矩阵实证。
    报告全部臂(透明披露臂间分歧), 深层汇总含 bootstrap 95% CI。"""
    import math
    res = {}
    deep_events_all = []
    for arm in ["zsbr_v1_500_s42", "zsbr_sfc_500_s42",
                "uniform_s42", "uniform_s123", "uniform_s7"]:
        p = os.path.join(ZDIR, arm, "reward_log.jsonl")
        if not os.path.exists(p):
            continue
        cells = load_cells(arm)
        seq = defaultdict(list)
        for st_ in sorted(cells):
            for idx, rs in cells[st_].items():
                seq[idx].append((st_, rs))
        strata = {"c0=1": [], "c0=2": [], "c0>=3": []}   # (Δt, converted), c0=连续纯零cell数
        for idx, obs in seq.items():
            zero_cells = 0
            for i in range(len(obs) - 1):
                st0, rs0 = obs[i]
                st1, rs1 = obs[i + 1]
                if all(r < 0.05 for r in rs0):           # 纯零 cell: 无格式分, 零梯度保证
                    zero_cells += 1
                    conv = any(r >= 0.9 for r in rs1)
                    key = "c0=1" if zero_cells == 1 else ("c0=2" if zero_cells == 2 else "c0>=3")
                    strata[key].append((st1 - st0, 1 if conv else 0))
                else:
                    zero_cells = 0                 # 重置: 非纯零后不再算 D 候选
        arm_res = {}
        for key, events in strata.items():
            if len(events) < 30:
                continue

            def mle(ev):
                def loglik(mu):
                    ll = 0.0
                    for dt, y in ev:
                        pc = 1 - (1 - mu) ** max(dt, 1)
                        pc = min(max(pc, 1e-9), 1 - 1e-9)
                        ll += math.log(pc) if y else math.log(1 - pc)
                    return ll
                lo, hi = 1e-5, 0.2
                for _ in range(80):
                    m1 = lo + (hi - lo) / 3
                    m2 = hi - (hi - lo) / 3
                    if loglik(m1) < loglik(m2):
                        lo = m1
                    else:
                        hi = m2
                return (lo + hi) / 2
            arm_res[key] = {"n_events": len(events),
                            "conv_frac": round(sum(y for _, y in events) / len(events), 4),
                            "mean_gap": round(sum(dt for dt, _ in events) / len(events), 1),
                            "mu0_mle": round(mle(events), 5)}
            if key == "c0>=3":
                deep_events_all.extend(events)
        if arm_res:
            res[arm] = arm_res
    # 深层汇总(全臂合并) + bootstrap CI(审计整改: 兼容首版承诺)
    agg = {}
    if len(deep_events_all) >= 30:
        import math as _m

        def mle_of(ev):
            def ll(mu):
                s = 0.0
                for dt, y in ev:
                    pc = min(max(1 - (1 - mu) ** max(dt, 1), 1e-9), 1 - 1e-9)
                    s += _m.log(pc) if y else _m.log(1 - pc)
                return s
            lo, hi = 1e-5, 0.2
            for _ in range(80):
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                if ll(m1) < ll(m2):
                    lo = m1
                else:
                    hi = m2
            return (lo + hi) / 2
        rng = random.Random(42)
        bs = sorted(mle_of([deep_events_all[rng.randrange(len(deep_events_all))]
                            for _ in range(len(deep_events_all))]) for _ in range(200))
        agg = {"n_events": len(deep_events_all),
               "mu0_mle": round(mle_of(deep_events_all), 5),
               "mu0_ci95_bootstrap": [round(bs[4], 5), round(bs[194], 5)]}
    return {"per_arm_stratified": res,
            "deep_stratum_pooled": agg,
            "slope_derived_reference": 0.002,
            "note": "v2 修订: 纯零 cell(all r<0.05, 零梯度保证'未被训练'); c0=连续纯零 cell 数; "
                    "递减收敛=混淆矩阵实证; 全臂透明披露"}


# ──────────────────────────── c) 小池构成 + τ₅₀₀ ────────────────────────────

def small_pool_estimate(pool_size=500):
    from analyze_signal_flux import invert_rho          # 复用 E-F0 EM(路径已在顶部注入)
    per = defaultdict(list)
    for s in [42, 123, 7]:
        p = os.path.join(ZDIR, f"uniform_s{s}", "reward_log.jsonl")
        if not os.path.exists(p):
            continue
        runp = defaultdict(lambda: [0, 0])
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r["idx"] >= pool_size:
                    continue
                runp[r["idx"]][0] += 1 if r["reward"] >= 0.9 else 0
                runp[r["idx"]][1] += 1
        for i, (s_, n_) in runp.items():
            per[i].append((s_, n_))
    inv = invert_rho(per)
    m_abs = inv["pools_true"]["M"] * pool_size
    harvest = (1 - EPSILON) * M_BUDGET
    tau = m_abs / max(MU_H * harvest, 1e-9)
    return {"pool_size": pool_size, "n_prompts_observed": len(per),
            "rho_inversion_subset": inv,
            "M_abs_est": round(m_abs, 1),
            "tau_500_cycles": round(tau, 1),
            "gate": "GO (tau<=400)" if tau <= 400 else "NO-GO (tau>400, 诚实条款: 停止阶段2)"}


# ──────── E-H0: 池尺寸扫描(τ_net 新公式, docs/21 §10.4; E-G 审计教训落地) ────────

MU0_RANGE = (0.002, 0.008)     # docs/21 §10.1b v2 工作区间
COOLDOWN = 3                   # 调度器默认(zsbr_scheduler.py)


def pool_sweep_tau_net(pools=(100, 150, 200, 250, 500)):
    """对候选池尺寸: 三池反演 + τ_net=|M|/(μ_h·槽−μ₀|D|)(μ₀ 区间两端) +
    调度器可行性(非冷却候选余量 ≈ N − (cooldown−1)·M_BUDGET ≥ 贪婪槽51;
    不足则贪婪槽饿死→v1≈uniform 混淆)。E-H1 闸门输入。"""
    harvest = MU_H * (1 - EPSILON) * M_BUDGET            # 1.37 题/cycle
    greedy_slots = int((1 - EPSILON) * M_BUDGET)         # 51
    out = []
    for n in pools:
        est = small_pool_estimate(n)
        d_abs = est["rho_inversion_subset"]["pools_true"]["D"] * n
        m_abs = est["M_abs_est"]
        row = {"pool": n, "D_abs": round(d_abs, 1), "M_abs": m_abs,
               "supply_mu0D": [round(MU0_RANGE[0] * d_abs, 2), round(MU0_RANGE[1] * d_abs, 2)],
               "harvest_rate": round(harvest, 2)}
        taus = []
        for mu0 in MU0_RANGE:
            denom = harvest - mu0 * d_abs
            taus.append(round(m_abs / denom, 1) if denom > 0 else None)   # None=补给主导
        row["tau_net_range"] = taus
        margin = n - (COOLDOWN - 1) * M_BUDGET           # 稳态非冷却余量(保守: 前2cycle全unique)
        row["scheduler_margin"] = margin
        row["scheduler_ok"] = margin >= greedy_slots
        gate_ok = (row["tau_net_range"][1] is not None and row["tau_net_range"][1] <= 400
                   and row["scheduler_ok"])
        row["gate"] = "GO" if gate_ok else "NO-GO"
        out.append(row)
    return {"criterion": "tau_net=|M|/(mu_h*51 - mu0*|D|); mu0 in [0.002,0.008]; "
                         "scheduler: N-(cooldown-1)*64 >= 51",
            "rows": out}


def main_sweep():
    out = {"meta": {"purpose": "E-H0 pool sweep (PREREGISTERED before E-H1, docs/21 §11)",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
           "sweep": pool_sweep_tau_net()}
    path = os.path.join(ZDIR, "flux_params_v3.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n✅ E-H0 落盘: {path}")


def main():
    out = {"meta": {"purpose": "E-G0 phase-diagram params (PREREGISTERED before E-G1)",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    print("[E-G0a] n_eff ...", flush=True)
    out["a_n_eff"] = measure_n_eff()
    print("[E-G0b] mu0 ...", flush=True)
    out["b_mu0"] = measure_mu0()
    print("[E-G0c] small pool ...", flush=True)
    out["c_small_pool"] = small_pool_estimate(500)
    path = os.path.join(ZDIR, "flux_params_v2.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n✅ E-G0 落盘: {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        main_sweep()                                     # E-H0 池扫描模式
    else:
        main()
