#!/usr/bin/env python3
"""
二维解耦批处理 GRPO 理论拟合与对照
====================================
理论模型 (docs/19):
  T_step(B_g,B_b) = T_gen(B_g) + (B_g/B_b)·T_bwd(B_b) + c
  取 T_bwd(B_b) = δ + γ·B_b (微批固定开销 + 每序列边际) 代入化简得三参数线性可辨识形式:
      T_step ≈ a1 + a2·B_g + a3·(B_g/B_b)
      a1 = c(常数开销), a2 = β_gen·L̄ + γ(B_g 的边际: 生成每序列 + 反向每序列),
      a3 = δ(每个微批的固定开销 —— 这正是 pdb=1 低效的根源)
  显存: M(B_g,B_b) = m0 + m_g·B_g + m_b·max(0, B_b−4)

模式:
  --preregister : 用 2D sweep 已有 7 点拟合, 盲预测 gen384/gen512 (E1 之前运行, 时间戳为证)
  --compare     : E1 落盘后, 全点残差 + 预测命中判定 + argmax 检验 + MFU 缺口瀑布

用法: python3 eval/fit_theory.py --preregister
      python3 eval/fit_theory.py --compare
"""

import os
import json
import time
import argparse

import numpy as np
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


BASE_DIR = REPO_ROOT
SWEEP_DIR = os.path.join(BASE_DIR, "results/efficiency/grpo_2d_sweep")
DECODE_JSON = os.path.join(BASE_DIR, "inference/results/batch_throughput.json")
OUT_DIR = os.path.join(BASE_DIR, "results/efficiency/theory")
os.makedirs(OUT_DIR, exist_ok=True)

PEAK_TFLOPS = 101.53
NUM_PARAMS = 3085697024
VRAM_LIMIT_GB = 24.0
OOM_THRESHOLD_GB = 23.5   # 留 0.5GB 余量的 OOM 预测阈值

# 预注册拟合只允许用这 7 个"旧"点(E1 之前已有)
PREREG_CONFIGS = [(1, 64), (2, 32), (4, 16), (8, 8), (2, 64), (4, 32), (4, 64)]
E1_CONFIGS = [(4, 96), (4, 128)]   # gen384 / gen512
PROBE_CONFIGS = [(8, 32)]          # P3 argmax 探针: gen256_bwd8 (理论预言 843.9, 实测裁决)


def load_config_obs(pdb: int, ga: int):
    """从 efficiency_report.json 的 step_records 差分出 steady 每优化步时间与 token 数。"""
    rep_path = os.path.join(SWEEP_DIR, f"pdb{pdb}_ga{ga}", "efficiency_report.json")
    if not os.path.exists(rep_path):
        return None
    rep = json.load(open(rep_path))
    recs = [r for r in rep.get("step_records", []) if r.get("elapsed_s", 0) > 0 and r.get("num_tokens", 0) > 0]
    if len(recs) < 2:
        return None
    steady_tokens = recs[-1]["num_tokens"] - recs[0]["num_tokens"]
    steady_time = sum(r["elapsed_s"] for r in recs[1:])
    steps = recs[-1]["step"] - recs[0]["step"]
    return {
        "pdb": pdb, "ga": ga, "gen_batch": pdb * ga, "bwd": pdb,
        "T_step_s": steady_time / steps,               # 每优化步墙钟
        "tok_per_step": steady_tokens / steps,         # 每优化步 token(TRL 口径, 含 prompt)
        "steady_tps": steady_tokens / steady_time,
        "peak_mem_gb": rep.get("peak_mem_gb"),
    }


def fit_time_model(obs):
    """最小二乘拟合 T_step = a1 + a2·B_g + a3·(B_g/B_b)。返回参数与逐点残差。"""
    A = np.array([[1.0, o["gen_batch"], o["gen_batch"] / o["bwd"]] for o in obs])
    y = np.array([o["T_step_s"] for o in obs])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid_pct = ((pred - y) / y * 100).tolist()
    return coef, resid_pct


def fit_mem_model(obs):
    """最小二乘拟合 M = m0 + m_g·B_g + m_b·max(0, B_b−4)。"""
    A = np.array([[1.0, o["gen_batch"], max(0, o["bwd"] - 4)] for o in obs])
    y = np.array([o["peak_mem_gb"] for o in obs])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid_gb = (pred - y).tolist()
    return coef, resid_gb


def predict(coef_t, coef_m, k_tok, pdb, ga):
    """预测 (pdb,ga) 的每步时间/吞吐/显存。k_tok = tok_per_step / B_g (从旧点估计)。"""
    bg, bb = pdb * ga, pdb
    T = coef_t[0] + coef_t[1] * bg + coef_t[2] * (bg / bb)
    tok = k_tok * bg
    mem = coef_m[0] + coef_m[1] * bg + coef_m[2] * max(0, bb - 4)
    return {
        "pdb": pdb, "ga": ga, "gen_batch": bg, "bwd": bb,
        "pred_T_step_s": round(float(T), 2),
        "pred_steady_tps": round(float(tok / T), 1),
        "pred_peak_mem_gb": round(float(mem), 2),
        "pred_oom": bool(mem > OOM_THRESHOLD_GB),
    }


def decode_curve():
    """纯解码微基准 t(B) [ms/解码步] —— 生成相 roofline 素材与 a2 一致性检验。"""
    d = json.load(open(DECODE_JSON))["results"]
    pts = [(v["batch"], v["s_per_step"]) for v in d.values()]  # s_per_step 实为 ms/token-step
    B = np.array([p[0] for p in pts]); t = np.array([p[1] for p in pts])
    A = np.vstack([np.ones_like(B, dtype=float), B]).T
    (alpha, beta), *_ = np.linalg.lstsq(A, t, rcond=None)
    return pts, float(alpha), float(beta)


def do_preregister():
    obs = [load_config_obs(p, g) for p, g in PREREG_CONFIGS]
    obs = [o for o in obs if o]
    assert len(obs) == 7, f"预注册要求恰好 7 个旧点, 实得 {len(obs)}"
    coef_t, resid_t = fit_time_model(obs)
    coef_m, resid_m = fit_mem_model(obs)
    k_tok = float(np.mean([o["tok_per_step"] / o["gen_batch"] for o in obs]))
    dec_pts, alpha, beta = decode_curve()

    preds = [predict(coef_t, coef_m, k_tok, p, g) for p, g in E1_CONFIGS]
    out = {
        "meta": {
            "purpose": "PREREGISTERED blind prediction for E1 (gen384/gen512) — written BEFORE experiments",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": "T_step = a1 + a2*Bg + a3*(Bg/Bb);  M = m0 + mg*Bg + mb*max(0,Bb-4)",
            "fit_points": PREREG_CONFIGS,
            "hit_criteria": {"tps_rel_err_pct": 10, "mem_abs_err_gb": 1.0},
        },
        "time_model": {
            "a1_const_s": round(float(coef_t[0]), 3),
            "a2_per_Bg_s": round(float(coef_t[1]), 4),
            "a3_per_microbatch_s": round(float(coef_t[2]), 4),
            "fit_resid_pct": [round(r, 2) for r in resid_t],
        },
        "mem_model": {
            "m0_gb": round(float(coef_m[0]), 3),
            "mg_gb_per_Bg": round(float(coef_m[1]), 5),
            "mb_gb_per_Bb_gt4": round(float(coef_m[2]), 4),
            "fit_resid_gb": [round(r, 3) for r in resid_m],
        },
        "decode_rooline_check": {
            "points_ms_per_step": dec_pts,
            "alpha_ms": round(alpha, 2), "beta_ms_per_seq": round(beta, 3),
            "note": "生成相 t(B)=alpha+beta*B; a2 应≈beta*L_gen/1000+gamma_bwd (docs/19 §2 交叉验证)",
        },
        "k_tok_per_Bg": round(k_tok, 1),
        "predictions": preds,
    }
    path = os.path.join(OUT_DIR, "prediction_preregistered.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n✅ 预注册预测已落盘(E1 前): {path}")


def do_compare():
    prereg = json.load(open(os.path.join(OUT_DIR, "prediction_preregistered.json")))
    all_cfgs = PREREG_CONFIGS + E1_CONFIGS + PROBE_CONFIGS
    obs = [load_config_obs(p, g) for p, g in all_cfgs]
    have = [o for o in obs if o]
    # 经验墙: E1 实测 OOM 的档(目录存在但无 report) → 可行 B_g 上限取实测最大可行档
    oom_cfgs = [(p, g) for (p, g), o in zip(all_cfgs, obs)
                if o is None and os.path.isdir(os.path.join(SWEEP_DIR, f"pdb{p}_ga{g}"))]
    empirical_wall_gen = max(o["gen_batch"] for o in have)

    # 1) 盲预测命中判定
    hits = []
    for pred in prereg["predictions"]:
        actual = next((o for o in have if o["gen_batch"] == pred["gen_batch"] and o["bwd"] == pred["bwd"]), None)
        if actual is None:
            oom = (pred["pdb"], pred["ga"]) in oom_cfgs
            hits.append({**pred, "actual": "OOM" if oom else "missing",
                         "hit": bool(oom and pred["pred_oom"]),
                         "note": "OOM预测命中" if (oom and pred["pred_oom"]) else
                                 ("预测可跑但实测OOM——显存模型漏算瞬时尖峰+非torch开销" if oom else "")})
        else:
            tps_err = (pred["pred_steady_tps"] - actual["steady_tps"]) / actual["steady_tps"] * 100
            mem_err = pred["pred_peak_mem_gb"] - actual["peak_mem_gb"]
            hits.append({**pred,
                         "actual_steady_tps": round(actual["steady_tps"], 1),
                         "actual_peak_mem_gb": actual["peak_mem_gb"],
                         "tps_rel_err_pct": round(tps_err, 2),
                         "mem_abs_err_gb": round(mem_err, 2),
                         "hit": bool(abs(tps_err) < 10 and abs(mem_err) < 1.0 and not pred["pred_oom"])})

    # 2) 全点重拟合 + 残差（含 argmax 探针点; 若探针证伪线性 T_bwd, 残差会暴露在 pdb8 点上）
    coef_t, resid_t = fit_time_model(have)
    coef_m, resid_m = fit_mem_model(have)
    k_tok = float(np.mean([o["tok_per_step"] / o["gen_batch"] for o in have]))
    max_resid = max(abs(r) for r in resid_t)

    # 3) argmax 检验: 离散可行域扫模型。双墙口径:
    #    (i) 经验墙(实测 OOM 约束, B_g ≤ empirical_wall_gen) —— 主判定
    #    (ii) 模型墙(当初预测, 仅供对照显存模型修正量)
    grid_emp, grid_model = [], []
    for bb in [1, 2, 4, 8]:
        for bg in [32, 64, 96, 128, 192, 256, 384, 512, 768]:
            if bg % bb:
                continue
            pr = predict(coef_t, coef_m, k_tok, bb, bg // bb)
            if not pr["pred_oom"]:
                grid_model.append(pr)
            if bg <= empirical_wall_gen:
                grid_emp.append(pr)
    theo_best = max(grid_emp, key=lambda r: r["pred_steady_tps"])
    theo_best_modelwall = max(grid_model, key=lambda r: r["pred_steady_tps"])
    actual_best = max(have, key=lambda o: o["steady_tps"])

    # 4) MFU 缺口瀑布: 用拟合参数分解最优点的每步时间
    o = actual_best
    bg, bb = o["gen_batch"], o["bwd"]
    # 模型项分解(以拟合参数口径):
    T_const = float(coef_t[0])
    T_bg = float(coef_t[1]) * bg                 # 生成每序列+反向每序列 合并边际
    T_micro = float(coef_t[2]) * bg / bb         # 微批固定开销(=pdb 杠杆的收益来源)
    T_tot = T_const + T_bg + T_micro
    waterfall = {
        "config": f"pdb{bb}_gen{bg}",
        "T_step_s_model": round(T_tot, 2),
        "T_step_s_actual": round(o["T_step_s"], 2),
        "share_const_pct": round(T_const / T_tot * 100, 1),
        "share_Bg_marginal_pct": round(T_bg / T_tot * 100, 1),
        "share_microbatch_overhead_pct": round(T_micro / T_tot * 100, 1),
        "mfu_actual_pct": round(6.0 * NUM_PARAMS * o["tok_per_step"] / o["T_step_s"] / 1e12 / PEAK_TFLOPS * 100, 2),
        "note": "share_microbatch→0 即 pdb→Bg(全批反向)的理论上限方向; share_const 为不可批开销",
    }
    # 渐近极限(无显存墙): TPS∞(B_b) = k_tok/(a2 + a3/B_b)
    tps_inf_b4 = k_tok / (float(coef_t[1]) + float(coef_t[2]) / 4)
    tps_inf = k_tok / float(coef_t[1])

    out = {
        "meta": {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "n_points_fit": len(have), "max_time_resid_pct": round(max_resid, 2),
                 "loopback_triggered": bool(max_resid > 15.0),
                 "empirical_wall": {"max_feasible_gen_batch": empirical_wall_gen,
                                    "oom_configs": [f"pdb{p}_ga{g}(gen{p*g})" for p, g in oom_cfgs],
                                    "note": "expandable_segments 已启用, OOM=真容量墙; 显存模型需补瞬时尖峰+非torch开销项"}},
        "blind_prediction_check": hits,
        "refit_time_model": {"a1": round(float(coef_t[0]), 3), "a2": round(float(coef_t[1]), 4),
                             "a3": round(float(coef_t[2]), 4),
                             "resid_pct": [round(r, 2) for r in resid_t]},
        "refit_mem_model": {"m0": round(float(coef_m[0]), 3), "mg": round(float(coef_m[1]), 5),
                            "mb": round(float(coef_m[2]), 4),
                            "resid_gb": [round(r, 3) for r in resid_m]},
        "argmax_check": {
            "theory_best_empirical_wall": theo_best,
            "theory_best_model_wall_obsolete": theo_best_modelwall,
            "actual_best": {"config": f"pdb{actual_best['bwd']}_gen{actual_best['gen_batch']}",
                            "steady_tps": round(actual_best["steady_tps"], 1),
                            "peak_mem_gb": actual_best["peak_mem_gb"]},
            "actual_vs_theory_max_pct": round(actual_best["steady_tps"] / theo_best["pred_steady_tps"] * 100, 1),
            "asymptotic_tps_bwd4": round(tps_inf_b4, 1),
            "asymptotic_tps_no_micro": round(tps_inf, 1),
            "actual_vs_asymptotic_pct": round(actual_best["steady_tps"] / tps_inf_b4 * 100, 1),
        },
        "mfu_waterfall": waterfall,
    }
    path = os.path.join(OUT_DIR, "theory_vs_experiment.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n✅ 理论-实验对照已落盘: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.preregister:
        do_preregister()
    elif a.compare:
        do_compare()
    else:
        print("指定 --preregister 或 --compare")
