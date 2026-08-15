#!/usr/bin/env python3
"""
B1/C 轮吞吐校正重算 — 第一篇论文审计修复 (P1-C1)
====================================================
背景: efficiency_profiler 旧版 tokens_per_s 用 sum(每区间累计值)/总时间 → 系统性
over-count (docs/18(g) 已确认该 bug, grpo_convergence 实测夸大 ~10.5×)。
B1/C 轮 JSON 无 step_records, 校正口径 = TRL trainer_state 末累计 num_tokens / 总墙钟
(与 docs/18 tokens_per_s_corrected 同口径, "末累计/总时间")。

零 GPU 数据考古; 不改历史 JSON, 产出新校正文件:
  results/efficiency/B1_C_corrected_summary.json
"""

import json, glob, os
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EFF = os.path.join(ROOT, "results", "efficiency")
N_PARAMS = 3.086e9          # Qwen2.5-3B
PEAK_PRAC = 101.53e12       # GEMM 实测峰值 TFLOPS
PEAK_THEO = 122.8e12

def load(p):
    with open(p) as f:
        return json.load(f)

def correct_run(run_dir):
    """单 run 校正: 末累计 num_tokens / total_time_s"""
    rep_path = os.path.join(run_dir, "efficiency_report.json")
    if not os.path.isdir(run_dir) or not os.path.exists(rep_path):
        return None
    rep = load(rep_path)
    # 找 checkpoint-*/trainer_state.json
    ts_paths = glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))
    if not ts_paths:
        return None
    ts = load(sorted(ts_paths)[-1])
    # TRL log_history 中 num_tokens 为累计值, 取末条非空
    final_tokens = None
    for rec in reversed(ts.get("log_history", [])):
        if "num_tokens" in rec:
            final_tokens = rec["num_tokens"]
            break
    if final_tokens is None:
        return None
    t = rep["total_time_s"]
    tps_corr = final_tokens / t
    # MFU (6ND 训练口径, 与原 profiler 同公式, 仅换正确 tokens)
    tflops = 6.0 * N_PARAMS * tps_corr
    return {
        "tokens_per_s_buggy": rep["tokens_per_s"],
        "tokens_per_s_corrected": round(tps_corr, 1),
        "overcount_factor": round(rep["tokens_per_s"] / tps_corr, 2),
        "mfu_practical_pct_buggy": rep["mfu_practical_pct"],
        "mfu_practical_pct_corrected": round(tflops / PEAK_PRAC * 100, 2),
        "mfu_theoretical_pct_corrected": round(tflops / PEAK_THEO * 100, 2),
        "final_cumulative_tokens": final_tokens,
        "total_time_s": t,
        "peak_mem_gb": rep["peak_mem_gb"],
        "steps": rep["total_steps"],
    }

def main():
    out = {"meta": {
        "purpose": "P1-C1 audit fix: corrected throughput/MFU for B1 & C rounds (paper 1)",
        "method": "final cumulative TRL num_tokens / total wall time (docs/18(g) corrected caliber)",
        "note": "original efficiency_report.json untouched (traceability rule)",
        "n_params": N_PARAMS, "peak_practical_tflops": 101.53,
    }, "B1": {}, "C": {}, "gap_L4": {}}

    for d in sorted(glob.glob(os.path.join(EFF, "B1_*"))):
        r = correct_run(d)
        if r:
            out["B1"][os.path.basename(d)] = r
            print(f"B1 {os.path.basename(d)}: {r['tokens_per_s_buggy']} -> {r['tokens_per_s_corrected']} tok/s (x{r['overcount_factor']}) | MFU {r['mfu_practical_pct_buggy']}% -> {r['mfu_practical_pct_corrected']}%")

    for d in sorted(glob.glob(os.path.join(EFF, "C_G*"))):
        r = correct_run(d)
        if r:
            out["C"][os.path.basename(d)] = r
            print(f"C  {os.path.basename(d)}: {r['tokens_per_s_buggy']} -> {r['tokens_per_s_corrected']} tok/s (x{r['overcount_factor']}) | MFU -> {r['mfu_practical_pct_corrected']}%")

    # gap L4 校正判定: gap_attribution.json 的 L4 tokens 来源检查
    gap = load(os.path.join(EFF, "B2_gap_attribution", "gap_attribution.json"))
    l4 = gap["levels"]["L4_grpo_full"]
    # L4 tokens=3.75M/2810s=1334.4 与 B1_unsloth_4bit buggy 值同源(同profiler口径)
    # 用 B1 unsloth_bf16 的平均 overcount factor 无法直接套用(L4是独立run);
    # 保守处置: 用 B1 全部 run 的 factor 均值作为 L4 校正系数并标注 estimated
    factors = [r["overcount_factor"] for r in out["B1"].values()]
    if factors:
        f_mean = sum(factors) / len(factors)
        l4_tps_corr = l4["tokens_per_s"] / f_mean
        l4_tflops_corr = 6.0 * N_PARAMS * l4_tps_corr / 1e12
        out["gap_L4"] = {
            "tokens_per_s_buggy": l4["tokens_per_s"],
            "correction_factor_estimated": round(f_mean, 2),
            "tokens_per_s_corrected_est": round(l4_tps_corr, 1),
            "achieved_tflops_corrected_est": round(l4_tflops_corr, 2),
            "mfu_practical_pct_corrected_est": round(l4_tflops_corr / 101.53 * 100, 2),
            "note": "L4 no trainer_state; factor = mean of B1 measured factors (estimated)",
        }
        print(f"\ngap L4: {l4['tokens_per_s']} -> ~{out['gap_L4']['tokens_per_s_corrected_est']} tok/s | TFLOPS 24.71 -> ~{out['gap_L4']['achieved_tflops_corrected_est']} | MFU 24.34% -> ~{out['gap_L4']['mfu_practical_pct_corrected_est']}%")

    # 相对结论复核 (A1.5 判定)
    import statistics as st
    def mean_std(prefix):
        vals = [r["tokens_per_s_corrected"] for k, r in out["B1"].items() if prefix in k]
        return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0, len(vals))
    u, us, un = mean_std("unsloth_bf16")
    h, hs, hn = mean_std("hf_bf16")
    q, qs, qn = mean_std("unsloth_4bit")
    out["meta"]["relative_check"] = {
        "unsloth_bf16_corr": f"{u:.1f}±{us:.1f} (n={un})",
        "hf_bf16_corr": f"{h:.1f}±{hs:.1f} (n={hn})",
        "unsloth_4bit_corr": f"{q:.1f}±{qs:.1f} (n={qn})",
        "speedup_u_vs_hf_pct": round((u - h) / h * 100, 1),
        "penalty_4bit_vs_ubf16_pct": round((q - u) / u * 100, 1),
    }
    print(f"\n相对结论复核: Unsloth {u:.1f} vs HF {h:.1f} -> +{(u-h)/h*100:.1f}% | 4bit {q:.1f} -> {(q-u)/u*100:.1f}%")

    with open(os.path.join(EFF, "B1_C_corrected_summary.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n✅ 落盘 results/efficiency/B1_C_corrected_summary.json")

if __name__ == "__main__":
    main()
