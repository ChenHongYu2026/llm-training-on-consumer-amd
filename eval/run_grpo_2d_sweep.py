#!/usr/bin/env python3
"""
实验A: 二维 (pdb × grad_accum) sweep —— 找解耦 GRPO 的吞吐极限
=============================================================
背景: ③ 证明解耦(pdb=1+grad_accum)在 8.87GB 跑到 gen_batch=64, 但 pdb=1 让训练反向
      退回 batch=1(batch-1 GEMV 低效), 且显存离 24GB 还远 → 极限没抓到。

本实验两个杠杆:
  1. 反向 batch 杠杆: 固定 gen_batch=64, 扫 pdb∈{1,2,4,8}(=backward batch), 看适度增大
     pdb 能否让反向更高效、总吞吐更高。
  2. gen_batch 上限: 用较优 pdb 把 gen_batch 推到 128/256, 找吞吐拐点/显存墙。

干净测量: train 脚本传 --logging-steps 2, profiler dump step_records; 本 runner 剔除
          首个 warmup 区间, 算 steady-state 吞吐(修正 ③ 的 warmup ~2× 低估)。

用法:
  sg render -c "cd {REPO_ROOT} && python3 -u eval/run_grpo_2d_sweep.py"
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
if "--model-path" in sys.argv:  # WS-C C.T1: E2B 等第二基座覆盖入口(不影响默认 Qwen 口径)
    MODEL = sys.argv[sys.argv.index("--model-path") + 1]
RESULTS = os.path.join(BASE_DIR, "results/efficiency/grpo_2d_sweep")
if "--results-dir" in sys.argv:  # WS-C C.T1: E2B 等第二基座扫描必须用独立目录(避免覆盖 Qwen 论文数据)
    RESULTS = sys.argv[sys.argv.index("--results-dir") + 1]
os.makedirs(RESULTS, exist_ok=True)

PEAK_TFLOPS_PRACTICAL = 101.53
G_FIXED = 2
MAX_STEPS = 8         # logging_steps=2 → records@2/4/6/8; 剔除首个 → 3 个 steady 区间
LOGGING_STEPS = 2

# (per_device_batch, grad_accum); gen_batch = pdb × grad_accum, backward_batch = pdb
CONFIGS = [
    (1, 64),   # gen64  bwd1  (= ③ GA64 清洁重测)
    (2, 32),   # gen64  bwd2
    (4, 16),   # gen64  bwd4
    (8, 8),    # gen64  bwd8   ← 反向 batch 杠杆
    (2, 64),   # gen128 bwd2
    (4, 32),   # gen128 bwd4   ← 推 gen_batch=128
    (4, 64),   # gen256 bwd4   ← gen_batch=256 极限探针(可能 OOM)
    (4, 96),   # gen384 bwd4   ← E1 显存墙中间点(预注册预测 829.7 tok/s @ 20.49GB)
    (4, 128),  # gen512 bwd4   ← E1 显存墙探针(预注册预测 25.13GB → OOM)
    (8, 32),   # gen256 bwd8   ← P3 理论 argmax 探针: 模型预言 843.9 tok/s @ 17.69GB (>pdb4 的 795.5);
               #   若实测≈pdb4 则证伪线性 T_bwd 外推, 需加 b>4 次线性项(docs/19 §7)
]

COMMON_ARGS = [
    "--framework", "hf",
    "--model-path", MODEL,
    "--no-sft-adapter",
    "--num-generations", str(G_FIXED),
    "--max-completion-length", "256",
    "--max-prompt-length", "256",
    "--max-steps", str(MAX_STEPS),
    "--logging-steps", str(LOGGING_STEPS),
    "--seed", "42",
]


def steady_state(step_records, num_params):
    """剔除首个 warmup 区间, 用差分法算 steady-state 吞吐与 MFU。
    注: TRL 日志的 num_tokens 是**累计值**, 必须用相邻区间差分得真实 token 数,
    否则 sum(累计值) 会系统性 over-count。"""
    recs = [r for r in (step_records or []) if r.get("elapsed_s", 0) > 0 and r.get("num_tokens", 0) > 0]
    if len(recs) < 2:
        return None
    # 剔除首区间(warmup): steady_tokens = 末累计 - 首区间末累计; time = 首区间之后各区间
    steady_tokens = recs[-1]["num_tokens"] - recs[0]["num_tokens"]
    steady_time = sum(r["elapsed_s"] for r in recs[1:])
    if steady_time <= 0 or steady_tokens <= 0:
        return None
    tps = steady_tokens / steady_time
    achieved_tflops = 6.0 * num_params * steady_tokens / steady_time / 1e12
    # 全程(含warmup)校正吞吐 = 末累计 / 总时间, 供参考
    full_tps = recs[-1]["num_tokens"] / sum(r["elapsed_s"] for r in recs)
    return {
        "steady_tps": round(tps, 1),
        "steady_mfu_practical_pct": round(achieved_tflops / PEAK_TFLOPS_PRACTICAL * 100, 3),
        "full_tps_corrected": round(full_tps, 1),
        "steady_intervals": len(recs) - 1,
    }


def run_one(pdb: int, ga: int) -> dict:
    gen_batch = pdb * ga
    tag = f"pdb{pdb}_ga{ga}"
    out_dir = os.path.join(RESULTS, tag)
    report_path = os.path.join(out_dir, "efficiency_report.json")
    base = {"pdb": pdb, "grad_accum": ga, "gen_batch": gen_batch, "backward_batch": pdb}

    if not os.path.exists(report_path):
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            sys.executable, os.path.join(BASE_DIR, "train/train_gsm8k_grpo.py"),
            "--per-device-batch", str(pdb),
            "--grad-accum", str(ga),
            "--output", out_dir,
            *COMMON_ARGS,
        ]
        env = os.environ.copy()
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        # E1 发现: gen384 在默认分配器下撞"碎片墙"(已分配16.8GB+申请3.8GB但 3.9GB reserved 碎片)
        # expandable_segments 消除碎片墙, 探真容量墙; 对吞吐影响可忽略
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print(f"\n{'='*60}\n🚀 {tag}: gen_batch={gen_batch}, backward_batch={pdb}\n{'='*60}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, cwd=BASE_DIR)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"  ✗ {tag} 退出码 {proc.returncode}")
            return {**base, "error": f"returncode={proc.returncode}"}
        print(f"  ✓ {tag} 完成, 墙钟 {elapsed:.0f}s")
    else:
        print(f"⏭️  跳过 {tag}（已有结果）")

    if not os.path.exists(report_path):
        return {**base, "error": "no efficiency_report.json"}
    with open(report_path) as f:
        rep = json.load(f)
    num_params = rep.get("num_params", 3085697024)
    ss = steady_state(rep.get("step_records"), num_params)
    return {
        **base,
        "profiler_avg_tps_BUGGY": rep.get("tokens_per_s"),   # profiler 累计求和 bug 值, 仅备查
        "steady_tps": ss["steady_tps"] if ss else None,
        "steady_mfu_practical_pct": ss["steady_mfu_practical_pct"] if ss else None,
        "full_tps_corrected": ss["full_tps_corrected"] if ss else None,
        "steady_intervals": ss["steady_intervals"] if ss else 0,
        "peak_mem_gb": rep.get("peak_mem_gb"),
        "gpu_busy_mean": (rep.get("gpu_busy", {}) or {}).get("gpu_busy_pct", {}).get("mean"),
    }


def main():
    print("=" * 60)
    print("实验A: 二维 (pdb × grad_accum) sweep — 找解耦 GRPO 吞吐极限")
    print("=" * 60)
    print(f"  配置数: {len(CONFIGS)} | G={G_FIXED} | max_steps={MAX_STEPS}, logging_steps={LOGGING_STEPS}")
    print(f"  steady-state = 剔除首个 warmup 区间后的吞吐")

    summary = []
    for pdb, ga in CONFIGS:
        try:
            summary.append(run_one(pdb, ga))
        except Exception as e:
            print(f"  ✗ pdb{pdb}_ga{ga} 异常: {e}")
            summary.append({"pdb": pdb, "grad_accum": ga, "gen_batch": pdb * ga, "error": str(e)})
        time.sleep(15)

    # 极限判定: steady_tps 最大者
    ok = [r for r in summary if r.get("steady_tps")]
    best = max(ok, key=lambda r: r["steady_tps"]) if ok else None
    # gen64 反向 batch 杠杆分析
    gen64 = sorted([r for r in ok if r["gen_batch"] == 64], key=lambda r: r["backward_batch"])
    verdict = "INCOMPLETE"
    if best:
        ref = "③(pdb=1) gen64=740(低估)"
        verdict = (f"极限点: {best['pdb']}×{best['grad_accum']} (gen_batch={best['gen_batch']}, "
                   f"bwd={best['backward_batch']}) → steady {best['steady_tps']} tok/s, "
                   f"MFU {best['steady_mfu_practical_pct']}%, 显存 {best['peak_mem_gb']}GB")
        if len(gen64) >= 2:
            lo = gen64[0]; hi = max(gen64, key=lambda r: r["steady_tps"])
            verdict += f" | gen64 反向杠杆: bwd{lo['backward_batch']}={lo['steady_tps']} → 最优 bwd{hi['backward_batch']}={hi['steady_tps']} tok/s"

    out = {
        "metadata": {
            "experiment": "2D (pdb x grad_accum) sweep for decoupled-GRPO throughput limit",
            "gpu": "RX 7900 XTX (RDNA3, 960 GB/s)",
            "model": "Qwen2.5-3B-Instruct",
            "framework": "hf",
            "G_fixed": G_FIXED,
            "max_steps": MAX_STEPS,
            "logging_steps": LOGGING_STEPS,
            "max_completion_length": 256,
            "note": "gen_batch=pdb×grad_accum, backward_batch=pdb; steady_tps 剔除首个 warmup 区间",
        },
        "results": summary,
        "verdict": verdict,
    }
    path = os.path.join(RESULTS, "summary.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n{'═'*84}\n  二维 sweep 结果 (steady-state)\n{'═'*84}")
    print(f"  {'pdb':>4}{'ga':>5}{'gen_b':>7}{'bwd':>5}{'steady_tps':>12}{'MFU%':>8}{'full_tps':>9}{'峰值GB':>9}")
    for r in summary:
        if r.get("error"):
            print(f"  {r.get('pdb',''):>4}{r.get('grad_accum',''):>5}{r.get('gen_batch',''):>7}  ERROR: {r['error'][:30]}")
        else:
            print(f"  {r['pdb']:>4}{r['grad_accum']:>5}{r['gen_batch']:>7}{r['backward_batch']:>5}"
                  f"{r.get('steady_tps',0):>12}{r.get('steady_mfu_practical_pct',0):>8}"
                  f"{r.get('full_tps_corrected',0):>9}{r.get('peak_mem_gb',0):>9}")
    print(f"\n  判定: {verdict}")
    print(f"  结果: {path}")


if __name__ == "__main__":
    main()
