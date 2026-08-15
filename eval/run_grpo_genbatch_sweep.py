#!/usr/bin/env python3
"""
实验②b: GRPO 生成 batch sweep (真正的杠杆 —— generation_batch_size)
====================================================================
②a 发现: 在固定 generation_batch_size (=per_device_batch×grad_accum=8) 下扫 num_generations G,
         吞吐/显存持平 (480→496 tok/s), 因为 G 只重分配固定的生成名额, 不增大生成 batch。
         (且 G > generation_batch_size 直接报错: G16 fail)

本实验: 固定 G=2、grad_accum=1, 扫 per_device_batch ∈ {2,8,16,32} (= generation_batch_size),
       直接测"生成 batch"这一真实杠杆能否把纯解码 batch_throughput 的 28.5× 迁移到真实 GRPO 训练。

判据: tokens/s 随 generation_batch 上升 → 杠杆迁移成立 (batch_throughput 结论在真实训练循环复现)。

用法:
  sg render -c "cd {REPO_ROOT} && python3 -u eval/run_grpo_genbatch_sweep.py"
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
RESULTS = os.path.join(BASE_DIR, "results/efficiency/grpo_genbatch_sweep")
os.makedirs(RESULTS, exist_ok=True)

# 扫 generation_batch = per_device_batch (grad_accum=1, G=2 固定); per_device_batch 必须可被 G 整除
GEN_BATCHES = [2, 8, 16, 32]
G_FIXED = 2
MAX_STEPS = 20
COMMON_ARGS = [
    "--framework", "hf",
    "--model-path", MODEL,
    "--no-sft-adapter",
    "--num-generations", str(G_FIXED),
    "--grad-accum", "1",
    "--max-steps", str(MAX_STEPS),
    "--max-completion-length", "256",
    "--max-prompt-length", "256",
    "--seed", "42",
]


def run_one(gb: int) -> dict:
    out_dir = os.path.join(RESULTS, f"GB{gb}")
    report_path = os.path.join(out_dir, "efficiency_report.json")

    if os.path.exists(report_path):
        print(f"⏭️  跳过 gen_batch={gb}（已有结果）")
    else:
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            sys.executable, os.path.join(BASE_DIR, "train/train_gsm8k_grpo.py"),
            "--per-device-batch", str(gb),
            "--output", out_dir,
            *COMMON_ARGS,
        ]
        env = os.environ.copy()
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"\n{'='*60}\n🚀 gen_batch={gb} (per_device_batch={gb}, G={G_FIXED}, grad_accum=1)\n{'='*60}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, cwd=BASE_DIR)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"  ✗ gen_batch={gb} 退出码 {proc.returncode}")
            return {"gen_batch": gb, "error": f"returncode={proc.returncode}"}
        print(f"  ✓ gen_batch={gb} 完成, 墙钟 {elapsed:.0f}s")

    if not os.path.exists(report_path):
        return {"gen_batch": gb, "error": "no efficiency_report.json"}
    with open(report_path) as f:
        rep = json.load(f)
    return {
        "gen_batch": gb,
        "tokens_per_s": rep.get("tokens_per_s"),
        "mfu_practical_pct": rep.get("mfu_practical_pct"),
        "s_per_step": rep.get("s_per_step"),
        "peak_mem_gb": rep.get("peak_mem_gb"),
        "valid_steps": rep.get("valid_steps"),
        "gpu_busy_mean": (rep.get("gpu_busy", {}) or {}).get("gpu_busy_pct", {}).get("mean"),
    }


def main():
    print("=" * 60)
    print("实验②b: GRPO 生成 batch sweep (真实训练循环, 真正杠杆)")
    print("=" * 60)
    print(f"  gen_batch 档: {GEN_BATCHES} | G={G_FIXED} 固定 | grad_accum=1 | HF+Qwen2.5-3B | completion=256")

    summary = []
    for gb in GEN_BATCHES:
        try:
            summary.append(run_one(gb))
        except Exception as e:
            print(f"  ✗ gen_batch={gb} 异常: {e}")
            summary.append({"gen_batch": gb, "error": str(e)})
        time.sleep(10)

    base = next((r for r in summary if r.get("gen_batch") == GEN_BATCHES[0] and r.get("tokens_per_s")), None)
    base_tps = base.get("tokens_per_s") if base else None
    for r in summary:
        if r.get("tokens_per_s") and base_tps:
            r["tps_vs_gb2"] = round(r["tokens_per_s"] / base_tps, 2)

    ok = [r for r in summary if r.get("tokens_per_s")]
    verdict = "INCOMPLETE"
    if len(ok) >= 2:
        lo, hi = ok[0]["tokens_per_s"], ok[-1]["tokens_per_s"]
        ratio = round(hi / lo, 2)
        if ratio >= 1.5:
            verdict = (f"✅ 杠杆迁移成立: gen_batch {ok[0]['gen_batch']}→{ok[-1]['gen_batch']} "
                       f"吞吐 {lo}→{hi} tok/s ({ratio}×) → batch_throughput 的算术强度杠杆在真实 GRPO 训练复现")
        else:
            verdict = (f"🔶 弱迁移: gen_batch {ok[0]['gen_batch']}→{ok[-1]['gen_batch']} "
                       f"吞吐 {lo}→{hi} tok/s ({ratio}×) → 训练阶段(反向)稀释了生成 batch 收益")

    out = {
        "metadata": {
            "experiment": "GRPO generation_batch_size Scaling (real training loop, true lever)",
            "gpu": "RX 7900 XTX (RDNA3, 960 GB/s)",
            "model": "Qwen2.5-3B-Instruct",
            "framework": "hf",
            "G_fixed": G_FIXED,
            "grad_accum": 1,
            "max_steps": MAX_STEPS,
            "max_completion_length": 256,
            "note": "generation_batch_size = per_device_batch × grad_accum; 此处 grad_accum=1 故 = per_device_batch",
        },
        "results": summary,
        "verdict": verdict,
    }
    path = os.path.join(RESULTS, "summary.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n{'═'*72}\n  GRPO 生成 batch sweep 结果\n{'═'*72}")
    print(f"  {'gen_batch':>10}{'tokens/s':>12}{'MFU实践%':>10}{'s/step':>10}{'峰值GB':>9}{'tps/gb2':>9}")
    for r in summary:
        if r.get("error"):
            print(f"  {r['gen_batch']:>10}{'ERROR':>12}  {r['error']}")
        else:
            print(f"  {r['gen_batch']:>10}{r.get('tokens_per_s', 0):>12}{r.get('mfu_practical_pct', 0):>10}"
                  f"{r.get('s_per_step', 0):>10}{r.get('peak_mem_gb', 0):>9}{r.get('tps_vs_gb2', ''):>9}")
    print(f"\n  判定: {verdict}")
    print(f"  结果: {path}")


if __name__ == "__main__":
    main()
