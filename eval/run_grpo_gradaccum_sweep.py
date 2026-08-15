#!/usr/bin/env python3
"""
实验③: 解耦批处理 GRPO —— grad_accum 承载生成 batch (突破训练显存墙)
=====================================================================
突破假设 (源自 ②a vs ②b 的显存对比):
  TRL generation_batch_size = per_device_batch × grad_accum, 生成阶段一次批量生成全部,
  但训练反向按 per_device_batch 微批做。
  → 固定 pdb=1 (反向永远 batch=1, 显存恒定 ~7GB), 用 grad_accum 放大生成 batch,
    即可拿到①的大 batch 生成加速, 又避免 ②b 中 pdb=32 反向 OOM。

对照:
  ②b (pdb=gen_batch, grad_accum=1): gen_batch=32 训练 OOM
  ③  (pdb=1, grad_accum=gen_batch): 预期 gen_batch=32/64 显存 ~7-8GB 不 OOM

判据: 吞吐随 gen_batch 上升 且 显存保持低位不 OOM → 解耦批处理突破成立。

用法:
  sg render -c "cd {REPO_ROOT} && python3 -u eval/run_grpo_gradaccum_sweep.py"
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
RESULTS = os.path.join(BASE_DIR, "results/efficiency/grpo_gradaccum_sweep")
os.makedirs(RESULTS, exist_ok=True)

# 固定 pdb=1, G=2; 用 grad_accum 放大生成 batch (= pdb × grad_accum = grad_accum)
GRAD_ACCUMS = [8, 16, 32, 64]
G_FIXED = 2
PER_DEVICE_BATCH = 1
MAX_STEPS = 10   # 优化器步 (logging_steps=5 → 2 个 log 点); grad_accum 越大每步生成越多
COMMON_ARGS = [
    "--framework", "hf",
    "--model-path", MODEL,
    "--no-sft-adapter",
    "--num-generations", str(G_FIXED),
    "--per-device-batch", str(PER_DEVICE_BATCH),
    "--max-completion-length", "256",
    "--max-prompt-length", "256",
    "--seed", "42",
]


def run_one(ga: int) -> dict:
    gen_batch = PER_DEVICE_BATCH * ga
    out_dir = os.path.join(RESULTS, f"GA{ga}")
    report_path = os.path.join(out_dir, "efficiency_report.json")

    if os.path.exists(report_path):
        print(f"⏭️  跳过 grad_accum={ga}（已有结果）")
    else:
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            sys.executable, os.path.join(BASE_DIR, "train/train_gsm8k_grpo.py"),
            "--grad-accum", str(ga),
            "--max-steps", str(MAX_STEPS),
            "--output", out_dir,
            *COMMON_ARGS,
        ]
        env = os.environ.copy()
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"\n{'='*60}\n🚀 grad_accum={ga} → gen_batch={gen_batch} (pdb=1, G={G_FIXED})\n{'='*60}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, cwd=BASE_DIR)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"  ✗ grad_accum={ga} 退出码 {proc.returncode}")
            return {"grad_accum": ga, "gen_batch": gen_batch, "error": f"returncode={proc.returncode}"}
        print(f"  ✓ grad_accum={ga} 完成, 墙钟 {elapsed:.0f}s")

    if not os.path.exists(report_path):
        return {"grad_accum": ga, "gen_batch": gen_batch, "error": "no efficiency_report.json"}
    with open(report_path) as f:
        rep = json.load(f)
    return {
        "grad_accum": ga,
        "gen_batch": gen_batch,
        "tokens_per_s": rep.get("tokens_per_s"),
        "mfu_practical_pct": rep.get("mfu_practical_pct"),
        "s_per_step": rep.get("s_per_step"),
        "peak_mem_gb": rep.get("peak_mem_gb"),
        "valid_steps": rep.get("valid_steps"),
    }


def main():
    print("=" * 60)
    print("实验③: 解耦批处理 GRPO (grad_accum 承载生成 batch)")
    print("=" * 60)
    print(f"  grad_accum 档: {GRAD_ACCUMS} | pdb={PER_DEVICE_BATCH} 固定 | G={G_FIXED} | completion=256")
    print(f"  对照 ②b: pdb=gen_batch 时 gen_batch=32 OOM; 本实验预期 gen_batch=32/64 不 OOM")

    summary = []
    for ga in GRAD_ACCUMS:
        try:
            summary.append(run_one(ga))
        except Exception as e:
            print(f"  ✗ grad_accum={ga} 异常: {e}")
            summary.append({"grad_accum": ga, "error": str(e)})
        time.sleep(10)

    base = next((r for r in summary if r.get("grad_accum") == GRAD_ACCUMS[0] and r.get("tokens_per_s")), None)
    base_tps = base.get("tokens_per_s") if base else None
    for r in summary:
        if r.get("tokens_per_s") and base_tps:
            r["tps_vs_ga8"] = round(r["tokens_per_s"] / base_tps, 2)

    ok = [r for r in summary if r.get("tokens_per_s")]
    max_gb = max((r["gen_batch"] for r in ok), default=0)
    max_mem = max((r["peak_mem_gb"] for r in ok), default=0)
    verdict = "INCOMPLETE"
    if ok:
        no_oom_32plus = any(r["gen_batch"] >= 32 and r["peak_mem_gb"] < 20 for r in ok)
        if len(ok) >= 2:
            lo, hi = ok[0]["tokens_per_s"], ok[-1]["tokens_per_s"]
            ratio = round(hi / lo, 2)
            if no_oom_32plus and ratio >= 1.5:
                verdict = (f"✅ 解耦批处理突破成立: gen_batch {ok[0]['gen_batch']}→{max_gb} 吞吐 {lo}→{hi} tok/s "
                           f"({ratio}×), 显存仅 {max_mem}GB (对比②b gen_batch=32 OOM) → grad_accum 解耦规避了训练显存墙")
            elif no_oom_32plus:
                verdict = (f"🔶 显存墙已绕过 (gen_batch≥32 不 OOM, 峰值 {max_mem}GB), 但吞吐仅 {ratio}× "
                           f"→ 生成加速被训练/反向稀释, 需进一步分析")
            else:
                verdict = f"❌ 未绕过显存墙或吞吐未升 (max gen_batch={max_gb}, 峰值 {max_mem}GB, {ratio if len(ok)>=2 else '?'}×)"

    out = {
        "metadata": {
            "experiment": "Decoupled-batch GRPO: scale generation batch via grad_accum (pdb=1)",
            "gpu": "RX 7900 XTX (RDNA3, 960 GB/s)",
            "model": "Qwen2.5-3B-Instruct",
            "framework": "hf",
            "per_device_batch": PER_DEVICE_BATCH,
            "G_fixed": G_FIXED,
            "max_steps": MAX_STEPS,
            "max_completion_length": 256,
            "note": "gen_batch = per_device_batch × grad_accum; pdb=1 → 训练反向恒 batch=1 (省显存), grad_accum 放大生成 batch",
        },
        "results": summary,
        "verdict": verdict,
    }
    path = os.path.join(RESULTS, "summary.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n{'═'*76}\n  解耦批处理 GRPO 结果\n{'═'*76}")
    print(f"  {'grad_accum':>11}{'gen_batch':>11}{'tokens/s':>11}{'MFU%':>8}{'s/step':>10}{'峰值GB':>9}{'vs_ga8':>8}")
    for r in summary:
        if r.get("error"):
            print(f"  {r['grad_accum']:>11}{r.get('gen_batch',''):>11}{'ERROR':>11}  {r['error']}")
        else:
            print(f"  {r['grad_accum']:>11}{r['gen_batch']:>11}{r.get('tokens_per_s',0):>11}{r.get('mfu_practical_pct',0):>8}"
                  f"{r.get('s_per_step',0):>10}{r.get('peak_mem_gb',0):>9}{r.get('tps_vs_ga8',''):>8}")
    print(f"\n  判定: {verdict}")
    print(f"  结果: {path}")


if __name__ == "__main__":
    main()
