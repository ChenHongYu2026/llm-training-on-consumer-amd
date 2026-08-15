#!/usr/bin/env python3
"""
实验B: 解耦 GRPO 收敛质量验证
==============================
用 A 阶段最优配置 + 小 batch 基线各跑长训练(~100 步), 验证:
  reward(=GSM8K 准确率)稳定上升 + held-out 准确率不低于基线 → 效率提升不损收敛质量。

配置来源: 自动读 results/efficiency/grpo_2d_sweep/summary.json 中 steady_tps 最高者为"最优";
          基线固定 pdb=1, grad_accum=8 (gen_batch=8)。

用法:
  sg render -c "cd {REPO_ROOT} && python3 -u eval/run_grpo_convergence.py"
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
RESULTS = os.path.join(BASE_DIR, "results/efficiency/grpo_convergence")
A_SUMMARY = os.path.join(BASE_DIR, "results/efficiency/grpo_2d_sweep/summary.json")
os.makedirs(RESULTS, exist_ok=True)

MAX_STEPS = 100
LOGGING_STEPS = 5
EVAL_N = 500
G_FIXED = 2
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


def pick_best_config():
    """从 A 的 summary.json 选 steady_tps 最高的 (pdb, grad_accum)。
    限 gen_batch<=128 以保 B 跑 100 步的可行运行时(gen256×100步太慢)。"""
    if os.path.exists(A_SUMMARY):
        with open(A_SUMMARY) as f:
            a = json.load(f)
        ok = [r for r in a.get("results", [])
              if r.get("steady_tps") and r.get("gen_batch", 999) <= 128]
        if ok:
            best = max(ok, key=lambda r: r["steady_tps"])
            return {"pdb": best["pdb"], "ga": best["grad_accum"],
                    "gen_batch": best["gen_batch"], "src": "A.summary(gen<=128)"}
    # 回退: A 已知最优(gen<=128) = pdb4 ga32 (gen128)
    print("  ⚠️ 未找到 A summary, 回退 pdb=4,ga=32 (gen128)")
    return {"pdb": 4, "ga": 32, "gen_batch": 128, "src": "fallback"}


def read_reward_curve(out_dir):
    """从 checkpoint-*/trainer_state.json 的 log_history 提取 reward 序列。"""
    for ck in sorted(os.listdir(out_dir), reverse=True):
        ts = os.path.join(out_dir, ck, "trainer_state.json")
        if os.path.exists(ts):
            with open(ts) as f:
                st = json.load(f)
            curve = [(h.get("step"), h.get("reward")) for h in st.get("log_history", [])
                     if h.get("reward") is not None]
            return curve
    return []


def run_train(tag, pdb, ga):
    out_dir = os.path.join(RESULTS, tag)
    report_path = os.path.join(out_dir, "efficiency_report.json")
    if not os.path.exists(report_path):
        os.makedirs(out_dir, exist_ok=True)
        cmd = [sys.executable, os.path.join(BASE_DIR, "train/train_gsm8k_grpo.py"),
               "--per-device-batch", str(pdb), "--grad-accum", str(ga),
               "--output", out_dir, *COMMON_ARGS]
        env = os.environ.copy(); env["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"\n{'='*60}\n🚀 训练 {tag}: pdb={pdb}, ga={ga}, gen_batch={pdb*ga}, {MAX_STEPS} 步\n{'='*60}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, cwd=BASE_DIR)
        if proc.returncode != 0:
            return {"tag": tag, "error": f"train returncode={proc.returncode}"}
        print(f"  ✓ {tag} 训练完成, 墙钟 {time.perf_counter()-t0:.0f}s")
    else:
        print(f"⏭️  跳过训练 {tag}（已有结果）")

    rep = json.load(open(report_path)) if os.path.exists(report_path) else {}
    # 校正吞吐: profiler.tokens_per_s 因累计求和 bug 会 over-count, 改用 step_records 末累计/总时间
    recs = [x for x in rep.get("step_records", []) if x.get("elapsed_s", 0) > 0 and x.get("num_tokens", 0) > 0]
    corrected_tps = None
    if recs:
        tot_t = sum(x["elapsed_s"] for x in recs)
        if tot_t > 0:
            corrected_tps = round(recs[-1]["num_tokens"] / tot_t, 1)
    # held-out 评估
    adapter = os.path.join(out_dir, "final_adapter")
    eval_json = os.path.join(out_dir, "gsm8k_eval.json")
    acc = None
    if os.path.exists(os.path.join(adapter, "adapter_config.json")):
        if not os.path.exists(eval_json):
            print(f"  📊 评估 {tag} held-out 准确率...")
            env = os.environ.copy(); env["HF_ENDPOINT"] = "https://hf-mirror.com"
            subprocess.run([sys.executable, os.path.join(BASE_DIR, "eval/evaluate_gsm8k.py"),
                            "--adapter", adapter, "--n", str(EVAL_N), "--output", eval_json],
                           env=env, cwd=BASE_DIR)
        if os.path.exists(eval_json):
            acc = json.load(open(eval_json)).get("accuracy")

    curve = read_reward_curve(out_dir)
    return {
        "tag": tag, "pdb": pdb, "grad_accum": ga, "gen_batch": pdb * ga,
        "tokens_per_s_corrected": corrected_tps,
        "profiler_tps_BUGGY": rep.get("tokens_per_s"),
        "mfu_practical_pct": rep.get("mfu_practical_pct"),
        "peak_mem_gb": rep.get("peak_mem_gb"),
        "reward_curve": curve,
        "reward_final": curve[-1][1] if curve else None,
        "heldout_accuracy": acc,
        "total_samples": MAX_STEPS * pdb * ga,
    }


def main():
    print("=" * 60)
    print("实验B: 解耦 GRPO 收敛质量验证")
    print("=" * 60)
    best = pick_best_config()
    print(f"  最优配置(来自 {best['src']}): pdb={best['pdb']}, ga={best['ga']}, gen_batch={best['gen_batch']}")
    print(f"  基线: pdb=1, ga=8, gen_batch=8 | 各 {MAX_STEPS} 步 | held-out test[:{EVAL_N}]")

    configs = [("baseline_gen8", 1, 8), (f"best_gen{best['gen_batch']}", best["pdb"], best["ga"])]
    summary = []
    for tag, pdb, ga in configs:
        try:
            summary.append(run_train(tag, pdb, ga))
        except Exception as e:
            print(f"  ✗ {tag} 异常: {e}")
            summary.append({"tag": tag, "error": str(e)})
        time.sleep(15)

    out = {
        "metadata": {
            "experiment": "Decoupled-GRPO convergence quality (reward + held-out accuracy)",
            "gpu": "RX 7900 XTX (RDNA3)", "model": "Qwen2.5-3B-Instruct",
            "max_steps": MAX_STEPS, "eval_n": EVAL_N, "best_config": best,
        },
        "results": summary,
    }
    path = os.path.join(RESULTS, "summary.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n{'═'*72}\n  收敛质量结果\n{'═'*72}")
    print(f"  {'配置':<18}{'gen_batch':>10}{'tok/s':>9}{'reward末':>10}{'held-out':>10}{'峰值GB':>9}")
    for r in summary:
        if r.get("error"):
            print(f"  {r['tag']:<18}  ERROR: {r['error'][:40]}")
        else:
            acc = f"{r['heldout_accuracy']:.1%}" if r.get("heldout_accuracy") is not None else "N/A"
            rf = f"{r['reward_final']:.3f}" if r.get("reward_final") is not None else "N/A"
            print(f"  {r['tag']:<18}{r['gen_batch']:>10}{r.get('tokens_per_s_corrected',0):>9}{rf:>10}{acc:>10}{r.get('peak_mem_gb',0):>9}")
    print(f"\n  结果: {path}")


if __name__ == "__main__":
    main()
