#!/usr/bin/env python3
"""
实验E2/E3: 多 seed 复验 + 等样本量对照
========================================
目的 (docs/18 §三):
  E2: seeds{42,123,7} × {gen8(pdb1,ga8), gen128(pdb4,ga32)} 各 100 步 → +7.2pp 稳健性
      (seed=42 两档复用 grpo_convergence 已有训练, 不重跑; held-out 统一 batch=32 口径重评)
  E3: gen8_eq = gen8 × 1600 步 (total_samples=12800 对齐 gen128×100 步) → 解耦
      "大 batch 方差效应" vs "数据量效应"

统计: mean±std / bootstrap 95% CI / 逐 seed 配对差 / Cohen's d (复用 utils.metrics)
判定 (verdict):
  - 3 seed 配对差全为正 且 差值 CI 下界>0 → "+7.2pp 稳健"
  - gen128(100步) ≥ gen8_eq(1600步) − 1pp → "方差效应成立(非纯数据量)"
    否则 → "提升主要来自样本量, 大 batch 价值在单位墙钟见更多样本"

用法:
  sg render -c "cd {REPO_ROOT} && python3 -u eval/run_grpo_multiseed.py"
"""

import os
import sys
import json
import time
import subprocess

BASE_DIR = REPO_ROOT
MODEL = os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct")
RESULTS = os.path.join(BASE_DIR, "results/efficiency/grpo_multiseed")
CONV_DIR = os.path.join(BASE_DIR, "results/efficiency/grpo_convergence")
os.makedirs(RESULTS, exist_ok=True)
sys.path.insert(0, BASE_DIR)
from utils.metrics import bootstrap_confidence_interval, cohens_d  # noqa: E402
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


SEEDS = [42, 123, 7]
EVAL_N = 500
EVAL_BATCH = 32
G_FIXED = 2

# (tag, pdb, ga, max_steps, seeds); seed=42 的 gen8/gen128 100步复用 grpo_convergence
CONFIGS = [
    ("gen8",    1, 8,  100,  SEEDS),
    ("gen128",  4, 32, 100,  SEEDS),
    ("gen8_eq", 1, 8,  1600, [42]),   # E3 等样本量对照: 12800 samples
]

# seed=42 复用映射: tag → 已有训练目录
REUSE_42 = {"gen8": os.path.join(CONV_DIR, "baseline_gen8"),
            "gen128": os.path.join(CONV_DIR, "best_gen128")}


def train_args(pdb, ga, max_steps, seed, out_dir):
    return [sys.executable, os.path.join(BASE_DIR, "train/train_gsm8k_grpo.py"),
            "--framework", "hf", "--model-path", MODEL, "--no-sft-adapter",
            "--num-generations", str(G_FIXED),
            "--max-completion-length", "256", "--max-prompt-length", "256",
            "--per-device-batch", str(pdb), "--grad-accum", str(ga),
            "--max-steps", str(max_steps), "--logging-steps", "5",
            "--save-steps", str(max_steps),   # 只留末 checkpoint
            "--seed", str(seed), "--output", out_dir]


def extract_train_stats(out_dir, max_steps):
    """从末 checkpoint trainer_state 提取 reward 末值/曲线统计 + 信号率/截断率。"""
    ts_path = os.path.join(out_dir, f"checkpoint-{max_steps}", "trainer_state.json")
    if not os.path.exists(ts_path):
        return {}
    hist = json.load(open(ts_path)).get("log_history", [])
    rew = [h["reward"] for h in hist if h.get("reward") is not None]
    fzs = [h["frac_reward_zero_std"] for h in hist if h.get("frac_reward_zero_std") is not None]
    clip = [h["completions/clipped_ratio"] for h in hist if h.get("completions/clipped_ratio") is not None]
    import statistics as st
    return {
        "reward_final": rew[-1] if rew else None,
        "reward_mean": round(st.mean(rew), 4) if rew else None,
        "reward_curve_std": round(st.stdev(rew), 4) if len(rew) > 1 else None,
        "frac_reward_zero_std_mean": round(st.mean(fzs), 4) if fzs else None,
        "clipped_ratio_mean": round(st.mean(clip), 4) if clip else None,
    }


def run_one(tag, pdb, ga, max_steps, seed):
    # seed=42 的 100 步档复用已有训练
    if seed == 42 and tag in REUSE_42:
        out_dir = REUSE_42[tag]
        reused = True
    else:
        out_dir = os.path.join(RESULTS, f"{tag}_seed{seed}")
        reused = False
        if not os.path.exists(os.path.join(out_dir, "final_adapter", "adapter_config.json")):
            os.makedirs(out_dir, exist_ok=True)
            env = os.environ.copy(); env["HF_ENDPOINT"] = "https://hf-mirror.com"
            print(f"\n{'='*60}\n🚀 训练 {tag}_seed{seed}: pdb={pdb}, ga={ga}, {max_steps} 步\n{'='*60}", flush=True)
            t0 = time.perf_counter()
            proc = subprocess.run(train_args(pdb, ga, max_steps, seed, out_dir), env=env, cwd=BASE_DIR)
            if proc.returncode != 0:
                return {"tag": tag, "seed": seed, "error": f"train rc={proc.returncode}"}
            print(f"  ✓ 训练完成, 墙钟 {time.perf_counter()-t0:.0f}s", flush=True)
        else:
            print(f"⏭️  跳过训练 {tag}_seed{seed}（已有 adapter）", flush=True)

    # held-out 评估: 统一 batch=32 口径(方法学一致); 结果文件带 batch32 后缀
    adapter = os.path.join(out_dir, "final_adapter")
    eval_json = os.path.join(out_dir, "gsm8k_eval_batch32.json")
    if not os.path.exists(eval_json):
        env = os.environ.copy(); env["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"  📊 评估 {tag}_seed{seed} (batch={EVAL_BATCH})...", flush=True)
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "eval/evaluate_gsm8k.py"),
                        "--adapter", adapter, "--n", str(EVAL_N),
                        "--batch-size", str(EVAL_BATCH), "--output", eval_json],
                       env=env, cwd=BASE_DIR)
    acc = json.load(open(eval_json)).get("accuracy") if os.path.exists(eval_json) else None

    return {"tag": tag, "seed": seed, "pdb": pdb, "grad_accum": ga,
            "gen_batch": pdb * ga, "max_steps": max_steps,
            "total_samples": max_steps * pdb * ga, "reused_train": reused,
            "heldout_accuracy": acc, **extract_train_stats(out_dir, max_steps)}


def main():
    print("=" * 60)
    print("实验E2/E3: 多 seed 复验 + 等样本量对照")
    print("=" * 60, flush=True)

    runs = []
    for tag, pdb, ga, max_steps, seeds in CONFIGS:
        for seed in seeds:
            try:
                runs.append(run_one(tag, pdb, ga, max_steps, seed))
            except Exception as e:
                print(f"  ✗ {tag}_seed{seed} 异常: {e}", flush=True)
                runs.append({"tag": tag, "seed": seed, "error": str(e)})
            time.sleep(10)
            # 每轮增量落盘, 便于中途监督
            with open(os.path.join(RESULTS, "summary.json"), "w") as f:
                json.dump({"runs": runs}, f, ensure_ascii=False, indent=2)

    # ── 统计汇总 ──
    def accs(tag):
        return [r["heldout_accuracy"] for r in runs
                if r.get("tag") == tag and r.get("heldout_accuracy") is not None]
    a8, a128 = accs("gen8"), accs("gen128")
    stats = {}
    if len(a8) >= 2 and len(a128) >= 2:
        import statistics as st
        paired = []
        for s in SEEDS:
            r8 = next((r for r in runs if r.get("tag") == "gen8" and r.get("seed") == s), None)
            r128 = next((r for r in runs if r.get("tag") == "gen128" and r.get("seed") == s), None)
            if r8 and r128 and r8.get("heldout_accuracy") is not None and r128.get("heldout_accuracy") is not None:
                paired.append({"seed": s, "diff_pp": round((r128["heldout_accuracy"] - r8["heldout_accuracy"]) * 100, 1)})
        diffs = [p["diff_pp"] for p in paired]
        ci8 = bootstrap_confidence_interval(a8); ci128 = bootstrap_confidence_interval(a128)
        cid = bootstrap_confidence_interval(diffs) if len(diffs) >= 2 else (None, None)
        stats = {
            "gen8": {"accs": a8, "mean": round(st.mean(a8), 4),
                     "std": round(st.stdev(a8), 4) if len(a8) > 1 else 0,
                     "ci95": [round(ci8[0], 4), round(ci8[1], 4)]},
            "gen128": {"accs": a128, "mean": round(st.mean(a128), 4),
                       "std": round(st.stdev(a128), 4) if len(a128) > 1 else 0,
                       "ci95": [round(ci128[0], 4), round(ci128[1], 4)]},
            "paired_diff_pp": paired,
            "diff_ci95_pp": [round(cid[0], 2), round(cid[1], 2)] if cid[0] is not None else None,
            "cohens_d": round(cohens_d(a128, a8), 2),
        }

    # ── 判定 ──
    verdicts = []
    if stats:
        diffs = [p["diff_pp"] for p in stats["paired_diff_pp"]]
        if diffs and all(d > 0 for d in diffs) and (stats["diff_ci95_pp"] and stats["diff_ci95_pp"][0] > 0):
            verdicts.append(f"✅ +7.2pp 稳健: 配对差全正 {diffs}, CI95 {stats['diff_ci95_pp']}pp, d={stats['cohens_d']}")
        else:
            verdicts.append(f"⚠️ 稳健性存疑: 配对差 {diffs}, CI95 {stats['diff_ci95_pp']}pp — 需追加 seed 或降级表述")
    eq = next((r for r in runs if r.get("tag") == "gen8_eq" and r.get("heldout_accuracy") is not None), None)
    a128_42 = next((r for r in runs if r.get("tag") == "gen128" and r.get("seed") == 42), None)
    if eq and a128_42 and a128_42.get("heldout_accuracy") is not None:
        d_eq = (a128_42["heldout_accuracy"] - eq["heldout_accuracy"]) * 100
        if d_eq >= -1.0:
            verdicts.append(f"✅ 方差效应成立(非纯数据量): gen128(100步)={a128_42['heldout_accuracy']:.1%} vs "
                            f"gen8_eq(1600步,同12800样本)={eq['heldout_accuracy']:.1%} (Δ={d_eq:+.1f}pp)")
        else:
            verdicts.append(f"⚠️ 提升主要来自样本量: gen8_eq={eq['heldout_accuracy']:.1%} > gen128={a128_42['heldout_accuracy']:.1%} "
                            f"(Δ={d_eq:+.1f}pp) — 大 batch 价值=单位墙钟见更多样本")

    out = {
        "metadata": {"experiment": "Multi-seed replication + equal-sample control (E2/E3)",
                     "gpu": "RX 7900 XTX (RDNA3)", "model": "Qwen2.5-3B-Instruct",
                     "eval": f"GSM8K test[:{EVAL_N}], greedy, batch={EVAL_BATCH} (统一口径)",
                     "seeds": SEEDS, "note": "seed=42 100步档复用 grpo_convergence 已有训练"},
        "runs": runs, "stats": stats, "verdicts": verdicts,
    }
    path = os.path.join(RESULTS, "summary.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n{'═'*72}\n  多 seed + 等样本量 结果\n{'═'*72}")
    for r in runs:
        if r.get("error"):
            print(f"  {r['tag']}_seed{r['seed']}: ERROR {r['error'][:40]}")
        else:
            acc = f"{r['heldout_accuracy']:.1%}" if r.get("heldout_accuracy") is not None else "N/A"
            print(f"  {r['tag']:<10} seed={r['seed']:<4} steps={r['max_steps']:<5} "
                  f"samples={r['total_samples']:<6} acc={acc}")
    for v in verdicts:
        print(f"\n  {v}")
    print(f"\n  结果: {path}")


if __name__ == "__main__":
    main()
