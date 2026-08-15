#!/usr/bin/env python3
"""
E-B/E-C: ZSBR 消融实验编排 (docs/20 §7.2/§7.3)
================================================
臂: uniform(基线,带reward_log=E-A素材) / zsbr_v1 / zsbr_v2(条件推进)
闸门(计划判定准则):
  - E-B s42 闸门: v1 后半程 frac_zero_std ≤0.60 且 T_step 差<3% → 补 seeds{123,7}
  - E-C 闸门: v2 后半程 fzs ≤0.45 (P5)
安全(硬中止标准): entropy 末值≥uniform 50% / kl<0.01 / reward 无崩溃 / held-out ≥ uniform−1pp
运行: sg render -c "cd {REPO_ROOT} && python3 -u eval/run_zsbr_ablation.py [--stage s42|full|v2]"
"""

import os
import re
import sys
import json
import time
import argparse
import subprocess
import statistics as st

BASE_DIR = REPO_ROOT
MODEL = os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct")
RESULTS = os.path.join(BASE_DIR, "results/efficiency/zsbr")
os.makedirs(RESULTS, exist_ok=True)
sys.path.insert(0, BASE_DIR)
from utils.metrics import bootstrap_confidence_interval, cohens_d  # noqa: E402
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


SEEDS = [42, 123, 7]
EVAL_N = 500
EVAL_BATCH = 32
G = 2
MAX_STEPS = 100          # 默认; flux500 阶段逐 run 覆盖(见 run_one 的 max_steps 参数)


def train_cmd(arm, seed, out_dir, max_steps):
    cmd = [sys.executable, os.path.join(BASE_DIR, "train/train_gsm8k_grpo.py"),
           "--framework", "hf", "--model-path", MODEL, "--no-sft-adapter",
           "--num-generations", str(G),
           "--max-completion-length", "256", "--max-prompt-length", "256",
           "--per-device-batch", "4", "--grad-accum", "32",     # gen128 统一配置
           "--max-steps", str(max_steps), "--logging-steps", "5",
           "--save-steps", str(max(max_steps // 2, 1)),          # 长训练留中间 checkpoint(过拟监控)
           "--seed", str(seed), "--output", out_dir,
           "--reward-log"]                                       # SFOC: 全臂开日志(转移观测基础设施)
    if arm.startswith("zsbr_v1"):
        cmd += ["--zsbr", "v1"]
    elif arm.startswith("zsbr_v2"):
        cmd += ["--zsbr", "v2"]
    elif arm.startswith("zsbr_sfc"):
        cmd += ["--zsbr", "sfc"]
    mp = re.search(r"_p(\d+)", arm)                          # 池截断臂: _p500/_p200/...
    if mp:                                                   # (审计修复: 原 _p500 硬编码会使
        cmd += ["--pool-size", mp.group(1)]                  #  _p200 臂静默跑成全池)
    return cmd


def extract_stats(out_dir, max_steps=MAX_STEPS):
    """trainer_state 指标 + step_records 差分 T_step + zsbr_state coverage/投资字段。"""
    r = {}
    ts_path = os.path.join(out_dir, f"checkpoint-{max_steps}", "trainer_state.json")
    if os.path.exists(ts_path):
        hist = json.load(open(ts_path)).get("log_history", [])
        fzs = [(h.get("step"), h["frac_reward_zero_std"]) for h in hist
               if h.get("frac_reward_zero_std") is not None]
        ent = [h["entropy"] for h in hist if h.get("entropy") is not None]
        kl = [h["kl"] for h in hist if h.get("kl") is not None]
        rew = [h["reward"] for h in hist if h.get("reward") is not None]
        half = len(fzs) // 2
        third = max(1, len(fzs) // 3)
        r.update({
            "fzs_curve": [(s, round(v, 4)) for s, v in fzs],
            "fzs_mean_all": round(st.mean(v for _, v in fzs), 4) if fzs else None,
            "fzs_mean_2nd_half": round(st.mean(v for _, v in fzs[half:]), 4) if fzs else None,
            # SFOC 衰减判定: 前/中/后三段均值(F-a 用末段)
            "fzs_seg_early": round(st.mean(v for _, v in fzs[:third]), 4) if fzs else None,
            "fzs_seg_mid": round(st.mean(v for _, v in fzs[third:2*third]), 4) if fzs else None,
            "fzs_seg_late": round(st.mean(v for _, v in fzs[2*third:]), 4) if fzs else None,
            "entropy_final": round(ent[-1], 4) if ent else None,
            "kl_mean": round(st.mean(kl), 5) if kl else None,
            "reward_final": round(rew[-1], 4) if rew else None,
            "reward_min": round(min(rew), 4) if rew else None,
        })
    rep_path = os.path.join(out_dir, "efficiency_report.json")
    if os.path.exists(rep_path):
        rep = json.load(open(rep_path))
        recs = [x for x in rep.get("step_records", []) if x.get("elapsed_s", 0) > 0 and x.get("num_tokens", 0) > 0]
        if len(recs) >= 2:
            steps = recs[-1]["step"] - recs[0]["step"]
            r["T_step_s"] = round(sum(x["elapsed_s"] for x in recs[1:]) / steps, 2)
            r["steady_tps"] = round((recs[-1]["num_tokens"] - recs[0]["num_tokens"]) /
                                    sum(x["elapsed_s"] for x in recs[1:]), 1)
        r["peak_mem_gb"] = rep.get("peak_mem_gb")
    zst_path = os.path.join(out_dir, "zsbr_state.json")
    if os.path.exists(zst_path):
        zst = json.load(open(zst_path))
        r["coverage_selected"] = round(zst.get("n_prompts_selected", 0) / zst["config"]["N"], 3)
        hist_z = zst.get("history", [])
        if hist_z:
            r["dup_slots_mean"] = round(st.mean(h["dup_slots"] for h in hist_z), 2)
            r["psig_hat_mean_2nd_half"] = round(
                st.mean(h["mean_psig_hat"] for h in hist_z[len(hist_z)//2:]), 4)
            inv = [h.get("invest_slots", 0) for h in hist_z]
            if any(inv):
                r["invest_slots_mean"] = round(st.mean(inv), 2)
        conv = zst.get("conversion_log", [])
        if conv:
            r["invest_n"] = len(conv)
            r["invest_converted"] = sum(1 for c in conv if c.get("converted"))
            r["invest_conv_rate"] = round(r["invest_converted"] / len(conv), 3)
    return r


def run_one(arm, seed):
    max_steps = 500 if "_500" in arm else MAX_STEPS
    out_dir = os.path.join(RESULTS, f"{arm}_s{seed}")
    done = os.path.exists(os.path.join(out_dir, "final_adapter", "adapter_config.json"))
    if not done:
        os.makedirs(out_dir, exist_ok=True)
        env = os.environ.copy(); env["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"\n{'='*60}\n🚀 {arm}_s{seed}: gen128, {max_steps} 步\n{'='*60}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(train_cmd(arm, seed, out_dir, max_steps), env=env, cwd=BASE_DIR)
        if proc.returncode != 0:
            return {"arm": arm, "seed": seed, "error": f"train rc={proc.returncode}"}
        print(f"  ✓ 训练完成 {time.perf_counter()-t0:.0f}s", flush=True)
    else:
        print(f"⏭️  跳过训练 {arm}_s{seed}", flush=True)

    eval_json = os.path.join(out_dir, "gsm8k_eval_batch32.json")
    if not os.path.exists(eval_json):
        env = os.environ.copy(); env["HF_ENDPOINT"] = "https://hf-mirror.com"
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "eval/evaluate_gsm8k.py"),
                        "--adapter", os.path.join(out_dir, "final_adapter"),
                        "--n", str(EVAL_N), "--batch-size", str(EVAL_BATCH),
                        "--output", eval_json], env=env, cwd=BASE_DIR)
    acc = json.load(open(eval_json)).get("accuracy") if os.path.exists(eval_json) else None
    return {"arm": arm, "seed": seed, "max_steps": max_steps,
            "heldout_accuracy": acc, **extract_stats(out_dir, max_steps)}


def summarize(runs):
    """统计 + 闸门/预言判定。"""
    def by(arm):
        return [r for r in runs if r.get("arm") == arm and not r.get("error")]
    uni, v1, v2 = by("uniform"), by("zsbr_v1"), by("zsbr_v2")
    verdicts = []

    # P1/P5 信号率
    for arm_runs, tag, thr, pn in [(v1, "V1", 0.60, "P1"), (v2, "V2", 0.45, "P5")]:
        if arm_runs:
            fzs = [r["fzs_mean_2nd_half"] for r in arm_runs if r.get("fzs_mean_2nd_half")]
            if fzs:
                ok = st.mean(fzs) <= thr
                verdicts.append(f"{'✅' if ok else '❌'} {pn}({tag} fzs后半≤{thr}): "
                                f"实测 {[round(x,3) for x in fzs]} mean={st.mean(fzs):.3f}")
    # P2 吞吐不变
    if uni and v1:
        t_u = st.mean(r["T_step_s"] for r in uni if r.get("T_step_s"))
        for arm_runs, tag in [(v1, "V1"), (v2, "V2")]:
            if arm_runs:
                t_a = st.mean(r["T_step_s"] for r in arm_runs if r.get("T_step_s"))
                d = (t_a - t_u) / t_u * 100
                verdicts.append(f"{'✅' if abs(d)<3 else '❌'} P2({tag} T_step差<3%): {d:+.1f}% "
                                f"({t_a:.1f}s vs {t_u:.1f}s)")
    # P3 held-out 配对
    stats = {}
    if len(uni) >= 2 and len(v1) >= 2:
        au = [r["heldout_accuracy"] for r in uni if r.get("heldout_accuracy") is not None]
        a1 = [r["heldout_accuracy"] for r in v1 if r.get("heldout_accuracy") is not None]
        paired = []
        for s in SEEDS:
            ru = next((r for r in uni if r["seed"] == s), None)
            r1 = next((r for r in v1 if r["seed"] == s), None)
            if ru and r1 and ru.get("heldout_accuracy") is not None and r1.get("heldout_accuracy") is not None:
                paired.append(round((r1["heldout_accuracy"] - ru["heldout_accuracy"]) * 100, 1))
        ciu = bootstrap_confidence_interval(au); ci1 = bootstrap_confidence_interval(a1)
        stats = {"uniform": {"accs": au, "mean": round(st.mean(au), 4),
                             "ci95": [round(ciu[0], 4), round(ciu[1], 4)]},
                 "zsbr_v1": {"accs": a1, "mean": round(st.mean(a1), 4),
                             "ci95": [round(ci1[0], 4), round(ci1[1], 4)]},
                 "paired_diff_pp": paired, "cohens_d": round(cohens_d(a1, au), 2)}
        ok = all(d >= 0 for d in paired)
        verdicts.append(f"{'✅' if ok else '⚠️'} P3(V1 held-out≥uniform 配对): {paired}pp")

    # 安全检查(逐 run)
    for r in runs:
        if r.get("error"):
            continue
        issues = []
        if r.get("kl_mean") and r["kl_mean"] >= 0.01:
            issues.append(f"kl={r['kl_mean']}")
        if r.get("reward_min") is not None and r["reward_min"] < 0.15:
            issues.append(f"reward_min={r['reward_min']}")
        if issues:
            verdicts.append(f"⚠️ 安全[{r['arm']}_s{r['seed']}]: {','.join(issues)}")
    if uni:
        e_u = st.mean(r["entropy_final"] for r in uni if r.get("entropy_final"))
        for arm_runs, tag in [(v1, "V1"), (v2, "V2")]:
            ents = [r["entropy_final"] for r in arm_runs if r.get("entropy_final")]
            if ents and st.mean(ents) < 0.5 * e_u:
                verdicts.append(f"⚠️ 安全: {tag} entropy {st.mean(ents):.3f} < uniform 50%({e_u:.3f})")

    # 有效信号组/秒
    eff = {}
    for arm_runs, tag in [(uni, "uniform"), (v1, "zsbr_v1"), (v2, "zsbr_v2")]:
        vals = [(1 - r["fzs_mean_2nd_half"]) * 64 / r["T_step_s"]
                for r in arm_runs if r.get("fzs_mean_2nd_half") and r.get("T_step_s")]
        if vals:
            eff[tag] = round(st.mean(vals), 3)
    if "uniform" in eff:
        for tag in ["zsbr_v1", "zsbr_v2"]:
            if tag in eff:
                verdicts.append(f"📈 有效信号组/秒 {tag}: {eff[tag]} vs uniform {eff['uniform']} "
                                f"= {eff[tag]/eff['uniform']:.2f}×")
    return stats, eff, verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["s42", "full", "v2", "flux500", "sfc500", "sfc500_full",
                                        "depl500", "depl200"],
                    default="s42",
                    help="s42/full/v2=docs20阶段; flux500=v1_500衰减判决(E-F1); "
                         "sfc500=sfc_500_s42对照(E-F2); sfc500_full=补seeds; "
                         "depl500=小池500 v1+sfc 链(E-G1); depl200=耗竭态终验链(E-H1, docs/21 §11)")
    args = ap.parse_args()

    plan = {"s42": [("uniform", 42), ("zsbr_v1", 42)],
            "full": [("uniform", s) for s in SEEDS] + [("zsbr_v1", s) for s in SEEDS],
            "v2": [("zsbr_v2", 42)],
            "flux500": [("zsbr_v1_500", 42)],
            "sfc500": [("zsbr_sfc_500", 42)],
            "sfc500_full": [("zsbr_v1_500", s) for s in [123, 7]] +
                           [("zsbr_sfc_500", s) for s in [123, 7]],
            "depl500": [("zsbr_v1_500_p500", 42), ("zsbr_sfc_500_p500", 42)],
            "depl200": [("zsbr_v1_500_p200", 42), ("zsbr_sfc_500_p200", 42)]}[args.stage]

    runs = []
    # 载入已有结果(增量)
    sum_path = os.path.join(RESULTS, "summary.json")
    if os.path.exists(sum_path):
        runs = [r for r in json.load(open(sum_path)).get("runs", []) if not r.get("error")]
    done_keys = {(r["arm"], r["seed"]) for r in runs}

    for arm, seed in plan:
        if (arm, seed) in done_keys:
            # 重新提取(可能有新文件), 替换旧记录
            runs = [r for r in runs if not (r["arm"] == arm and r["seed"] == seed)]
        try:
            runs.append(run_one(arm, seed))
        except Exception as e:
            print(f"  ✗ {arm}_s{seed} 异常: {e}", flush=True)
            runs.append({"arm": arm, "seed": seed, "error": str(e)})
        stats, eff, verdicts = summarize(runs)
        with open(sum_path, "w") as f:
            json.dump({"metadata": {"experiment": "ZSBR/SFOC ablation",
                                    "last_stage": args.stage,
                                    "config": "gen128 pdb4 G=2, eval batch32 n500; _500臂=500步, _p500臂=训练池前500题",
                                    "gates": "docs/20 P1-P5(仅 uniform/v1/v2 臂); "
                                             "E-F/E-G 预言判定在 docs/21 预注册+原始数据复算, 不经本 verdicts"},
                       "runs": runs, "stats": stats, "signal_groups_per_s": eff,
                       "verdicts": verdicts}, f, ensure_ascii=False, indent=2)
        time.sleep(10)

    print(f"\n{'═'*72}\n  ZSBR 消融结果 ({args.stage})\n{'═'*72}")
    for r in runs:
        if r.get("error"):
            print(f"  {r['arm']}_s{r['seed']}: ERROR {r['error'][:40]}")
        else:
            acc = f"{r['heldout_accuracy']:.1%}" if r.get("heldout_accuracy") is not None else "N/A"
            print(f"  {r['arm']:<10} s{r['seed']:<4} fzs2nd={r.get('fzs_mean_2nd_half','?')} "
                  f"T_step={r.get('T_step_s','?')}s acc={acc} cov={r.get('coverage_selected','-')}")
    _, _, verdicts = summarize(runs)
    for v in verdicts:
        print(f"  {v}")
    print(f"\n  结果: {sum_path}")


if __name__ == "__main__":
    main()
