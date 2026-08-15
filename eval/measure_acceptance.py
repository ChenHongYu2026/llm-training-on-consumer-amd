#!/usr/bin/env python3
"""
B.T2: 策略漂移 vs drafter 接受率 — 零成本数据考古测量脚本
============================================================
对多个训练阶段 checkpoint 的策略, 用固定 n-gram drafter 测量接受率 α(policy_t):
  - 接受率 α = 接受 token 数 / draft token 数 (每步接受长度 τ 一并统计)
  - 曲线: α(t) for t ∈ {0, 100, 200, 300, 400, 500} (checkpoint 可用性见 docs/35 §6)
  - 200 条 GSM8K prompt: 训练分布 100 + held-out 100, temperature 与训练一致(0.8)

预注册预言: docs/35 §1 (P-B2a: α 单调下降; P-B2b: 训练分布下降快于 held-out;
P-B2c: 与 KL 相关, 若可反推)

用法:
  sg render -c "cd {REPO_ROOT} && python3 eval/measure_acceptance.py \
      --checkpoints base:{MODELS_DIR}/Qwen2.5-3B-Instruct,ckpt100:/path/ckpt100 \
      --prompts-file data/gsm8k_bt2_prompts.json --output results/efficiency/bt2/acceptance.json"

drafter: n-gram prompt-lookup (与 B.T1 候选 1 一致, 零训练);
依赖 GSM8K 数字与公式的重复性, 在 base 分布上构建(与策略无关)。
"""

import os
import sys
import json
import time
import argparse

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


# ═══════════════════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════════════════
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoints", type=str, required=True,
                help="逗号分隔 tag:path 列表, 如 base:/models/Qwen2.5-3B-Instruct,ckpt100:/path")
ap.add_argument("--prompts-file", type=str, required=True, help="200 条 prompt 的 JSON 文件")
ap.add_argument("--output", type=str, required=True, help="结果 JSON 路径")
ap.add_argument("--max-new-tokens", type=int, default=256, help="每条 prompt 最大生成长度(与训练一致)")
ap.add_argument("--n-gram", type=int, default=5, help="n-gram 匹配长度")
ap.add_argument("--max-draft", type=int, default=5, help="每次最多 draft token 数")
ap.add_argument("--temperature", type=float, default=0.8, help="采样温度(与训练 rollout 一致)")
ap.add_argument("--max-prompts", type=int, default=0, help="调试用: 只跑前 N 条(0=全部)")
args = None  # main() 中赋值; 模块可导入供单测


def load_prompts(path: str, max_prompts: int = 0):
    """JSON 格式: [{"prompt": "...", "split": "train|heldout"}, ...]"""
    with open(path) as f:
        data = json.load(f)
    if max_prompts:
        data = data[:max_prompts]
    return data


def ngram_draft(prompt_ids: torch.Tensor, gen_ids: torch.Tensor, n: int, max_draft: int):
    """n-gram prompt-lookup draft: 末尾 n-gram 在 prompt 历史中匹配, 返回匹配点后续 token。
    语义: 模型复现 prompt 内模式后, 倾向于继续该模式后续 → draft = prompt 匹配点后的
    未生成 token (仅限匹配点在 prompt 内; 匹配点在已生成区域无新信息, 不 draft)。
    返回 (draft token 列表, 匹配位置 idx); 无匹配返回 ([], -1)。
    实现注: 2026-08-12 修复——原实现用 GPU tensor 逐位 torch.equal 搜索(每次触发
    GPU 同步, 200 prompts×256 步需数小时); 改为 CPU int list 搜索(纯 Python, 无同步)。"""
    seq = prompt_ids.tolist() + gen_ids.tolist()
    if len(seq) < n:
        return [], -1
    tail = seq[-n:]
    idx = -1
    for i in range(len(seq) - n - 1, -1, -1):
        if seq[i : i + n] == tail:
            idx = i
            break
    if idx < 0 or idx + n >= len(prompt_ids):
        return [], idx  # 无匹配或匹配点在已生成区域
    start = idx + n
    end = min(start + max_draft, len(prompt_ids))
    return seq[start:end], idx


def measure_checkpoint(model, tok, prompts, tag: str):
    """对单一 checkpoint 策略测量接受率。"""
    accepted_total = 0
    drafted_total = 0
    accept_len_sum = 0
    n_prompts = 0
    t0 = time.time()

    for item in prompts:
        text = item["prompt"]
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to("cuda:0")
        prompt_ids = enc["input_ids"][0]
        gen_ids = torch.tensor([], dtype=torch.long, device="cuda:0")
        step_accepted = 0
        step_drafted = 0
        # 逐步生成 + 手动 draft/verify
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                draft, _ = ngram_draft(prompt_ids, gen_ids, args.n_gram, args.max_draft)
                need_forward = True
                if draft:
                    # verify: 一次 forward 拿完整 logits (2026-08-12 优化: 拒绝位置采样复用此 logits)
                    draft_t = torch.tensor(draft, device="cuda:0")
                    full = torch.cat([prompt_ids, gen_ids])
                    inputs = torch.cat([full, draft_t]).unsqueeze(0)
                    logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits[0]
                    drafted_total += len(draft)
                    step_drafted += len(draft)
                    probs = torch.softmax(logits / args.temperature, dim=-1)
                    acc = 0
                    for i in range(len(draft)):
                        p = probs[full.size(0) + i - 1, draft_t[i]].item()
                        # 概率接受近似: p > 0.5 即接受 (n-gram 无显式 draft 分布;
                        # 论文口径用 B.T1 选型后的真实 drafter 校准)
                        if p > 0.5:
                            acc += 1
                        else:
                            break
                    accepted_total += acc
                    step_accepted += acc
                    accept_len_sum += acc
                    if acc == len(draft):
                        gen_ids = torch.cat([gen_ids, draft_t])
                        need_forward = False
                    elif acc > 0:
                        gen_ids = torch.cat([gen_ids, draft_t[:acc]])
                        # 部分接受: 从拒绝位置(acc)复用 logits 采样
                        p_rej = probs[full.size(0) + acc - 1] / args.temperature
                        nxt = torch.multinomial(torch.softmax(p_rej, dim=-1), 1)
                        gen_ids = torch.cat([gen_ids, nxt])
                        need_forward = False
                        if nxt.item() == tok.eos_token_id:
                            break
                if need_forward:
                    # 无 draft 或全部拒绝: 确定性 argmax (2026-08-12: multinomial 采样
                    # 在 s7_ckpt100 场景触发 ROCm 无痕崩溃, 改 argmax 提高稳定性;
                    # α 口径 = draft 与策略 argmax 的一致性, 确定性可复现)
                    inputs = torch.cat([prompt_ids, gen_ids]).unsqueeze(0)
                    logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits[0, -1]
                    nxt = torch.argmax(logits, dim=-1, keepdim=True)
                    gen_ids = torch.cat([gen_ids, nxt])
                    if nxt.item() == tok.eos_token_id:
                        break
        n_prompts += 1
        if n_prompts % 20 == 0:
            print(f"  [{tag}] {n_prompts}/{len(prompts)} prompts, 累计 α={accepted_total/max(drafted_total,1):.3f}", flush=True)

    dt = time.time() - t0
    alpha = accepted_total / max(drafted_total, 1)
    return {
        "tag": tag,
        "n_prompts": n_prompts,
        "drafted_tokens": drafted_total,
        "accepted_tokens": accepted_total,
        "alpha": round(alpha, 4),
        "mean_accept_len": round(accept_len_sum / max(n_prompts, 1), 2),
        "time_s": round(dt, 1),
        "note": "n-gram draft + 确定性接受近似(p>0.5); 与 B.T1 选型后真实 drafter 校准口径待定",
    }


def main():
    global args
    args = ap.parse_args()
    ckpts = [c.split(":", 1) for c in args.checkpoints.split(",")]
    prompts = load_prompts(args.prompts_file, args.max_prompts)
    print(f"checkpoints: {[t for t, _ in ckpts]} | prompts: {len(prompts)}")

    results = []
    for tag, path in ckpts:
        print(f"加载 {tag} <- {path} ...", flush=True)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
            # LoRA adapter: 显式 base + adapter (2026-08-12 修复)
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct"), torch_dtype=torch.bfloat16,
                device_map="cuda:0")
            model = PeftModel.from_pretrained(base_model, path)
            tok = AutoTokenizer.from_pretrained(os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct"))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="cuda:0")
            tok = AutoTokenizer.from_pretrained(path)
        tok.pad_token = tok.eos_token
        # 分训练/held-out 统计
        train_prompts = [p for p in prompts if p.get("split", "train") == "train"]
        held_prompts = [p for p in prompts if p.get("split") == "heldout"]
        r_train = measure_checkpoint(model, tok, train_prompts, f"{tag}/train")
        r_held = measure_checkpoint(model, tok, held_prompts, f"{tag}/heldout")
        results.append({"train": r_train, "heldout": r_held})
        del model
        torch.cuda.empty_cache()

    out = {"config": vars(args), "results": results}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"✅ 结果: {args.output}")
    for r in results:
        print(f"  {r['train']['tag']}: α_train={r['train']['alpha']} α_held={r['heldout']['alpha']}")


if __name__ == "__main__":
    main()
