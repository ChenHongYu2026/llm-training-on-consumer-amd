#!/usr/bin/env python3
"""C.T2 步骤 1: E2B base 在 GSM8K 训练池上的通过率摸底 (docs/36 §2 P-C2 前置)。

零训练纯推理: 对训练池全部 7473 题跑 greedy 解码(非 thinking 模板, 256 token),
记录逐题 pass/fail → 输出 p 分布统计(均值/直方图/S₀ 估计), 供 ZSBR 容量阶梯
先验折算使用。

用法:
  venv_torch211/bin/python eval/pass_rate_probe.py \
      --base {MODELS_DIR}/gemma-4-E2B-it \
      --output results/efficiency/e2b/pass_rate_train_pool.json
"""
import argparse
import json
import os
import re
import sys
import time

import torch
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))



def normalize_number(text: str) -> str:
    """提取数字并规范化(与 evaluate_gsm8k.py 同口径)。"""
    text = text.replace(",", "").replace("$", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return ""
    f = float(nums[-1])
    if f.is_integer():
        return str(int(f))
    return f"{f:.4f}".rstrip("0").rstrip(".")


def extract_model_answer(text: str) -> str:
    """从模型输出尾部提取答案数字(同 evaluate_gsm8k.py)。"""
    text = text.strip()
    # 优先取最后一行(模型最后陈述答案)
    lines = [l for l in text.split("\n") if l.strip()]
    for line in reversed(lines):
        n = normalize_number(line)
        if n:
            return n
    return normalize_number(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--max-prompts", type=int, default=0,
                    help="0=全池 7473; 否则只测前 N 题(冒烟用)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    is_gemma4 = AutoConfig.from_pretrained(args.base).model_type == "gemma4"
    print(f"[probe] base={args.base} gemma4={is_gemma4}", flush=True)

    if is_gemma4:
        from transformers import AutoModelForMultimodalLM
        model = AutoModelForMultimodalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="eager", low_cpu_mem_usage=True)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="eager", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    n = args.max_prompts if args.max_prompts > 0 else len(ds)
    ds = ds.select(range(n))
    print(f"[probe] 训练池题数: {n}", flush=True)

    per_item = []
    t0 = time.perf_counter()
    for s in range(0, len(ds), args.batch_size):
        batch = ds.select(range(s, min(s + args.batch_size, len(ds))))
        texts, gts = [], []
        for ex in batch:
            gts.append(normalize_number(re.findall(r"-?\d+(?:\.\d+)?",
                         ex["answer"].replace(",", "").replace("$", ""))[-1]))
            messages = [{"role": "user", "content":
                         f"Solve the following math problem step by step. Give your final answer as a number.\n\nProblem: {ex['question']}"}]
            if is_gemma4:
                texts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False))
            else:
                texts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True))
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for j, gt in enumerate(gts):
            gen = tokenizer.decode(out[j][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            pred = extract_model_answer(gen)
            per_item.append(1 if (gt and pred and pred == gt) else 0)
        done = min(s + args.batch_size, len(ds))
        if done % 256 == 0 or done == len(ds):
            print(f"  {done}/{len(ds)} pass={sum(per_item)/len(per_item):.1%}", flush=True)

    elapsed = time.perf_counter() - t0
    n_ok = sum(per_item)
    p_mean = n_ok / len(per_item)
    # S₀ 估计(G=2): E[2p(1-p)]
    s0 = 2.0 * p_mean * (1.0 - p_mean)
    hist = {}
    for v in per_item:
        hist[str(v)] = hist.get(str(v), 0) + 1
    result = {
        "base": args.base, "n": len(per_item),
        "pass_count": n_ok, "p_mean": round(p_mean, 4),
        "S0_estimate_G2": round(s0, 4),
        "per_item": per_item, "histogram": hist,
        "elapsed_s": round(elapsed, 1),
        "note": "非 thinking 模板, greedy, max_new_tokens=256",
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[probe] p_mean={p_mean:.1%} | S₀(G=2)≈{s0:.3f} | {elapsed:.0f}s")
    print(f"[probe] 结果: {args.output}")


if __name__ == "__main__":
    main()
