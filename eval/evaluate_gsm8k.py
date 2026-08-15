#!/usr/bin/env python3
"""
GSM8K held-out 准确率评估
==========================
加载 base(Qwen2.5-3B) + 可选 final_adapter, 在 GSM8K test[:N] 上 greedy 生成并判对。
复用 train/train_gsm8k_grpo.py 的答案匹配逻辑, 保证与训练 reward 口径一致。

用法:
  sg render -c "cd {REPO_ROOT} && python3 -u eval/evaluate_gsm8k.py \
      --adapter results/efficiency/grpo_convergence/best/final_adapter --n 500 --output <json>"
  # 不传 --adapter 则评估 base 模型 (基准锚点)
  # --batch-size 32 (默认): 左 pad 批量 greedy 生成, 较单流快 ~10×; --batch-size 1 退回单流
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import re
import json
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


BASE_MODEL = os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct")


# ── 复用 train 脚本的答案匹配逻辑 ──
def extract_gsm8k_answer(answer_text: str) -> str:
    m = re.search(r"####\s*([\d,]+(?:\.\d+)?)", answer_text)
    return m.group(1).replace(",", "").strip() if m else ""


def normalize_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return str(float(s)) if "." in s else str(int(s))
    except (ValueError, TypeError):
        return s


def extract_model_answer(text: str) -> str:
    text = text.strip()
    m = re.search(r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*\$?([\d,]+(?:\.\d+)?)", text, re.I)
    if m:
        return normalize_number(m.group(1))
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return normalize_number(m.group(1))
    nums = re.findall(r"[\d,]+(?:\.\d+)?", text)
    return normalize_number(nums[-1]) if nums else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None, help="final_adapter 路径; 缺省评估 base")
    ap.add_argument("--base", type=str, default=BASE_MODEL)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="批量生成大小(左pad); 1=单流(与历史结果同口径)")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--enable-thinking", action="store_true", default=False,
                    help="Gemma4: thinking 模板开关(C.T0 锚点需关/开各一次)")
    args = ap.parse_args()

    # Gemma4 检测: 多模态 API 适配 (docs/36 §0)
    from transformers import AutoConfig as _AC
    _is_gemma4 = _AC.from_pretrained(args.base).model_type == "gemma4"

    print(f"[评估] base={args.base} | adapter={args.adapter} | n={args.n} | gemma4={_is_gemma4}")
    if _is_gemma4:
        from transformers import AutoModelForMultimodalLM
        model = AutoModelForMultimodalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager", low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.adapter and os.path.exists(os.path.join(args.adapter, "adapter_config.json")):
        from peft import PeftModel
        print(f"  加载 adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    # 批量生成需左 pad(保证生成紧接 prompt 末尾); 逐条提取/判对逻辑与单流完全一致
    tokenizer.padding_side = "left"
    correct = 0
    done = 0
    per_item = []          # 逐题 0/1(配对检验用, 审计整改项 docs/21 §9.5-2)
    t0 = time.perf_counter()
    for s in range(0, len(ds), args.batch_size):
        batch = ds.select(range(s, min(s + args.batch_size, len(ds))))
        texts = []
        gts = []
        for ex in batch:
            gts.append(normalize_number(extract_gsm8k_answer(ex["answer"])))
            messages = [{"role": "user", "content":
                         f"Solve the following math problem step by step. Give your final answer as a number.\n\nProblem: {ex['question']}"}]
            if _is_gemma4:
                texts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=args.enable_thinking))
            else:
                texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for j, gt in enumerate(gts):
            gen = tokenizer.decode(out[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_model_answer(gen)
            ok = 1 if (gt and pred and pred == gt) else 0
            correct += ok
            per_item.append(ok)
        done += len(gts)
        print(f"  {done}/{args.n}  acc={correct/done:.1%}", flush=True)
    elapsed = time.perf_counter() - t0

    acc = correct / len(ds)
    result = {
        "adapter": args.adapter, "n": len(ds), "correct": correct,
        "accuracy": round(acc, 4), "elapsed_s": round(elapsed, 1),
        "batch_size": args.batch_size,
        "per_item_correct": per_item,
    }
    print(f"\n准确率: {acc:.1%} ({correct}/{len(ds)}) | {elapsed:.0f}s")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果: {args.output}")


if __name__ == "__main__":
    main()
