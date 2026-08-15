#!/usr/bin/env python3
"""Legal-PT-01 评估 v5 — 直接加载合并后完整模型 (无 PEFT)"""

import os, re, json, gc, time
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
import torch

PROJ = REPO_ROOT
BASE = os.path.join(MODELS_DIR, "Qwen3.5-4B/models/Qwen--Qwen3.5-4B/snapshots/master")

models = {
    "Baseline": BASE,
    "SFT-only": f"{PROJ}/results/sft_merged",
    "GRPO-v3": f"{PROJ}/results/grpo_v3_merged",
    "H2-G4": f"{PROJ}/results/h2_merged",
}

eval_dir = f"{PROJ}/eval/prompts"
datasets = {}
for fname in ["casehold_us.jsonl","lawbench_cn.jsonl","gdpr_eu.jsonl"]:
    path = os.path.join(eval_dir, fname)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            datasets[fname.replace(".jsonl","")] = [json.loads(l) for l in f]
if not datasets:
    print("❌ 无评估数据"); exit(1)

def extract_answer(text):
    t = text.strip()
    m = re.search(r"(?:^|[^A-Za-z])([A-D])(?:[^A-Za-z]|$)", t)
    if m: return m.group(1)
    for yn in ["YES","NO","是","否","有效","无效","成立","不成立"]:
        if yn in t.upper() or yn in t: return yn.upper()
    m = re.search(r"(?:答案|结论|综上)[：:\s]+(.+?)(?:[。！？\n]|$)", t, re.IGNORECASE)
    if m:
        content = m.group(1).strip()[:30]
        lm = re.search(r"\b([A-D])\b", content)
        if lm: return lm.group(1)
        return content
    return "N/A"

def match_answer(pred, gold):
    p, g = pred.strip().upper(), str(gold).strip().upper()
    if not p or p == "N/A": return False
    if p == g: return True
    pl, gl = re.fullmatch(r"([A-D])", p), re.fullmatch(r"([A-D])", g)
    if pl and gl: return pl.group(1) == gl.group(1)
    return False

print("=" * 55); print("Legal-PT-01 评估 v5 (合并模型)")
for ds, s in datasets.items(): print(f"  {ds}: {len(s)} prompts")
print("=" * 55)

from transformers import AutoModelForCausalLM, AutoTokenizer
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))

results = {}; model = None

for model_name, model_path in models.items():
    if model is not None: del model, tokenizer
    gc.collect(); torch.cuda.empty_cache(); time.sleep(2)
    print(f"\n[{model_name}]", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM: {vram:.1f} GB", flush=True)

    jur_scores = {}; all_c = all_t = 0
    for ds_name, samples in datasets.items():
        c = t = 0
        for i, s in enumerate(samples):
            prompt = s.get("prompt", s.get("question",""))
            answer = s.get("answer", s.get("output",""))
            if not prompt or not answer: continue
            messages = [{"role":"user","content":prompt}]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, max_length=512, truncation=True,
                enable_thinking=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            if match_answer(extract_answer(gen), answer): c += 1
            t += 1
            if (i+1) % 50 == 0: print(f"  {ds_name} {i+1}/{len(samples)}", flush=True)
        acc = c / max(t,1); jur_scores[ds_name] = (c,t,acc); all_c += c; all_t += t
        print(f"  {ds_name} = {acc:.1%} ({c}/{t})", flush=True)
    avg = all_c / max(all_t,1); print(f"  avg = {avg:.1%}", flush=True)
    results[model_name] = (jur_scores, avg)

print(f"\n{'='*55}")
header = f"{'Model':<15s}" + "".join(f"{d:>14s}" for d in datasets) + f"{'Avg':>10s}"
print(header); print("-"*len(header))
for mn, (jd, avg) in results.items():
    row = f"{mn:<15s}"
    for ds in datasets:
        c,t,a = jd.get(ds,(0,1,0))
        row += f"{a:>13.1%} "
    row += f"{avg:>9.1%}"
    print(row)
print("="*55)

with open(f"{PROJ}/results/eval_v5.json","w",encoding="utf-8") as f:
    json.dump({k:{dk:{"c":dv[0],"t":dv[1],"acc":round(dv[2],4)} for dk,dv in v[0].items()} for k,v in results.items()}, f, ensure_ascii=False, indent=2)
print("\n✅ 已保存")
