#!/usr/bin/env python3
"""
GSM8K GRPO 训练 — Legal-PT-01 效率研究 Phase B/C 主 workload
============================================================
角色：
  - GRPO 效率主战场（生成密集型，最考验框架效率）
  - 质量锚点（数学答案唯一，准确率可靠）
  - 框架横向对比载体（--framework unsloth|hf 切换）

奖励函数：纯规则（精确匹配最终数字答案），不依赖 LLM judge。

效率测量：集成 utils/efficiency_profiler.py 双峰值 MFU + 测量有效性保障。

用法：
  # Unsloth 模式（默认）
  python3 train/train_gsm8k_grpo.py --max-steps 500

  # HF 原生模式（框架对比基线）
  python3 train/train_gsm8k_grpo.py --framework hf --max-steps 500

  # Smoke test（验证管线 + profiler，20步）
  python3 train/train_gsm8k_grpo.py --smoke-test
"""

import os, re, sys, json, time, argparse
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像

# =============================================================================
# CLI 参数
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="GSM8K GRPO 效率实验")
    ap.add_argument("--framework", choices=["unsloth", "hf"], default="unsloth",
                    help="训练框架: unsloth(默认) 或 hf(原生 Transformers+PEFT)")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--num-generations", type=int, default=4, help="GRPO G")
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--max-prompt-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1, help="KL 系数")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--model-path", type=str, default=None,
                    help="模型路径（默认用 Qwen3.5-4B，可切换为 Qwen2.5 等标准架构）")
    ap.add_argument("--no-sft-adapter", action="store_true",
                    help="不加载 SFT adapter（用于非法律模型的纯效率测试）")
    ap.add_argument("--smoke-test", action="store_true", help="20步快速验证")
    ap.add_argument("--no-profiler", action="store_true", help="禁用效率采集")
    ap.add_argument("--no-4bit", action="store_true",
                    help="Unsloth 不用 4-bit 量化（BF16 公平对比模式）")
    ap.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="禁用 gradient checkpointing（Phase C sweep 用）")
    ap.add_argument("--logging-steps", type=int, default=5,
                    help="日志/profiler 采样间隔步数（默认 5；sweep 用 2 以小 max_steps 拿多个 steady 区间）")
    ap.add_argument("--save-steps", type=int, default=100,
                    help="checkpoint 保存间隔（长训练如 1600 步设大值免中间 checkpoint 堆积）")
    # ── ZSBR: 零信号预算重分配 (docs/20) ──
    ap.add_argument("--zsbr", choices=["off", "v1", "v2", "sfc"], default="off",
                    help="零信号预算重分配: off=均匀, v1=选择器, v2=多重入选, sfc=信号通量控制(docs/21)")
    ap.add_argument("--zsbr-epsilon", type=float, default=0.2, help="ε-greedy 覆盖比例")
    ap.add_argument("--zsbr-cooldown", type=int, default=3,
                    help="冷却周期数(0=禁用冷却, A.T1 去冷却变体; docs/34 P-A1a)")
    ap.add_argument("--zsbr-kmax", type=int, default=3, help="V2/SFC 同 prompt 最大组槽位数")
    ap.add_argument("--zsbr-gamma", type=float, default=0.98, help="p̂ 时间衰减系数")
    ap.add_argument("--zsbr-invest-ratio", type=float, default=0.25,
                    help="SFC 投资槽比例 ι(仅 sfc 模式生效; 0=严格退化为 v1)")
    ap.add_argument("--reward-log", action="store_true",
                    help="逐条记录 (step,idx,reward) 到 output/reward_log.jsonl(E-A 校准用, 独立于 zsbr)")
    ap.add_argument("--pool-size", type=int, default=0,
                    help="训练池截断前 N 题(E-G1 耗竭 regime 实验, docs/21 §10; 0=全量)")
    return ap.parse_args()

args = parse_args()

# Smoke test 覆盖
if args.smoke_test:
    args.max_steps = 20
    args.num_generations = 2
    args.grad_accum = 2

# 输出目录
if args.output is None:
    tag = f"gsm8k-grpo-{args.framework}-{time.strftime('%Y%m%d-%H%M%S')}-seed{args.seed}"
    args.output = f"{REPO_ROOT}/results/efficiency/{tag}"
os.makedirs(args.output, exist_ok=True)

# 保存实验配置
with open(os.path.join(args.output, "config.json"), "w") as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=2)

print("=" * 60)
print(f"GSM8K GRPO | framework={args.framework} | G={args.num_generations}")
print(f"max_steps={args.max_steps} | lr={args.lr} | beta={args.beta}")
print(f"completion={args.max_completion_length} | batch={args.per_device_batch}×{args.grad_accum}")
print(f"output: {args.output}")
print("=" * 60)

# =============================================================================
# 框架加载（Unsloth 必须最先 import）
# =============================================================================
if args.framework == "unsloth":
    import unsloth  # noqa: F401 — 必须最先
    print("✅ Unsloth 已加载")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# =============================================================================
# 数据准备：GSM8K
# =============================================================================
print("\n📦 加载 GSM8K 数据集...")
dataset = load_dataset("openai/gsm8k", "main", split="train")
if args.pool_size and args.pool_size < len(dataset):
    dataset = dataset.select(range(args.pool_size))   # 前 N 题; idx 语义不变(0..N-1)
    print(f"   ✂️ 训练池截断: 前 {args.pool_size} 题 (E-G1 耗竭 regime)")
print(f"   训练集: {len(dataset)} 条")

# 提取标准答案数字
def extract_gsm8k_answer(answer_text: str) -> str:
    """从 GSM8K answer 字段提取最终数字（#### 后面）。"""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""

def normalize_number(s: str) -> str:
    """标准化数字字符串用于比较。"""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return str(float(s)) if "." in s else str(int(s))
    except (ValueError, TypeError):
        return s

# 构建 prompt 格式
def format_prompt(example):
    """将 GSM8K 问题格式化为 GRPO prompt。"""
    return [{"role": "user", "content": f"Solve the following math problem step by step. Give your final answer as a number.\n\nProblem: {example['question']}"}]

# 提取 ground truth 映射
gt_answers = {}
for i, ex in enumerate(dataset):
    gt_answers[i] = normalize_number(extract_gsm8k_answer(ex["answer"]))

# 转为 TRL 格式
def prepare_dataset(ds):
    """转为 GRPOTrainer 需要的 prompt 格式。"""
    def transform(example, idx):
        return {"prompt": format_prompt(example), "idx": idx}
    return ds.map(lambda ex, idx: transform(ex, idx), with_indices=True,
                  remove_columns=ds.column_names)

train_ds = prepare_dataset(dataset)
print(f"✅ 数据准备完成: {len(train_ds)} prompts")

# =============================================================================
# ZSBR 调度器 (docs/20; --zsbr off 时完全旧行为)
# =============================================================================
zsbr_scheduler = None
if args.zsbr != "off":
    sys.path.insert(0, REPO_ROOT)
    from train.zsbr_scheduler import ZSBRScheduler, ZSBRSampler  # noqa: F401
    zsbr_scheduler = ZSBRScheduler(
        num_prompts=len(train_ds), G=args.num_generations, mode=args.zsbr,
        epsilon=args.zsbr_epsilon, k_max=args.zsbr_kmax,
        cooldown=args.zsbr_cooldown,
        gamma=args.zsbr_gamma, invest_ratio=args.zsbr_invest_ratio, seed=args.seed)
    print(f"✅ ZSBR 调度器: mode={args.zsbr} ε={args.zsbr_epsilon} kmax={args.zsbr_kmax} "
          f"cooldown={args.zsbr_cooldown} γ={args.zsbr_gamma} "
          f"ι={args.zsbr_invest_ratio if args.zsbr=='sfc' else '-'}")

_reward_log_path = os.path.join(args.output, "reward_log.jsonl") if args.reward_log else None

# =============================================================================
# 奖励函数：精确匹配最终数字
# =============================================================================
def extract_model_answer(text: str) -> str:
    """从模型生成中提取最终数字答案。多策略。"""
    text = text.strip()
    # 策略1: "The answer is X" 模式
    m = re.search(r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*\$?([\d,]+(?:\.\d+)?)", text, re.I)
    if m:
        return normalize_number(m.group(1))
    # 策略2: \boxed{X}
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return normalize_number(m.group(1))
    # 策略3: 最后一个独立数字
    nums = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if nums:
        return normalize_number(nums[-1])
    return ""

def reward_function(completions, prompts=None, **kwargs):
    """GSM8K 规则奖励：正确=1.0，格式奖励=0.1，错误=0.0。

    返回 list[float]，每个 completion 一个分数。
    """
    rewards = []
    # completions 可能是 list[list[dict]] 或 list[str]
    batch_indices = kwargs.get("idx", None)

    for i, comp in enumerate(completions):
        # 提取文本
        if isinstance(comp, list) and comp:
            text = comp[-1].get("content", "") if isinstance(comp[-1], dict) else str(comp[-1])
        elif isinstance(comp, str):
            text = comp
        else:
            text = str(comp)

        # 获取 ground truth
        idx = batch_indices[i] if batch_indices is not None and i < len(batch_indices) else i
        gt = gt_answers.get(int(idx) if idx is not None else i, "")

        # 提取模型答案
        pred = extract_model_answer(text)

        # 打分
        score = 0.0
        if gt and pred:
            if pred == gt:
                score = 1.0  # 精确匹配
            else:
                # 容错：数值接近（±1%）
                try:
                    if abs(float(pred) - float(gt)) / max(abs(float(gt)), 1e-9) < 0.01:
                        score = 0.9
                except (ValueError, TypeError):
                    pass
        # 格式奖励：有明确答案表述
        if re.search(r"(?:answer|答案)\s+is|\\boxed", text, re.I):
            score = max(score, 0.1)  # 至少给格式分

        rewards.append(score)

    # ── ZSBR 钩子: 在线更新 p̂ + 同组同 prompt 断言 + 可选 (idx,reward) 日志 ──
    if batch_indices is not None:
        G = args.num_generations
        if zsbr_scheduler is not None and len(batch_indices) % G == 0:
            # 断言1: 每组 G 条必须同 prompt(保 advantage 计算正确; 违例=采样器结构被破坏)
            for g in range(0, len(batch_indices), G):
                grp = [int(batch_indices[g + j]) for j in range(G)]
                assert len(set(grp)) == 1, f"ZSBR 同组不同 prompt: {grp} @ 组{g//G}"
            for i2, idx2 in enumerate(batch_indices):
                zsbr_scheduler.update(int(idx2), float(rewards[i2]))
        if _reward_log_path:
            _st = kwargs.get("trainer_state")
            _step = getattr(_st, "global_step", -1) if _st is not None else -1
            with open(_reward_log_path, "a") as _f:
                for i2, idx2 in enumerate(batch_indices):
                    _f.write(json.dumps({"step": _step, "idx": int(idx2),
                                         "reward": float(rewards[i2])}) + "\n")

    return rewards

# =============================================================================
# 模型加载
# =============================================================================
# 模型路径（支持切换）
DEFAULT_MODEL = os.path.join(MODELS_DIR, "Qwen3.5-4B/models/Qwen--Qwen3.5-4B/snapshots/master")
BASE = args.model_path or DEFAULT_MODEL
SFT_ADAPTER = REPO_ROOT + "/results/sft"

# Gemma4 多模态检测 (WS-C C.T0, docs/36 §0): 需 MultimodalLM 加载 + 文本模板字符串
from transformers import AutoConfig as _AC_PRE
IS_GEMMA4 = _AC_PRE.from_pretrained(BASE).model_type == "gemma4"

print(f"\n🔧 加载模型: {BASE}")
print(f"   框架: {args.framework}")

if args.framework == "unsloth":
    from unsloth import FastLanguageModel
    _load_4bit = not args.no_4bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE,
        max_seq_length=args.max_prompt_length + args.max_completion_length,
        load_in_4bit=_load_4bit,
        dtype=torch.bfloat16,
    )
    print(f"   Unsloth 量化: {'4-bit' if _load_4bit else 'BF16(无量化)'}")
    # 加载 SFT adapter 作为起点（公平对比：两框架都从 SFT 模型开始）
    if not args.no_sft_adapter and os.path.exists(os.path.join(SFT_ADAPTER, "adapter_config.json")):
        print(f"   加载 SFT adapter: {SFT_ADAPTER}")
        model.load_adapter(SFT_ADAPTER, adapter_name="default")
    # LoRA 配置（GRPO 继续训练）
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=(False if args.no_gradient_checkpointing else "unsloth"),
        random_state=args.seed,
    )
else:
    # HF 原生模式（BF16 无量化——排除 BnB 混淆变量，3B 完全放入 24GB）
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    if IS_GEMMA4:
        # Gemma4: 官方多模态 API (AutoModelForCausalLM 的 generate 会 KeyError, docs/36 §0)
        from transformers import AutoModelForMultimodalLM
        model = AutoModelForMultimodalLM.from_pretrained(
            BASE,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",  # AMD 兼容
            low_cpu_mem_usage=True,  # 必须: 整份 10GB 先入 CPU RAM 会被沙箱内存限制杀
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",  # AMD 兼容
        )
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    # 加载 SFT adapter
    if not args.no_sft_adapter and os.path.exists(os.path.join(SFT_ADAPTER, "adapter_config.json")):
        from peft import PeftModel
        print(f"   加载 SFT adapter: {SFT_ADAPTER}")
        model = PeftModel.from_pretrained(model, SFT_ADAPTER)
        model = model.merge_and_unload()
    # 新 LoRA
    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    if IS_GEMMA4:
        # Gemma4: vision/audio 塔的同名模块是 ClippableLinear(非 nn.Linear), PEFT 拒绝;
        # 用全路径精确限定 language_model 内的标准 Linear (docs/36 §0)
        _allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        lora_config.target_modules = [
            n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in _allowed
            and "language_model" in n
        ]
        print(f"   Gemma4 LoRA 目标: {len(lora_config.target_modules)} 个 language_model 线性层")
    model = get_peft_model(model, lora_config)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Gemma4: 预渲染为非 thinking 文本字符串 (TRL processing_class 不传 enable_thinking, docs/36 §0)
if IS_GEMMA4:
    def _render_text(ex):
        msgs = ex["prompt"] if isinstance(ex["prompt"], list) else [{"role": "user", "content": ex["prompt"]}]
        return {"prompt": tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False), "idx": ex["idx"]}
    train_ds = train_ds.map(_render_text)
    print(f"✅ Gemma4: prompts 已预渲染为非 thinking 文本模板 ({len(train_ds)} 条)")

# 打印参数量
# 注意：4-bit 量化下 model.parameters() 的 numel() 会少计（packed storage），
# MFU 计算必须用模型真实参数量（从 config 推算），而非量化后的存储元素数。
from transformers import AutoConfig as _AC
_cfg = _AC.from_pretrained(BASE)
_tc = getattr(_cfg, "text_config", _cfg) if IS_GEMMA4 else _cfg
_h = _tc.hidden_size
_n = _tc.num_hidden_layers
_nh = _tc.num_attention_heads
_nkv = getattr(_tc, 'num_key_value_heads', _nh)
_inter = _tc.intermediate_size
_vocab = _tc.vocab_size
_hd = _h // _nh
_attn_per_layer = _h * (_nh * _hd) + 2 * _h * (_nkv * _hd) + (_nh * _hd) * _h
_mlp_per_layer = _h * _inter * 3  # SwiGLU: gate + up + down
_emb = _vocab * _h if not getattr(_cfg, 'tie_word_embeddings', False) else _vocab * _h
total_params = _emb + _n * (_attn_per_layer + _mlp_per_layer)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   总参数(config推算): {total_params/1e9:.3f}B | 可训练(LoRA): {trainable_params/1e6:.1f}M")

# =============================================================================
# 效率 Profiler 集成
# =============================================================================
profiler = None
callbacks = []

if not args.no_profiler:
    sys.path.insert(0, REPO_ROOT)
    from utils.efficiency_profiler import EfficiencyProfiler, EfficiencyCallback

    # 读取实测 GEMM 峰值
    gemm_peak = 101.53  # 默认值
    gemm_json = REPO_ROOT + "/results/gemm_peak_bf16.json"
    if os.path.exists(gemm_json):
        with open(gemm_json) as f:
            gemm_peak = json.load(f).get("practical_peak_tflops", 101.53)

    profiler = EfficiencyProfiler(
        num_params=total_params,
        peak_tflops_practical=gemm_peak,
        poll_gpu=True,
        poll_interval=1.0,
    )
    callbacks.append(EfficiencyCallback(profiler))
    print(f"✅ Profiler 已挂载 (GEMM peak={gemm_peak} TFLOPS)")

# ── 显存碎片预防 (2026-08-08, s7 长 run 显存单调涨至 99.8% 教训) ──
# 长 run 每步张量大小变化大 → 默认分配器空闲段碎片不可复用 → 驱动 Used 单调涨。
# 周期性 empty_cache 归还空闲段; 每 50 步一次, 开销 <1s, 不影响测量口径。
if not args.no_profiler:
    from transformers import TrainerCallback

    class PeriodicEmptyCacheCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 50 == 0:
                import torch
                torch.cuda.empty_cache()

    callbacks.append(PeriodicEmptyCacheCallback())
    print("✅ PeriodicEmptyCacheCallback 已挂载 (每 50 步 empty_cache)")

# =============================================================================
# GRPO 训练
# =============================================================================
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir=args.output,
    num_generations=args.num_generations,
    max_prompt_length=args.max_prompt_length,
    max_completion_length=args.max_completion_length,
    per_device_train_batch_size=args.per_device_batch,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    beta=args.beta,
    temperature=args.temperature,
    max_steps=args.max_steps,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    seed=args.seed,
    bf16=True,
    gradient_checkpointing=not args.no_gradient_checkpointing,
    max_grad_norm=0.5,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    report_to="none",
    remove_unused_columns=False,
)

# TRL 0.24 兼容：PEFT 包装后 model.warnings_issued 不可达，手动挂载
if not hasattr(model, 'warnings_issued'):
    model.warnings_issued = {}

# ZSBR: 3 行子类覆写采样器(零 TRL 手术); 结构复刻 RepeatSampler, chunk 由调度器给出
trainer_cls = GRPOTrainer
if zsbr_scheduler is not None:
    assert training_args.dataloader_num_workers == 0, "ZSBR 要求 num_workers=0(调度器单进程状态)"

    class _ZSBRTrainer(GRPOTrainer):
        def _get_train_sampler(self, dataset=None):
            return ZSBRSampler(
                num_samples=len(self.train_dataset),
                scheduler=zsbr_scheduler,
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
            )
    trainer_cls = _ZSBRTrainer

trainer = trainer_cls(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    reward_funcs=reward_function,
    processing_class=tokenizer,  # 强制用 tokenizer 而非 VLM processor
    callbacks=callbacks,
)

print(f"\n🚀 开始训练 | {args.max_steps} steps | {args.framework}")
t_start = time.perf_counter()
trainer.train()
t_total = time.perf_counter() - t_start
print(f"\n✅ 训练完成 | 耗时 {t_total:.1f}s ({t_total/60:.1f}min)")

# =============================================================================
# 效率报告
# =============================================================================
if profiler:
    report_path = os.path.join(args.output, "efficiency_report.json")
    profiler.report(path=report_path)
    print(f"📊 效率报告: {report_path}")

# 保存最终 adapter
final_path = os.path.join(args.output, "final_adapter")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"💾 Adapter 已保存: {final_path}")

# ZSBR 状态落盘(p̂ 全量 + 选择历史, 供复盘/coverage 分析)
if zsbr_scheduler is not None:
    zsbr_scheduler.dump(os.path.join(args.output, "zsbr_state.json"))
    print(f"💾 ZSBR 状态: {os.path.join(args.output, 'zsbr_state.json')}")

# 保存训练总耗时
with open(os.path.join(args.output, "timing.json"), "w") as f:
    json.dump({"total_seconds": round(t_total, 1),
               "max_steps": args.max_steps,
               "framework": args.framework,
               "s_per_step": round(t_total / args.max_steps, 2)}, f, indent=2)

print(f"\n{'=' * 60}")
print(f"实验完成: {args.output}")
print(f"{'=' * 60}")
