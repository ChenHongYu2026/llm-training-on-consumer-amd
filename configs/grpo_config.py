"""
GRPO 训练配置 — Legal-PT-01 Stage 2
====================================
基座模型: Qwen3.5-4B, 方法参考 DeepSeek-R1 的 GRPO 框架。
单卡 24GB 适配版。
"""

from dataclasses import dataclass, field
from typing import List, Optional
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))



@dataclass
class GRPOConfig:
    """Stage 2: Rule-based GRPO 全量配置"""

    # === 模型路径 ===
    sft_checkpoint_path: str = REPO_ROOT + "/results/sft"
    base_model_path: str = (
        os.path.join(MODELS_DIR, "Qwen3.5-4B/models/Qwen--Qwen3.5-4B/snapshots/master")
    )

    # === QLoRA ===
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    load_adapter_from_sft: bool = True

    # === GRPO 核心 ===
    num_generations: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    max_completion_length: int = 3072
    max_prompt_length: int = 512

    # === 奖励函数权重 ===
    reward_accuracy_weight: float = 0.7
    reward_format_weight: float = 0.2
    reward_citation_weight: float = 0.1

    # === 训练 ===
    output_dir: str = REPO_ROOT + "/results/grpo"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 25
    max_steps: int = 500

    # === 序列 ===
    max_length: int = 3584  # prompt(512) + completion(3072)

    # === 精度 ===
    bf16: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})
    optim: str = "paged_adamw_8bit"

    # === 日志 ===
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 5
    report_to: str = "tensorboard"

    # === DeepSpeed (单卡不需要) ===
    deepspeed: Optional[str] = None

    # === 数据 ===
    data_path: str = REPO_ROOT + "/data/datasets/processed/grpo_train.jsonl"

    # === 确定性 ===
    seed: int = 42

    # === KL 监控 ===
    kl_monitoring: bool = True
    kl_target: float = 3.0
    kl_beta_initial: float = 0.1
    kl_check_steps: int = 20
    kl_upper_threshold: float = 5.0
    kl_lower_threshold: float = 0.5

    # === Reward Hacking 检测 ===
    hacking_detection: bool = True
    hacking_format_window: int = 50
    hacking_accuracy_threshold: float = 0.3
    hacking_length_collapse: int = 100

    # === 实验组 ===
    experiment_group: str = "legal-grpo-multi"
    jurisdictions: List[str] = field(default_factory=lambda: ["CN", "US", "EU"])


@dataclass
class RewardFunctionConfig:
    accuracy_exact_match_score: float = 1.0
    accuracy_fuzzy_match_score: float = 0.5
    accuracy_no_match_score: float = 0.0
    format_full_match_score: float = 0.3
    format_partial_match_score: float = 0.1
    format_no_match_score: float = 0.0
    citation_match_score: float = 0.1
    citation_no_match_score: float = 0.0
    citation_patterns_cn: List[str] = field(default_factory=lambda: [
        r"第[零一二三四五六七八九十百千万\d]+[条款章节]", r"《.*?法》", r"最高.*?法院.*?(?:解释|批复|意见|通知)",
    ])
    citation_patterns_us: List[str] = field(default_factory=lambda: [
        r"\b\w+\s+v\.\s+\w+\b", r"\([12][0-9]{3}\)", r"\d+\s+U\.S\.\s+\d+", r"\d+\s+F\.[234]d\s+\d+",
    ])
    citation_patterns_eu: List[str] = field(default_factory=lambda: [
        r"Article\s+\d+", r"GDPR", r"Directive\s+\d+/\d+/EU", r"Regulation\s*\(EU\)\s*\d+/\d+",
        r"ECtHR|European Court of Human Rights", r"CJEU|Court of Justice",
    ])
