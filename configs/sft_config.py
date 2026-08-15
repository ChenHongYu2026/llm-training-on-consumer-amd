"""
SFT 训练配置 — Legal-PT-01 Stage 1
====================================
所有可调参数集中管理，确保实验可复现。
运行方式: python3 train/train_sft.py (自动读取本文件)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))



@dataclass
class SFTConfig:
    """Stage 1: 法律 SFT 全量配置"""

    # =========================================================================
    # 模型
    # =========================================================================
    model_name_or_path: str = (
        os.path.join(MODELS_DIR, "Qwen3.5-4B/models/Qwen--Qwen3.5-4B/snapshots/master")
    )
    use_4bit: bool = True  # QLoRA: 底座 NF4 4-bit 量化
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True  # 双重量化省 ~0.4GB

    # =========================================================================
    # LoRA
    # =========================================================================
    lora_r: int = 32  # Rank: 32 是法律推理任务的推荐值
    lora_alpha: int = 64  # alpha/r = 2, 适中的 LoRA 强度
    lora_dropout: float = 0.05
    # v2 修复：DeltaNet 全投影覆盖
    # in_proj_qkv/out_proj: 序列投影 | in_proj_z/a/b: 门控(写入/遗忘/输出门)
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",                 # Attention
        "in_proj_qkv", "out_proj", "in_proj_z", "in_proj_a", "in_proj_b",  # DeltaNet
        "gate_proj", "up_proj", "down_proj",                     # MLP
    ])

    # =========================================================================
    # 训练
    # =========================================================================
    output_dir: str = REPO_ROOT + "/results/sft"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4  # 有效 batch = 4
    learning_rate: float = 2e-4  # LoRA 学习率, 比全量高 10x
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # =========================================================================
    # 序列长度
    # =========================================================================
    max_seq_length: int = 1024
    packing: bool = False  # 法律数据长度差异大，不打包

    # =========================================================================
    # 精度 & 显存
    # =========================================================================
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {
        "use_reentrant": False
    })
    optim: str = "paged_adamw_8bit"  # 8-bit 优化器省显存

    # =========================================================================
    # 日志 & 保存
    # =========================================================================
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    evaluation_strategy: str = "no"  # SFT 不做 eval（数据量小）
    report_to: str = "tensorboard"
    run_name: str = "legal-sft-v1"

    # =========================================================================
    # DeepSpeed
    # =========================================================================
    deepspeed: str = REPO_ROOT + "/configs/deepspeed_zero2.json"

    # =========================================================================
    # 数据
    # =========================================================================
    data_path: str = REPO_ROOT + "/data/datasets/processed/sft_train.jsonl"
    jurisdictions: List[str] = field(default_factory=lambda: ["CN", "US", "EU"])
    data_temperature: float = 2.0  # 温度平滑采样
    max_samples: int = 30000  # 总训练样本数

    # =========================================================================
    # 确定性 & 种子
    # =========================================================================
    seed: int = 42
    data_seed: int = 42
    full_determinism: bool = False  # 单卡训练不需要完全确定性（速度优先）

    # =========================================================================
    # GRPO Stage 2 准备
    # =========================================================================
    # SFT checkpoint 保存为可加载的 PEFT 模型
    # train_grpo.py 会从 output_dir 加载做 Policy Model
    save_adapter_only: bool = True  # 只保存 LoRA 权重（省磁盘）


@dataclass
class DataConfig:
    """数据管线配置 — 独立于训练配置"""

    # DISC-Law-SFT (中国)
    disc_law_sft_path: str = (
        REPO_ROOT + "/data/datasets/disc_law_sft"
    )
    disc_law_subset_size: int = 15000

    # CaseHOLD (美国)
    casehold_path: str = (
        REPO_ROOT + "/data/datasets/casehold"
    )
    casehold_subset_size: int = 10000

    # AQuAECHR (欧盟)
    aqueachr_path: str = (
        REPO_ROOT + "/data/datasets/aqueachr"
    )
    aqueachr_subset_size: int = 5000

    # 数据质量过滤
    min_text_length: int = 50
    max_text_length: int = 4096
    dedup_threshold: float = 0.8  # MinHash 相似度阈值

    # 法律领域关键词（确保样本确实是法律相关）
    legal_keywords_cn: List[str] = field(default_factory=lambda: [
        "法", "罪", "刑", "诉", "判", "裁", "合同", "侵权", "继承",
        "婚姻", "劳动", "公司", "知识", "产权", "证据", "诉讼",
        "仲裁", "行政", "处罚", "赔偿", "债务", "担保", "物权",
    ])

    legal_keywords_en: List[str] = field(default_factory=lambda: [
        "law", "court", "statute", "plaintiff", "defendant", "contract",
        "tort", "crime", "constitution", "regulation", "directive",
        "appeal", "jurisdiction", "liability", "damages", "injunction",
        "precedent", "doctrine", "clause", "provision", "article",
        "GDPR", "charter", "treaty", "convention",
    ])


@dataclass
class CoTGenerationConfig:
    """Stage 0: CoT 推理链合成配置"""

    # API 配置
    api_provider: str = "deepseek"  # "deepseek" | "openai"
    api_model: str = "deepseek-chat"  # 或 "gpt-4o-mini"
    api_key: Optional[str] = None  # 从环境变量 DEEPSEEK_API_KEY 读取
    api_base_url: str = "https://api.deepseek.com/v1"

    # 合成参数
    max_concurrent: int = 10  # 并发请求数
    max_retries: int = 3
    temperature: float = 0.3  # CoT 合成需要低温度保证一致性
    max_output_tokens: int = 2048

    # Few-shot 示例数量
    n_fewshot_examples: int = 3  # 每个请求给出的示例数

    # 输出验证
    require_think_tag: bool = True  # 必须包含 <think>
    require_answer_tag: bool = True  # 必须包含 <answer>

    # 输出路径
    output_path: str = REPO_ROOT + "/data/datasets/processed/sft_train.jsonl"
    failed_output_path: str = REPO_ROOT + "/data/datasets/sft_cot_failed.jsonl"

    # 断点续传
    resume: bool = True  # 支持中断后继续
    checkpoint_interval: int = 100  # 每 100 条存一次 checkpoint


@dataclass
class GRPODataConfig:
    """Stage 2: GRPO 数据准备配置"""

    sft_data_path: str = REPO_ROOT + "/data/datasets/processed/sft_train.jsonl"
    output_path: str = REPO_ROOT + "/data/datasets/grpo_data.jsonl"

    # 筛选条件: 只保留客观题（有明确答案的）
    objective_answer_patterns: List[str] = field(default_factory=lambda: [
        r"^[A-D]$",  # 单选题
        r"^[A-D][).]",  # A) B) 格式
        r"^(正确|错误|对|错|是|否|True|False|Yes|No)$",
        r"^\d+[\d.,]*\s*(?:平方|厘米|公里|元|万|年|天|人|件|项|次)",  # 数字答案
    ])

    # 最少保留样本数
    min_samples_per_jurisdiction: int = 500
