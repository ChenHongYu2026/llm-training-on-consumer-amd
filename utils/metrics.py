"""
法律评估指标 — Legal-PT-01
============================
所有指标函数接受 prediction 和 reference, 返回 float。
论文 Results 部分的核心数据来源。
"""

import re
import math
from typing import List, Tuple, Dict, Optional
from collections import Counter
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))



# =============================================================================
# 核心法律指标
# =============================================================================


def legal_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    法律准确率: 严格提取 <answer> 内容后与标准答案比对。

    适配格式:
      - 选择题: "A" vs "A", "Option B" vs "B"
      - 判断题: "正确" vs "正确", "True" vs "True"
      - 短答案: "3年" vs "3年"
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, ref in zip(predictions, references):
        pred_clean = _extract_answer(pred).strip().lower()
        ref_clean = ref.strip().lower()

        if pred_clean == ref_clean:
            correct += 1
        elif ref_clean in pred_clean or pred_clean in ref_clean:
            correct += 1

    return correct / len(predictions)


def legal_f1(
    predictions: List[str],
    references: List[str],
    tokenize: bool = True,
) -> float:
    """
    法律 F1: 基于 token 级别的精确率和召回率。
    适用于长答案的场景 (如法律摘要、法条解释)。
    """
    if not predictions:
        return 0.0

    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_clean = _extract_answer(pred)
        ref_clean = ref.strip()

        if tokenize:
            pred_tokens = set(pred_clean.split())
            ref_tokens = set(ref_clean.split())
        else:
            pred_tokens = set(pred_clean)
            ref_tokens = set(ref_clean)

        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue

        tp = len(pred_tokens & ref_tokens)
        precision = tp / len(pred_tokens) if pred_tokens else 0
        recall = tp / len(ref_tokens) if ref_tokens else 0

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / len(f1_scores)


def format_compliance_rate(completions: List[str]) -> float:
    """
    格式合规率: 输出中同时包含 <think> 和 <answer> 的比例。
    这是判断模型是否"学会格式"的核心指标。
    """
    if not completions:
        return 0.0

    compliant = 0
    pattern = re.compile(
        r"^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$", re.DOTALL
    )

    for comp in completions:
        if pattern.match(comp):
            compliant += 1

    return compliant / len(completions)


def citation_recall_rate(
    prompts: List[str],
    completions: List[str],
) -> Dict[str, float]:
    """
    法源引用率: 按法域统计 <think> 中是否包含法源引用。
    返回 {"CN": 0.85, "US": 0.72, "EU": 0.68} 格式。
    """
    jurisdiction_counts: Dict[str, int] = Counter()
    citation_counts: Dict[str, int] = Counter()

    cn_patterns = [
        r"第[零一二三四五六七八九十百千万\d]+[条款章节]",
        r"《.*?法》",
    ]
    us_patterns = [
        r"\b\w+\s+v\.\s+\w+\b",
        r"\([12][0-9]{3}\)",
        r"\d+\s+U\.S\.\s+\d+",
    ]
    eu_patterns = [
        r"Article\s+\d+",
        r"GDPR",
        r"Directive\s+\d+/\d+/EU",
        r"Regulation\s*\(EU\)",
    ]

    for prompt, comp in zip(prompts, completions):
        # 判断法域
        if "[Jurisdiction: CN]" in prompt:
            jurisdiction_counts["CN"] += 1
            if any(re.search(p, comp, re.IGNORECASE) for p in cn_patterns):
                citation_counts["CN"] += 1
        elif "[Jurisdiction: US]" in prompt:
            jurisdiction_counts["US"] += 1
            if any(re.search(p, comp, re.IGNORECASE) for p in us_patterns):
                citation_counts["US"] += 1
        elif "[Jurisdiction: EU]" in prompt:
            jurisdiction_counts["EU"] += 1
            if any(re.search(p, comp, re.IGNORECASE) for p in eu_patterns):
                citation_counts["EU"] += 1

    rates = {}
    for jur in ["CN", "US", "EU"]:
        if jurisdiction_counts[jur] > 0:
            rates[jur] = citation_counts[jur] / jurisdiction_counts[jur]
        else:
            rates[jur] = 0.0

    rates["total"] = (
        sum(citation_counts.values()) / sum(jurisdiction_counts.values())
        if sum(jurisdiction_counts.values()) > 0
        else 0.0
    )

    return rates


def cross_jurisdiction_confusion(
    prompts: List[str],
    completions: List[str],
) -> float:
    """
    法域混淆率: CN prompt 下错误引用 US 判例 / EU prompt 下错误引用 CN 法条 的比例。
    越低越好。监控 Anti-Hallucination 效果。
    """
    if not prompts:
        return 0.0

    confusion_count = 0
    total = len(prompts)

    cn_pattern = re.compile(r"第[零一二三四五六七八九十百千万\d]+[条款]|《")
    us_pattern = re.compile(r"\b\w+\s+v\.\s+\w+\b|U\.S\.\s+\d+")
    eu_pattern = re.compile(r"GDPR|Directive|Regulation\s*\(EU\)", re.IGNORECASE)

    for prompt, comp in zip(prompts, completions):
        if "[Jurisdiction: CN]" in prompt:
            # CN prompt 下不应该出现 US 判例引用
            if us_pattern.search(comp):
                confusion_count += 1
        elif "[Jurisdiction: US]" in prompt:
            # US prompt 下不应该出现 CN 法条引用
            if cn_pattern.search(comp):
                confusion_count += 1
        elif "[Jurisdiction: EU]" in prompt:
            # EU prompt 下不应该出现 CN 条文格式引用
            if cn_pattern.search(comp):
                confusion_count += 1

    return confusion_count / total


def general_capability_retention(
    baseline_scores: Dict[str, float],
    current_scores: Dict[str, float],
) -> Dict[str, float]:
    """
    通用能力保持率: 对比训练前后 MMLU 等基准的分数。
    retention = current_score / baseline_score
    >1.0 表示提升, <1.0 表示退化。
    """
    retention = {}
    for key in baseline_scores:
        if key in current_scores and baseline_scores[key] > 0:
            retention[key] = current_scores[key] / baseline_scores[key]
        else:
            retention[key] = 1.0
    retention["avg"] = (
        sum(retention.values()) / len(retention) if retention else 1.0
    )
    return retention


def thinking_depth_score(completions: List[str]) -> Dict[str, float]:
    """
    推理深度统计: 衡量 <think> 块的质量。
    返回:
      - avg_think_length: 平均思考长度 (字符)
      - avg_steps: 平均推理步骤数 (通过 "1." / "首先" 等标记计数)
      - has_legal_terms_ratio: 包含法律术语的比例
    """
    if not completions:
        return {"avg_think_length": 0, "avg_steps": 0, "has_legal_terms_ratio": 0}

    think_lengths = []
    step_counts = []
    legal_term_count = 0

    legal_terms = re.compile(
        r"(法[条规律]|构成要件|罪|刑[罚事]|诉[讼权]|判[决例]|裁[判决]|"
        r"合同|侵权|继承|证据|抗辩|责任|义务|权利|statute|precedent|"
        r"jurisdiction|liability|article|directive|regulation)",
        re.IGNORECASE,
    )
    step_pattern = re.compile(
        r"(?:\d+\.|第[一二三四五六七八九十\d]+[步点]|首先|其次|然后|最后|"
        r"first|second|finally|therefore|hence|thus)",
        re.IGNORECASE,
    )

    for comp in completions:
        think_match = re.search(r"<think>(.*?)</think>", comp, re.DOTALL)
        if think_match:
            think_text = think_match.group(1)
            think_lengths.append(len(think_text))
            step_counts.append(len(step_pattern.findall(think_text)))
            if legal_terms.search(think_text):
                legal_term_count += 1
        else:
            think_lengths.append(0)
            step_counts.append(0)

    n = len(completions)
    return {
        "avg_think_length": sum(think_lengths) / n,
        "avg_steps": sum(step_counts) / n,
        "has_legal_terms_ratio": legal_term_count / n,
    }


# =============================================================================
# 统计检验
# =============================================================================


def bootstrap_confidence_interval(
    scores: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Bootstrap 置信区间 — 论文 Table 中的 ± 值来源。
    """
    import random

    random.seed(42)
    n = len(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    lower = means[int(n_bootstrap * alpha / 2)]
    upper = means[int(n_bootstrap * (1 - alpha / 2))]
    return lower, upper


def cohens_d(scores_a: List[float], scores_b: List[float]) -> float:
    """
    Cohen's d 效应量 — 衡量两组结果的差异大小。
    d > 0.8: 大效应; 0.5: 中效应; 0.2: 小效应。
    """
    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)

    var_a = sum((x - mean_a) ** 2 for x in scores_a) / len(scores_a)
    var_b = sum((x - mean_b) ** 2 for x in scores_b) / len(scores_b)

    pooled_std = math.sqrt((var_a + var_b) / 2)
    if pooled_std == 0:
        return 0.0

    return (mean_a - mean_b) / pooled_std


# =============================================================================
# 工具函数
# =============================================================================


def _extract_answer(text: str) -> str:
    """从完整输出中提取 <answer> 内容"""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: 没有标签时返回最后一段
    lines = text.strip().split("\n")
    return lines[-1] if lines else text


def format_metric_table(
    metrics: Dict[str, float],
    title: str = "Evaluation Results",
) -> str:
    """
    格式化指标为 LaTeX 表格 — 直接可插入论文。
    """
    lines = [f"\\begin{{table}}[h]", f"\\caption{{{title}}}", "\\centering"]
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\hline")
    lines.append("Metric & Value \\\\")
    lines.append("\\hline")

    for k, v in metrics.items():
        lines.append(f"{k.replace('_', '\\_')} & {v:.4f} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)
