"""
组合规则奖励函数 — Legal-PT-01 Stage 2
========================================
兼容 TRL v0.24 的 list[dict] completion 格式。
"""

import re
from typing import List
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


W_ACCURACY = 0.7
W_FORMAT = 0.2
W_CITATION = 0.1
ACCURACY_MAX = 1.0
FORMAT_MAX = 0.3
CITATION_MAX = 0.1


def _extract_text(completion) -> str:
    """兼容 TRL v0.24 的 completion 格式: str | list[dict] | dict"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        if completion and isinstance(completion[-1], dict):
            return completion[-1].get("content", "")
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


# =============================================================================
# R_format (20%)
# =============================================================================

def format_reward_func(completions: List, **kwargs) -> List[float]:
    rewards = []
    pattern = re.compile(
        r"^\s*<(?:think|reasoning)>(.+?)</(?:think|reasoning)>\s*<answer>(.+?)</answer>\s*$",
        re.DOTALL,
    )
    for completion in completions:
        text = _extract_text(completion)
        match = pattern.match(text)
        if match and len(match.group(1).strip()) > 0 and len(match.group(2).strip()) > 0:
            rewards.append(FORMAT_MAX)
        elif "<think>" in text and "<answer>" in text:
            rewards.append(0.1)
        elif "<think>" in text or "<answer>" in text:
            rewards.append(0.05)
        else:
            rewards.append(0.0)
    return rewards


# =============================================================================
# R_accuracy (70%)
# =============================================================================

def accuracy_reward_func(completions: List, answer: List, **kwargs) -> List[float]:
    rewards = []
    for completion, truth in zip(completions, answer):
        text = _extract_text(completion)
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue
        pred = match.group(1).strip().lower()
        truth_str = str(truth).strip().lower()
        if pred == truth_str:
            rewards.append(ACCURACY_MAX)
        elif truth_str in pred or pred in truth_str:
            rewards.append(0.5)
        elif _extract_option_letter(pred) == _extract_option_letter(truth_str):
            rewards.append(ACCURACY_MAX)
        elif _numeric_match(pred, truth_str):
            rewards.append(ACCURACY_MAX)
        else:
            rewards.append(0.0)
    return rewards


def _extract_option_letter(t: str) -> str:
    m = re.match(r"(?:option|选项|choice)?\s*\(?([a-dA-D])\)?", t)
    return m.group(1).lower() if m else ""


def _numeric_match(pred: str, truth: str) -> bool:
    pn = re.findall(r"[\d.]+", pred)
    tn = re.findall(r"[\d.]+", truth)
    try:
        if pn and tn: return float(pn[0]) == float(tn[0])
    except ValueError: pass
    return False


# =============================================================================
# R_citation (10%)
# =============================================================================

def citation_reward_func(prompts: List, completions: List, **kwargs) -> List[float]:
    cn_pats = [r"第[零一二三四五六七八九十百千万\d]+[条款章节]", r"《.*?法》", r"最高.*?法院.*?(?:解释|批复|意见|通知)"]
    us_pats = [r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.\s+[A-Z][a-z]+", r"\([12][0-9]{3}\)", r"\d+\s+U\.S\.\s+\d+", r"\d+\s+F\.[234]d\s+\d+"]
    eu_pats = [r"Article\s+\d+", r"GDPR", r"Directive\s+\d+/\d+/E[UC]", r"Regulation\s*\(EU\)\s*\d+/\d+", r"ECtHR|European Court of Human Rights", r"CJEU|Court of Justice"]

    rewards = []
    for prompt, completion in zip(prompts, completions):
        text = _extract_text(completion)
        if "[Jurisdiction: CN]" in prompt:
            pats = cn_pats
        elif "[Jurisdiction: US]" in prompt:
            pats = us_pats
        elif "[Jurisdiction: EU]" in prompt:
            pats = eu_pats
        else:
            pats = cn_pats + us_pats + eu_pats
        score = CITATION_MAX if any(re.search(p, text, re.IGNORECASE) for p in pats) else 0.0
        rewards.append(score)
    return rewards


# =============================================================================
# 组合奖励
# =============================================================================

def composite_reward(completions: List, prompts: List, answers: List, **kwargs) -> List[float]:
    r_acc = accuracy_reward_func(completions, answer=answers)
    r_fmt = format_reward_func(completions)
    r_cit = citation_reward_func(prompts=prompts, completions=completions)
    return [W_ACCURACY * a + W_FORMAT * f + W_CITATION * c for a, f, c in zip(r_acc, r_fmt, r_cit)]


def analyze_reward_distribution(completions, prompts, answers) -> dict:
    r_acc = accuracy_reward_func(completions, answer=answers)
    r_fmt = format_reward_func(completions)
    r_cit = citation_reward_func(prompts=prompts, completions=completions)
    def stats(arr):
        arr = sorted(arr)
        n = len(arr); m = sum(arr)/n; v = sum((x-m)**2 for x in arr)/n
        return {"mean": round(m,4), "std": round(v**0.5,4), "max": max(arr), "min": min(arr), "median": arr[n//2]}
    return {"accuracy": stats(r_acc), "format": stats(r_fmt), "citation": stats(r_cit)}
