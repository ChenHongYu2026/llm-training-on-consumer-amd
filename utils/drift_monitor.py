"""
KL 散度与 Reward Hacking 监控 — Legal-PT-01
=============================================
GRPO 阶段的安全阀。防止模型在奖励函数引导下走向作弊或崩溃。

核心机制:
  1. KL Divergence Tracking — 每 N 步计算 Policy 与 Reference 的距离
  2. Adaptive Beta Control — 根据 KL 动态调整 KL 惩罚系数
  3. Reward Hacking Detection — 检测格式合规但答案随机的不良行为
  4. Length Collapse Detection — 检测生成质量坍缩
  5. Early Stopping — 异常自动触发

使用方式:
    monitor = DriftMonitor(kl_target=3.0, kl_check_steps=20)

    # 每步训练后调用
    monitor.step(
        step=50,
        completions=["<think>推理</think><answer>B</answer>"],
        rewards=[0.8],
        policy_logps=p_logps,
        ref_logps=r_logps,
    )

    if monitor.should_stop():
        print(f"Early stop at step {monitor.stop_step}")

    # 获取当前 KL beta
    beta = monitor.get_beta()
"""

import json
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))



class DriftMonitor:
    """
    GRPO 训练漂移监控器。
    论文 Method §KL Monitoring 部分的核心实现。
    """

    def __init__(
        self,
        kl_target: float = 3.0,
        kl_beta_initial: float = 0.1,
        kl_check_steps: int = 20,
        kl_upper_threshold: float = 5.0,
        kl_lower_threshold: float = 0.5,
        kl_consecutive_trigger: int = 3,
        hacking_format_window: int = 50,
        hacking_accuracy_threshold: float = 0.3,
        hacking_length_collapse: int = 100,
        log_dir: str = "results/grpo",
    ):
        # KL 参数
        self.kl_target = kl_target
        self.beta = kl_beta_initial
        self.beta_initial = kl_beta_initial
        self.kl_check_steps = kl_check_steps
        self.kl_upper_threshold = kl_upper_threshold
        self.kl_lower_threshold = kl_lower_threshold
        self.kl_consecutive_trigger = kl_consecutive_trigger

        # Hacking 检测参数
        self.hacking_format_window = hacking_format_window
        self.hacking_accuracy_threshold = hacking_accuracy_threshold
        self.hacking_length_collapse = hacking_length_collapse

        # 状态
        self.kl_history: List[Dict] = []
        self.beta_history: List[float] = [kl_beta_initial]
        self.format_rewards: deque = deque(maxlen=hacking_format_window)
        self.accuracy_rewards: deque = deque(maxlen=hacking_format_window)
        self.completion_lengths: deque = deque(maxlen=hacking_format_window)

        self.kl_upper_consecutive: int = 0
        self.kl_lower_consecutive: int = 0
        self.hacking_flag_consecutive: int = 0
        self.length_collapse_consecutive: int = 0

        # Early stopping
        self._should_stop: bool = False
        self.stop_reason: str = ""
        self.stop_step: Optional[int] = None

        # 日志
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.drift_log_path = self.log_dir / "drift_log.json"
        self._drift_records: List[Dict] = []

        # 事件记录
        self.events: List[Dict] = []

    def step(
        self,
        step: int,
        completions: List[str],
        rewards: List[float],
        policy_logps: Optional[torch.Tensor] = None,
        ref_logps: Optional[torch.Tensor] = None,
        kl_div: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        每步训练后调用。返回监控状态字典。

        Returns:
            Dict with keys:
              - "kl_div": float or None
              - "beta": current beta
              - "hacking_score": 0.0 to 1.0 (越高越可疑)
              - "should_stop": bool
              - "stop_reason": str
              - "alert_level": "normal" | "warning" | "critical"
        """
        result = {
            "kl_div": None,
            "beta": self.beta,
            "hacking_score": 0.0,
            "should_stop": False,
            "stop_reason": "",
            "alert_level": "normal",
        }

        # 1. KL 散度检查
        if (policy_logps is not None and ref_logps is not None) or kl_div is not None:
            result["kl_div"] = self._check_kl_divergence(
                step, policy_logps, ref_logps, kl_div
            )

        # 2. Reward Hacking 检测
        result["hacking_score"] = self._detect_hacking(step, completions, rewards)

        # 3. 长度坍缩检测
        self._detect_length_collapse(step, completions)

        # 4. 评估警报级别
        if self._should_stop:
            result["should_stop"] = True
            result["stop_reason"] = self.stop_reason
            result["alert_level"] = "critical"
        elif result["hacking_score"] > 0.5:
            result["alert_level"] = "warning"
        elif (result["kl_div"] or 0) > self.kl_upper_threshold:
            result["alert_level"] = "warning"

        return result

    def _check_kl_divergence(
        self,
        step: int,
        policy_logps: Optional[torch.Tensor],
        ref_logps: Optional[torch.Tensor],
        kl_div_provided: Optional[float],
    ) -> Optional[float]:
        """计算或接收 KL 散度, 执行 Adaptive Beta 调整"""

        kl_div = kl_div_provided
        if kl_div is None and policy_logps is not None and ref_logps is not None:
            # KL(P || Ref) = sum(P * (log P - log Ref))
            with torch.no_grad():
                log_ratio = policy_logps - ref_logps
                kl_div = torch.mean(torch.sum(torch.exp(policy_logps) * log_ratio, dim=-1)).item()

        if kl_div is None:
            return None

        # 记录
        self.kl_history.append(
            {"step": step, "kl_div": kl_div, "beta": self.beta,
             "timestamp": datetime.now().isoformat()}
        )
        self._drift_records.append(self.kl_history[-1])

        # Adaptive Beta: KL 太大 → 增大 beta (拉回参考模型)
        if kl_div > self.kl_upper_threshold:
            self.kl_upper_consecutive += 1
            self.kl_lower_consecutive = 0

            if self.kl_upper_consecutive >= self.kl_consecutive_trigger:
                old_beta = self.beta
                self.beta *= 2.0
                self._log_event(
                    step, "adaptive_beta_up",
                    f"KL={kl_div:.2f} > upper={self.kl_upper_threshold} "
                    f"(连续{self.kl_upper_consecutive}次), beta: {old_beta:.4f} → {self.beta:.4f}"
                )
                self.kl_upper_consecutive = 0

                # 如果 beta 已经很大了, 触发 Early Stop
                if self.beta > self.beta_initial * 32:
                    self._should_stop = True
                    self.stop_reason = (
                        f"KL divergence too high ({kl_div:.2f}) despite beta={self.beta:.4f}. "
                        "Model is diverging from reference."
                    )
                    self.stop_step = step

        # KL 太小 → 减小 beta (允许更大探索)
        elif kl_div < self.kl_lower_threshold:
            self.kl_lower_consecutive += 1
            self.kl_upper_consecutive = 0

            if self.kl_lower_consecutive >= self.kl_consecutive_trigger:
                old_beta = self.beta
                self.beta = max(self.beta * 0.5, 0.001)
                self._log_event(
                    step, "adaptive_beta_down",
                    f"KL={kl_div:.2f} < lower={self.kl_lower_threshold} "
                    f"(连续{self.kl_lower_consecutive}次), beta: {old_beta:.4f} → {self.beta:.4f}"
                )
                self.kl_lower_consecutive = 0
        else:
            self.kl_upper_consecutive = 0
            self.kl_lower_consecutive = 0

        self.beta_history.append(self.beta)
        return kl_div

    def _detect_hacking(
        self,
        step: int,
        completions: List[str],
        rewards: List[float],
    ) -> float:
        """
        Reward Hacking 检测。

        检测信号:
          1. 格式奖励高但准确率低且剧烈震荡
             → 模型学会了格式但答案靠猜
          2. 回答长度极短
             → 模型用最短回答骗取格式分

        Returns:
            hacking_score: 0.0 (安全) 到 1.0 (严重作弊)
        """
        import re

        # 解析各奖励分量 (假设 rewards 是总分的列表, 我们需要从 completions 推算)
        format_ok = 0
        answer_lengths = []

        for comp in completions:
            has_think = bool(re.search(r"<think>.*?</think>", comp, re.DOTALL))
            has_answer = bool(re.search(r"<answer>.*?</answer>", comp, re.DOTALL))
            if has_think and has_answer:
                format_ok += 1

            think_match = re.search(r"<think>(.*?)</think>", comp, re.DOTALL)
            if think_match:
                think_len = len(think_match.group(1))
                answer_lengths.append(think_len)

        format_rate = format_ok / max(len(completions), 1)
        self.format_rewards.append(format_rate)

        avg_reward = sum(rewards) / max(len(rewards), 1)
        self.accuracy_rewards.append(avg_reward)

        hacking_score = 0.0

        # 条件 1: 格式合规但奖励低 → 在猜
        if len(self.format_rewards) >= 10:
            avg_format = sum(self.format_rewards) / len(self.format_rewards)
            avg_acc = sum(self.accuracy_rewards) / len(self.accuracy_rewards)

            if avg_format > 0.7 and avg_acc < self.hacking_accuracy_threshold:
                hacking_score += 0.6
                self.hacking_flag_consecutive += 1

                if self.hacking_flag_consecutive >= 20:
                    self._log_event(
                        step, "hacking_detected_format_only",
                        f"格式合规率={avg_format:.2f}但准确率={avg_acc:.2f}, "
                        f"疑似模型仅输出格式标签不作实质推理"
                    )
            else:
                self.hacking_flag_consecutive = max(0, self.hacking_flag_consecutive - 1)

        # 条件 2: 回答极短
        avg_len = sum(answer_lengths) / max(len(answer_lengths), 1)
        self.completion_lengths.append(avg_len)

        if avg_len < 50 and len(self.completion_lengths) >= 10:
            self.length_collapse_consecutive += 1
            if self.length_collapse_consecutive >= 10:
                hacking_score += 0.4
                self._log_event(
                    step, "length_collapse_warning",
                    f"平均 think 长度仅 {avg_len:.0f} 字符，模型可能在偷懒"
                )
                if self.length_collapse_consecutive >= 30:
                    self._should_stop = True
                    self.stop_reason = (
                        f"Generation length collapsed to {avg_len:.0f} chars "
                        f"for {self.length_collapse_consecutive} consecutive steps."
                    )
                    self.stop_step = step
        else:
            self.length_collapse_consecutive = 0

        return hacking_score

    def _detect_length_collapse(
        self, step: int, completions: List[str]
    ) -> None:
        """检测生成文本长度坍缩"""
        lengths = [len(c) for c in completions]
        avg_len = sum(lengths) / max(len(lengths), 1)

        if avg_len < self.hacking_length_collapse:
            self.length_collapse_consecutive += 1
            if self.length_collapse_consecutive >= 30:
                self._should_stop = True
                self.stop_reason = (
                    f"输出长度坍缩至平均 {avg_len:.0f} 字符，"
                    f"连续 {self.length_collapse_consecutive} 步"
                )
                self.stop_step = step
        else:
            self.length_collapse_consecutive = 0

    def get_beta(self) -> float:
        """获取当前 KL 惩罚系数"""
        return self.beta

    def should_stop(self) -> bool:
        """是否应触发 Early Stopping"""
        return self._should_stop

    def save_log(self) -> str:
        """保存漂流日志到 JSON"""
        with open(self.drift_log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "kl_history": self.kl_history,
                    "beta_history": self.beta_history,
                    "events": self.events,
                    "final_state": {
                        "should_stop": self._should_stop,
                        "stop_reason": self.stop_reason,
                        "stop_step": self.stop_step,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return str(self.drift_log_path)

    def get_summary(self) -> Dict:
        """获取监控摘要 — 用于论文附录"""
        kl_values = [h["kl_div"] for h in self.kl_history]
        return {
            "kl_mean": sum(kl_values) / len(kl_values) if kl_values else 0,
            "kl_max": max(kl_values) if kl_values else 0,
            "kl_min": min(kl_values) if kl_values else 0,
            "beta_final": self.beta,
            "num_adaptive_adjustments": len(self.events),
            "should_stop": self._should_stop,
            "stop_reason": self.stop_reason,
            "total_kl_checks": len(self.kl_history),
        }

    def _log_event(self, step: int, event_type: str, message: str) -> None:
        """记录监控事件"""
        event = {
            "step": step,
            "type": event_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        self.events.append(event)
        print(f"[DriftMonitor] Step {step} | {event_type}: {message}")
