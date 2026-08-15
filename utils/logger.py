"""
实验日志系统 — Legal-PT-01
===========================
双层架构: TensorBoard (训练曲线) + JSON (结构化记录)
确保实验结果完整可追溯, 支持论文复现。
"""

import json
import time
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))



class ExperimentLogger:
    """
    实验日志器 — 同时写入 TensorBoard 和结构化 JSON。

    使用方式:
        logger = ExperimentLogger("results/grpo", experiment_group="legal-grpo-multi")
        logger.log_config({...})  # 训练前调用一次
        logger.log_step(step=10, metrics={"loss": 0.5, "reward": 0.8})
        logger.log_eval(step=100, metrics={"legal_accuracy": 0.85})
        logger.finalize()  # 训练结束调用
    """

    def __init__(
        self,
        output_dir: str,
        experiment_group: str,
        seed: int = 42,
        run_id: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_group = experiment_group
        self.seed = seed

        # 生成唯一 run_id
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_id = f"{experiment_group}-{timestamp}-seed{seed}"
        else:
            self.run_id = run_id

        # 创建子目录
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

        # JSON 日志
        self.json_log_path = self.run_dir / "run_log.json"
        self.json_log: Dict[str, Any] = {
            "run_id": self.run_id,
            "experiment_group": experiment_group,
            "seed": seed,
            "created_at": datetime.now().isoformat(),
            "hardware": self._capture_hardware(),
            "git_commit": self._capture_git_commit(),
            "config": {},
            "steps": [],
            "evaluations": [],
            "final_metrics": {},
            "duration_seconds": 0,
        }

        self.start_time = time.time()
        print(f"[Logger] Run ID: {self.run_id}")
        print(f"[Logger] Log dir: {self.run_dir}")

    def _capture_hardware(self) -> Dict[str, Any]:
        """捕获硬件环境 — 论文附录必需"""
        info = {
            "gpu_name": "N/A",
            "gpu_vram_gb": 0,
            "gpu_count": 0,
            "cpu_count": os.cpu_count(),
            "ram_gb": "N/A",
        }
        try:
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
                )
                info["gpu_count"] = torch.cuda.device_count()
        except Exception:
            pass

        try:
            import psutil

            info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
        except ImportError:
            pass

        info["pytorch_version"] = torch.__version__
        info["rocm_version"] = os.environ.get("ROCM_VERSION", "unknown")

        return info

    def _capture_git_commit(self) -> str:
        """捕获 Git commit hash — 确保代码版本可追溯"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "no-git"
        except Exception:
            return "no-git"

    def log_config(self, config: Dict[str, Any]) -> None:
        """训练开始前记录完整配置"""
        # 将 dataclass 或 dict 序列化
        serializable = {}
        for k, v in config.items():
            if hasattr(v, "__dict__"):
                serializable[k] = {kk: str(vv) for kk, vv in v.__dict__.items()}
            elif isinstance(v, (list, tuple)):
                serializable[k] = [str(x) for x in v]
            else:
                serializable[k] = str(v)
        self.json_log["config"] = serializable
        self._flush_json()

    def log_step(
        self,
        step: int,
        metrics: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录单步训练指标"""
        # TensorBoard
        for name, value in metrics.items():
            self.tb_writer.add_scalar(f"train/{name}", value, step)

        # 显存
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            self.tb_writer.add_scalar("system/vram_gb", vram, step)

        # JSON
        step_record = {
            "step": step,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            step_record["extra"] = extra
        self.json_log["steps"].append(step_record)

        # 每 50 步刷盘
        if step % 50 == 0:
            self._flush_json()

    def log_kl_divergence(
        self,
        step: int,
        kl_div: float,
        beta: float,
        hacking_score: float,
    ) -> None:
        """记录 KL 散度监控数据 — drift_monitor 专用"""
        self.tb_writer.add_scalar("monitor/kl_divergence", kl_div, step)
        self.tb_writer.add_scalar("monitor/beta", beta, step)
        self.tb_writer.add_scalar("monitor/hacking_score", hacking_score, step)

    def log_eval(
        self,
        step: int,
        metrics: Dict[str, float],
        dataset_name: str = "benchmark",
    ) -> None:
        """记录评估结果"""
        for name, value in metrics.items():
            self.tb_writer.add_scalar(f"eval/{dataset_name}/{name}", value, step)

        self.json_log["evaluations"].append(
            {
                "step": step,
                "dataset": dataset_name,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self._flush_json()

    def log_convergence(self, steps: List[int], rewards: List[float]) -> None:
        """记录奖励收敛曲线 — 用于论文 Figure"""
        for s, r in zip(steps, rewards):
            self.tb_writer.add_scalar("convergence/reward_vs_step", r, s)

    def finalize(
        self,
        final_metrics: Optional[Dict[str, float]] = None,
        best_checkpoint_step: Optional[int] = None,
    ) -> str:
        """训练结束, 封存日志"""
        duration = time.time() - self.start_time
        self.json_log["duration_seconds"] = round(duration, 1)
        self.json_log["duration_hours"] = round(duration / 3600, 2)
        self.json_log["final_metrics"] = final_metrics or {}
        self.json_log["best_checkpoint_step"] = best_checkpoint_step
        recorded_steps = len(self.json_log["steps"])
        if recorded_steps > 0:
            self.json_log["total_steps_recorded"] = recorded_steps

        # 计算统计摘要
        if self.json_log["steps"]:
            losses = [
                s["metrics"].get("loss", 0)
                for s in self.json_log["steps"]
                if "loss" in s["metrics"]
            ]
            if losses:
                self.json_log["summary"] = {
                    "final_loss": losses[-1],
                    "min_loss": min(losses),
                    "loss_at_min_step": losses.index(min(losses)),
                }

        self._flush_json()
        self.tb_writer.close()

        summary_path = self.run_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Duration: {duration/3600:.1f} hours\n")
            f.write(f"Hardware: {self.json_log['hardware']['gpu_name']}\n")
            if final_metrics:
                for k, v in final_metrics.items():
                    if isinstance(v, (int, float)):
                        f.write(f"{k}: {v:.4f}\n")
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            if isinstance(vv, (int, float)):
                                f.write(f"{k}/{kk}: {vv:.4f}\n")
                    else:
                        f.write(f"{k}: {v}\n")

        print(f"\n[Logger] 日志已保存: {self.run_dir}")
        print(f"[Logger] 摘要: {summary_path}")
        print(f"[Logger] TensorBoard: tensorboard --logdir {self.run_dir}/tensorboard")
        return str(self.run_dir)

    def _flush_json(self) -> None:
        """刷盘 JSON 日志"""
        with open(self.json_log_path, "w", encoding="utf-8") as f:
            json.dump(self.json_log, f, indent=2, ensure_ascii=False)

    def get_run_dir(self) -> str:
        return str(self.run_dir)
