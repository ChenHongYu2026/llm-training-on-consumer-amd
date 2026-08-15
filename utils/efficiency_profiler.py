"""
效率 Profiler — Legal-PT-01 新方向 Phase A 测量基建
====================================================
测量消费级 AMD GPU 上 LLM 后训练（SFT/GRPO）的效率指标。

核心指标
--------
- MFU (Model FLOPS Utilization)，采用**双峰值法**：
    * mfu_theoretical : 对标硬件白纸峰值（7900 XTX BF16 = 122.8 TFLOPS）
    * mfu_practical   : 对标本机实测 GEMM 峰值（更能反映真实优化空间）
  说明：RDNA3 无专用 Tensor Core，WMMA 为微码化向量指令，白纸峰值实践不可达，
        故必须同时报告两种 MFU，避免单一理论 MFU 人为偏低产生误导。
- GPU busy%      : rocm-smi 后台轮询
- 吞吐           : tokens/s, samples/s, s/step
- 峰值显存       : torch.cuda.max_memory_allocated + rocm-smi
- time-to-target : 达到目标 loss/acc 的墙钟时间（由调用方记录）

FLOPS 估算方法学（透明可审计）
------------------------------
训练步（SFT / GRPO 的策略更新）: FLOPS ≈ 6 · N · D
    N = 模型总参数量（前向+反向触达全部权重，含冻结部分；LoRA 下为近似）
    D = 该步处理的 token 数
    系数 6 = 前向 2 + 反向 4（激活梯度 2 + 权重梯度 2）
GRPO 生成阶段: 自回归逐 token，主要受显存带宽限制而非算力，
    MFU 意义有限，故单独报告生成吞吐(tokens/s)，不与训练步 MFU 混算。

使用方式
--------
1) 独立测本机 GEMM 实践峰值:
       python3 utils/efficiency_profiler.py --benchmark
2) 挂载到 HF Trainer 作为 Callback:
       from utils.efficiency_profiler import EfficiencyProfiler, EfficiencyCallback
       prof = EfficiencyProfiler(num_params=4.2e9, peak_tflops_practical=45.0)
       trainer = Trainer(..., callbacks=[EfficiencyCallback(prof)])
       prof.report()
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


# =============================================================================
# GPU 规格注册表（理论峰值）
# =============================================================================
# bf16_tflops: BF16/FP16 WMMA 矩阵理论峰值；bw_gbps: 显存带宽
GPU_SPECS = {
    "AMD Radeon RX 7900 XTX": {"bf16_tflops": 122.8, "fp32_tflops": 61.4, "bw_gbps": 960, "vram_gb": 24.0},
    "AMD Radeon RX 7900 XT":  {"bf16_tflops": 103.2, "fp32_tflops": 51.6, "bw_gbps": 800, "vram_gb": 20.0},
}
DEFAULT_BF16_PEAK_TFLOPS = 122.8  # 7900 XTX


def detect_gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "Unknown"


def theoretical_peak_tflops(gpu_name: Optional[str] = None) -> float:
    name = gpu_name or detect_gpu_name()
    for key, spec in GPU_SPECS.items():
        if key in name:
            return spec["bf16_tflops"]
    return DEFAULT_BF16_PEAK_TFLOPS


# =============================================================================
# 实践 GEMM 峰值 benchmark
# =============================================================================

def measure_practical_gemm_peak(dtype_str: str = "bf16", size: int = 8192,
                                warmup: int = 10, iters: int = 50) -> dict:
    """跑大矩阵乘法，测本机实际可达 BF16/FP16 峰值（TFLOPS）。

    这是"实践峰值"——比白纸峰值低，但是真实优化能逼近的上限。
    """
    import torch
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]
    dev = torch.device("cuda")
    a = torch.randn(size, size, dtype=dtype, device=dev)
    b = torch.randn(size, size, dtype=dtype, device=dev)

    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    flops_per_matmul = 2.0 * size ** 3
    total_flops = flops_per_matmul * iters
    tflops = total_flops / elapsed / 1e12
    return {
        "dtype": dtype_str, "size": size, "iters": iters,
        "elapsed_s": round(elapsed, 4),
        "practical_peak_tflops": round(tflops, 2),
        "theoretical_peak_tflops": theoretical_peak_tflops(),
        "fraction_of_theoretical": round(tflops / theoretical_peak_tflops(), 3),
    }


# =============================================================================
# FLOPS 估算
# =============================================================================

def estimate_train_flops(num_params: float, num_tokens: int) -> float:
    """训练步 FLOPS ≈ 6·N·D。返回 FLOPS（非 TFLOPS）。"""
    return 6.0 * num_params * num_tokens


# =============================================================================
# rocm-smi 后台轮询
# =============================================================================

class RocmSmiPoller:
    """后台线程周期轮询 GPU 利用率与显存，记录均值/峰值。"""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.gpu_use_samples: list[float] = []
        self.mem_used_samples: list[float] = []  # MB

    def _query(self) -> Optional[dict]:
        try:
            out = subprocess.run(
                ["rocm-smi", "--showuse", "--showmemuse", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            return json.loads(out.stdout)
        except Exception:
            return None

    def _parse(self, data: dict) -> tuple[Optional[float], Optional[float]]:
        # rocm-smi JSON: {"card0": {"GPU use (%)": "12", "GPU memory use (%)": "34"}}
        for card, info in data.items():
            if not isinstance(info, dict):
                continue
            use = mem = None
            for k, v in info.items():
                if "GPU use" in k and "memory" not in k.lower():
                    try: use = float(str(v).strip().rstrip("%"))
                    except (ValueError, TypeError): pass
                if "memory use" in k.lower():
                    try: mem = float(str(v).strip().rstrip("%"))
                    except (ValueError, TypeError): pass
            if use is not None:
                return use, mem
        return None, None

    def _loop(self):
        while not self._stop.is_set():
            data = self._query()
            if data:
                use, mem = self._parse(data)
                if use is not None:
                    self.gpu_use_samples.append(use)
                if mem is not None:
                    self.mem_used_samples.append(mem)
            self._stop.wait(self.interval)

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        def stats(arr):
            if not arr:
                return {"mean": None, "max": None, "n": 0}
            return {"mean": round(sum(arr) / len(arr), 1),
                    "max": round(max(arr), 1), "n": len(arr)}
        return {"gpu_busy_pct": stats(self.gpu_use_samples),
                "mem_use_pct": stats(self.mem_used_samples)}


# =============================================================================
# 效率数据容器
# =============================================================================

@dataclass
class StepRecord:
    step: int
    elapsed_s: float
    num_tokens: int
    samples: int = 1
    peak_mem_gb: float = 0.0
    # v1.1 测量有效性保障字段
    grad_norm: Optional[float] = None
    loss: Optional[float] = None
    lr: Optional[float] = None
    valid: bool = True  # 由 validate() 判定


@dataclass
class EfficiencySummary:
    num_params: float
    peak_tflops_theoretical: float
    peak_tflops_practical: Optional[float]
    total_steps: int
    valid_steps: int  # v1.1: 通过有效性判定的步数
    total_tokens: int
    total_time_s: float
    tokens_per_s: float
    s_per_step: float
    achieved_tflops: float
    mfu_theoretical_pct: float
    mfu_practical_pct: Optional[float]
    peak_mem_gb: float
    gpu_busy: dict = field(default_factory=dict)
    invalid_steps: int = 0  # v1.1: 无效步数（发散/冻结/空跑）


# =============================================================================
# 主 Profiler
# =============================================================================

class EfficiencyProfiler:
    """非侵入式效率采集器。配合 EfficiencyCallback 使用，或手动 record_step。"""

    def __init__(self, num_params: float,
                 peak_tflops_practical: Optional[float] = None,
                 peak_tflops_theoretical: Optional[float] = None,
                 poll_gpu: bool = True, poll_interval: float = 0.5):
        self.num_params = num_params
        self.peak_theoretical = peak_tflops_theoretical or theoretical_peak_tflops()
        self.peak_practical = peak_tflops_practical
        self.records: list[StepRecord] = []
        self._step_t0: Optional[float] = None
        self._step_tokens: int = 0
        self.poller = RocmSmiPoller(poll_interval) if poll_gpu else None
        if self.poller:
            self.poller.start()

    # --- 手动记录接口 ---
    def begin_step(self, num_tokens: int):
        self._step_t0 = time.perf_counter()
        self._step_tokens = num_tokens

    def end_step(self, step: int, samples: int = 1):
        if self._step_t0 is None:
            return
        elapsed = time.perf_counter() - self._step_t0
        mem = self._peak_mem_gb()
        self.records.append(StepRecord(step, elapsed, self._step_tokens, samples, mem))
        self._step_t0 = None

    def record_step(self, step: int, elapsed_s: float, num_tokens: int, samples: int = 1,
                    grad_norm: Optional[float] = None, loss: Optional[float] = None,
                    lr: Optional[float] = None):
        rec = StepRecord(step, elapsed_s, num_tokens, samples, self._peak_mem_gb(),
                         grad_norm=grad_norm, loss=loss, lr=lr)
        rec.valid = self._check_validity(rec)
        self.records.append(rec)

    # --- v1.1 测量有效性准则 ---
    def _check_validity(self, rec: StepRecord) -> bool:
        """Valid measurement 判定准则（吸取 Chronicals 教训）：
        1. grad_norm 必须 > 0（排除零梯度/模型未训练）
        2. num_tokens 必须 > 0（排除空跑）
        3. elapsed_s 必须 > 0（排除时钟异常）
        若 grad_norm 未提供（旧日志），仅检查 2/3，标记为 valid（宽容模式）。
        """
        if rec.num_tokens <= 0 or rec.elapsed_s <= 0:
            return False
        if rec.grad_norm is not None and rec.grad_norm <= 0:
            return False
        return True

    @staticmethod
    def _peak_mem_gb() -> float:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / 1024 ** 3
        except Exception:
            pass
        return 0.0

    # --- 汇总 ---
    def summarize(self, only_valid: bool = True) -> EfficiencySummary:
        """汇总效率指标。only_valid=True 时仅统计通过有效性判定的步。"""
        recs = [r for r in self.records if (r.valid or not only_valid)]
        valid_count = sum(1 for r in self.records if r.valid)
        invalid_count = len(self.records) - valid_count
        total_time = sum(r.elapsed_s for r in recs)
        total_tokens = sum(r.num_tokens for r in recs)
        peak_mem = max((r.peak_mem_gb for r in recs), default=0.0)
        tps = total_tokens / total_time if total_time else 0.0
        # 实测 TFLOPS = 6·N·D / time
        achieved = (6.0 * self.num_params * total_tokens / total_time / 1e12) if total_time else 0.0
        mfu_th = achieved / self.peak_theoretical * 100 if self.peak_theoretical else 0.0
        mfu_pr = (achieved / self.peak_practical * 100) if self.peak_practical else None
        return EfficiencySummary(
            num_params=self.num_params,
            peak_tflops_theoretical=self.peak_theoretical,
            peak_tflops_practical=self.peak_practical,
            total_steps=len(recs),
            valid_steps=valid_count,
            total_tokens=total_tokens,
            total_time_s=round(total_time, 2),
            tokens_per_s=round(tps, 1),
            s_per_step=round(total_time / len(recs), 3) if recs else 0.0,
            achieved_tflops=round(achieved, 2),
            mfu_theoretical_pct=round(mfu_th, 2),
            mfu_practical_pct=round(mfu_pr, 2) if mfu_pr is not None else None,
            peak_mem_gb=round(peak_mem, 2),
            gpu_busy=self.poller.summary() if self.poller else {},
            invalid_steps=invalid_count,
        )

    def report(self, path: Optional[str] = None) -> dict:
        s = self.summarize(only_valid=True)
        d = asdict(s)
        d["gpu_name"] = detect_gpu_name()
        # 增量: dump 每区间原始记录, 供下游剔除 warmup 区间算 steady-state (不影响既有字段)
        d["step_records"] = [asdict(r) for r in self.records]
        if self.poller:
            self.poller.stop()
        txt = (
            f"\n{'=' * 60}\n效率报告 | {d['gpu_name']}\n{'=' * 60}\n"
            f"参数量 N        : {s.num_params:.3e}\n"
            f"有效步 / 总步   : {s.valid_steps} / {s.valid_steps + s.invalid_steps}"
            f"  (invalid={s.invalid_steps})\n"
            f"总 tokens       : {s.total_tokens:,}\n"
            f"总耗时          : {s.total_time_s} s\n"
            f"吞吐            : {s.tokens_per_s} tokens/s | {s.s_per_step} s/step\n"
            f"实测算力        : {s.achieved_tflops} TFLOPS\n"
            f"理论峰值        : {s.peak_tflops_theoretical} TFLOPS → MFU(理论) = {s.mfu_theoretical_pct}%\n"
        )
        if s.peak_tflops_practical:
            txt += f"实践峰值        : {s.peak_tflops_practical} TFLOPS → MFU(实践) = {s.mfu_practical_pct}%\n"
        txt += f"峰值显存        : {s.peak_mem_gb} GB\n"
        if s.gpu_busy:
            gb = s.gpu_busy.get("gpu_busy_pct", {})
            txt += f"GPU busy%       : mean={gb.get('mean')} max={gb.get('max')} (n={gb.get('n')})\n"
        txt += "=" * 60 + "\n"
        print(txt)
        if path:
            with open(path, "w") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
        return d


# =============================================================================
# HF Trainer Callback（可选，惰性导入 transformers）
# =============================================================================

def EfficiencyCallback(profiler: EfficiencyProfiler):
    """构造一个 HF TrainerCallback，自动从训练日志采集每步耗时与 token 数。

    依赖 Trainer 在 log 中提供 'num_tokens'（TRL/GRPO 提供）或回退到
    'train_samples_per_second'×batch 估算。SFT 若无 num_tokens，需用
    train_on_responses_only 的 token 计数或手动设置。
    """
    from transformers import TrainerCallback

    class _Cb(TrainerCallback):
        def __init__(self):
            self._last_log_time = time.perf_counter()
            self._last_step = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            now = time.perf_counter()
            step = state.global_step
            # 优先用 num_tokens（GRPO/TRL 提供）
            num_tokens = logs.get("num_tokens")
            if num_tokens is None:
                # 回退：用 logging_steps × batch × seq 估算（粗略）
                num_tokens = int(logs.get("train_samples_per_second", 0) *
                                 (now - self._last_log_time) *
                                 getattr(args, "per_device_train_batch_size", 1)) or 0
            elapsed = now - self._last_log_time
            # v1.1: 同步采集 grad_norm / loss / lr 用于有效性判定
            grad_norm = logs.get("grad_norm")
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            if step > self._last_step and elapsed > 0:
                profiler.record_step(step, elapsed, int(num_tokens),
                                     samples=getattr(args, "per_device_train_batch_size", 1),
                                     grad_norm=grad_norm, loss=loss, lr=lr)
            self._last_log_time = now
            self._last_step = step

    return _Cb()


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="效率 Profiler / GEMM 峰值 benchmark")
    ap.add_argument("--benchmark", action="store_true", help="测本机 GEMM 实践峰值")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--size", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--json", default=None, help="结果写入 JSON")
    args = ap.parse_args()

    if args.benchmark:
        print(f"GPU: {detect_gpu_name()}")
        print(f"理论 BF16 峰值: {theoretical_peak_tflops()} TFLOPS")
        print(f"运行 GEMM benchmark ({args.dtype}, size={args.size}, iters={args.iters}) ...")
        res = measure_practical_gemm_peak(args.dtype, args.size, iters=args.iters)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
