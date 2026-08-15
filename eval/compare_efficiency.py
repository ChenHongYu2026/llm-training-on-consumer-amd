#!/usr/bin/env python3
"""
Phase B1 框架横向对比分析
========================
读取多组 efficiency_report.json，生成对比表格与统计摘要。

用法：
  python3 eval/compare_efficiency.py results/efficiency/B1_*/efficiency_report.json
  python3 eval/compare_efficiency.py --dir results/efficiency/  # 自动发现 B1_ 前缀
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))



def load_report(path: str) -> dict:
    """加载单个 efficiency_report.json。"""
    with open(path) as f:
        return json.load(f)


def discover_reports(base_dir: str) -> list[tuple[str, dict]]:
    """自动发现 base_dir 下所有 B1_ 开头的实验目录。"""
    reports = []
    base = Path(base_dir)
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("B1_"):
            rpath = d / "efficiency_report.json"
            if rpath.exists():
                reports.append((d.name, load_report(str(rpath))))
    return reports


def format_table(reports: list[tuple[str, dict]]) -> str:
    """生成 Markdown 对比表格。"""
    lines = []
    lines.append("# Phase B1 框架横向对比结果\n")
    lines.append(f"> 生成时间: {__import__('time').strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> 硬件: AMD Radeon RX 7900 XTX 24GB (RDNA3/gfx1100)")
    lines.append(f"> 模型: Qwen2.5-3B-Instruct (3.086B params)")
    lines.append(f"> Workload: GSM8K GRPO (G=4, batch=1×8, completion≤512)\n")

    # 表头
    header = "| 配置 | tokens/s | s/step | MFU(理论)% | MFU(实践)% | VRAM(GB) | GPU busy% | 有效步/总步 |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    for name, r in reports:
        s = r.get("summary", r)
        tps = s.get("tokens_per_s", 0)
        sps = s.get("s_per_step", 0)
        mfu_t = s.get("mfu_theoretical_pct", 0)
        mfu_p = s.get("mfu_practical_pct", 0)
        vram = s.get("peak_mem_gb", 0)
        gpu_busy = s.get("gpu_busy", {}).get("mean", 0)
        valid = s.get("valid_steps", "?")
        total = s.get("total_steps", "?")
        invalid = s.get("invalid_steps", 0)

        lines.append(
            f"| {name} | {tps:.1f} | {sps:.1f} | {mfu_t:.2f} | {mfu_p:.2f} "
            f"| {vram:.1f} | {gpu_busy:.1f} | {valid}/{total} (inv={invalid}) |"
        )

    return "\n".join(lines)


def compute_speedup(reports: list[tuple[str, dict]]) -> str:
    """计算相对加速比。"""
    if len(reports) < 2:
        return ""

    lines = ["\n## 加速比分析\n"]

    # 以 HF BF16 为基准
    baseline = None
    for name, r in reports:
        if "hf" in name.lower() and "bf16" in name.lower():
            baseline = (name, r)
            break
    if baseline is None:
        baseline = reports[0]  # fallback

    base_tps = baseline[1].get("summary", baseline[1]).get("tokens_per_s", 1)
    lines.append(f"基准: {baseline[0]} ({base_tps:.1f} tokens/s)\n")
    lines.append("| 配置 | 相对吞吐比 | 解读 |")
    lines.append("|---|---|---|")

    for name, r in reports:
        s = r.get("summary", r)
        tps = s.get("tokens_per_s", 0)
        ratio = tps / base_tps if base_tps > 0 else 0
        if ratio > 1.05:
            interp = f"快 {(ratio-1)*100:.0f}%"
        elif ratio < 0.95:
            interp = f"慢 {(1-ratio)*100:.0f}%"
        else:
            interp = "持平"
        lines.append(f"| {name} | {ratio:.2f}× | {interp} |")

    return "\n".join(lines)


def measurement_validity_check(reports: list[tuple[str, dict]]) -> str:
    """测量有效性审计。"""
    lines = ["\n## 测量有效性审计\n"]
    lines.append("| 配置 | 有效步 | 无效步 | 有效率 | 判定 |")
    lines.append("|---|---|---|---|---|")

    for name, r in reports:
        s = r.get("summary", r)
        valid = s.get("valid_steps", 0)
        total = s.get("total_steps", 0)
        invalid = s.get("invalid_steps", 0)
        rate = valid / total * 100 if total > 0 else 0
        verdict = "✅ PASS" if rate >= 95 else "⚠️ WARN" if rate >= 80 else "❌ FAIL"
        lines.append(f"| {name} | {valid} | {invalid} | {rate:.0f}% | {verdict} |")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Phase B1 效率对比分析")
    ap.add_argument("reports", nargs="*", help="efficiency_report.json 路径列表")
    ap.add_argument("--dir", type=str, default=None, help="自动发现目录")
    ap.add_argument("--output", type=str, default=None, help="输出 Markdown 路径")
    args = ap.parse_args()

    # 收集报告
    if args.dir:
        reports = discover_reports(args.dir)
    elif args.reports:
        reports = [(Path(p).parent.name, load_report(p)) for p in args.reports]
    else:
        # 默认发现
        reports = discover_reports("results/efficiency/")

    if not reports:
        print("❌ 未找到任何效率报告。请先运行实验。")
        sys.exit(1)

    print(f"📊 发现 {len(reports)} 组实验报告:")
    for name, _ in reports:
        print(f"   - {name}")

    # 生成分析
    md = format_table(reports)
    md += "\n" + compute_speedup(reports)
    md += "\n" + measurement_validity_check(reports)

    # 输出
    output_path = args.output or "results/efficiency/B1_comparison.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md)
    print(f"\n✅ 对比分析已保存: {output_path}")
    print("\n" + md)


if __name__ == "__main__":
    main()
