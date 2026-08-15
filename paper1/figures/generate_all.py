#!/usr/bin/env python3
"""
Generate all publication figures from experiment JSON data.
Data-driven pipeline: JSON → matplotlib → vector PDF

Usage:
    python3 generate_all.py

Outputs:
    fig_waterfall.pdf  — Gap attribution waterfall (vertical, linear)
    fig_framework.pdf  — Framework comparison bars (3 panels)
    fig_sweep.pdf      — Hyperparameter sweep (2×2 layout)
    fig_pareto.pdf     — VRAM-Throughput Pareto (smart labels)
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adjustText import adjust_text

import style
style.apply_style()

# Paths
ROOT = Path(__file__).parent.parent.parent  # project root
RESULTS = ROOT / "results" / "efficiency"
FIG_DIR = Path(__file__).parent


def load_json(path):
    with open(path) as f:
        return json.load(f)


# P1-C1 audit fix: corrected caliber (final cumulative tokens / wall time).
# Original efficiency_report.json values retained on disk for traceability.
CORR = load_json(RESULTS / "B1_C_corrected_summary.json")


def corrected_b1(run_name):
    """Return corrected (tps, mfu, vram) for a B1 run dir name."""
    r = CORR["B1"][run_name]
    return (r["tokens_per_s_corrected"], r["mfu_practical_pct_corrected"],
            r["peak_mem_gb"])


def corrected_c(cfg_name):
    """Return corrected dict for a C sweep config name."""
    r = CORR["C"][cfg_name]
    return {"name": cfg_name, "tokens_per_s": r["tokens_per_s_corrected"],
            "mfu_practical_pct": r["mfu_practical_pct_corrected"],
            "peak_mem_gb": r["peak_mem_gb"]}


# ═══════════════════════════════════════════════════════════
# Figure 1: Gap Attribution Waterfall (vertical, linear)
# ═══════════════════════════════════════════════════════════
def fig_waterfall():
    """Vertical waterfall chart showing TFLOPS at each level."""
    data = load_json(RESULTS / "B2_gap_attribution" / "gap_attribution.json")
    waterfall = data["waterfall"]

    labels = ["L0\nGEMM", "L1\nInfer", "L2\nSFT", "L3\n+ckpt", "L4\nGRPO"]
    tflops = [w["tflops"] for w in waterfall]
    # P1-C1: L4 used the buggy profiler caliber; replace with corrected estimate
    l4_corr = CORR.get("gap_L4", {}).get("achieved_tflops_corrected_est")
    if l4_corr:
        tflops[-1] = l4_corr
    peak = tflops[0]
    total_gap_pct = (peak - tflops[-1]) / peak * 100

    fig, ax = plt.subplots(figsize=(style.COL_WIDTH_2, 3.0))

    colors = [style.SEMANTIC['gemm'], style.COLORS['gray'],
              style.SEMANTIC['unsloth'], style.SEMANTIC['unsloth'],
              style.SEMANTIC['gap']]

    x = np.arange(len(labels))
    bars = ax.bar(x, tflops, color=colors, edgecolor='white', width=0.55)

    # Value labels on top of each bar
    for i, (bar, val) in enumerate(zip(bars, tflops)):
        mfu = val / peak * 100
        label = f'{val:.1f}\n({mfu:.1f}%)' if val > 1 else f'{val:.2f}\n({mfu:.2f}%)'
        ax.text(bar.get_x() + bar.get_width() / 2, val + peak * 0.02,
                label, ha='center', va='bottom', fontsize=6.5)

    # Dashed connectors between adjacent levels
    for i in range(len(tflops) - 1):
        ax.plot([x[i] + 0.28, x[i + 1] - 0.28],
                [tflops[i], tflops[i]], 'k--', linewidth=0.5, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Achieved TFLOPS')
    ax.set_ylim(0, peak * 1.18)
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', visible=False)

    # Total gap annotation (top-right, away from data)
    ax.text(0.98, 0.95, f'Total gap: {total_gap_pct:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor=style.COLORS['yellow'], alpha=0.7))

    style.save_fig(fig, FIG_DIR / "fig_waterfall.pdf")


# ═══════════════════════════════════════════════════════════
# Figure 2: Framework Comparison (3 panels, taller)
# ═══════════════════════════════════════════════════════════
def fig_framework():
    """3-panel bar chart: Throughput, MFU, VRAM with annotate."""
    unsloth_bf16, hf_bf16, unsloth_4bit = [], [], []

    # P1-C1: use corrected caliber values from B1_C_corrected_summary.json
    for run_name in CORR["B1"]:
        tps, mfu, vram = corrected_b1(run_name)
        rec = {"tokens_per_s": tps, "mfu_practical_pct": mfu, "peak_mem_gb": vram}
        if "unsloth_bf16" in run_name:
            unsloth_bf16.append(rec)
        elif "hf_bf16" in run_name:
            hf_bf16.append(rec)
        elif "unsloth_4bit" in run_name:
            unsloth_4bit.append(rec)

    def stats(reports):
        tps = [r["tokens_per_s"] for r in reports]
        mfu = [r["mfu_practical_pct"] for r in reports]
        vram = [r["peak_mem_gb"] for r in reports]
        return (np.mean(tps), np.std(tps),
                np.mean(mfu), np.std(mfu),
                np.mean(vram), np.std(vram))

    u_tps, u_tps_s, u_mfu, u_mfu_s, u_vram, u_vram_s = stats(unsloth_bf16)
    h_tps, h_tps_s, h_mfu, h_mfu_s, h_vram, h_vram_s = stats(hf_bf16)
    q_tps, q_tps_s, q_mfu, q_mfu_s, q_vram, q_vram_s = stats(unsloth_4bit)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(style.COL_WIDTH_2, 2.8))

    x = np.arange(3)
    labels = ['Unsloth\nBF16', 'HF\nBF16', 'Unsloth\n4-bit']
    colors = [style.SEMANTIC['unsloth'], style.SEMANTIC['hf'], style.SEMANTIC['quant']]

    # Panel 1: Throughput
    means = [u_tps, h_tps, q_tps]
    stds = [u_tps_s, h_tps_s, q_tps_s]
    bars = ax1.bar(x, means, yerr=stds, color=colors, edgecolor='white',
                   capsize=3, error_kw={'linewidth': 0.8})
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=6.5)
    ax1.set_ylabel('tok/s $\\uparrow$')
    ax1.set_ylim(0, max(means) * 1.18)
    for bar, m, s in zip(bars, means, stds):
        ax1.annotate(f'{m:.0f}', xy=(bar.get_x() + bar.get_width()/2, m + s),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=6.5)

    # Panel 2: MFU
    means_m = [u_mfu, h_mfu, q_mfu]
    stds_m = [u_mfu_s, h_mfu_s, q_mfu_s]
    bars2 = ax2.bar(x, means_m, yerr=stds_m, color=colors, edgecolor='white',
                    capsize=3, error_kw={'linewidth': 0.8})
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=6.5)
    ax2.set_ylabel('MFU$_{\\mathrm{prac}}$ (%) $\\uparrow$')
    ax2.set_ylim(0, max(means_m) * 1.22)
    for bar, m, s in zip(bars2, means_m, stds_m):
        ax2.annotate(f'{m:.1f}', xy=(bar.get_x() + bar.get_width()/2, m + s),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=6.5)

    # Panel 3: VRAM
    means_v = [u_vram, h_vram, q_vram]
    stds_v = [u_vram_s, h_vram_s, q_vram_s]
    bars3 = ax3.bar(x, means_v, yerr=stds_v, color=colors, edgecolor='white',
                    capsize=3, error_kw={'linewidth': 0.8})
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=6.5)
    ax3.set_ylabel('VRAM (GB) $\\downarrow$')
    ax3.set_ylim(0, max(means_v) * 1.22)
    for bar, m, s in zip(bars3, means_v, stds_v):
        ax3.annotate(f'{m:.1f}', xy=(bar.get_x() + bar.get_width()/2, m + s),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=6.5)

    fig.tight_layout()
    style.save_fig(fig, FIG_DIR / "fig_framework.pdf")


# ═══════════════════════════════════════════════════════════
# Figure 3: Hyperparameter Sweep (2×2 layout)
# ═══════════════════════════════════════════════════════════
def fig_sweep():
    """2×2 dot-line chart: sweep results by dimension."""
    # P1-C1: corrected caliber
    data = [corrected_c(name) for name in CORR["C"]]

    dims = {
        'Generations (G)': [
            ('G=2', 'C_G2_b1x8_ckpt_512'),
            ('G=4', 'C_G4_b1x8_ckpt_512'),
            ('G=8', 'C_G8_b1x8_ckpt_512'),
        ],
        'Effective Batch': [
            ('1×4', 'C_G4_b1x4_ckpt_512'),
            ('1×8', 'C_G4_b1x8_ckpt_512'),
            ('2×4', 'C_G4_b2x4_ckpt_512'),
        ],
        'Completion Length': [
            ('256', 'C_G4_b1x8_ckpt_256'),
            ('512', 'C_G4_b1x8_ckpt_512'),
            ('1024', 'C_G4_b1x8_ckpt_1024'),
        ],
        'Gradient Checkpoint': [
            ('On', 'C_G4_b1x8_ckpt_512'),
            ('Off', 'C_G4_b1x8_noCkpt_512'),
        ],
    }

    lookup = {d['name']: d for d in data}

    fig, axes = plt.subplots(2, 2, figsize=(style.COL_WIDTH_2, 4.5))
    axes_flat = axes.flatten()

    dim_colors = [style.COLORS['blue'], style.COLORS['orange'],
                  style.COLORS['green'], style.COLORS['purple']]

    for ax, (dim_name, configs), color in zip(axes_flat, dims.items(), dim_colors):
        tps_vals = []
        tick_labels = []
        for label, cfg_name in configs:
            if cfg_name in lookup:
                tps_vals.append(lookup[cfg_name]['tokens_per_s'])
                tick_labels.append(label)

        xi = np.arange(len(tps_vals))
        ax.plot(xi, tps_vals, 'o-', color=color, markersize=7, linewidth=1.5)
        ax.set_xticks(xi)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_title(dim_name, fontsize=8, pad=6)
        ax.grid(axis='y', alpha=0.3)

        # Independent y-axis to highlight within-dimension differences
        y_min, y_max = min(tps_vals), max(tps_vals)
        margin = max((y_max - y_min) * 0.35, y_max * 0.05)
        ax.set_ylim(y_min - margin, y_max + margin)

        # Annotate with points offset (no overlap)
        for x_val, val in zip(xi, tps_vals):
            ax.annotate(f'{val:.0f}', (x_val, val),
                       xytext=(0, 9), textcoords='offset points',
                       ha='center', fontsize=6.5)

    # Optimal annotation at bottom
    opt = lookup.get('C_G2_b2x4_ckpt_256')
    if opt:
        fig.text(0.5, 0.01,
                f'Optimal: G=2, b2×4, seq=256 → {opt["tokens_per_s"]:.0f} tok/s '
                f'({opt["mfu_practical_pct"]:.1f}% MFU)',
                ha='center', fontsize=7, color=style.SEMANTIC['optimal'])

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    style.save_fig(fig, FIG_DIR / "fig_sweep.pdf")


# ═══════════════════════════════════════════════════════════
# Figure 4: VRAM-Throughput Pareto (smart labels)
# ═══════════════════════════════════════════════════════════
def fig_pareto():
    """Horizontal bar chart: throughput per config, colored by MFU, with VRAM annotations."""
    # P1-C1: corrected caliber
    data = [corrected_c(name) for name in CORR["C"]]

    # Sort by throughput ascending (bottom=worst, top=best)
    data_sorted = sorted(data, key=lambda d: d['tokens_per_s'])

    names = []
    for d in data_sorted:
        n = d['name'].replace('C_', '')
        parts = n.split('_')
        if 'noCkpt' in n:
            names.append(f"{parts[0]}, {parts[1]}, noCkpt")
        elif len(parts) >= 4:
            names.append(f"{parts[0]}, {parts[1]}, {parts[3]}")
        else:
            names.append(n)

    tps = [d['tokens_per_s'] for d in data_sorted]
    mfus = [d['mfu_practical_pct'] for d in data_sorted]
    vrams = [d['peak_mem_gb'] for d in data_sorted]

    fig, ax = plt.subplots(figsize=(style.COL_WIDTH_2, 3.2))

    y = np.arange(len(names))
    # Color bars by MFU using colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(mfus) - 1, max(mfus) + 1)
    colors = [cmap(norm(m)) for m in mfus]

    bars = ax.barh(y, tps, color=colors, edgecolor='white', height=0.6)

    # Highlight optimal config
    opt_idx = next((i for i, d in enumerate(data_sorted)
                   if d['name'] == 'C_G2_b2x4_ckpt_256'), None)
    if opt_idx is not None:
        bars[opt_idx].set_edgecolor('red')
        bars[opt_idx].set_linewidth(1.5)

    # Value labels: tok/s inside bar, VRAM at bar end
    for i, (bar, t, v) in enumerate(zip(bars, tps, vrams)):
        # tok/s value at bar end
        ax.text(t + 1.5, bar.get_y() + bar.get_height()/2,
                f'{t:.0f}', va='center', fontsize=6.5)
        # VRAM annotation
        ax.text(t - 3, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}G', va='center', ha='right',
                fontsize=5.5, color='white', alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel('Throughput (tok/s) $\\uparrow$')
    ax.set_xlim(0, max(tps) * 1.12)
    ax.grid(axis='x', alpha=0.3)
    ax.grid(axis='y', visible=False)

    # Colorbar for MFU
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                       shrink=0.6, pad=0.18, aspect=30)
    cbar.set_label('MFU$_{\\mathrm{prac}}$ (%)', fontsize=7)

    fig.tight_layout()
    style.save_fig(fig, FIG_DIR / "fig_pareto.pdf")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating publication figures...")
    print(f"  Data source: {RESULTS}")
    print(f"  Output dir:  {FIG_DIR}\n")

    fig_waterfall()
    fig_framework()
    fig_sweep()
    fig_pareto()

    print("\n✓ All figures generated successfully.")
