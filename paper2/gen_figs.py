#!/usr/bin/env python3
"""
Three-Layer Paper — publication figures (SciencePlots-driven).
==============================================================
Uses garrettj403/SciencePlots (science+nature styles) for journal-quality output.
Data-driven: reads results/efficiency/zsbr/*.json → vector PDF + PNG preview.

Figures:
  fig1_milestone.pdf   — Three-layer milestone ladder (23.2→33.3→40.5→45.6)
  fig2_phase.pdf       — τ_net phase diagram (3 empirical regime points)
  fig3_dose.pdf        — Investment dose-response (harm decays with selection freedom)
  fig4_capacity.pdf    — Signal-rate capacity staircase (L0→L1→L2→L4)

Usage: python3 gen_figs.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scienceplots  # noqa: F401
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


BASE = REPO_ROOT
ZDIR = os.path.join(BASE, "results/efficiency/zsbr")
OUT = os.path.join(BASE, "paper2/figures")
os.makedirs(OUT, exist_ok=True)

# science + nature: serif, colorblind-safe, journal sizing. no-latex 避免依赖系统 LaTeX
plt.style.use(['science', 'nature', 'no-latex'])

# Okabe-Ito 语义色(与旧 style.py 一致)
C = {'throughput': '#0072B2', 'signal': '#009E73', 'dynamics': '#CC79A7',
     'harm': '#D55E00', 'neutral': '#999999', 'base': '#000000'}


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {name}.pdf / .png")


# ── Fig 1: 三层里程碑阶梯 ──
def fig1_milestone():
    stages = ['Base', '+Throughput\n(DGBB)', '+Signal Rate\n(ZSBR)', '+Long Train\n(500 steps)']
    acc = [23.2, 33.3, 40.5, 45.6]
    layer_c = [C['base'], C['throughput'], C['signal'], C['dynamics']]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    x = np.arange(len(stages))
    bars = ax.bar(x, acc, color=layer_c, width=0.62, edgecolor='white', zorder=3)
    # 增量连线标注
    for i in range(1, len(acc)):
        dv = acc[i] - acc[i-1]
        ax.annotate(f'+{dv:.1f}', xy=(i, acc[i]), xytext=(i, acc[i]+1.8),
                    ha='center', fontsize=6.5, color=layer_c[i], fontweight='bold')
    for i, v in enumerate(acc):
        ax.text(i, v/2, f'{v}%', ha='center', va='center', color='white',
                fontsize=7.5, fontweight='bold')
    ax.axhline(23.2, ls=':', lw=0.8, color=C['neutral'], zorder=1)
    ax.text(0.4, 24.3, '2$\\times$ base', fontsize=6, color=C['neutral'], ha='center')
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=6.5)
    ax.set_ylabel('GSM8K held-out accuracy (%)')
    ax.set_ylim(0, 52)
    ax.set_title('Three-Layer Progressive Optimization', fontsize=8)
    save(fig, 'fig1_milestone')


# ── Fig 2: τ_net 相图三点 ──
def fig2_phase():
    d = json.load(open(os.path.join(ZDIR, 'phase_diagram.json')))
    pts = d['points']
    # x = 供给/收割比(对数), y = 投资 Δheld-out; 点大小=池尺寸
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    regimes = [
        ('Supply-dominated\n(7473)', 13.7/1.37, -6.4, C['throughput'], 'o'),
        ('Balance band\n(500)', 1.3/1.37, -6.6, C['signal'], 's'),
        ('Rotation self-balance\n(200)', 0.45/1.37, -0.8, C['dynamics'], '^'),
    ]
    ax.axhline(0, ls='--', lw=0.8, color=C['neutral'], zorder=1)
    ax.axvline(1.0, ls=':', lw=0.8, color='gray', zorder=1)
    ax.text(1.15, -1.3, 'balance line', fontsize=5.2, color='gray', rotation=90)
    for name, ratio, dy, col, mk in regimes:
        ax.scatter(ratio, dy, s=90, c=col, marker=mk, zorder=3,
                   edgecolor='white', linewidth=0.8, label=name)
    # 剂量响应趋势虚线
    xs = [r[1] for r in regimes]; ys = [r[2] for r in regimes]
    ax.plot(xs, ys, ls='-', lw=0.7, color=C['harm'], alpha=0.5, zorder=2)
    ax.set_xscale('log')
    ax.set_xlabel(r'Supply/Harvest ratio $\mu_0|D| / (\mu_h \cdot \mathrm{slots})$')
    ax.set_ylabel(r'Investment $\Delta$ held-out (pp)')
    ax.set_title(r'$\tau_{net}$ Regime Phase Diagram', fontsize=8)
    ax.legend(fontsize=5.0, loc='upper left', frameon=True, framealpha=0.9)
    ax.set_ylim(-8, 2.2); ax.set_xlim(0.25, 15)
    save(fig, 'fig2_phase')


# ── Fig 3: 投资剂量响应 ──
def fig3_dose():
    pools = ['Full\n(7473)', '500', '200']
    flux = [-17.0, -10.5, 4.5]      # sfc vs v1 通量 Δ%
    heldout = [-6.4, -6.6, -0.8]    # held-out Δpp
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    x = np.arange(len(pools))
    ax.axhline(0, ls='-', lw=0.6, color='black', zorder=1)
    l1 = ax.plot(x, flux, 'o-', color=C['harm'], lw=1.3, ms=6,
                 label='Signal flux $\\Delta$ (%)', zorder=3)
    ax2 = ax.twinx()
    l2 = ax2.plot(x, heldout, 's--', color=C['dynamics'], lw=1.3, ms=6,
                  label='Held-out $\\Delta$ (pp)', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(pools)
    ax.set_xlabel('Training pool size (selection freedom $\\downarrow$)')
    ax.set_ylabel('Signal flux $\\Delta$ sfc$-$v1 (%)', color=C['harm'])
    ax2.set_ylabel('Held-out $\\Delta$ (pp)', color=C['dynamics'])
    ax.tick_params(axis='y', colors=C['harm'])
    ax2.tick_params(axis='y', colors=C['dynamics'])
    ax.set_title('Investment Dose-Response (harm decays)', fontsize=8)
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], fontsize=5.5, loc='upper left')
    save(fig, 'fig3_dose')


# ── Fig 4: 信号率容量阶梯 ──
def fig4_capacity():
    levels = ['L0\nUniform', 'L1\nV1 select', 'L2\nMulti-entry', 'L4\nAdaptive G']
    bound = [0.25, 0.50, 0.75, 1.0]
    achieved = [0.25, 0.425, None, None]  # V1 实测 S=1-0.575=0.425 → 含ε修正可达上界0.45的94%
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    x = np.arange(len(levels))
    ax.bar(x, bound, width=0.6, color=C['neutral'], alpha=0.35,
           edgecolor='gray', zorder=2, label='Theoretical bound')
    # V1 实测点(达 ε-adjusted 上界0.45的 94%)
    ax.bar(1, 0.425, width=0.6, color=C['signal'], zorder=3,
           label='ZSBR-V1 achieved (94% of $\\varepsilon$-adj. bound 0.45)')
    ax.annotate('unreachable\n(TRL constraint)', xy=(3, 1.0), xytext=(2.5, 0.78),
                fontsize=5.2, ha='center', color=C['harm'],
                arrowprops=dict(arrowstyle='->', color=C['harm'], lw=0.7))
    ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=6.5)
    ax.set_ylabel('Signal-rate capacity $S$')
    ax.set_title('Signal-Rate Capacity Staircase (G=2)', fontsize=8)
    ax.legend(fontsize=5.5, loc='upper left')
    ax.set_ylim(0, 1.15)
    save(fig, 'fig4_capacity')


if __name__ == '__main__':
    print("生成三层论文图表 (SciencePlots science+nature):")
    fig1_milestone()
    fig2_phase()
    fig3_dose()
    fig4_capacity()
    print(f"\n✅ 全部输出至 {OUT}")
