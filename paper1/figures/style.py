"""
Unified figure style for MLSys 2025 paper.
Follows Orchestra AI-Research-SKILLs figure standards:
- Okabe-Ito colorblind-friendly palette
- Vector PDF output with TrueType embedding
- MLSys column-width sizing (3.25in single, 6.75in double)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ═══════════════════════════════════════════════════════════
# Okabe-Ito colorblind-friendly palette
# ═══════════════════════════════════════════════════════════
COLORS = {
    'blue':    '#0072B2',
    'orange':  '#D55E00',
    'green':   '#009E73',
    'purple':  '#CC79A7',
    'yellow':  '#F0E442',
    'cyan':    '#56B4E9',
    'red':     '#E69F00',
    'black':   '#000000',
    'gray':    '#999999',
}

# Semantic colors for this paper
SEMANTIC = {
    'unsloth':  COLORS['blue'],
    'hf':       COLORS['orange'],
    'quant':    COLORS['purple'],
    'gemm':     COLORS['green'],
    'gap':      COLORS['red'],
    'optimal':  COLORS['green'],
    'baseline': COLORS['gray'],
}

# ═══════════════════════════════════════════════════════════
# MLSys column widths
# ═══════════════════════════════════════════════════════════
COL_WIDTH = 3.25   # single column (inches)
COL_WIDTH_2 = 6.75  # double column (inches)

# ═══════════════════════════════════════════════════════════
# Apply style
# ═══════════════════════════════════════════════════════════
def apply_style():
    """Apply unified style to matplotlib."""
    plt.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        
        # Figure size (default single column)
        'figure.figsize': (COL_WIDTH, COL_WIDTH * 0.75),
        'figure.dpi': 300,
        
        # PDF output
        'pdf.fonttype': 42,  # TrueType embedding
        'ps.fonttype': 42,
        
        # Axes
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        
        # Lines
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
        
        # Legend
        'legend.frameon': False,
        'legend.borderaxespad': 0.5,
        
        # Saving
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.transparent': False,
    })


def save_fig(fig, path):
    """Save figure as vector PDF."""
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
