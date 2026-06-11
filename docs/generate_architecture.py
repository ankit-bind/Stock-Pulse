"""
Generate architecture diagram for Stock-Pulse.
Run this to create docs/architecture_diagram.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set dark background
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#0a0e1a')
ax.set_facecolor('#0a0e1a')
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
COLORS = {
    'etl': '#00d4ff',      # cyan
    'features': '#7c3aed', # violet
    'ml': '#10b981',       # green
    'portfolio': '#f59e0b', # amber
    'dashboard': '#e2e8f0', # light
    'data': '#ef4444',      # red
    'arrow': '#94a3b8',     # muted
}

def draw_box(ax, x, y, w, h, color, label, sublabel=None, fontsize=11):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15",
                          facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.1, label, ha='center', va='center', fontsize=fontsize,
            color=color, fontweight='bold', fontfamily='sans-serif')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha='center', va='center', fontsize=9,
                color='#94a3b8', fontfamily='sans-serif')

def draw_arrow(ax, x1, y1, x2, y2, color='white'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Title
ax.text(7, 9.5, 'Stock-Pulse Architecture', ha='center', va='center', fontsize=24,
        color='#e2e8f0', fontweight='bold', fontfamily='sans-serif')
ax.text(7, 9.1, 'End-to-End Quantitative Research & ML Portfolio System', ha='center', va='center',
        fontsize=14, color='#94a3b8', fontfamily='sans-serif')

# Layer 1: Data Sources
ax.text(1.5, 8.2, 'DATA SOURCES', ha='center', va='center', fontsize=10,
        color=COLORS['data'], fontweight='bold', alpha=0.7)
draw_box(ax, 0.5, 7.0, 2.0, 0.8, COLORS['data'], 'CSV Files', '10 Stock Symbols', 10)

# Arrow to ETL
ax.annotate('', xy=(3.2, 7.4), xytext=(2.5, 7.4),
            arrowprops=dict(arrowstyle='->', color=COLORS['data'], lw=2))

# Layer 2: ETL Pipeline
ax.text(5.5, 8.2, 'ETL PIPELINE', ha='center', va='center', fontsize=10,
        color=COLORS['etl'], fontweight='bold', alpha=0.7)
draw_box(ax, 3.2, 7.0, 1.4, 0.8, COLORS['etl'], 'Bronze', 'Raw CSV', 9)
draw_box(ax, 4.8, 7.0, 1.4, 0.8, COLORS['etl'], 'Silver', 'Cleaned', 9)
draw_box(ax, 6.4, 7.0, 1.4, 0.8, COLORS['etl'], 'Gold', 'Features', 9)

ax.annotate('', xy=(4.8, 7.4), xytext=(4.6, 7.4),
            arrowprops=dict(arrowstyle='->', color=COLORS['etl'], lw=2))
ax.annotate('', xy=(6.4, 7.4), xytext=(6.2, 7.4),
            arrowprops=dict(arrowstyle='->', color=COLORS['etl'], lw=2))

# Arrow to DB
ax.annotate('', xy=(8.2, 7.4), xytext=(7.8, 7.4),
            arrowprops=dict(arrowstyle='->', color=COLORS['etl'], lw=2))

# Database
ax.text(10.5, 8.2, 'DATABASE', ha='center', va='center', fontsize=10,
        color=COLORS['dashboard'], fontweight='bold', alpha=0.7)
draw_box(ax, 8.2, 7.0, 1.6, 0.8, COLORS['dashboard'], 'SQL Server', 'SQLite (dev)', 10)
draw_box(ax, 10.0, 7.0, 1.8, 0.8, COLORS['dashboard'], 'Schema', 'Bronze/Silver/Gold', 10)

# Vertical arrow down
ax.annotate('', xy=(7.0, 6.2), xytext=(7.0, 7.0),
            arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=2, linestyle='--'))

# Middle Layer: Feature Engineering
ax.text(3.5, 6.0, 'FEATURE ENGINEERING', ha='center', va='center', fontsize=10,
        color=COLORS['features'], fontweight='bold', alpha=0.7)
draw_box(ax, 2.5, 5.0, 2.0, 0.8, COLORS['features'], 'Technical', 'RSI, MACD, SMA', 9)
draw_box(ax, 5.0, 5.0, 2.0, 0.8, COLORS['features'], 'Returns', 'Daily, Rolling', 9)
draw_box(ax, 7.5, 5.0, 2.0, 0.8, COLORS['features'], 'Momentum', 'Cross-sectional', 9)

# Vertical arrow down
ax.annotate('', xy=(7.0, 4.2), xytext=(7.0, 5.0),
            arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=2, linestyle='--'))

# ML Layer
ax.text(5.5, 4.0, 'ML PREDICTION', ha='center', va='center', fontsize=10,
        color=COLORS['ml'], fontweight='bold', alpha=0.7)
draw_box(ax, 2.5, 3.0, 2.2, 0.8, COLORS['ml'], 'RandomForest', 'Walk-Forward', 10)
draw_box(ax, 5.0, 3.0, 2.2, 0.8, COLORS['ml'], 'XGBoost', 'Optional', 10)
draw_box(ax, 7.5, 3.0, 2.2, 0.8, COLORS['ml'], 'Validation', 'Expanding Window', 10)

# Vertical arrow down
ax.annotate('', xy=(7.0, 2.2), xytext=(7.0, 3.0),
            arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=2, linestyle='--'))

# Portfolio Layer
ax.text(5.5, 2.0, 'PORTFOLIO', ha='center', va='center', fontsize=10,
        color=COLORS['portfolio'], fontweight='bold', alpha=0.7)
draw_box(ax, 2.0, 1.0, 2.0, 0.8, COLORS['portfolio'], 'Equal Weight', 'Inverse Vol', 9)
draw_box(ax, 4.2, 1.0, 2.0, 0.8, COLORS['portfolio'], 'Cost-Aware', 'L1 Proximal', 9)
draw_box(ax, 6.4, 1.0, 2.0, 0.8, COLORS['portfolio'], 'Beta Neutral', 'Vol Target', 9)
draw_box(ax, 8.6, 1.0, 2.0, 0.8, COLORS['portfolio'], 'Evaluation', 'Sharpe, CAGR', 9)

# Arrow to Dashboard
ax.annotate('', xy=(12.0, 1.4), xytext=(10.6, 1.4),
            arrowprops=dict(arrowstyle='->', color=COLORS['dashboard'], lw=2))

# Dashboard
ax.text(12.8, 2.0, 'DASHBOARD', ha='center', va='center', fontsize=10,
        color=COLORS['etl'], fontweight='bold', alpha=0.7)
draw_box(ax, 11.8, 1.0, 2.2, 0.8, COLORS['etl'], 'Streamlit', 'Plotly Charts', 10)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=COLORS['etl'], alpha=0.15, edgecolor=COLORS['etl'], label='ETL'),
    mpatches.Patch(facecolor=COLORS['features'], alpha=0.15, edgecolor=COLORS['features'], label='Features'),
    mpatches.Patch(facecolor=COLORS['ml'], alpha=0.15, edgecolor=COLORS['ml'], label='ML'),
    mpatches.Patch(facecolor=COLORS['portfolio'], alpha=0.15, edgecolor=COLORS['portfolio'], label='Portfolio'),
    mpatches.Patch(facecolor=COLORS['dashboard'], alpha=0.15, edgecolor=COLORS['dashboard'], label='Database'),
    mpatches.Patch(facecolor=COLORS['data'], alpha=0.15, edgecolor=COLORS['data'], label='Data'),
]
ax.legend(handles=legend_elements, loc='lower left', facecolor='#0a0e1a', edgecolor='#1a2235',
          labelcolor='#94a3b8', fontsize=10)

plt.tight_layout()
plt.savefig('docs/architecture_diagram.png', dpi=150, facecolor='#0a0e1a', edgecolor='none',
            bbox_inches='tight', pad_inches=0.3)
print("Architecture diagram saved to docs/architecture_diagram.png")
