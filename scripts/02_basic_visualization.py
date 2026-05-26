"""
Step 2: Basic Visualization Suite
Publication-quality figures following Nature/Science guidelines
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from config import *

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(DATA_DIR, 'outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_excel(os.path.join(DATA_DIR, 'data', 'raw', '数据集3_空气质量.xlsx'))

# ─── Figure 1: Distribution of All Variables ─────────────────────────────────
print("Generating Figure 1: Variable Distributions...")
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
fig.suptitle('Distribution of Air Quality and Meteorological Variables',
             fontsize=14, fontweight='bold', y=0.98)

for idx, col in enumerate(NUMERIC_COLS):
    ax = axes[idx // 3][idx % 3]
    data = df[col].dropna()

    # Histogram with KDE
    n, bins, patches = ax.hist(data, bins=15, density=True, alpha=0.6,
                                color=PALETTE_MAIN[idx], edgecolor='white',
                                linewidth=0.5)

    # KDE overlay
    kde_x = np.linspace(data.min() - data.std(), data.max() + data.std(), 200)
    kde = stats.gaussian_kde(data)
    ax.plot(kde_x, kde(kde_x), color=PALETTE_MAIN[idx], linewidth=1.8)

    # Add mean and median lines
    ax.axvline(data.mean(), color='#E74C3C', linestyle='--', linewidth=1.2, label=f'Mean={data.mean():.1f}')
    ax.axvline(data.median(), color='#2C3E50', linestyle=':', linewidth=1.2, label=f'Median={data.median():.1f}')

    ax.set_xlabel(COL_NAMES.get(col, col), fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Remove empty subplot
axes[1][2].axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(FIG_DIR, 'fig1_distributions.png'), dpi=300, bbox_inches='tight')
plt.close()

# ─── Figure 2: Seasonal Box Plots ────────────────────────────────────────────
print("Generating Figure 2: Seasonal Box Plots...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Seasonal Distribution of Variables',
             fontsize=14, fontweight='bold', y=0.98)

for idx, col in enumerate(NUMERIC_COLS):
    ax = axes[idx // 3][idx % 3]
    season_data = [df[df['season'] == s][col].values for s in SEASON_ORDER]
    bp = ax.boxplot(season_data, labels=[f'{s}\n({SEASON_NAMES[s]})' for s in SEASON_ORDER],
                    patch_artist=True, widths=0.6,
                    medianprops=dict(color='#2C3E50', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markerfacecolor='#E74C3C', markersize=4, alpha=0.6))

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(SEASON_COLORS[SEASON_ORDER[i]])
        patch.set_alpha(0.7)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1.2)

    ax.set_ylabel(COL_NAMES.get(col, col), fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

axes[1][2].axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(FIG_DIR, 'fig2_seasonal_boxplots.png'), dpi=300, bbox_inches='tight')
plt.close()

# ─── Figure 3: Correlation Heatmap ───────────────────────────────────────────
print("Generating Figure 3: Correlation Heatmap...")
corr = df[NUMERIC_COLS].corr(method='pearson')
pvals = pd.DataFrame(np.zeros_like(corr), index=corr.index, columns=corr.columns)
for i in range(len(NUMERIC_COLS)):
    for j in range(len(NUMERIC_COLS)):
        if i != j:
            _, p = stats.pearsonr(df[NUMERIC_COLS[i]], df[NUMERIC_COLS[j]])
            pvals.iloc[i, j] = p

fig, ax = plt.subplots(figsize=(8, 6.5))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
labels_short = [COL_NAMES_SHORT.get(c, c) for c in NUMERIC_COLS]

# Create annotation with significance stars
annot = np.empty_like(corr, dtype=object)
for i in range(len(NUMERIC_COLS)):
    for j in range(len(NUMERIC_COLS)):
        r = corr.iloc[i, j]
        p = pvals.iloc[i, j]
        if i == j:
            annot[i, j] = '1.00'
        else:
            stars = ''
            if p < 0.001: stars = '***'
            elif p < 0.01: stars = '**'
            elif p < 0.05: stars = '*'
            annot[i, j] = f'{r:.2f}{stars}'

sns.heatmap(corr, mask=mask, annot=annot, fmt='', cmap=CORR_CMAP,
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=1, linecolor='white',
            xticklabels=labels_short, yticklabels=labels_short,
            ax=ax, cbar_kws={'shrink': 0.8, 'label': "Pearson's r"})

ax.set_title('Pearson Correlation Matrix with Significance Levels',
             fontsize=13, fontweight='bold', pad=15)
ax.tick_params(labelsize=10)

# Add significance legend
ax.text(0.98, -0.02, '* p<0.05  ** p<0.01  *** p<0.001',
        transform=ax.transAxes, ha='right', fontsize=8, style='italic',
        color='#666666')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# ─── Figure 4: Scatter Plot Matrix (Predictors vs PM2.5) ─────────────────────
print("Generating Figure 4: Scatter Plots (Predictors vs PM2.5)...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Relationship between PM2.5 and Predictor Variables',
             fontsize=14, fontweight='bold', y=1.05)

for idx, pred in enumerate(PREDICTORS):
    ax = axes[idx]
    for season in SEASON_ORDER:
        mask = df['season'] == season
        ax.scatter(df.loc[mask, pred], df.loc[mask, 'pm25'],
                   c=SEASON_COLORS[season], label=SEASON_NAMES[season],
                   alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

    # Regression line
    z = np.polyfit(df[pred], df['pm25'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[pred].min(), df[pred].max(), 100)
    ax.plot(x_line, p(x_line), color='#E74C3C', linewidth=1.5, linestyle='--')

    # Pearson r
    r, pval = stats.pearsonr(df[pred], df['pm25'])
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {pval:.4f}' if pval >= 0.001 else f'r = {r:.3f}\np < 0.001',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#CCCCCC'))

    ax.set_xlabel(COL_NAMES.get(pred, pred), fontsize=10)
    if idx == 0:
        ax.set_ylabel(COL_NAMES['pm25'], fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_scatter_pm25_predictors.png'), dpi=300, bbox_inches='tight')
plt.close()

# ─── Figure 5: Seasonal Comparison Bar Charts ────────────────────────────────
print("Generating Figure 5: Seasonal Means Comparison...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Seasonal Mean Comparison with 95% Confidence Intervals',
             fontsize=14, fontweight='bold', y=0.98)

for idx, col in enumerate(NUMERIC_COLS):
    ax = axes[idx // 3][idx % 3]
    means = []
    sems = []
    for season in SEASON_ORDER:
        data = df[df['season'] == season][col]
        means.append(data.mean())
        sems.append(data.sem())

    x = np.arange(len(SEASON_ORDER))
    bars = ax.bar(x, means, yerr=[1.96 * s for s in sems],
                  color=[SEASON_COLORS[s] for s in SEASON_ORDER],
                  alpha=0.8, edgecolor='#333333', linewidth=0.8,
                  capsize=4, error_kw={'linewidth': 1.2, 'capthick': 1.2},
                  width=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}\n({SEASON_NAMES[s]})' for s in SEASON_ORDER], fontsize=9)
    ax.set_ylabel(COL_NAMES.get(col, col), fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{m:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

axes[1][2].axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(FIG_DIR, 'fig5_seasonal_means.png'), dpi=300, bbox_inches='tight')
plt.close()

# ─── Figure 6: PM2.5 Time Trend with Season Coloring ────────────────────────
print("Generating Figure 6: PM2.5 Ordered Values...")
fig, ax = plt.subplots(figsize=(12, 4.5))

# Sort by PM2.5
df_sorted = df.sort_values('pm25').reset_index(drop=True)
colors = [SEASON_COLORS[s] for s in df_sorted['season']]

ax.bar(range(len(df_sorted)), df_sorted['pm25'], color=colors,
       alpha=0.85, edgecolor='white', linewidth=0.3, width=1.0)

# WHO guideline
ax.axhline(y=25, color='#E74C3C', linestyle='--', linewidth=1.5,
           label='WHO Guideline (25 μg/m³)')

# Add median line
median_pm25 = df['pm25'].median()
ax.axhline(y=median_pm25, color='#2C3E50', linestyle=':', linewidth=1.2,
           label=f'Median ({median_pm25:.1f} μg/m³)')

# Legend for seasons
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=SEASON_COLORS[s], alpha=0.85, label=f'{SEASON_NAMES[s]} ({s})')
                   for s in SEASON_ORDER]
legend_elements.append(plt.Line2D([0], [0], color='#E74C3C', linestyle='--', linewidth=1.5, label='WHO Guideline'))
legend_elements.append(plt.Line2D([0], [0], color='#2C3E50', linestyle=':', linewidth=1.2, label=f'Median'))

ax.legend(handles=legend_elements, fontsize=9, loc='upper left', ncol=2)
ax.set_xlabel('Observation Rank (sorted by PM2.5)', fontsize=11)
ax.set_ylabel('PM2.5 (μg/m³)', fontsize=11)
ax.set_title('PM2.5 Distribution Ranked by Concentration', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig6_pm25_ranked.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ All 6 basic figures saved to outputs/figures/")
print("  fig1_distributions.png")
print("  fig2_seasonal_boxplots.png")
print("  fig3_correlation_heatmap.png")
print("  fig4_scatter_pm25_predictors.png")
print("  fig5_seasonal_means.png")
print("  fig6_pm25_ranked.png")
